# --------------------------------------------------------
# HermiteFlow-VFI — Config utilities
# Adapted from GIMM-VFI's config.py
# --------------------------------------------------------

from omegaconf import OmegaConf
from easydict import EasyDict as edict
import yaml

from models.hermite_vfi.configs import HermiteFlowConfig
import os.path as osp


def easydict_to_dict(obj):
    if not isinstance(obj, edict):
        return obj
    else:
        return {k: easydict_to_dict(v) for k, v in obj.items()}


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = easydict_to_dict(config)
        config = OmegaConf.create(config)
    return config


def augment_arch_defaults(arch_config):
    if arch_config.type in ["hermiteflow_r", "hermiteflow_f"]:
        arch_defaults = HermiteFlowConfig.create(arch_config)
    else:
        raise ValueError(f"{arch_config.type} is not implemented for default arguments")

    return OmegaConf.merge(arch_defaults, arch_config)


def augment_optimizer_defaults(optim_config):
    defaults = OmegaConf.create(
        {
            "type": "adamW",
            "max_gn": None,
            "warmup": {
                "mode": "linear",
                "start_from_zero": (True if optim_config.warmup.epoch > 0 else False),
            },
        }
    )
    return OmegaConf.merge(defaults, optim_config)


def augment_defaults(config):
    defaults = OmegaConf.create(
        {
            "arch": augment_arch_defaults(config.arch),
            "dataset": {
                "transforms": {"type": None},
            },
            "optimizer": augment_optimizer_defaults(config.optimizer),
            "experiment": {
                "test_freq": 10,
                "amp": False,
            },
        }
    )

    if "hermite" in config.arch.type:
        subsample_defaults = OmegaConf.create({"type": None, "ratio": 1.0})
        loss_defaults = OmegaConf.create(
            {
                "loss": {
                    "type": "mse",
                    "subsample": subsample_defaults,
                    "coord_noise": None,
                    "perceptual_loss": False,
                }
            }
        )
        defaults = OmegaConf.merge(defaults, loss_defaults)
    config = OmegaConf.merge(defaults, config)
    return config


def augment_dist_defaults(config, distenv):
    config = config.copy()
    local_batch_size = config.experiment.batch_size
    world_batch_size = distenv.world_size * local_batch_size
    total_batch_size = config.experiment.get("total_batch_size", world_batch_size)

    if total_batch_size % world_batch_size != 0:
        # experiment.batch_size is PER PROCESS, so a config written for
        # one GPU stops fitting the moment it is launched on two. Say
        # what the numbers are and what to set - the bare assertion sent
        # people hunting through the config for a value that was fine.
        if total_batch_size % distenv.world_size == 0:
            fix = (
                f"set experiment.batch_size="
                f"{total_batch_size // distenv.world_size} to keep the same "
                f"effective batch of {total_batch_size}"
            )
        else:
            fix = (
                f"raise experiment.total_batch_size to a multiple of "
                f"{world_batch_size}"
            )
        raise ValueError(
            f"total_batch_size ({total_batch_size}) must be divisible by the "
            f"world batch size ({world_batch_size} = batch_size "
            f"{local_batch_size} x world_size {distenv.world_size}); {fix}"
        )
    else:
        grad_accm_steps = total_batch_size // world_batch_size

    config.optimizer.grad_accm_steps = grad_accm_steps
    config.experiment.total_batch_size = total_batch_size
    return config


# Command-line flags that override a config field wherever the config came
# from (fresh YAML, a resumed run, or a saved eval config). Paths are the
# only things that genuinely differ between machines, so they get first-class
# flags; anything else can still be set with a trailing dotlist argument.
CLI_OVERRIDES = {
    "data_path": "dataset.path",
    "val_path": "dataset.val_path",
    "num_timesteps": "dataset.num_timesteps",
    "raft_ckpt": "arch.pretrained_raft_ckpt",
    "flowformer_ckpt": "arch.pretrained_flowformer_ckpt",
}


def cli_override_dotlist(args):
    """Turn the path-style CLI flags into an OmegaConf dotlist."""
    dotlist = []
    for arg_name, config_key in CLI_OVERRIDES.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            dotlist.append(f"{config_key}={value}")
    return dotlist


def apply_cli_overrides(config, args):
    overrides = cli_override_dotlist(args)
    if not overrides:
        return config
    return OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))


def config_setup(args, distenv, config_path, extra_args=()):
    if not osp.isfile(config_path):
        config_path = args.model_config
    if args.eval:
        config = load_config(config_path)
        config = augment_defaults(config)
        if hasattr(args, "test_batch_size"):
            config.experiment.batch_size = args.test_batch_size
        if not hasattr(config, "seed"):
            config.seed = args.seed
        config = apply_cli_overrides(config, args)

    elif args.resume:
        config = load_config(config_path)
        if distenv.world_size != config.runtime.distenv.world_size:
            raise ValueError("world_size not identical to the resuming config")
        config = apply_cli_overrides(config, args)
        config.runtime = {"args": vars(args), "distenv": distenv}

    else:  # training
        config_path = args.model_config
        config = load_config(config_path)

        extra_config = OmegaConf.from_dotlist(
            list(extra_args) + cli_override_dotlist(args)
        )
        config = OmegaConf.merge(config, extra_config)

        config = augment_defaults(config)
        config = augment_dist_defaults(config, distenv)

        config.seed = args.seed
        config.runtime = {
            "args": vars(args),
            "extra_config": extra_config,
            "distenv": distenv,
        }

    return config
