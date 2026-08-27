# --------------------------------------------------------
# HermiteFlow — optimizer factory
# Adapted from GIMM-VFI's optimizer.py
#
# Everything trainable is trained: CoeffNet (Phase 2), RefineNet
# (Phase 4) and SynthNet (Phase 5). The flow estimator (Phase 1) is
# frozen and Phase 3 has no parameters, so both are absent from the
# parameter list by construction.
#
# `optimizer.ft: true` reproduces GIMM-VFI's stage-2 grouping. There,
# the freshly-initialised synthesis module (`amt_*`) trains at the full
# learning rate while the *pre-trained* motion module fine-tunes at
# 0.01x with 0.01x weight decay, so stage-1 knowledge is not destroyed
# in the first few hundred steps. The same split applies here: Phase 5
# is new in stage 2, Phases 2 and 4 arrive pre-trained.
# --------------------------------------------------------

import torch

# Modules that arrive pre-trained from stage 1 and should therefore be
# fine-tuned gently rather than retrained.
PRETRAINED_PREFIXES = ("coeff_net", "flow_reversal")


def _split_params(model):
    fresh, pretrained = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(PRETRAINED_PREFIXES):
            pretrained.append(param)
        else:
            fresh.append(param)
    return fresh, pretrained


def create_hermiteflow_optimizer(model, config):
    optimizer_type = config.type.lower()
    finetune = bool(getattr(config, "ft", False))

    if finetune:
        fresh, pretrained = _split_params(model)
        param_dicts = []
        if fresh:
            param_dicts.append({"params": fresh})
        if pretrained:
            param_dicts.append(
                {
                    "params": pretrained,
                    "lr": config.init_lr * 0.01,
                    "weight_decay": config.weight_decay * 0.01,
                }
            )
        if not param_dicts:
            raise ValueError("no trainable parameters found")
    else:
        param_dicts = [p for p in model.parameters() if p.requires_grad]
        if len(param_dicts) == 0:
            raise ValueError("no trainable parameters found")

    kwargs = dict(lr=config.init_lr, weight_decay=config.weight_decay)
    if optimizer_type in ("adamw", "adam"):
        kwargs["betas"] = tuple(config.betas)

    if optimizer_type == "adamw":
        return torch.optim.AdamW(param_dicts, **kwargs)
    if optimizer_type == "adam":
        return torch.optim.Adam(param_dicts, **kwargs)
    if optimizer_type == "sgd":
        return torch.optim.SGD(param_dicts, momentum=0.9, **kwargs)

    raise ValueError(f"{optimizer_type} invalid..")


def create_inr_optimizer(model, config):
    """
    GIMM-VFI's own optimizer grouping, vendored verbatim from its
    optimizer.py: `optimizer.ft: true` splits on the "amt_" prefix
    (their synthesis decoder's own naming) instead of this project's
    PRETRAINED_PREFIXES, since that is what their state_dict actually
    uses - HermiteFlow's split and GIMM's split are the same IDEA
    (fresh module at full LR, pretrained motion module fine-tuned
    gently) applied to two differently-named module trees.
    """
    optimizer_type = config.type.lower()
    if not config.ft:
        param_dicts = model.parameters()
    else:
        param_dicts = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "amt_" in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if "amt_" not in n and p.requires_grad
                ],
                "lr": config.init_lr * 0.01,
                "weight_decay": config.weight_decay * 0.01,
            },
        ]
        if len(param_dicts[1]["params"]) == 0:
            print("only amt_part will be trained")

    kwargs = dict(lr=config.init_lr, weight_decay=config.weight_decay)
    if optimizer_type in ("adamw", "adam"):
        kwargs["betas"] = tuple(config.betas)

    if optimizer_type == "adamw":
        return torch.optim.AdamW(param_dicts, **kwargs)
    if optimizer_type == "adam":
        return torch.optim.Adam(param_dicts, **kwargs)
    if optimizer_type == "sgd":
        return torch.optim.SGD(param_dicts, momentum=0.9, **kwargs)

    raise ValueError(f"{optimizer_type} invalid..")


def create_optimizer(model, config):
    arch_type = config.arch.type.lower()
    if "hermite" in arch_type:
        return create_hermiteflow_optimizer(model, config.optimizer)
    if "inr" in arch_type or "dnn" in arch_type or "gimm" in arch_type:
        return create_inr_optimizer(model, config.optimizer)
    raise ValueError(f"{arch_type} invalid..")
