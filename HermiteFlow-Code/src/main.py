# --------------------------------------------------------
# HermiteFlow-VFI — Main entry point
# Adapted from GIMM-VFI's main.py
# --------------------------------------------------------

import ctypes

try:
    # Inherited from GIMM-VFI: preloading libgcc works around a glibc
    # threading issue on some Linux images. Absent on Windows/macOS.
    libgcc_s = ctypes.CDLL("libgcc_s.so.1")
except OSError:
    libgcc_s = None

import argparse
import math
import os
import sys

import torch
import torch.distributed as dist

import utils.dist as dist_utils

from models import create_model
from trainers import create_trainer
from datasets import create_dataset
from optimizer import create_optimizer, create_scheduler
from utils.utils import set_seed
from utils.profiler import Profiler
from utils.setup import setup


def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-config",
        type=str,
        default="./configs/hermiteflow/hermiteflow_r.yaml",
    )
    parser.add_argument("-r", "--result-path", type=str, default="./results.tmp")
    parser.add_argument("-l", "--load-path", type=str, default="")
    parser.add_argument("-p", "--postfix", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true")

    # Paths that are environment-specific and therefore overridable from
    # the command line. Each one wins over the value in the YAML config.
    # Equivalent dotlist overrides (e.g. `dataset.path=/data/...`) also
    # work and can be passed as trailing arguments.
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="dataset root; overrides dataset.path",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="validation data root; overrides dataset.val_path "
        "(X4K keeps train and val in separate trees)",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=None,
        help="ground-truth middle frames supervised per clip; "
        "overrides dataset.num_timesteps",
    )
    parser.add_argument(
        "--raft-ckpt",
        type=str,
        default=None,
        help="RAFT weights; overrides arch.pretrained_raft_ckpt",
    )
    parser.add_argument(
        "--flowformer-ckpt",
        type=str,
        default=None,
        help="FlowFormer weights; overrides arch.pretrained_flowformer_ckpt",
    )
    return parser


def add_dist_arguments(parser):
    parser.add_argument(
        "--world_size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="local rank for distributed training"
    )
    parser.add_argument(
        "--node_rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--nnodes", default=-1, type=int)
    parser.add_argument(
        "--nproc_per_node",
        default=-1,
        type=int,
        help="GPUs to use. Greater than 1 re-launches this script once "
        "per GPU, so a plain `python src/main.py` drives DDP without "
        "torchrun; harmless under torchrun, which sets WORLD_SIZE itself",
    )
    parser.add_argument(
        "--master-port",
        "--master_port",
        dest="master_port",
        default=16890,
        type=int,
        help="rendezvous port for the self-launched workers",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=86400,
        help="time limit (s) to wait for other nodes in DDP",
    )
    return parser


def parse_args():
    parser = default_parser()
    parser = add_dist_arguments(parser)
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def spawn_workers(args):
    """
    Re-launch this script once per GPU with the env torchrun would set.

    dist_utils.initialize reads RANK, WORLD_SIZE and LOCAL_RANK and
    nothing else, so exporting those three is the whole of what torchrun
    was doing here. Without this, `python src/main.py --nproc_per_node=2`
    quietly trains on one GPU - the flag was parsed and never used.
    """
    import subprocess
    import time

    env = os.environ.copy()
    env["WORLD_SIZE"] = str(args.nproc_per_node)
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", str(args.master_port))

    workers = []
    failure = 1
    try:
        for rank in range(args.nproc_per_node):
            workers.append(subprocess.Popen(
                [sys.executable] + sys.argv,
                env={**env, "RANK": str(rank), "LOCAL_RANK": str(rank)},
            ))
        while True:
            codes = [w.poll() for w in workers]
            # Poll rather than wait() in order: a rank that dies leaves
            # its peers blocked forever on an all-reduce that will never
            # arrive, and waiting on rank 0 first would not notice. On a
            # time-limited session that hang costs the rest of the run.
            bad = [(i, c) for i, c in enumerate(codes) if c not in (None, 0)]
            if bad:
                # Record the real exit code here, before the peers are
                # terminated below - their signal codes are also non-zero
                # and would otherwise mask which rank actually failed.
                rank, failure = bad[0]
                print(f"[dist] rank {rank} exited with {failure}; "
                      f"stopping {len(workers) - 1} peer(s)")
                break
            if all(c is not None for c in codes):
                return 0
            time.sleep(2.0)
    except KeyboardInterrupt:
        failure = 130
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.kill()
    return failure


if __name__ == "__main__":
    args, extra_args = parse_args()
    # Only the outermost invocation spawns: torchrun and our own workers
    # both arrive with WORLD_SIZE already in the environment.
    if args.nproc_per_node > 1 and "WORLD_SIZE" not in os.environ:
        sys.exit(spawn_workers(args))
    set_seed(args.seed)
    config, logger, writer = setup(args, extra_args)
    distenv = config.runtime.distenv
    profiler = Profiler(logger)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", distenv.local_rank)
    torch.cuda.set_device(device)

    dataset_trn, dataset_val = create_dataset(config, is_eval=args.eval, logger=logger)

    model, model_ema = create_model(config.arch, ema=config.arch.ema is not None)
    model = model.to(device)
    model_ema = model_ema.to(device) if model_ema is not None else None

    if distenv.master:
        print(model)
        profiler.get_model_size(model)
        profiler.get_model_size(model, opt="trainable-only")
        profiler.get_module_breakdown(model)

    # Checkpoint loading
    if not args.load_path == "":
        ckpt = torch.load(args.load_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        if model_ema is not None:
            if "state_dict_ema" in ckpt.keys():
                ckpt["state_dict_ema"] = {
                    "module." + k: v for k, v in ckpt["state_dict_ema"].items()
                }
                model_ema.load_state_dict(ckpt["state_dict_ema"])
            else:
                model_ema.load_state_dict(ckpt["state_dict"], strict=False)

        if distenv.master:
            logger.info(f"{args.load_path} model is loaded")
    else:
        ckpt = None
        if args.eval or args.resume:
            raise ValueError(
                "--load-path must be specified in evaluation or resume mode"
            )

    # Optimizer definition
    if args.eval:
        optimizer, scheduler, epoch_st = None, None, None
    else:
        steps_per_epoch = math.ceil(
            len(dataset_trn) / (config.experiment.batch_size * distenv.world_size)
        )
        steps_per_epoch = steps_per_epoch // config.optimizer.grad_accm_steps

        # `optimizer.ft` is GIMM-VFI's stage-2 grouping: it drops the
        # PRE-TRAINED modules to 0.01x lr so a stage-1 solution is not
        # destroyed. With no stage-1 checkpoint to protect there is
        # nothing pre-trained, and the only effect is to train CoeffNet
        # and RefineNet 100x too slowly - which looks exactly like the
        # curvature failing to learn, only after hours of wall clock.
        if config.optimizer.get("ft", False) and not args.load_path:
            raise ValueError(
                "optimizer.ft=true fine-tunes the motion modules at 0.01x lr, "
                "but no --load-path was given, so there is nothing pre-trained "
                "to protect. For stage 2 from a stage-1 checkpoint, pass "
                "--load-path <stage1-run>/epochN_model.pt ; to train from "
                "scratch in a single stage, pass optimizer.ft=false"
            )

        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(
            optimizer,
            config.optimizer.warmup,
            steps_per_epoch,
            config.experiment.epochs,
            distenv,
        )

        if distenv.master:
            print(optimizer)

        if args.resume:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            epoch_st = ckpt["epoch"]

            if distenv.master:
                logger.info(f"Optimizer, scheduler, and epoch is resumed")
                logger.info(f"resuming from {epoch_st}..")
        else:
            epoch_st = 0

    # Usual DDP setting
    static_graph = False
    model = dist_utils.dataparallel_and_sync(
        distenv, model, static_graph=static_graph, find_unused_parameters=False
    )
    if model_ema is not None:
        model_ema = dist_utils.dataparallel_and_sync(
            distenv, model_ema, static_graph=static_graph
        )

    trainer = create_trainer(config)
    trainer = trainer(
        model, model_ema, dataset_trn, dataset_val, config, writer, device, distenv
    )

    if distenv.master:
        logger.info(f"Trainer created. type: {trainer.__class__}")

    if args.eval:
        trainer.config.experiment.subsample_during_eval = False
        trainer.eval(valid=False, verbose=True)
        trainer.eval(valid=True, verbose=True)
        if model_ema is not None:
            trainer.eval(valid=True, ema=True, verbose=True)
    else:
        trainer.run_epoch(optimizer, scheduler, epoch_st)

    if dist.is_initialized():
        dist.barrier()

    if distenv.master:
        writer.close()
