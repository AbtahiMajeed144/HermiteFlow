"""
Precompute the frozen-teacher trajectory targets for X4K1000FPS.

Stage 1's only objective is trajectory distillation against RAFT run on
the ground-truth middle frames (HermiteFlowBase.teacher_flows). That is
2*K frozen flow passes per optimizer step - measured at ~900 ms against
~660 ms for the entire rest of the step - so it is worth paying once,
offline, exactly as GIMM-VFI caches its Vimeo flows.

WHAT IS AND IS NOT BAKED IN

Cache the UNAUGMENTED sample and augment at read time. The augmentations
in X4KMultiT._augment are all exact transforms of a flow field, so the
loader can apply them to the images and the cached flows together:

    horizontal flip   flip cols, negate u
    vertical flip     flip rows, negate v
    rot90 / 180 / 270 rotate the field, permute and sign the components
    channel reverse   nothing - RGB->BGR does not move anything
    time reversal     free: both directions are cached, so it is just
                      t -> 1-t with flow_0_t/flow_1_t swapped and k
                      reversed
    random crop       already a no-op at downsample 3.0 (768/3 = 256)

Caching one pre-augmented draw per `repeat` instead would multiply the
disk by that factor to freeze that many particular augmentations
forever - which is what the first version of this script did, at 8x,
and why it did not fit.

The one thing that genuinely cannot move online is the random 32-frame
window (33 starts out of 65): the flow depends on which frames were
picked. This uses the deterministic centre window, via split="test".

ALL SEVEN TIMESTEPS

Recovering (A, B) from samples of Phi(t) is ill-conditioned, and how
badly depends on which t are drawn - 18.6x amplification at K=7 versus
26.5x at K=5 (see the X4KMultiT docstring). Caching the whole k/8 grid
costs 1.4x the flow bytes and makes K a training-time choice: the
loader picks any subset it likes without regenerating anything.

    python scripts/generate_offline_gt.py \
        -m configs/hermiteflow/hermiteflow_r_x4k_stage1.yaml \
        --output-dir /kaggle/working/offline_gt \
        --split-idx 0 --num-splits 1 \
        --data-path /kaggle/input/.../x4k1000fps/encoded_train \
        --raft-ckpt /kaggle/input/.../raft-things.pth \
        --batch-size 32

Resumable: rerun the same command after a session timeout and it picks
up where it stopped. Indexing is deterministic, so sample N is the same
clip and window in every session.
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import cv2

# Add src to pythonpath so it can resolve models/datasets/etc. before global packages
sys.path.insert(0, os.path.abspath("src"))

from models import create_model
from datasets.x4k_multi_t import X4KMultiT
from utils.utils import set_seed
from utils.setup import single_setup
from trainers.trainer_hermiteflow import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Offline GT for X4K1000FPS")
    parser.add_argument(
        "-m", "--model-config", type=str,
        default="configs/hermiteflow/hermiteflow_r_x4k_stage1.yaml",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-idx", type=int, default=0, help="0 .. num-splits-1")
    parser.add_argument("--num-splits", type=int, default=1)
    # Default None so `experiment.batch_size=N` on the command line works
    # too: an explicit --batch-size wins, otherwise the config decides.
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("-r", "--result-path", type=str, default="./results.tmp")
    parser.add_argument("-l", "--load-path", type=str, default="")
    parser.add_argument("-p", "--postfix", type=str, default="")

    # Overrides matching main.py
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--raft-ckpt", type=str, default=None)
    parser.add_argument("--flowformer-ckpt", type=str, default=None)

    args, extra_args = parser.parse_known_args()
    return args, extra_args


def already_done(split_dir, index):
    prefix = os.path.join(split_dir, f"sample_{index:06d}")
    return all(
        os.path.exists(prefix + suffix)
        for suffix in ("_img0.png", "_img1.png", "_flow.npz")
    )


def main():
    args, extra_args = parse_args()
    set_seed(args.seed)

    # single_setup requires args.eval to be True
    args.eval = True

    # Setup configuration (works for non-ddp scripts)
    config = single_setup(args, extra_args)
    # config_setup DISCARDS extra_args on the args.eval branch - only the
    # training branch merges the dotlist (src/utils/config.py:153). We set
    # args.eval above purely to satisfy an assertion, so without this every
    # `key=value` on the command line would be silently ignored: a run
    # launched with `loss.teacher_raft_iter=6` would quietly cache at
    # whatever the yaml said instead. Merge them here rather than changing
    # the shared eval path that the benchmarks also use.
    if extra_args:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(extra_args)))
        print("overrides:", " ".join(extra_args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(config.experiment.batch_size)
    )
    num_divisions = int(config.dataset.get("num_divisions", 8))

    # split="test" is what makes this deterministic and therefore
    # resumable: a fixed centre window, the full k/8 grid, a centre crop
    # and no augmentation. `repeat` is forced to 1 for a non-train split,
    # so this is exactly one entry per clip.
    print("Indexing clips (deterministic, unaugmented)...")
    dataset = X4KMultiT(
        "test",
        config.dataset.path,
        num_timesteps=num_divisions - 1,
        aug=False,
        crop_size=config.dataset.get("crop_size", 256),
        frame_gap=config.dataset.get("frame_gap", 32),
        num_divisions=num_divisions,
        clip_length=config.dataset.get("clip_length", 65),
        source=config.dataset.get("source", "auto"),
        downsample=config.dataset.get("downsample", 1.0),
    )

    split_dir = os.path.join(args.output_dir, f"split_{args.split_idx}")
    os.makedirs(split_dir, exist_ok=True)

    indices = np.array_split(np.arange(len(dataset)), args.num_splits)[args.split_idx]
    # Sample i of this split always comes from the same clip and window,
    # so a finished sample never needs recomputing.
    todo = [
        (out, int(src))
        for out, src in enumerate(indices)
        if not (args.resume and already_done(split_dir, out))
    ]
    done = len(indices) - len(todo)

    print(
        f"Split {args.split_idx}/{args.num_splits}: {len(indices)} samples "
        f"({done} already on disk, {len(todo)} to go), K={num_divisions - 1}, "
        f"batch {batch_size}"
    )
    if not todo:
        print("Nothing to do.")
        return

    loader = DataLoader(
        Subset(dataset, [src for _, src in todo]),
        batch_size=batch_size,
        shuffle=False,  # must stay False: output index is position in `todo`
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("Loading Teacher Network (RAFT/FlowFormer)...")
    model, _ = create_model(config.arch, ema=False)
    model = model.to(device).eval()

    teacher_raft_iter = int(getattr(config.loss, "teacher_raft_iter", 20))

    cursor = 0
    written = 0
    skipped = []
    spatial = None
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"split {args.split_idx}"):
            # img_xs: (B, 3, 2, H, W), gts: list of K (B, 3, H, W)
            img_xs, gts, t_list = Trainer.unpack(batch, device)

            if spatial is None:
                spatial = tuple(img_xs.shape[-2:])
                # RAFT correlates on a H/8 grid and returns all-NaN once
                # that grid gets too small - measured finite at 128x128
                # and 256x256, all-NaN at 64x64. Checking the EFFECTIVE
                # size catches a downsample/crop_size combination that
                # silently shrank the frames, which is not visible from
                # the config alone.
                if min(spatial) < 128:
                    raise SystemExit(
                        f"frames are {spatial[0]}x{spatial[1]} after "
                        f"downsample={config.dataset.get('downsample', 1.0)} and "
                        f"crop_size={config.dataset.get('crop_size', 256)}. RAFT "
                        f"returns NaN below ~128 px; lower dataset.downsample."
                    )

            # f_{0->t} and f_{1->t} from the frozen estimator run on the
            # ground-truth middle frame - the privileged teacher.
            target_flows = model.teacher_flows(
                img_xs[:, :, 0], img_xs[:, :, 1], gts, iters=teacher_raft_iter
            )
            phi = torch.stack([pair[0] for pair in target_flows], dim=1)
            psi = torch.stack([pair[1] for pair in target_flows], dim=1)
            times = torch.stack(t_list, dim=1)  # (B, K)
            # A NaN written here is permanent and silent: it would poison
            # every epoch that ever reads this cache, and the trainer's
            # own non-finite guard would just skip those micro-batches
            # forever without saying why. Cheaper to drop the sample. The
            # loader indexes by glob, so a hole costs nothing.
            ok = (
                torch.isfinite(phi).flatten(1).all(dim=1)
                & torch.isfinite(psi).flatten(1).all(dim=1)
            )

            for b in range(img_xs.shape[0]):
                out_idx = todo[cursor][0]
                cursor += 1
                if not bool(ok[b]):
                    skipped.append(out_idx)
                    continue
                prefix = os.path.join(split_dir, f"sample_{out_idx:06d}")

                # Endpoints only. The middle frames are what PRODUCED the
                # flows above, and stage 1 has no image term - compute_loss
                # returns before `gts` is touched - so storing them would
                # be 5/7 of the PNG bytes for something never read again.
                for name, img in (("img0", img_xs[b, :, 0]), ("img1", img_xs[b, :, 1])):
                    rgb = img.permute(1, 2, 0).cpu().numpy() * 255.0
                    cv2.imwrite(prefix + f"_{name}.png", rgb[:, :, ::-1].astype(np.uint8))

                # One file for the whole grid, not one per timestep: 3
                # files per sample instead of 2 + 2K, the difference
                # between 13k and 423k files in the output directory.
                # Uncompressed on purpose - zlib on float16 flow measured
                # 460 vs 512 KiB, a 10% saving paid for with main-thread
                # CPU time, which is the actual bottleneck here.
                np.savez(
                    prefix + "_flow.npz",
                    flow_0_t=phi[b].cpu().numpy().astype(np.float16),  # (K, 2, H, W)
                    flow_1_t=psi[b].cpu().numpy().astype(np.float16),  # (K, 2, H, W)
                    t=times[b].cpu().numpy().astype(np.float32),       # (K,)
                )
                written += 1

    # Everything the loader needs to validate the cache instead of
    # guessing at it, so a cache paired with a config it was not
    # generated for fails loudly rather than training on the wrong grid.
    manifest = {
        "num_samples": len(indices),
        "num_written": written + done,
        "num_timesteps": num_divisions - 1,
        "num_divisions": num_divisions,
        "frame_gap": int(config.dataset.get("frame_gap", 32)),
        "downsample": float(config.dataset.get("downsample", 1.0)),
        # The EFFECTIVE size, not config.crop_size. _crop_extent takes
        # min(crop_size, H, W) and rounds down to a multiple of 8, so the
        # two disagree whenever downsample leaves the frame smaller than
        # the crop - and the loader needs the one that is actually on disk.
        "height": None if spatial is None else int(spatial[0]),
        "width": None if spatial is None else int(spatial[1]),
        "crop_size": int(config.dataset.get("crop_size", 256)),
        "teacher_raft_iter": teacher_raft_iter,
        "augmented": False,
        "split_idx": args.split_idx,
        "num_splits": args.num_splits,
        "skipped_nonfinite": skipped,
    }
    with open(os.path.join(split_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    if skipped:
        print(f"WARNING: dropped {len(skipped)} samples with non-finite teacher flow")
    print(f"Wrote {written} samples ({written + done} on disk) to {split_dir}")


if __name__ == "__main__":
    main()
