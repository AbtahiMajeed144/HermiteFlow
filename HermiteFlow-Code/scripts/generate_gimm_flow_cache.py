"""
Precompute GIMM stage-1's flow targets for X4K1000FPS.

GIMM-VFI's own stage 1 (arch.type "gimm", trainer_gimm.py) is trained
against fast_vimeo_flow, which reads THREE flow observations per sample
from precomputed .flo files:

    F  = flow(im1 -> im3)                     "im1_im3.flo"
    M  = flow(im2 -> im3) - flow(im2 -> im1)   "im2_im3.flo" - "im2_im1.flo"
    F' = -flow(im3 -> im1)                     "-im3_im1.flo"

im1/im2/im3 are Vimeo's own naming for (first frame, middle frame, last
frame) - i.e. (I0, GT, I1). This script computes the same three
quantities for X4K, on the fly via RAFT (batched, on GPU) instead of
reading pre-baked files, and caches them - the trainer stays byte-for-
byte the vendored original either way, only the flow SOURCE differs.

trainer_gimm.py hardcodes its three supervised positions at t in
{0, 0.5, 1} (see its train()/eval(): `t_id = random.randint(0, 2)`,
`timesteps = 0.5 * t_id`) - it has no continuous-t path. A real Vimeo
triplet's middle frame is always at true t=0.5, which is the assumption
that formula bakes in, so this uses X4K's exact grid centre
(k = num_divisions // 2) as the middle frame - NOT a random k like
generate_offline_gt.py's K=7 grid - so the cached M really is the flow
to/from a frame at t=0.5, matching what the trainer thinks it is.

Deterministic and resumable via the same trick generate_offline_gt.py
uses: split="test" gives a fixed centre window and no augmentation, so
sample N is always the same clip; augmentation (flip/rotate) is
reapplied on read by X4KGimmFlowCache instead of being baked in.

    python scripts/generate_gimm_flow_cache.py \
        -m configs/hermiteflow/hermiteflow_r_x4k_stage1.yaml \
        --output-dir /kaggle/working/gimm_flow_cache \
        --split-idx 0 --num-splits 1 \
        --data-path /kaggle/input/.../x4k1000fps/encoded_train \
        --raft-ckpt /kaggle/input/.../raft-things.pth \
        --batch-size 32

Any hermiteflow_r_x4k*.yaml works as -m: only its dataset: block and
arch.pretrained_raft_ckpt/raft_iter are read, matching
generate_offline_gt.py's own convention - this does not build a
HermiteFlow model at all, just RAFT and the X4K dataset.
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

sys.path.insert(0, os.path.abspath("src"))

from datasets.x4k_multi_t import X4KMultiT
from models.hermite_vfi.raft import initialize_RAFT
from utils.utils import set_seed
from utils.setup import single_setup


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GIMM stage-1 flow cache for X4K1000FPS"
    )
    parser.add_argument(
        "-m", "--model-config", type=str,
        default="configs/hermiteflow/hermiteflow_r_x4k_stage1.yaml",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-idx", type=int, default=0, help="0 .. num-splits-1")
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--raft-iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("-r", "--result-path", type=str, default="./results.tmp")
    parser.add_argument("-l", "--load-path", type=str, default="")
    parser.add_argument("-p", "--postfix", type=str, default="")

    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--raft-ckpt", type=str, default=None)

    args, extra_args = parser.parse_known_args()
    return args, extra_args


def already_done(split_dir, index):
    return os.path.exists(os.path.join(split_dir, f"sample_{index:06d}_flow.npz"))


def main():
    args, extra_args = parse_args()
    set_seed(args.seed)

    # single_setup requires args.eval; config_setup's eval branch drops
    # extra_args (src/utils/config.py), so merge them ourselves - same
    # workaround generate_offline_gt.py uses, for the same reason.
    args.eval = True
    config = single_setup(args, extra_args)
    if extra_args:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(extra_args)))
        print("overrides:", " ".join(extra_args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_divisions = int(config.dataset.get("num_divisions", 8))
    raft_iter = args.raft_iter or int(getattr(config.arch, "raft_iter", 20))
    raft_ckpt = args.raft_ckpt or config.arch.pretrained_raft_ckpt

    print("Indexing clips (deterministic, unaugmented, grid centre only)...")
    dataset = X4KMultiT(
        "test",
        config.dataset.path,
        num_timesteps=1,  # _eval_steps()'s num_timesteps==1 case is
                           # exactly [num_divisions // 2] - X4K's true
                           # centre, t=0.5 - see the module docstring.
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
    todo = [
        (out, int(src))
        for out, src in enumerate(indices)
        if not (args.resume and already_done(split_dir, out))
    ]
    done = len(indices) - len(todo)

    print(
        f"Split {args.split_idx}/{args.num_splits}: {len(indices)} samples "
        f"({done} already on disk, {len(todo)} to go), t=0.5 only, "
        f"batch {args.batch_size}"
    )
    if not todo:
        print("Nothing to do.")
        return

    loader = DataLoader(
        Subset(dataset, [src for _, src in todo]),
        batch_size=args.batch_size,
        shuffle=False,  # must stay False: output index is position in `todo`
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Loading RAFT from {raft_ckpt} ...")
    raft = initialize_RAFT(raft_ckpt).to(device).eval()

    @torch.no_grad()
    def flow_once(img_a, img_b):
        flow, _feats, _fmap = raft(
            255.0 * img_a, 255.0 * img_b, return_feat=True, iters=raft_iter
        )
        return flow

    cursor = 0
    written = 0
    skipped = []
    spatial = None
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"split {args.split_idx}"):
            xs = batch["xs"].to(device, non_blocking=True)  # (B, 3, 3, H, W)
            assert xs.shape[2] == 3, (
                f"expected exactly one middle frame (num_timesteps=1), got "
                f"xs.shape[2]={xs.shape[2]}"
            )
            img0, img1, gt = xs[:, :, 0], xs[:, :, 1], xs[:, :, 2]

            if spatial is None:
                spatial = tuple(img0.shape[-2:])
                if min(spatial) < 128:
                    raise SystemExit(
                        f"frames are {spatial[0]}x{spatial[1]} after "
                        f"downsample={config.dataset.get('downsample', 1.0)} and "
                        f"crop_size={config.dataset.get('crop_size', 256)}. RAFT "
                        f"returns NaN below ~128 px; lower dataset.downsample."
                    )

            flow_f = flow_once(img0, img1)      # F  = RAFT(I0, I1)
            flow_b = flow_once(img1, img0)       # F' = RAFT(I1, I0)
            f_gt0 = flow_once(gt, img0)          # RAFT(Igt, I0)
            f_gt1 = flow_once(gt, img1)          # RAFT(Igt, I1)
            flow_m = f_gt1 - f_gt0               # M

            ok = (
                torch.isfinite(flow_f).flatten(1).all(dim=1)
                & torch.isfinite(flow_b).flatten(1).all(dim=1)
                & torch.isfinite(flow_m).flatten(1).all(dim=1)
            )

            for b in range(xs.shape[0]):
                out_idx = todo[cursor][0]
                cursor += 1
                if not bool(ok[b]):
                    skipped.append(out_idx)
                    continue
                prefix = os.path.join(split_dir, f"sample_{out_idx:06d}")
                np.savez(
                    prefix + "_flow.npz",
                    flow_f=flow_f[b].cpu().numpy().astype(np.float16),
                    flow_m=flow_m[b].cpu().numpy().astype(np.float16),
                    flow_b=flow_b[b].cpu().numpy().astype(np.float16),
                )
                written += 1

    manifest = {
        "num_samples": len(indices),
        "num_written": written + done,
        "num_divisions": num_divisions,
        "frame_gap": int(config.dataset.get("frame_gap", 32)),
        "downsample": float(config.dataset.get("downsample", 1.0)),
        "height": None if spatial is None else int(spatial[0]),
        "width": None if spatial is None else int(spatial[1]),
        "crop_size": int(config.dataset.get("crop_size", 256)),
        "raft_iter": raft_iter,
        "split_idx": args.split_idx,
        "num_splits": args.num_splits,
        "skipped_nonfinite": skipped,
    }
    with open(os.path.join(split_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    if skipped:
        print(f"WARNING: dropped {len(skipped)} samples with non-finite flow")
    print(f"Wrote {written} samples ({written + done} on disk) to {split_dir}")


if __name__ == "__main__":
    main()
