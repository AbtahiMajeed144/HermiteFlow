"""
Is the training data actually trackable at this frame_gap and crop_size?

`frame_gap: 32` comes from X-TEST, where it is applied to 4K frames
(4096x2160). We apply it to a 256x256 crop of a 768x768 X-TRAIN clip.
The temporal distance is the same; the SPATIAL context is ~16x smaller.
If objects move further than the crop, RAFT has nothing to match, the
forward-backward error explodes, the occlusion gate closes, and the
distillation targets are noise - which looks exactly like "the model
will not converge".

This script measures that directly, with no training involved.

    python others/inspect_x4k_motion.py \
        --data-path /kaggle/input/.../x4k1000fps/encoded_train \
        --raft-ckpt /kaggle/input/.../raft-things.pth \
        --frame-gaps 4 8 16 32 --crop-size 256 --num-samples 24

Read the output as:
  |F| p95        typical large displacement, in pixels
  out-of-crop    fraction of pixels displaced further than crop/2;
                 above a few percent, most of the crop has no
                 correspondence inside the crop at all
  U/s            forward-backward inconsistency, normalised
  gate           sigmoid(5 - 20*U/s) - the value Phase 2 will multiply
                 the curvature by. Near 0 means the model is being told
                 to ignore its own flow everywhere.
  curvature      dB by which the true midpoint flow departs from the
                 linear guess t*F. This is the HEADROOM: with no
                 curvature in the data there is nothing for d0, d1 to
                 learn, no matter how long you train.
"""

import argparse
import os
import sys

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from datasets.x4k_multi_t import X4KMultiT  # noqa: E402
from models.hermite_vfi.modules.phase1_measure import (  # noqa: E402
    flow_scale,
    forward_backward_error,
)
from models.hermite_vfi.raft import initialize_RAFT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--raft-ckpt", default="pretrained/raft-things.pth")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--num-samples", type=int, default=24)
    p.add_argument("--raft-iter", type=int, default=20)
    p.add_argument(
        "--frame-gaps", type=int, nargs="+", default=[4, 8, 16, 32],
        help="must each be divisible by --num-divisions",
    )
    p.add_argument("--num-divisions", type=int, default=8)
    p.add_argument("--downsample", type=float, default=1.0,
                   help="shrink each frame by this factor before cropping")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raft = initialize_RAFT(args.raft_ckpt).to(device).eval()
    for prm in raft.parameters():
        prm.requires_grad = False

    def flow(a, b):
        out = raft(255.0 * a, 255.0 * b, return_feat=True, iters=args.raft_iter)
        return out[0]

    half = args.crop_size / 2
    print(f"\ncrop {args.crop_size}x{args.crop_size}, {args.num_samples} samples per gap, "
          f"RAFT iters {args.raft_iter}")
    print(f"\n{'gap':>5}{'|F| mean':>10}{'|F| p95':>9}{'|F| max':>9}"
          f"{'out-of-crop':>13}{'U/s':>8}{'gate':>7}{'curvature':>11}")

    for gap in args.frame_gaps:
        torch.manual_seed(args.seed)
        ds = X4KMultiT(
            "train", args.data_path, num_timesteps=1, aug=True,
            crop_size=args.crop_size, frame_gap=gap,
            num_divisions=args.num_divisions, downsample=args.downsample,
        )
        stats = {k: [] for k in ("mean", "p95", "max", "oob", "us", "gate", "curv")}

        for i in range(args.num_samples):
            item = ds[i % len(ds)]
            xs = item["xs"].unsqueeze(0).to(device)
            t_val = float(item["t"][0])
            img0, img1, gt = xs[:, :, 0], xs[:, :, 1], xs[:, :, 2]

            with torch.no_grad():
                f01 = flow(img0, img1)
                f10 = flow(img1, img0)
                mag = f01.norm(dim=1)
                occ = forward_backward_error(f01, f10)
                s = flow_scale(f01, f10)
                us = (occ / s).mean()
                gate = torch.sigmoid(5.0 - 20.0 * us)

                # How far is the TRUE midpoint flow from the linear guess?
                f0t = flow(img0, gt)
                linear = t_val * f01
                err_lin = ((f0t - linear) / s).pow(2).mean()
                ref = (f0t / s).pow(2).mean()
                curv = 10 * torch.log10((ref / err_lin.clamp_min(1e-12)).clamp_min(1e-12))

            stats["mean"].append(mag.mean().item())
            stats["p95"].append(mag.flatten().quantile(0.95).item())
            stats["max"].append(mag.max().item())
            stats["oob"].append((mag > half).float().mean().item())
            stats["us"].append(us.item())
            stats["gate"].append(gate.item())
            stats["curv"].append(curv.item())

        m = {k: sum(v) / len(v) for k, v in stats.items()}
        warn = ""
        if m["oob"] > 0.05:
            warn = "  <-- motion leaves the crop"
        elif m["gate"] < 0.5:
            warn = "  <-- gate mostly shut"
        print(f"{gap:>5}{m['mean']:>10.1f}{m['p95']:>9.1f}{m['max']:>9.1f}"
              f"{m['oob']:>12.1%}{m['us']:>8.3f}{m['gate']:>7.3f}"
              f"{m['curv']:>10.1f}dB{warn}")

    print("\ncurvature = how much signal there is for d0/d1 to capture.")
    print("  high dB  -> the linear guess is already right, nothing to learn")
    print("  low dB   -> the motion genuinely bends; this is where the model wins")
    print("Pick the largest gap that keeps out-of-crop small and the gate open.")


if __name__ == "__main__":
    main()
