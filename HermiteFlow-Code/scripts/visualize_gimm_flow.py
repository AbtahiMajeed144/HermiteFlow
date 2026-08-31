"""
Paper-ready flow visualisation for a stage-1 GIMM motion checkpoint on X4K1000FPS.

`epoch31_model.pt.zip` is a stage-1 GIMM motion model (`arch.type: gimm`): a
flow-interpolation INR, not the full gimmvfi_r frame model. Given the two
endpoint flows F = RAFT(I0, I1) and F' = RAFT(I1, I0) it predicts a flow field
at any query time t in [0, 1]. It was trained against three anchor observations,
exactly as GIMM's own fast_vimeo_flow supplies them:

    t = 0   ->  F  = RAFT(I0, I1)
    t = 0.5 ->  M  = RAFT(I_gt, I1) - RAFT(I_gt, I0)      (the GT-frame anchor)
    t = 1   ->  F' = RAFT(I1, I0)

so a *ground-truth* flow is only well defined at those three anchors - that is
what this script draws GT for. At the in-between times (0.25, 0.75, ...) it
shows the prediction alone, which is the point: a continuous motion field
generated from two frames.

This reproduces the training-time inputs precisely (same "test"-split centre
window at true t = 0.5, same F/M/F' computation as
scripts/generate_gimm_flow_cache.py, same (flow/s + 1)/2 normalisation as
datasets/x4k_single_t.py's X4KGimmFlowCache), so what the model sees here is
byte-for-byte what it saw in training - no cache needed.

    python scripts/visualize_gimm_flow.py \
        -m configs/gimm/gimm_x4k.yaml \
        --ckpt   kaggle/epoch31_model.pt.zip \
        --data-path /kaggle/input/.../x4k1000fps/val \
        --raft-ckpt /kaggle/input/.../raft-things.pth \
        --output-dir /kaggle/working/gimm_flow_figs \
        --num-samples 6 --timesteps 0,0.25,0.5,0.75,1.0

Needs a CUDA GPU with the CUDA toolkit headers present (GIMM's softsplat kernel
compiles at first call via cupy) - Kaggle's GPU images have this; a laptop with
only the PyTorch runtime does not.
"""

import sys
import os
import argparse

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("src"))

from omegaconf import OmegaConf
from datasets.x4k_multi_t import X4KMultiT
from models import create_model
from models.generalizable_INR.configs import GIMMConfig
from models.hermite_vfi.raft import initialize_RAFT
from utils.flow_viz import flow_to_image
from utils.utils import set_seed


ANCHOR_T = {0.0: "F", 0.5: "M", 1.0: "F'"}  # the only times GT is defined
ANCHOR_TOL = 1e-3


def parse_args():
    p = argparse.ArgumentParser(description="GIMM stage-1 flow visualisation on X4K")
    p.add_argument("-m", "--model-config", type=str, default="configs/gimm/gimm_x4k.yaml")
    p.add_argument("--ckpt", type=str, required=True, help="stage-1 gimm checkpoint")
    p.add_argument("--data-path", type=str, required=True, help="X4K clips (val/train)")
    p.add_argument("--raft-ckpt", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument(
        "--sample-stride", type=int, default=None,
        help="pick every Nth clip instead of the first N (spreads variety); "
             "default spaces samples evenly across the split",
    )
    p.add_argument(
        "--timesteps", type=str, default="0,0.25,0.5,0.75,1.0",
        help="comma-separated query times in [0,1]; 0/0.5/1 get a GT panel",
    )
    p.add_argument("--raft-iter", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--per-column-scale", action="store_true",
        help="scale flow colour per timestep instead of one shared scale; the "
             "shared default makes brighter = faster, so motion growth reads "
             "across the row",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--no-pdf", action="store_true", help="skip the vector PDF copy")
    return p.parse_args()


@torch.no_grad()
def raft_flow(raft, img_a, img_b, iters):
    # Same call convention as generate_gimm_flow_cache.py: RAFT wants 0-255.
    flow, _feat, _fmap = raft(255.0 * img_a, 255.0 * img_b, return_feat=True, iters=iters)
    return flow


def to_uv(flow_bchw):
    """(1,2,H,W) tensor -> (H,W,2) float32 numpy for flow_to_image."""
    return flow_bchw[0].permute(1, 2, 0).detach().cpu().float().numpy()


def epe(pred_bchw, gt_bchw):
    """Mean end-point error in pixels between two (1,2,H,W) flows."""
    return torch.linalg.vector_norm(pred_bchw - gt_bchw, dim=1).mean().item()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit(
            "GIMM's softsplat kernel needs a CUDA GPU (cupy compiles it at "
            "first call). Run this on the GPU box, not a CPU-only machine."
        )

    timesteps = [float(x) for x in args.timesteps.split(",")]

    # ---- config, model, checkpoint -----------------------------------------
    file_cfg = OmegaConf.load(args.model_config)
    arch = OmegaConf.merge(OmegaConf.structured(GIMMConfig(ema=False)), file_cfg.arch)
    model, _ = create_model(arch, ema=False)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

    raft_iter = args.raft_iter or int(getattr(arch, "raft_iter", 20))
    raft = initialize_RAFT(args.raft_ckpt).to(device).eval()

    ds_cfg = file_cfg.dataset
    dataset = X4KMultiT(
        "test",                # deterministic centre window, no augmentation
        args.data_path,
        num_timesteps=1,        # the single centre frame at true t = 0.5
        aug=False,
        crop_size=int(ds_cfg.get("crop_size", 256)),
        frame_gap=int(ds_cfg.get("frame_gap", 32)),
        num_divisions=int(ds_cfg.get("num_divisions", 8)),
        clip_length=int(ds_cfg.get("clip_length", 65)),
        source=ds_cfg.get("source", "auto"),
        downsample=float(ds_cfg.get("downsample", 1.0)),
    )
    n = len(dataset)
    if n == 0:
        raise SystemExit(f"no clips found under {args.data_path}")

    if args.sample_stride:
        idxs = list(range(0, n, args.sample_stride))[: args.num_samples]
    else:  # evenly spaced across the split, so the figures are not all clip 0
        k = min(args.num_samples, n)
        idxs = [int(round(i * (n - 1) / max(1, k - 1))) for i in range(k)] if k > 1 else [0]
    print(f"{n} clips available; rendering {len(idxs)}: {idxs}")

    for fig_i, idx in enumerate(idxs):
        sample = dataset[idx]
        xs = sample["xs"].unsqueeze(0).to(device)   # (1, 3, 3, H, W): [I0, I1, GT]
        img0, img1, gt = xs[:, :, 0], xs[:, :, 1], xs[:, :, 2]

        # F / F' / M exactly as the flow cache defines them.
        F = raft_flow(raft, img0, img1, raft_iter)          # (1,2,H,W)
        Fp = raft_flow(raft, img1, img0, raft_iter)
        M = raft_flow(raft, gt, img1, raft_iter) - raft_flow(raft, gt, img0, raft_iter)
        if not (torch.isfinite(F).all() and torch.isfinite(Fp).all() and torch.isfinite(M).all()):
            print(f"  clip {idx}: non-finite RAFT flow, skipped")
            continue

        # Scale and normalise into the model's input space (X4KGimmFlowCache).
        scaler = torch.stack([F.abs(), Fp.abs()]).max().clamp_min(1.0)
        F_n, Fp_n = (F / scaler + 1) / 2, (Fp / scaler + 1) / 2
        input_xs = torch.stack([F_n, Fp_n], dim=2)          # (1,2,2,H,W) = [F,F']
        ori_flow = torch.stack([F, Fp], dim=2)              # (1,2,2,H,W) raw
        spatial = xs.shape[-2:]

        gt_by_anchor = {0.0: F, 0.5: M, 1.0: Fp}

        preds, gts, epes = [], [], []
        for t in timesteps:
            t_vec = torch.full((1,), t, device=device, dtype=torch.float)
            coord = model.sample_coord_input(1, spatial, t_vec, device=device)
            out = model(input_xs, coord=coord, ori_flow=ori_flow, timesteps=t_vec)
            pred = (out[:, :, 0] * 2 - 1) * scaler          # denormalise -> pixels
            preds.append(pred)

            anchor = next((a for a in ANCHOR_T if abs(t - a) < ANCHOR_TOL), None)
            if anchor is not None:
                g = gt_by_anchor[anchor]
                gts.append(g)
                epes.append(epe(pred, g))
            else:
                gts.append(None)
                epes.append(None)

        # One shared colour scale across the whole sample unless asked otherwise,
        # so a brighter panel genuinely means faster motion.
        if args.per_column_scale:
            col_max = [
                max(
                    float(torch.linalg.vector_norm(p, dim=1).max()),
                    float(torch.linalg.vector_norm(g, dim=1).max()) if g is not None else 0.0,
                )
                for p, g in zip(preds, gts)
            ]
        else:
            allmax = max(float(torch.linalg.vector_norm(p, dim=1).max()) for p in preds)
            allmax = max(allmax, *[
                float(torch.linalg.vector_norm(g, dim=1).max()) for g in gts if g is not None
            ])
            col_max = [allmax] * len(timesteps)

        _render(args, fig_i, idx, timesteps, preds, gts, epes, col_max)

    print(f"done -> {args.output_dir}")


def _render(args, fig_i, idx, timesteps, preds, gts, epes, col_max):
    ncol = len(timesteps)
    fig, axes = plt.subplots(2, ncol, figsize=(2.4 * ncol, 4.9), squeeze=False)

    for c, t in enumerate(timesteps):
        mx = col_max[c] + 1e-6

        ax = axes[0][c]
        ax.imshow(flow_to_image(to_uv(preds[c]), max_flow=mx))
        ax.set_title(f"$t={t:g}$", fontsize=11)
        _strip(ax)
        if c == 0:
            ax.set_ylabel("GIMM (pred)", fontsize=11)

        ax = axes[1][c]
        if gts[c] is not None:
            ax.imshow(flow_to_image(to_uv(gts[c]), max_flow=mx))
            ax.set_xlabel(f"EPE {epes[c]:.2f}px", fontsize=9)
        else:
            ax.imshow(np.full((*to_uv(preds[c]).shape[:2], 3), 245, dtype=np.uint8))
            ax.text(0.5, 0.5, "no GT\nat this $t$", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="0.5")
        _strip(ax)
        if c == 0:
            ax.set_ylabel("Ground truth", fontsize=11)

    fig.suptitle(f"X4K clip #{idx} — GIMM flow generation vs anchor GT", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stem = os.path.join(args.output_dir, f"gimm_flow_sample{fig_i:02d}_clip{idx:04d}")
    fig.savefig(stem + ".png", dpi=args.dpi, bbox_inches="tight")
    if not args.no_pdf:
        fig.savefig(stem + ".pdf", bbox_inches="tight")  # vector, for LaTeX
    plt.close(fig)

    # Also drop the bare colour panels, so a custom LaTeX figure can compose them.
    paneldir = os.path.join(args.output_dir, "panels")
    os.makedirs(paneldir, exist_ok=True)
    for c, t in enumerate(timesteps):
        mx = col_max[c] + 1e-6
        _imwrite(os.path.join(paneldir, f"s{fig_i:02d}_clip{idx:04d}_t{t:g}_pred.png"),
                 flow_to_image(to_uv(preds[c]), max_flow=mx))
        if gts[c] is not None:
            _imwrite(os.path.join(paneldir, f"s{fig_i:02d}_clip{idx:04d}_t{t:g}_gt.png"),
                     flow_to_image(to_uv(gts[c]), max_flow=mx))


def _strip(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _imwrite(path, rgb_uint8):
    plt.imsave(path, rgb_uint8)


if __name__ == "__main__":
    main()
