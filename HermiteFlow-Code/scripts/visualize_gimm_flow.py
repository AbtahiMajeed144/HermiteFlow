"""
Paper-ready flow visualisation for a stage-1 GIMM motion checkpoint on X4K1000FPS.

`epoch31_model.pt.zip` is a stage-1 GIMM motion model (`arch.type: gimm`): a
flow-interpolation INR, not the full gimmvfi_r frame model. Given the two
endpoint flows F = RAFT(I0, I1) and F' = RAFT(I1, I0) it predicts a flow field
at any query time t in [0, 1].

Ground truth at every timestep comes from RAFT on the REAL X4K frame at that
time. X4K clips are dense, and the two endpoints sit `frame_gap` frames apart
with real frames on the t = k/num_divisions grid in between (k = 1..7 for the
default 8-way grid). So for a query t that lands on the grid we take the real
frame I_t and form GT exactly as GIMM's own stage-1 target M is formed at the
centre frame:

    GT(t) = RAFT(I_t, I1) - RAFT(I_t, I0)       (interior t = k/num_divisions)
    GT(0) = F  = RAFT(I0, I1)                    (start frame)
    GT(1) = F' = RAFT(I1, I0)                    (end frame)

At t = 0.5 this is byte-for-byte the flow the model was trained against; at
0.25 / 0.75 it is the same recipe on the real intermediate frame, i.e. a
genuine held-out target the model never saw. Off-grid times (e.g. 0.3) have no
real frame, so they show the prediction alone.

Each figure shows three rows over the requested timesteps: the real frame at t
(with the start/end frames as the t=0 and t=1 columns), the GIMM prediction,
and the RAFT ground truth, with per-t EPE.

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
        help="pick every Nth clip; default spaces samples evenly across the split",
    )
    p.add_argument(
        "--timesteps", type=str, default="0,0.25,0.5,0.75,1.0",
        help="comma-separated query times in [0,1]; those landing on the "
             "k/num_divisions grid (and the 0/1 endpoints) get a RAFT GT panel",
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


def to_rgb(img_bchw):
    """(1,3,H,W) tensor in [0,1] -> (H,W,3) float numpy for imshow."""
    return img_bchw[0].permute(1, 2, 0).clamp(0, 1).detach().cpu().float().numpy()


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
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch', '?')})")

    raft_iter = args.raft_iter or int(getattr(arch, "raft_iter", 20))
    raft = initialize_RAFT(args.raft_ckpt).to(device).eval()

    ds_cfg = file_cfg.dataset
    num_div = int(ds_cfg.get("num_divisions", 8))
    dataset = X4KMultiT(
        "test",                      # deterministic centre window, no augmentation
        args.data_path,
        num_timesteps=num_div - 1,   # expose EVERY interior real frame (t = k/num_div)
        aug=False,
        crop_size=int(ds_cfg.get("crop_size", 256)),
        frame_gap=int(ds_cfg.get("frame_gap", 32)),
        num_divisions=num_div,
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

    results = []  # (fig_i, idx, cols) for every sample that rendered
    for fig_i, idx in enumerate(idxs):
        cols = _process(args, model, raft, raft_iter, dataset, num_div, timesteps, idx, device)
        if cols is None:
            continue
        _render(args, fig_i, idx, cols)        # per-sample figure
        results.append((fig_i, idx, cols))

    if results:
        _render_combined(args, results)        # one image of all samples
    print(f"done -> {args.output_dir}")


def _resolve_frame(t, num_div, times, middles, img0, img1):
    """
    Map a query t to (real_frame, kind). kind is 'start', 'end', 'interior',
    or None (off-grid, no real frame). times/middles are the K interior frames.
    """
    k = t * num_div
    if abs(k - round(k)) > 1e-6:
        return None, None                       # not on the k/num_div grid
    k = int(round(k))
    if k == 0:
        return img0, "start"
    if k == num_div:
        return img1, "end"
    for j, tj in enumerate(times):
        if abs(tj * num_div - k) < 1e-6:
            return middles[j], "interior"
    return None, None


def _process(args, model, raft, raft_iter, dataset, num_div, timesteps, idx, device):
    """Run one clip through the model and build its per-timestep column data.
    Returns the list of column dicts, or None if the clip could not be used."""
    sample = dataset[idx]
    xs = sample["xs"].unsqueeze(0).to(device)   # (1, 3, 2 + K, H, W)
    times = [float(v) for v in sample["t"].tolist()]
    img0, img1 = xs[:, :, 0], xs[:, :, 1]
    middles = [xs[:, :, 2 + j] for j in range(len(times))]

    # Endpoint flows: the model's input, and the GT at t = 0 / t = 1.
    F = raft_flow(raft, img0, img1, raft_iter)
    Fp = raft_flow(raft, img1, img0, raft_iter)
    if not (torch.isfinite(F).all() and torch.isfinite(Fp).all()):
        print(f"  clip {idx}: non-finite endpoint flow, skipped")
        return None

    scaler = torch.stack([F.abs(), Fp.abs()]).max().clamp_min(1.0)
    F_n, Fp_n = (F / scaler + 1) / 2, (Fp / scaler + 1) / 2
    input_xs = torch.stack([F_n, Fp_n], dim=2)  # (1,2,2,H,W) = [F, F']
    ori_flow = torch.stack([F, Fp], dim=2)      # (1,2,2,H,W) raw
    spatial = xs.shape[-2:]

    cols = []  # one dict per requested timestep
    for t in timesteps:
        # Prediction (defined at any t).
        t_vec = torch.full((1,), t, device=device, dtype=torch.float)
        coord = model.sample_coord_input(1, spatial, t_vec, device=device)
        out = model(input_xs, coord=coord, ori_flow=ori_flow, timesteps=t_vec)
        pred = (out[:, :, 0] * 2 - 1) * scaler   # denormalise -> pixels

        # Ground truth from RAFT on the real frame at t, where one exists.
        frame, kind = _resolve_frame(t, num_div, times, middles, img0, img1)
        if kind == "start":
            gt, real = F, frame
        elif kind == "end":
            gt, real = Fp, frame
        elif kind == "interior":
            # Same construction as the trained t=0.5 target M, on the real I_t.
            gt = raft_flow(raft, frame, img1, raft_iter) - raft_flow(raft, frame, img0, raft_iter)
            real = frame
            if not torch.isfinite(gt).all():
                gt = None
        else:
            gt, real = None, None
        cols.append(dict(t=t, pred=pred, gt=gt, real=real, kind=kind,
                         epe=(epe(pred, gt) if gt is not None else None)))

    # Colour scale: one shared value (brighter = faster, reads across the row)
    # unless asked otherwise.
    def col_mag(c):
        m = float(torch.linalg.vector_norm(c["pred"], dim=1).max())
        if c["gt"] is not None:
            m = max(m, float(torch.linalg.vector_norm(c["gt"], dim=1).max()))
        return m
    if args.per_column_scale:
        for c in cols:
            c["mx"] = col_mag(c) + 1e-6
    else:
        shared = max(col_mag(c) for c in cols) + 1e-6
        for c in cols:
            c["mx"] = shared

    return cols


def _render(args, fig_i, idx, cols):
    ncol = len(cols)
    fig, axes = plt.subplots(3, ncol, figsize=(2.4 * ncol, 7.1), squeeze=False)

    for c, col in enumerate(cols):
        t, mx = col["t"], col["mx"]

        # Row 0: the real frame at t (t=0 / t=1 are the input start / end frames).
        ax = axes[0][c]
        if col["real"] is not None:
            ax.imshow(to_rgb(col["real"]))
            if col["kind"] == "start":
                ax.set_xlabel("input: frame 0", fontsize=9)
            elif col["kind"] == "end":
                ax.set_xlabel("input: frame 1", fontsize=9)
        else:
            _blank(ax, "off-grid\n(no real frame)")
        ax.set_title(f"$t={t:g}$", fontsize=11)
        _strip(ax)
        if c == 0:
            ax.set_ylabel("Frame $I_t$", fontsize=11)

        # Row 1: GIMM predicted flow.
        ax = axes[1][c]
        ax.imshow(flow_to_image(to_uv(col["pred"]), max_flow=mx))
        _strip(ax)
        if c == 0:
            ax.set_ylabel("Predicted", fontsize=11)

        # Row 2: RAFT ground-truth flow at t.
        ax = axes[2][c]
        if col["gt"] is not None:
            ax.imshow(flow_to_image(to_uv(col["gt"]), max_flow=mx))
            ax.set_xlabel(f"EPE {col['epe']:.2f}px", fontsize=9)
        else:
            _blank(ax, "no GT\nat this $t$")
        _strip(ax)
        if c == 0:
            ax.set_ylabel("GT (RAFT)", fontsize=11)

    fig.suptitle(f"X4K clip #{idx} — GIMM flow generation vs RAFT ground truth", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stem = os.path.join(args.output_dir, f"final_gimm_flow_sample{fig_i:02d}_clip{idx:04d}")
    fig.savefig(stem + ".png", dpi=args.dpi, bbox_inches="tight")
    if not args.no_pdf:
        fig.savefig(stem + ".pdf", bbox_inches="tight")  # vector, for LaTeX
    plt.close(fig)

    # Bare panels too, so a custom LaTeX figure can compose them.
    paneldir = os.path.join(args.output_dir, "panels")
    os.makedirs(paneldir, exist_ok=True)
    for col in cols:
        tag = f"s{fig_i:02d}_clip{idx:04d}_t{col['t']:g}"
        plt.imsave(os.path.join(paneldir, tag + "_pred.png"),
                   flow_to_image(to_uv(col["pred"]), max_flow=col["mx"]))
        if col["gt"] is not None:
            plt.imsave(os.path.join(paneldir, tag + "_gt.png"),
                       flow_to_image(to_uv(col["gt"]), max_flow=col["mx"]))
        if col["real"] is not None:
            plt.imsave(os.path.join(paneldir, tag + "_frame.png"), to_rgb(col["real"]))


def _render_combined(args, results):
    """One image holding every sample: a 3-row (frame / predicted / GT) block
    per clip, stacked vertically, with the timestep columns shared."""
    ncol = len(results[0][2])
    nblock = len(results)
    nrow = 3 * nblock
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.3 * nrow), squeeze=False)
    row_names = ("Frame $I_t$", "Predicted", "GT (RAFT)")

    for b, (_fig_i, idx, cols) in enumerate(results):
        for c, col in enumerate(cols):
            t, mx = col["t"], col["mx"]
            r0 = 3 * b

            ax = axes[r0][c]                       # frame
            if col["real"] is not None:
                ax.imshow(to_rgb(col["real"]))
                if col["kind"] == "start":
                    ax.set_xlabel("input: frame 0", fontsize=8)
                elif col["kind"] == "end":
                    ax.set_xlabel("input: frame 1", fontsize=8)
            else:
                _blank(ax, "off-grid")
            if b == 0:
                ax.set_title(f"$t={t:g}$", fontsize=11)

            axes[r0 + 1][c].imshow(flow_to_image(to_uv(col["pred"]), max_flow=mx))  # pred

            ax = axes[r0 + 2][c]                   # GT
            if col["gt"] is not None:
                ax.imshow(flow_to_image(to_uv(col["gt"]), max_flow=mx))
                ax.set_xlabel(f"EPE {col['epe']:.2f}px", fontsize=8)
            else:
                _blank(ax, "no GT")

            for rr, name in enumerate(row_names):
                a = axes[r0 + rr][c]
                _strip(a)
                if c == 0:
                    label = f"clip {idx}\n{name}" if rr == 0 else name
                    a.set_ylabel(label, fontsize=10)

    fig.suptitle("GIMM flow generation vs RAFT ground truth — X4K samples", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    stem = os.path.join(args.output_dir, "final_gimm_flow_all")
    fig.savefig(stem + ".png", dpi=args.dpi, bbox_inches="tight")
    if not args.no_pdf:
        fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"combined figure: {stem}.png ({nblock} samples)")


def _strip(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _blank(ax, text):
    ax.imshow(np.full((16, 16, 3), 245, dtype=np.uint8))
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="0.5")


if __name__ == "__main__":
    main()
