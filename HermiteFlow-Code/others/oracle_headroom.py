"""
How much can the cubic model possibly gain over linear on this data?

Stage 1 starts at delta = 0, i.e. exactly the linear baseline, so its
loss at init is just "how wrong is t*F". If that is 0.015, the question
is not whether 0.015 is small - it is how much of it the model is even
ABLE to remove.

This fits the BEST possible (A, B) per pixel by least squares against
the teacher targets, with no network involved. That is a strict upper
bound on what CoeffNet can learn, reachable only if it predicts the
optimum everywhere.

Three numbers per degree:

  in-sample   fit and evaluate on the same K timesteps. Optimistic:
              with 2 free parameters and K targets it partly fits RAFT's
              own noise.
  hold-out    fit on K-1 timesteps, evaluate on the held-out one,
              averaged over which one is held out. This is the honest
              number - it is what generalisation looks like.
  gap         in-sample minus hold-out. A large gap means the apparent
              headroom is mostly noise, not motion.

Read the hold-out column:

  > 1.0 dB    real, learnable curvature. Worth the GPU.
  0.3-1.0 dB  marginal; the model may win but slowly.
  < 0.3 dB    the data is essentially linear at this frame_gap. No
              amount of training will produce a result, and the right
              response is to change the data, not the model.

    python others/oracle_headroom.py \
        --data-path /kaggle/input/.../x4k1000fps/encoded_train \
        --raft-ckpt /kaggle/input/.../raft-things.pth \
        --frame-gap 32 --downsample 3.0 --num-timesteps 5 --num-samples 16
"""

import argparse
import os
import sys

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from datasets.x4k_multi_t import X4KMultiT  # noqa: E402
from models.hermite_vfi.modules.phase1_measure import flow_scale  # noqa: E402
from models.hermite_vfi.raft import initialize_RAFT  # noqa: E402

# Which basis columns each degree may use. beta_p(s) = s^p - s.
DEGREE_BASIS = {
    "linear": [],
    "quadratic": [2],
    "cubic": [2, 3],
    "quartic": [2, 3, 4],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--raft-ckpt", default="pretrained/raft-things.pth")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--downsample", type=float, default=3.0)
    p.add_argument("--frame-gap", type=int, default=32)
    p.add_argument("--num-divisions", type=int, default=8)
    p.add_argument("--num-timesteps", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--raft-iter", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def basis_matrix(times, powers):
    """(K, P) design matrix of beta_p(t) = t^p - t."""
    if not powers:
        return torch.zeros(len(times), 0)
    return torch.tensor(
        [[t**p - t for p in powers] for t in times], dtype=torch.float32
    )


def fit_and_eval(residuals, times, powers, train_idx, eval_idx):
    """
    Least-squares fit of the basis coefficients on `train_idx`, evaluated
    on `eval_idx`. Returns summed squared error over the eval set.

    The design depends only on t, not on position, so the pseudo-inverse
    is a tiny (P, K) matrix applied once to the whole residual stack.
    """
    if not powers:
        return sum(residuals[k].pow(2).sum() for k in eval_idx), sum(
            residuals[k].numel() for k in eval_idx
        )

    design = basis_matrix([times[k] for k in train_idx], powers)  # (K', P)
    pinv = torch.linalg.pinv(design).to(residuals[0].device)      # (P, K')
    stack = torch.stack([residuals[k] for k in train_idx])        # (K', C,H,W)
    coeffs = torch.einsum("pk,k...->p...", pinv.to(stack.dtype), stack)

    err, n = 0.0, 0
    for k in eval_idx:
        pred = sum(
            (times[k] ** p - times[k]) * coeffs[i] for i, p in enumerate(powers)
        )
        err = err + (pred - residuals[k]).pow(2).sum()
        n += residuals[k].numel()
    return err, n


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raft = initialize_RAFT(args.raft_ckpt).to(device).eval()
    for prm in raft.parameters():
        prm.requires_grad = False

    def flow(a, b):
        with torch.no_grad():
            return raft(255.0 * a, 255.0 * b, return_feat=True, iters=args.raft_iter)[0]

    torch.manual_seed(args.seed)
    ds = X4KMultiT(
        "train", args.data_path, num_timesteps=args.num_timesteps, aug=True,
        crop_size=args.crop_size, frame_gap=args.frame_gap,
        num_divisions=args.num_divisions, downsample=args.downsample,
    )

    degrees = list(DEGREE_BASIS)
    acc = {d: {"ins": 0.0, "ins_n": 0, "out": 0.0, "out_n": 0} for d in degrees}

    for i in range(args.num_samples):
        item = ds[i % len(ds)]
        xs = item["xs"].unsqueeze(0).to(device)
        times = [float(v) for v in item["t"]]
        img0, img1 = xs[:, :, 0], xs[:, :, 1]
        gts = [xs[:, :, 2 + k] for k in range(len(times))]

        f01 = flow(img0, img1)
        s = flow_scale(f01, flow(img1, img0))
        # Residual the LINEAR model leaves at each t, in units of s.
        residuals = [(flow(img0, g) - t * f01) / s for t, g in zip(times, gts)]

        idx = list(range(len(times)))
        for d in degrees:
            powers = DEGREE_BASIS[d]
            e, n = fit_and_eval(residuals, times, powers, idx, idx)
            acc[d]["ins"] += float(e); acc[d]["ins_n"] += n
            # leave-one-out
            if len(idx) > len(powers) + 1:
                for held in idx:
                    tr = [k for k in idx if k != held]
                    e, n = fit_and_eval(residuals, times, powers, tr, [held])
                    acc[d]["out"] += float(e); acc[d]["out_n"] += n

    def db(num, den):
        return -10 * torch.log10(torch.tensor(max(num / den, 1e-12))).item()

    lin_ins = db(acc["linear"]["ins"], acc["linear"]["ins_n"])
    lin_out = db(acc["linear"]["out"], max(acc["linear"]["out_n"], 1))

    print(f"\ngap {args.frame_gap}, downsample {args.downsample}, K={args.num_timesteps}, "
          f"crop {args.crop_size}, {args.num_samples} samples")
    print("\nOracle: best possible (A, B) per pixel, no network.\n")
    print(f"{'degree':<12}{'in-sample dB':>14}{'hold-out dB':>13}"
          f"{'gain vs linear':>16}{'gap':>8}")
    for d in degrees:
        ins = db(acc[d]["ins"], acc[d]["ins_n"])
        has_out = acc[d]["out_n"] > 0
        out = db(acc[d]["out"], max(acc[d]["out_n"], 1)) if has_out else float("nan")
        g = out - lin_out if has_out else float("nan")
        gap = ins - out if has_out else float("nan")
        print(f"{d:<12}{ins:>14.3f}{out:>13.3f}{g:>15.3f}{gap:>8.2f}")

    print("\nThe hold-out `gain vs linear` column is the ceiling for stage 1.")
    print("  >1.0 dB    real curvature, worth training")
    print("  0.3-1.0    marginal")
    print("  <0.3 dB    data is effectively linear here; change the data")
    print("A large `gap` means the in-sample gain is mostly fitted RAFT noise.")


if __name__ == "__main__":
    main()
