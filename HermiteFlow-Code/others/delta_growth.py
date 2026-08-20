"""
Is the velocity residual still growing, and how long to the target?

The gain a checkpoint achieves is bounded by how large |d0| is. The
oracle wants |d0| ~ 0.375*s (see others/oracle_headroom.py), i.e. tens
of pixels, while epoch 1 typically ends around 0.3 px. The question
that decides what to do next is not "is the gain small" - it is
whether |d0| is STILL CLIMBING.

  still climbing  -> the run is working, it just started at zero.
                     Extrapolate to get an honest epoch budget.
  plateaued       -> an optimisation problem, not a data problem.
                     Head gradients were measured at the same scale as
                     the trunk, so the lever is a HIGHER head learning
                     rate, not a lower one.

Reads the TensorBoard scalars the trainer already writes; no GPU, no
re-run.

    python others/delta_growth.py --run /kaggle/working/runs/<run> \
        --target-px 20
"""

import argparse
import glob
import os
import sys

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError:
    sys.exit("tensorboard is required: pip install tensorboard")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory (holds train/)")
    p.add_argument(
        "--target-px", type=float, default=20.0,
        help="|d0| the oracle wants, in pixels: roughly 0.375 * motion scale",
    )
    p.add_argument("--steps-per-epoch", type=int, default=None)
    return p.parse_args()


def load(run, tag):
    candidates = [os.path.join(run, "train"), run]
    for directory in candidates:
        if not os.path.isdir(directory):
            continue
        acc = EventAccumulator(directory)
        acc.Reload()
        if tag in acc.Tags().get("scalars", []):
            events = acc.Scalars(tag)
            return np.array([e.step for e in events]), np.array(
                [e.value for e in events]
            )
    return None, None


def r2(y, yhat):
    resid = ((y - yhat) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    return 1.0 - resid / total if total > 0 else 0.0


def fit_models(steps, delta):
    """{name: (r2, predict_fn, steps_to_reach_fn, formula)}."""
    steps = steps.astype(float)
    models = {}

    b, a = np.polyfit(steps, delta, 1)
    models["linear"] = (
        r2(delta, a + b * steps),
        lambda t: a + b * t,
        lambda tgt: (tgt - a) / b if b > 0 else np.inf,
        f"d = {a:.4f} + {b:.3e} t",
    )

    ok = delta > 0
    if ok.sum() > 3:
        k, c = np.polyfit(steps[ok], np.log(delta[ok]), 1)
        models["exponential"] = (
            r2(delta, np.exp(c) * np.exp(k * steps)),
            lambda t: np.exp(c) * np.exp(k * t),
            lambda tgt: (np.log(tgt) - c) / k if k > 0 else np.inf,
            f"d = {np.exp(c):.4f} exp({k:.3e} t)",
        )
        p, q = np.polyfit(np.log(steps[ok]), np.log(delta[ok]), 1)
        models["power law"] = (
            r2(delta, np.exp(q) * steps ** p),
            lambda t: np.exp(q) * t ** p,
            lambda tgt: np.exp((np.log(tgt) - q) / p) if p > 0 else np.inf,
            f"d = {np.exp(q):.4f} t^{p:.3f}",
        )
    return models


def main():
    args = parse_args()
    # step/* is written every log_every optimizer steps; the
    # velocity_residual/* tag is the per-epoch fallback for runs whose
    # step logging never fired.
    for tag in ("step/delta_0", "velocity_residual/delta_0"):
        steps, delta = load(args.run, tag)
        if steps is not None:
            print("")
            print(f"source tag: {tag}")
            break
    if steps is None:
        tags = []
        for d in (os.path.join(args.run, "train"), args.run):
            if os.path.isdir(d):
                acc = EventAccumulator(d)
                acc.Reload()
                tags += acc.Tags().get("scalars", [])
        sys.exit(
            f"no 'step/delta_0' scalar under {args.run}. Found: "
            f"{sorted(set(tags))[:12]}"
        )

    print(f"\n{len(steps)} points, steps {steps[0]}..{steps[-1]}\n")
    print(f"{'step':>8}{'|d0| px':>12}")
    stride = max(1, len(steps) // 12)
    for i in range(0, len(steps), stride):
        print(f"{steps[i]:>8}{delta[i]:>12.5f}")
    if (len(steps) - 1) % stride:
        print(f"{steps[-1]:>8}{delta[-1]:>12.5f}")

    half = len(steps) // 2
    early, late = delta[:half], delta[half:]
    growing = late.mean() > early.mean() * 1.05

    print(f"\nmean |d0| first half {early.mean():.5f} px, "
          f"last half {late.mean():.5f} px   "
          f"({'rising' if growing else 'flat or falling'})")

    if not growing:
        print("\nPLATEAUED or falling. That is an optimisation problem, not a")
        print("data problem: if the oracle headroom is real and smooth, the")
        print("residuals should be able to grow. Head gradients measure at")
        print("the same scale as the trunk, so the lever is a HIGHER head")
        print("learning rate - and NOT head_lr_divisor, which would slow the")
        print("very parameters that are stuck.")
        return

    # Extrapolating from one noisy epoch is exactly where this goes
    # wrong: a linear and an exponential fit can agree on the data and
    # disagree by two orders of magnitude on the forecast. Fit several,
    # report how well each explains the curve, and when they cannot be
    # told apart, say so rather than inventing a budget.
    models = fit_models(steps, delta)
    print(f"\n{'model':<14}{'R2':>7}   {'fit':<34}{'to target':>13}")
    for name, (score, _, solve, formula) in models.items():
        need = solve(args.target_px) - steps[-1]
        if need <= 0:
            label = "reached"
        elif args.steps_per_epoch:
            label = f"{need / args.steps_per_epoch:,.1f} ep"
        else:
            label = f"{need:,.0f} st"
        print(f"{name:<14}{score:>7.3f}   {formula:<34}{label:>13}")

    scores = [m[0] for m in models.values()]
    if len(models) > 1 and max(scores) - min(scores) < 0.15:
        nxt = steps[-1] + (args.steps_per_epoch or len(steps) * 50)
        print("\nThese fits are statistically indistinguishable on this much")
        print("data, and they disagree by orders of magnitude. Do not plan a")
        print("budget from them. Run one more epoch and re-run this - the")
        print("models predict different values and separate quickly:")
        print(f"\n  predicted |d0| at step {nxt:,.0f}")
        for name, (_, predict, _, _) in models.items():
            print(f"    {name:<14}{predict(nxt):>9.3f} px")
    else:
        best = max(models.items(), key=lambda kv: kv[1][0])
        print(f"\nBest fit: {best[0]} (R2 {best[1][0]:.3f}).")


if __name__ == "__main__":
    main()
