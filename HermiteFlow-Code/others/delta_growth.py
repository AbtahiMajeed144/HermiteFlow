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

    # Growth over the last half, which is where the trend matters.
    half = len(steps) // 2
    early, late = delta[:half], delta[half:]
    slope = np.polyfit(steps[half:], late, 1)[0] if len(late) > 2 else 0.0

    print(f"\nmean |d0| first half {early.mean():.5f} px, "
          f"last half {late.mean():.5f} px")
    print(f"slope over the last half: {slope:.3e} px/step")

    growing = slope > 0 and late.mean() > early.mean() * 1.05
    print()
    if growing:
        remaining = args.target_px - delta[-1]
        steps_needed = remaining / slope if slope > 0 else float("inf")
        print(f"STILL CLIMBING. At this rate, reaching |d0| = "
              f"{args.target_px:g} px takes ~{steps_needed:,.0f} more steps.")
        if args.steps_per_epoch:
            print(f"  = ~{steps_needed / args.steps_per_epoch:,.0f} more epochs "
                  f"at {args.steps_per_epoch} steps/epoch")
        print("  Linear extrapolation is optimistic if growth is saturating")
        print("  and pessimistic if it is still accelerating - re-run this")
        print("  after the next epoch to see which.")
    else:
        print("PLATEAUED or falling. This is an optimisation problem, not a")
        print("data problem: the oracle headroom is real and smooth, so the")
        print("residuals should be able to grow. Head gradients measured at")
        print("the same scale as the trunk, so try a HIGHER head learning")
        print("rate (and do NOT apply head_lr_divisor, which would slow the")
        print("very parameters that are stuck).")


if __name__ == "__main__":
    main()
