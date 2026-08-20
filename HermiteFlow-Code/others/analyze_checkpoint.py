"""
Deep analysis of a stage-1 checkpoint.

Scores the model against the CEILING this data actually has, not
against zero. Five questions in one pass:

  1. Did it learn curvature, and what fraction of the available
     headroom did it capture?  (model vs linear vs oracle)
  2. Where does the gain come from, per timestep?
  3. Are d0 and d1 doing different jobs, or has the network collapsed
     onto the symmetric mode - which barely bends the curve at all?
  4. Is the occlusion gate open, or suppressing everything?
  5. How much does the RGB branch contribute?

    python others/analyze_checkpoint.py \
        --checkpoint /kaggle/working/runs/<run>/epoch1_model.pt \
        --model-config configs/hermiteflow/hermiteflow_r_x4k_stage1.yaml \
        --data-path /kaggle/input/.../x4k1000fps/val \
        --raft-ckpt /kaggle/input/.../raft-things.pth --num-samples 16
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from datasets.x4k_multi_t import X4KMultiT  # noqa: E402
from models import create_model  # noqa: E402
from utils.config import load_config, augment_arch_defaults  # noqa: E402
from oracle_headroom import DEGREE_BASIS, fit_and_eval  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--model-config", default=None,
        help="defaults to config.yaml beside the checkpoint, which is the "
             "fully-resolved config of that run (including dotlist overrides)",
    )
    p.add_argument("--data-path", required=True, help="validation root")
    p.add_argument("--raft-ckpt", default=None)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--crop-size", type=int, default=None)
    p.add_argument("--downsample", type=float, default=None)
    p.add_argument("--frame-gap", type=int, default=None)
    p.add_argument("--num-timesteps", type=int, default=None)
    p.add_argument(
        "--teacher-raft-iter", type=int, default=None,
        help="defaults to loss.teacher_raft_iter from the run config; must "
             "match training or the baseline is measured against a "
             "different teacher than the model was fitted to",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def match_architecture(model, state):
    """
    Make the constructed model match the checkpoint's architecture.

    Checkpoints written before the GroupNorm branches have no
    `flow_branch.layers.3` and no `gain`. Loading those with
    strict=False "works" - and is silently wrong: GroupNorm at init is
    NOT the identity (weight=1, bias=0 still normalises), so the trunk
    would run on normalised features it was never trained for and the
    analysis would report a model that learned nothing.

    Detect it and drop the GroupNorm instead, so the evaluated function
    is exactly the one that was trained.
    """
    has_groupnorm = any(k.endswith("flow_branch.layers.3.weight") for k in state)
    if has_groupnorm:
        return "current (GroupNorm branches)", set()
    for branch in (model.coeff_net.flow_branch, model.coeff_net.rgb_branch):
        if len(branch.layers) > 3:
            branch.layers = nn.Sequential(*list(branch.layers)[:3])
        with torch.no_grad():
            branch.gain.fill_(1.0)   # unit gain == the legacy behaviour
    handled = {
        "coeff_net.flow_branch.gain",
        "coeff_net.rgb_branch.gain",
    }
    return "legacy (pre-GroupNorm; GroupNorm dropped, gains pinned to 1)", handled


class Meter:
    """Accumulates squared error and element count -> PSNR in dB."""

    def __init__(self):
        self.sq = 0.0
        self.n = 0

    def add(self, err):
        self.sq += float(err.pow(2).sum())
        self.n += err.numel()

    def add_raw(self, sq, n):
        self.sq += float(sq)
        self.n += n

    @property
    def db(self):
        if self.n == 0:
            return float("nan")
        return -10 * torch.log10(torch.tensor(max(self.sq / self.n, 1e-12))).item()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The run directory holds the resolved config that produced this
    # checkpoint - including any dotlist overrides passed at launch.
    # Reading the original YAML instead silently rebuilds a DIFFERENT
    # architecture whenever the run was launched with overrides.
    config_path = args.model_config
    if config_path is None:
        sibling = os.path.join(os.path.dirname(args.checkpoint), "config.yaml")
        if not os.path.isfile(sibling):
            sys.exit(
                f"no config.yaml beside the checkpoint ({sibling}); pass "
                f"--model-config explicitly"
            )
        config_path = sibling
    cfg = load_config(config_path)
    if args.raft_ckpt:
        cfg.arch.pretrained_raft_ckpt = args.raft_ckpt
    arch = augment_arch_defaults(cfg.arch)
    d = cfg.dataset

    ds = X4KMultiT(
        "test", args.data_path,
        num_timesteps=args.num_timesteps or d.get("num_timesteps", 5),
        aug=False,
        crop_size=args.crop_size or d.get("crop_size", 256),
        frame_gap=args.frame_gap or d.get("frame_gap", 32),
        num_divisions=d.get("num_divisions", 8),
        downsample=(
            args.downsample if args.downsample is not None
            else d.get("downsample", 1.0)
        ),
    )

    # The teacher defines the target, so it must be the SAME teacher the
    # run trained against. Using arch.raft_iter here instead of
    # loss.teacher_raft_iter silently scores the model against a
    # different ground truth - it shifted the linear baseline by ~3.5 dB
    # in practice, which is 100x the effect being measured.
    teacher_iters = (
        args.teacher_raft_iter
        if args.teacher_raft_iter is not None
        else cfg.get("loss", {}).get("teacher_raft_iter", arch.raft_iter)
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variants = {"weights": ckpt["state_dict"]}
    if "state_dict_ema" in ckpt:
        variants["EMA"] = ckpt["state_dict_ema"]

    print(f"\ncheckpoint : {os.path.basename(args.checkpoint)}  "
          f"(epoch {ckpt.get('epoch', '?')})")
    print(f"data       : gap {ds.frame_gap}, downsample {ds.downsample}, "
          f"K={ds.num_timesteps}, crop {ds.crop_size}, "
          f"{args.num_samples} samples from {args.data_path}")
    print(f"teacher    : RAFT @ {teacher_iters} iterations "
          f"(must match training)")

    model, _ = create_model(arch, ema=False)
    flavour, handled = match_architecture(model, variants[list(variants)[0]])
    print(f"architecture: {flavour}")
    model = model.to(device).eval()

    # ------------------------------------------------------------------
    # Phase 1 + teacher are model-independent, so compute them ONCE and
    # reuse across every variant. Otherwise each variant would repeat
    # 2 + 2K RAFT passes per sample.
    # ------------------------------------------------------------------
    cache = []
    for i in range(args.num_samples):
        item = ds[i % len(ds)]
        xs = item["xs"].unsqueeze(0).to(device)
        times = [float(v) for v in item["t"]]
        img_xs = xs[:, :, :2]
        gts = [xs[:, :, 2 + k] for k in range(len(times))]
        with torch.no_grad():
            measured = model.measure(img_xs[:, :, 0], img_xs[:, :, 1])
            teacher = model.teacher_flows(
                img_xs[:, :, 0], img_xs[:, :, 1], gts, iters=teacher_iters
            )
        cache.append({
            "img_xs": img_xs, "times": times, "measured": measured,
            "targets": [tp for tp, _ in teacher],
        })

    # ---------------- linear baseline and oracle ceiling ----------------
    linear, oracle = Meter(), Meter()
    per_t = {}
    for c in cache:
        s = c["measured"]["scale"]
        f01 = c["measured"]["flow_fwd"]
        residuals = [
            (tp - t * f01) / s for t, tp in zip(c["times"], c["targets"])
        ]
        idx = list(range(len(c["times"])))
        for k in idx:
            linear.add(residuals[k])
            per_t.setdefault(c["times"][k], {}).setdefault("linear", Meter()).add(
                residuals[k]
            )
        # Leave-one-out cubic fit: the honest ceiling. Needs K > 3 so the
        # fit still has more equations than the 2 free parameters.
        if len(idx) > 3:
            for held in idx:
                train = [j for j in idx if j != held]
                e, n = fit_and_eval(
                    residuals, c["times"], DEGREE_BASIS["cubic"], train, [held]
                )
                oracle.add_raw(e, n)

    # ---------------- the model, per variant ----------------
    results, delta_report, gate_report = {}, {}, {}
    for label, state in variants.items():
        clean = {k.replace("module.", ""): v for k, v in state.items()}
        try:
            missing, _ = model.load_state_dict(clean, strict=False)
        except RuntimeError as exc:
            sys.exit(
                "checkpoint does not match the architecture in "
                f"{config_path}. Point --model-config at the config.yaml "
                f"from the RUN that produced this checkpoint. Detail: {exc}"
            )
        # Anything still missing after the architecture match is a real
        # mismatch and would be evaluated at random init - refuse rather
        # than quietly report a meaningless number.
        starved = [
            k for k in missing if k.startswith("coeff_net") and k not in handled
        ]
        if starved:
            sys.exit(
                f"'{label}' is missing {len(starved)} coeff_net parameters, e.g. "
                f"{starved[:3]}. These would be evaluated at initialisation, so "
                f"the result would be meaningless. The checkpoint and the config "
                f"do not describe the same network."
            )

        for rgb in (True, False):
            model.coeff_net.set_rgb_branch(rgb)
            key = label if rgb else f"{label} -RGB"
            meter = Meter()
            d0s, d1s, gates = [], [], []

            for c in cache:
                t_list = [torch.tensor([v], device=device) for v in c["times"]]
                with torch.no_grad():
                    out = model(c["img_xs"], t=t_list, trajectory_only=True)
                    s = out["flow_scale"]
                    for k, tp in enumerate(c["targets"]):
                        err = (out["phi"][k] - tp) / s
                        meter.add(err)
                        per_t.setdefault(c["times"][k], {}).setdefault(
                            key, Meter()
                        ).add(err)
                    if rgb:
                        res0, _ = model.predict_velocities(
                            c["img_xs"][:, :, 0], c["img_xs"][:, :, 1], c["measured"]
                        )
                        d0s.append(res0[0]); d1s.append(res0[1])
                        gates.append(model.coeff_net.gate(
                            c["measured"]["occ_fwd"] / c["measured"]["scale"]
                        ))

            results[key] = meter.db
            if rgb:
                d0 = torch.cat([x.flatten() for x in d0s])
                d1 = torch.cat([x.flatten() for x in d1s])
                sym = (d0 + d1) / 2
                anti = (d0 - d1) / 2
                delta_report[label] = {
                    "d0_mean": d0.abs().mean().item(),
                    "d0_p95": d0.abs().quantile(0.95).item(),
                    "d1_mean": d1.abs().mean().item(),
                    "sym": sym.pow(2).sum().item(),
                    "anti": anti.pow(2).sum().item(),
                }
                g = torch.cat([x.flatten() for x in gates])
                gate_report[label] = (g.mean().item(), g.quantile(0.05).item())

    # ---------------- report ----------------
    lin_db = linear.db
    ora_db = oracle.db if oracle.n else float("nan")

    print("\nTRAJECTORY QUALITY   flow-PSNR vs teacher, higher is better\n")
    print(f"  {'linear baseline (d=0)':<26}{lin_db:>9.3f} dB")
    best = max(results.items(), key=lambda kv: kv[1])
    for key, v in results.items():
        mark = "  <-- best" if key == best[0] else ""
        print(f"  {key:<26}{v:>9.3f} dB   gain {v - lin_db:+.3f}{mark}")
    if oracle.n:
        print(f"  {'oracle ceiling (LOO)':<26}{ora_db:>9.3f} dB   "
              f"gain {ora_db - lin_db:+.3f}")
        ceiling = ora_db - lin_db
        if ceiling > 1e-6:
            print(f"\n  ==> captured {(best[1] - lin_db) / ceiling:>6.1%} of the "
                  f"available headroom  ({best[1] - lin_db:+.3f} of {ceiling:+.3f} dB)")
    else:
        print("  oracle: needs num_timesteps > 3 for a leave-one-out fit")

    print("\nPER-TIMESTEP   gain over linear, dB\n")
    cols = [k for k in results if "-RGB" not in k]
    print(f"  {'t':<9}" + "".join(f"{k:>14}" for k in cols))
    for t in sorted(per_t):
        if "linear" not in per_t[t]:
            continue
        base = per_t[t]["linear"].db
        row = "".join(
            f"{per_t[t][k].db - base:>14.3f}" if k in per_t[t] else f"{'-':>14}"
            for k in cols
        )
        print(f"  {t:<9.3f}{row}")

    print("\nVELOCITY RESIDUALS   (px)\n")
    for label, r in delta_report.items():
        total = r["sym"] + r["anti"]
        frac = r["anti"] / total if total > 0 else 0.0
        print(f"  {label:<10} |d0| mean {r['d0_mean']:.4f}  p95 {r['d0_p95']:.4f}"
              f"   |d1| mean {r['d1_mean']:.4f}")
        print(f"  {'':<10} antisymmetric energy fraction {frac:.1%}")
    print("  d0 = d1 (symmetric) bends the curve by only 0.096*|d|;")
    print("  d0 = -d1 (antisymmetric) bends it by 0.250*|d|, 2.6x harder.")
    print("  A low antisymmetric fraction means the network is using the")
    print("  weak mode and the gain will be small however large |d| gets.")

    print("\nOCCLUSION GATE\n")
    for label, (mean, p05) in gate_report.items():
        print(f"  {label:<10} alpha mean {mean:.4f}   5th pct {p05:.4f}"
              f"   (1.0 = fully trusting the flow)")

    print("\nNote: the '-RGB' rows ablate the branch AT INFERENCE on a model")
    print("trained WITH it, so they measure reliance, not necessity.")
    print("Experiment (1) still needs a flow-only model trained from scratch.")


if __name__ == "__main__":
    main()
