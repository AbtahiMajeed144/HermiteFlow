# --------------------------------------------------------
# HermiteFlow — architecture configuration
# --------------------------------------------------------

from typing import Optional
from dataclasses import dataclass

from omegaconf import OmegaConf


@dataclass
class HermiteFlowConfig:
    type: str = "hermiteflow_r"
    ema: Optional[bool] = None
    ema_value: Optional[float] = None

    # ---- Phase 1: measure ----
    # Frozen flow estimator. Both paths can be overridden from the
    # command line (--raft-ckpt / --flowformer-ckpt) or with a dotlist
    # override (arch.pretrained_raft_ckpt=...).
    pretrained_raft_ckpt: str = "pretrained/raft-things.pth"
    pretrained_flowformer_ckpt: str = "pretrained/flowformer_sintel.pth"
    raft_iter: int = 20
    # Floor on the per-sample motion scale s = max(|F|, |F'|), in pixels.
    # Everything in pixel units is normalised by s before entering a CNN.
    min_flow_scale: float = 1.0

    # ---- Phase 2: endpoint velocities (v2.1: AppNet + CoeffHead) ----
    # AppNet's final-stage width; internal stem is (base//2, base, base),
    # i.e. 32 -> 64 -> 64 at the default 64. CoeffHead's fusion + trunk
    # width and its LateralBlock count - the doc's falsification test
    # asks for both a "3x128" and a "2x96" head to be run and compared,
    # so these are knobs, not hardcoded.
    appnet_channels: int = 64
    coeff_head_channels: int = 96
    coeff_head_blocks: int = 2
    # Occlusion gate alpha = sigmoid(w1 - w2 * U), initial values. v2.1
    # drops Phase 2's per-clip motion-scale normalisation (see
    # residual_bound below), so U here is RAW, full-resolution pixels,
    # not U/s. w1=2.0, w2=4.0 puts the half-way point at U = 0.5px, a
    # rough "well-tracked" threshold at the downsample the X4K configs
    # use (see hermiteflow_r_x4k_stage1.yaml's U/s measurements). Both
    # are learned, so this is a starting point, not a calibration that
    # needs to be exactly right.
    gate_init_bias: float = 2.0
    gate_init_scale: float = 4.0
    # Experiment (1), v2.1 form: the "no-appearance" ablation. False
    # zeroes AppNet's contribution (S_i) before fusion. AppNet's output
    # is one group among several concatenated into a single 1x1 fusion
    # conv, and a 1x1 conv over a concatenation is exactly the sum of
    # per-group sub-convs - so zeroing a group's input is still a pure,
    # retrain-free runtime switch, the same guarantee v1's additive
    # fusion gave under a different mechanism.
    use_appearance: bool = True
    # "Flow-only (strict)" ablation: also zero c_i, h_i^(N) (RAFT's
    # context features and final GRU state) before fusion, on top of
    # use_appearance=False. h_i^(N) is initialised from RAFT's own
    # appearance-derived cnet, so it must be zeroed alongside c_i for
    # the strict gate to be interpretable - see the doc's ablation table.
    use_context: bool = True
    # "No-blur" ablation: zero only the blur-descriptor channels feeding
    # AppNet (B_i, backwarp(B_j,F)), leaving I_i/backwarp(I_j,F) intact.
    use_blur: bool = True
    # Hard cap on |d_i^down8|, in RAFT's own 1/8-pixel-unit convention
    # (v2.1 has no per-clip motion scale in Phase 2 - c_i/h_i^(N) were
    # trained in that unit system, so Phase 2's flow inputs are converted
    # to match it instead of being normalised by s). Applied with a tanh
    # so the map is exactly linear for small residuals, at the 1/8-res
    # head output, before the x8 ConvexUp step.
    #
    # Phase 3 observes the residuals only through
    #     Phi(t) - t F = beta2(t) [ (t - 1) d0 + t d1 ]
    # whose singular values over t in k/8 are 0.334 and 0.097: the
    # symmetric mode d0 ~ +d1 shifts the trajectory 7.7x less than the
    # antisymmetric one. The loss barely constrains that direction, but
    # Adam normalises away gradient magnitude and steps along it at full
    # size regardless, so the residuals random-walk - v1's CoeffNet hit
    # exactly this (|d0| = 1183 px at loss 0.026, then NaN) before its
    # own residual_bound was added; the same failure mode applies here
    # in the new units.
    #
    # 12.0 in 1/8-pixel units is ~96px at full resolution - generously
    # clear of realistic content on the X4K configs (native |F| ~61px,
    # ds 3.0 |F| ~22px), binding only once a run is already diverging.
    # Freshly recalibrated for v2.1; watch delta_0/delta_1 in the first
    # few thousand steps the way v1's bound was tuned.
    residual_bound: float = 12.0

    # ---- Training stage ----
    # Mirrors GIMM-VFI's two-stage recipe (paper Tab. 5, repo
    # configs/gimm + configs/gimmvfi):
    #   1 = motion only. Phases 1-3 run, phases 4-5 are skipped and
    #       frozen, and the only objective is trajectory distillation
    #       against pseudo-ground-truth flow. GIMM's stage 1 is exactly
    #       this: train the motion module alone on an MSE flow loss.
    #   2 = joint. Everything trains on image losses, with the flow
    #       term retained as a regulariser - GIMM's L_rec, which exists
    #       to stop the pre-trained motion module drifting once the
    #       synthesis module starts absorbing error.
    # Stage 2 is initialised from a stage-1 checkpoint via --load-path.
    train_stage: int = 2

    # ---- Phase 3: trajectory degree ----
    # Experiment (2): "linear" (d_i = 0, RIFE-style) | "quadratic"
    # (B = 0, IQ-VFI) | "cubic" (ours) | "quartic" (adds C*beta4, the
    # ablation upper end). CoeffNet always builds enough heads for the
    # widest degree, so one checkpoint serves the whole ablation.
    degree: str = "cubic"

    # ---- Phase 4: reverse ----
    refine_net_channels: int = 64
    refine_net_blocks: int = 3
    # Splat with softmax importance e^Z, Z = -|I0 - backwarp(I1,F)|_1,
    # so photometrically convincing correspondences win collisions and
    # foreground lands in front of background. False falls back to plain
    # average splatting.
    use_splat_importance: bool = True
    # "torch" is the portable scatter implementation: it runs on CPU and
    # CUDA, needs no extra dependency, and is the one covered by
    # others/verify_hermiteflow.py. "cupy" uses the CUDA
    # softmax-splatting kernel and is faster; "auto" picks it whenever
    # cupy imports. Run the verification script on the target machine
    # first - it compares the two backends whenever cupy is present -
    # then switch this to "auto" for the speedup.
    splat_impl: str = "torch"

    # ---- Phase 5: synthesize ----
    synth_net_channels: int = 64

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls(ema=False))
        config = OmegaConf.merge(defaults, config)
        return config
