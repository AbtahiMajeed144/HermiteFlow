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

    # ---- Phase 2: endpoint velocities ----
    coeff_net_channels: int = 64
    # Occlusion gate alpha = sigmoid(w1 - w2 * U/s), initial values.
    # w1 = 5.0 and w2 = 20.0 put the half-way point at U/s = 0.25,
    # i.e. a forward-backward error of a quarter of the clip's peak
    # motion, and start the gate wide open for consistent flow.
    gate_init_bias: float = 5.0
    gate_init_scale: float = 20.0
    # Experiment (1): the CoeffNet input ablation the headline claim
    # rests on. False keeps the flow branch (F, backwarp(F',F), U) and
    # drops the RGB branch (I0, backwarp(I1,F)). The branches are fused
    # by addition, so this is a pure runtime switch - the same
    # checkpoint runs either way and the flow branch needs no retraining.
    use_rgb_branch: bool = True
    # Hard cap on |d_i|, in units of the motion scale s, applied with a
    # tanh so the map is exactly linear for small residuals.
    #
    # Phase 3 observes the residuals only through
    #     Phi(t) - t F = beta2(t) [ (t - 1) d0 + t d1 ]
    # whose singular values over t in k/8 are 0.334 and 0.097: the
    # symmetric mode d0 ~ +d1 shifts the trajectory 7.7x less than the
    # antisymmetric one, by 0.043 px per pixel of residual. The loss
    # barely constrains that direction, but Adam normalises away
    # gradient magnitude and steps along it at full size regardless, so
    # the residuals random-walk. A stage-1 run reached |d0| = 1183 px
    # while its loss was still 0.026, then overflowed to NaN.
    #
    # The oracle wants ~0.375 s, so the default 2.0 sits ~5x clear of
    # any real solution: distortion is 7.5e-06 at the |d| training
    # currently reaches. It binds only when a run is already diverging.
    residual_bound: float = 2.0
    # Optional third downsample in CoeffNet, bottleneck at H/8 instead
    # of H/4. Off by default - costs ~1.6M params on a network whose
    # count is already the awkward number next to GIMM's 0.25M motion
    # module, and the case for it (object-scale context at a fast
    # object) is speculation, not measurement. Turn on only if
    # large-displacement benchmarks underperform.
    coeff_net_deep: bool = False
    # Linear-cost global attention at the CoeffNet bottleneck: every
    # location cross-attends to a FIXED T x T pool of the whole feature
    # map (see GlobalContext in phase2_coeffnet.py), rather than to
    # every other location - plain pixel self-attention is quadratic
    # in token count, and this model is evaluated at 2K/4K while it
    # trains at 256x256, where a quadratic cost is not slow, it does
    # not run. Zero-initialised, so it is the identity until training
    # moves it - off by default until it has been A/B'd against a
    # checkpoint that did not have it.
    use_global_context: bool = False
    global_context_tokens: int = 8
    global_context_heads: int = 4

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
