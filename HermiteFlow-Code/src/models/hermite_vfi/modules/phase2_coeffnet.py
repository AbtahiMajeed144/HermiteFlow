# --------------------------------------------------------
# HermiteFlow v2.1 - Phase 2: Predict endpoint velocities
#
#   In:  I0, I1, F, F', U, B0, B1, c0, c1, h0^(N), h1^(N), W0, W1
#   Out: m0, m1                                            (no t)
#
# Rebuilt per Learned_Hermite_VFI_v2.md: no U-Net anywhere in Phase 2.
# AppNet is a small encoder-only CNN; CoeffHead is single-scale at 1/8
# resolution; upsampling reuses RAFT's OWN convex mask instead of a
# learned one. ~0.43M trainable params total (vs. the full-res U-Net
# this replaces), because the trunk reads RAFT's own context features
# and final GRU state directly rather than re-deriving them.
#
# 2.1 AppNet (trainable, ~60K, gateable). Encodes appearance + blur,
#     aligned to the source lattice:
#
#       S_i = AppNet( I_i, backwarp(I_j,F), B_i, backwarp(B_j,F) )
#
#     RAFT's encoder learned WHICH PIXEL MATCHES WHICH; AppNet learns
#     WHAT KIND OF THING THIS IS AND HOW SUCH THINGS MOVE - a different
#     feature, not a subset of c_i/h_i^(N).
#
# 2.2 Fuse at 1/8 res via a single 1x1 conv over the concatenation
#
#       Psi_i = [ c_i, h_i^(N), F/8, backwarp(F',F)/8, U^{down8}, S_i ]
#
#     "Divide flow by 8": RAFT's internal flow lives in 1/8-resolution
#     pixel units - the same convention c_i/h_i^(N) were trained
#     against - so F and its aligned counterpart are spatially
#     downsampled AND value-scaled by 1/8 before concatenation. U has
#     no such pretrained convention and is only spatially downsampled.
#     A 1x1 conv over a concatenation is exactly the sum of per-group
#     sub-convs, so this is mathematically the same "additive fusion"
#     v1's CoeffNet used, just written as one conv: zeroing an input
#     group at inference removes exactly that group's linear
#     contribution and nothing else - still a retrain-free ablation
#     switch (set_appearance / set_context / set_blur below).
#
# 2.3 CoeffHead (trainable, ~336K). Single scale, no encoder/decoder:
#
#       Xi = LateralBlock^xN( Conv1x1(Psi_i) )
#       d_i^down8 = bound * tanh( head_i(Xi) / bound )     per residual
#
#     Zero-init heads: training starts exactly at d = 0, i.e. linear
#     motion. tanh caps |d_i^down8| the same way v1's CoeffNet capped
#     |d_i| - see CoeffNet's docstring for why an unconstrained
#     symmetric mode (d0 ~ d1) can random-walk under Adam regardless of
#     gradient magnitude. v2.1 has no per-clip motion scale in Phase 2
#     (RAFT's own 1/8-pixel-unit convention takes its place), so the
#     bound is now a fixed value in those units, not `bound * scale`.
#
# 2.4 ConvexUp, reusing RAFT's OWN mask - no learned upsample head.
#     d is a correction TO F, so it shares F's motion boundaries and
#     should share F's upsampler; reusing W_i is the more principled
#     choice as well as the cheaper one, and guarantees d cannot leak
#     across an edge F itself respects.
#
# 2.5 Occlusion gate, applied AFTER upsampling, at full resolution,
#     on the full-res U (sharper occlusion boundaries than gating the
#     smoothed U^{down8} would give):
#
#       alpha = sigmoid(w1 - w2 U),   d_i <- alpha * d_i
#       m_i = F + d_i
#
# 2.6 Basis conversion - see phase3_evaluate.py, unchanged.
#
# 2.7 Swapped pass: same weights, lattice-1 inputs, no extra RAFT call
#     (c1, h1^(N), W1 already exist from Phase 1).
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fi_components import LateralBlock

CONTEXT_CHANNELS = 128  # c_i, RAFT's context encoder (ReLU half of cnet)
HIDDEN_CHANNELS = 128  # h_i^(N), RAFT's final GRU hidden state
# I_i (3) + backwarp(I_j,F) (3) + B_i (3) + backwarp(B_j,F) (3)
APPNET_INPUT_CHANNELS = 12


class OcclusionGate(nn.Module):
    """
    alpha = sigmoid(w1 - w2 * x)

    Two learnable scalars, as specified, with one guarantee added that
    preserves the meaning: w2 is held non-negative via softplus. With an
    unconstrained w2 the sign can flip during training and the gate would
    then *amplify* the velocity residual precisely where its input says to
    distrust it.

    Callers decide what `x` means and whether it needs normalising first -
    v2.1's Phase 2 passes it raw, full-resolution U (pixels); nothing here
    assumes a particular scale.
    """

    def __init__(self, init_bias=5.0, init_scale=20.0):
        super().__init__()
        w2_raw = float(init_scale) if init_scale > 20.0 else float(
            torch.log(torch.expm1(torch.tensor(float(init_scale))))
        )
        self.w1 = nn.Parameter(torch.tensor(float(init_bias)))
        self.w2_raw = nn.Parameter(torch.tensor(w2_raw))

    def forward(self, x):
        w2 = F.softplus(self.w2_raw)
        return torch.sigmoid(self.w1 - w2 * x)


class AppNet(nn.Module):
    """
    Phase 2.1 - appearance encoder (Appendix A4). Trainable, ~60K params.

    Three stride-2 convs down to 1/8 resolution, channel progression
    base/2 -> base -> base (32 -> 64 -> 64 at the default base=64).
    Encoder only - no decoder, no skips: v2.1 has no U-Net anywhere in
    Phase 2.
    """

    def __init__(self, base_channels=64, in_channels=APPNET_INPUT_CHANNELS):
        super().__init__()
        c1, c2 = base_channels // 2, base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c2, c2, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.out_channels = c2

    def forward(self, x):
        return self.net(x)


def convex_upsample(field, mask):
    """
    Phase 2.4 / Appendix A8 - ConvexUp. Reuses RAFT's own convex-upsample
    formula verbatim (`RAFT.upsample_flow`), generalised to any channel
    count so every active residual head can be upsampled in one call (the
    mask/softmax depend only on the lattice, not the head).

    Args:
        field: (B, C, h, w) - already scaled to what RAFT itself passes
               here (`8 * flow`; CoeffHead does `8 * delta_downsampled`).
        mask:  (B, 576, h, w) - raw (pre-softmax) mask logits, W_i.
    Returns:
        (B, C, 8h, 8w)
    """
    n, c, h, w = field.shape
    mask = mask.view(n, 1, 9, 8, 8, h, w)
    mask = torch.softmax(mask, dim=2)

    up = F.unfold(field, [3, 3], padding=1)
    up = up.view(n, c, 9, 1, 1, h, w)
    up = torch.sum(mask * up, dim=2)
    up = up.permute(0, 1, 4, 2, 5, 3)
    return up.reshape(n, c, 8 * h, 8 * w)


class CoeffNet(nn.Module):
    """
    Phase 2 (v2.1): AppNet + CoeffHead. Predicts the velocity residuals
    d0, d1 (and, for the quartic ablation, a third residual feeding
    coefficient C), reading RAFT's own context features, final GRU hidden
    state and convex-upsample mask directly instead of re-deriving them.
    Runs entirely at 1/8 resolution until the ConvexUp step.

    Units. Flow-valued inputs (F, backwarp(F',F)) are converted to RAFT's
    own 1/8-pixel-unit convention (spatial downsample + /8) rather than
    normalised by a per-clip motion scale as v1's CoeffNet did - c_i and
    h_i^(N) were trained in that convention, so matching it is what makes
    them directly usable. `residual_bound` is therefore a fixed cap in
    1/8-pixel units, applied to the head output BEFORE the x8 ConvexUp
    step, not `bound * scale`.

    Bounded output, same reasoning as v1: Phase 3 observes the residuals
    only through beta2(t)[(t-1)d0 + t*d1], an ill-conditioned map under
    which the symmetric mode d0~d1 is barely constrained by the loss, so
    Adam's per-parameter gradient normalisation lets it random-walk away
    unless something else stops it. tanh caps |d_i^down8| while staying
    exactly linear for small arguments, so at realistic magnitudes it is
    numerically inert and only binds once a run is already diverging.

    Ablation gates (doc's table: full / no-appearance / flow-only-strict /
    no-blur), same DDP-safe requires_grad-toggling pattern v1 used for
    `use_rgb_branch`:

      set_appearance(False)  zeroes S_i (skips running AppNet, freezes it)
      set_context(False)     zeroes c_i, h_i^(N) (no parameters to freeze -
                              they come from the frozen flow estimator)
      set_blur(False)        zeroes only the B_i channels feeding AppNet,
                              leaving I_i / backwarp(I_j,F) intact

    "Flow-only (strict)" is set_appearance(False) + set_context(False)
    together.
    """

    def __init__(
        self,
        appnet_channels=64,
        coeff_head_channels=96,
        coeff_head_blocks=2,
        gate_init_bias=2.0,
        gate_init_scale=4.0,
        use_appearance=True,
        use_context=True,
        use_blur=True,
        num_residuals=2,
        residual_bound=12.0,
        context_channels=CONTEXT_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
    ):
        super().__init__()
        self.num_residuals = num_residuals
        self.residual_bound = float(residual_bound)
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels

        self.app_net = AppNet(base_channels=appnet_channels)

        fused_channels = (
            context_channels
            + hidden_channels
            + 2  # F/8
            + 2  # backwarp(F',F)/8
            + 1  # U^{down8}
            + self.app_net.out_channels  # S_i
        )
        self.fuse = nn.Conv2d(fused_channels, coeff_head_channels, 1)
        self.trunk = nn.Sequential(
            *[LateralBlock(coeff_head_channels) for _ in range(coeff_head_blocks)]
        )

        self.heads = nn.ModuleList(
            [
                nn.Conv2d(coeff_head_channels, 2, 3, 1, 1, padding_mode="reflect")
                for _ in range(num_residuals)
            ]
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.gate = OcclusionGate(gate_init_bias, gate_init_scale)

        self.num_active_heads = num_residuals
        self.set_appearance(use_appearance)
        self.set_context(use_context)
        self.set_blur(use_blur)

    # ---------------- runtime switches ----------------

    def set_appearance(self, enabled):
        """Ablation (1)'s v2.1 form: zero AppNet's contribution entirely."""
        self.use_appearance = bool(enabled)
        for param in self.app_net.parameters():
            param.requires_grad = self.use_appearance

    def set_context(self, enabled):
        """Zero c_i, h_i^(N) before fusion - no parameters to freeze."""
        self.use_context = bool(enabled)

    def set_blur(self, enabled):
        """Zero only the blur-descriptor channels feeding AppNet."""
        self.use_blur = bool(enabled)

    def set_active_heads(self, num_used):
        """
        Keep only the heads the configured degree actually consumes. See
        phase3_evaluate.RESIDUALS_PER_DEGREE. Rebuild the optimizer after
        calling this.
        """
        assert 1 <= num_used <= len(self.heads)
        self.num_active_heads = num_used
        for index, head in enumerate(self.heads):
            for param in head.parameters():
                param.requires_grad = index < num_used

    def param_groups(self, lr, head_lr_divisor=50.0, weight_decay=0.0):
        """
        Optimizer parameter groups. Heads train at a fraction of the trunk's
        learning rate for the same reason v1's CoeffNet needed it: the
        residuals are returned as `bound * tanh(head(x)/bound)`, so near
        zero (where training starts and mostly stays) the gradient reaching
        the head weights is scaled by the x8 ConvexUp and by whatever
        downstream loss weight applies - large enough that a shared
        learning rate lets the zero-initialised heads destabilise before
        the trunk has learned anything to feed them.

        Weight decay is disabled on the heads and on the gate's scalars -
        decaying a zero-initialised head fights the signal you are trying
        to grow.
        """
        head_params, scalar_params, trunk_params = [], [], []
        for name, param in self.named_parameters():
            if name.startswith("heads."):
                head_params.append(param)
            elif name.startswith("gate."):
                scalar_params.append(param)
            else:
                trunk_params.append(param)
        return [
            {"params": trunk_params, "lr": lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": lr / head_lr_divisor, "weight_decay": 0.0},
            {"params": scalar_params, "lr": lr, "weight_decay": 0.0},
        ]

    # ---------------- forward ----------------

    def forward(
        self,
        img_src,
        img_dst_aligned,
        flow,
        flow_bwd_aligned,
        occlusion,
        blur_src,
        blur_dst_aligned,
        context,
        hidden,
        mask,
    ):
        """
        Every tensor except `context`/`hidden`/`mask` is on the SOURCE
        lattice; those three are already lattice-native - computed
        directly by RAFT(img_src, img_dst) in Phase 1, not warped in from
        the other side.

        Args:
            img_src, img_dst_aligned: (B, 3, H, W)  I0, backwarp(I1, F)
            flow:             (B, 2, H, W)  F,        pixels
            flow_bwd_aligned: (B, 2, H, W)  backwarp(F', F), pixels
            occlusion:        (B, 1, H, W)  U,        pixels, full res
            blur_src, blur_dst_aligned: (B, 3, H, W)  B0, backwarp(B1, F)
            context: (B, 128, H/8, W/8)  c_i
            hidden:  (B, 128, H/8, W/8)  h_i^(N)
            mask:    (B, 576, H/8, W/8)  W_i, pre-softmax

        Returns:
            residuals: list of `num_active_heads` tensors (B, 2, H, W),
                       in pixels, occlusion-gated. [d0, d1] (+ d2).
            alpha:     (B, 1, H, W), the gate map - needed by the
                       cross-lattice velocity-consistency loss.
        """
        h, w = context.shape[-2:]

        blur_src_in = blur_src if self.use_blur else torch.zeros_like(blur_src)
        blur_dst_in = (
            blur_dst_aligned if self.use_blur else torch.zeros_like(blur_dst_aligned)
        )
        appearance = torch.cat(
            [img_src, img_dst_aligned, blur_src_in, blur_dst_in], dim=1
        )
        if self.use_appearance:
            s_feat = self.app_net(appearance)
            if s_feat.shape[-2:] != (h, w):
                # Guards against off-by-one rounding when H/W are not
                # exact multiples of 8; a no-op whenever they are.
                s_feat = F.interpolate(
                    s_feat, size=(h, w), mode="bilinear", align_corners=False
                )
        else:
            s_feat = torch.zeros(
                img_src.shape[0], self.app_net.out_channels, h, w,
                device=img_src.device, dtype=img_src.dtype,
            )

        # "Divide flow by 8": spatial downsample to RAFT's own working
        # resolution AND rescale into its 1/8-pixel-unit convention.
        flow_ds = F.interpolate(
            flow, size=(h, w), mode="bilinear", align_corners=False
        ) / 8.0
        flow_bwd_ds = F.interpolate(
            flow_bwd_aligned, size=(h, w), mode="bilinear", align_corners=False
        ) / 8.0
        occ_ds = F.interpolate(
            occlusion, size=(h, w), mode="bilinear", align_corners=False
        )

        ctx_in = context if self.use_context else torch.zeros_like(context)
        hid_in = hidden if self.use_context else torch.zeros_like(hidden)

        fused = torch.cat([ctx_in, hid_in, flow_ds, flow_bwd_ds, occ_ds, s_feat], dim=1)
        feat = self.trunk(self.fuse(fused))

        bound = self.residual_bound
        raw = [
            bound * torch.tanh(self.heads[i](feat) / bound)
            for i in range(self.num_active_heads)
        ]

        # One ConvexUp call for every active head: the mask/softmax only
        # depend on the lattice, not the head, so batching on the channel
        # dimension avoids recomputing it per head.
        stacked = convex_upsample(8.0 * torch.cat(raw, dim=1), mask)
        upsampled = torch.chunk(stacked, self.num_active_heads, dim=1)

        # Gate AFTER upsampling, at full resolution, on the full-res U -
        # sharper occlusion boundaries than the smoothed U^{down8} would
        # give. Where RAFT is unreliable the correction dies and the
        # trajectory reverts to linear.
        alpha = self.gate(occlusion)
        residuals = [alpha * d for d in upsampled]
        return residuals, alpha


# ---------------- helpers ----------------


def run_both_sides(
    coeff_net,
    img0,
    img1,
    flow_fwd,
    flow_bwd,
    occ_fwd,
    occ_bwd,
    blur0,
    blur1,
    context0,
    context1,
    hidden0,
    hidden1,
    mask0,
    mask1,
    align,
):
    """
    Run the same weights on the forward inputs and on the swapped inputs,
    batched into a single forward pass.

        lattice 0:  (I0, backwarp(I1,F ), F , backwarp(F',F ), U ,
                     B0, backwarp(B1,F ), c0, h0^(N), W0)
        lattice 1:  (I1, backwarp(I0,F'), F', backwarp(F ,F'), U',
                     B1, backwarp(B0,F'), c1, h1^(N), W1)

    Note that occ_bwd must be computed ON LATTICE 1, i.e.
    occ_bwd = || F' + backwarp(F, F') ||_1, and is NOT occ_fwd warped
    across - see phase1_measure.forward_backward_error. Getting this
    wrong is silent: the shapes match and training proceeds, but the gate
    fires in the wrong places on the frame-1 side.

    context0/hidden0/mask0 and context1/hidden1/mask1 are already
    lattice-native (Phase 1 computed them directly on each side) and are
    NOT warped by `align` - only the RGB/flow/blur terms are.

    Args:
        align: callable(tensor_on_other_lattice, flow) -> aligned tensor.

    Returns:
        (residuals_0, residuals_1, alpha_0, alpha_1) - residuals are each
        a list of (B, 2, H, W) tensors on lattice 0 and lattice 1
        respectively; alpha_0, alpha_1 are (B, 1, H, W) gate maps.
    """
    batch = img0.shape[0]

    residuals, alpha = coeff_net(
        torch.cat([img0, img1], dim=0),
        torch.cat([align(img1, flow_fwd), align(img0, flow_bwd)], dim=0),
        torch.cat([flow_fwd, flow_bwd], dim=0),
        torch.cat([align(flow_bwd, flow_fwd), align(flow_fwd, flow_bwd)], dim=0),
        torch.cat([occ_fwd, occ_bwd], dim=0),
        torch.cat([blur0, blur1], dim=0),
        torch.cat([align(blur1, flow_fwd), align(blur0, flow_bwd)], dim=0),
        torch.cat([context0, context1], dim=0),
        torch.cat([hidden0, hidden1], dim=0),
        torch.cat([mask0, mask1], dim=0),
    )

    return (
        [r[:batch] for r in residuals],
        [r[batch:] for r in residuals],
        alpha[:batch],
        alpha[batch:],
    )


@torch.no_grad()
def residual_stats(residuals, scale, prefix="coeff"):
    """
    Diagnostic for the headline claim. Log this from the first few
    thousand steps, not after a full run.

    If the residual magnitudes stay near zero the network has fallen
    back to linear motion and the contribution does not exist yet - the
    failure mode the Time Lens++ ablation predicts, and one that
    zero-init makes easy to slide into.

    `scale` is Phase 4/5's per-clip motion scale (flow_scale) - Phase 2
    itself no longer uses it, but it is still the right normaliser for
    reading these numbers: absolute pixel magnitude is meaningless
    without knowing how much the clip moved.
    """
    stats = {}
    for i, residual in enumerate(residuals):
        magnitude = residual.norm(dim=1)
        stats[f"{prefix}/d{i}_px"] = magnitude.mean().item()
        stats[f"{prefix}/d{i}_rel"] = (magnitude.mean() / scale.mean()).item()
        stats[f"{prefix}/d{i}_p99_px"] = magnitude.flatten().quantile(0.99).item()
    return stats
