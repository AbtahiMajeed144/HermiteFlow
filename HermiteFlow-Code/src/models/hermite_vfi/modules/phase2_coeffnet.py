# --------------------------------------------------------
# HermiteFlow - Phase 2: Predict endpoint velocities
#
#   In:  I0, I1, F, F', U      Out: m0, m1      (no t)
#
# 2.1 Encode. Align every input to lattice 0, then encode through
#     two branches:
#
#       Xi = CoeffNet( F, backwarp(F', F), U  |  I0, backwarp(I1, F) )
#                      \___ flow branch ___/     \__ RGB branch __/
#
#     The RGB branch is separately gateable: zeroing it must not
#     require retraining the flow branch. This is the ablation the
#     headline claim rests on, so the two branches are fused by
#     ADDITION, not concatenation - with concat, zeroing the RGB
#     half changes the input distribution of the fusion conv and
#     the flow branch would have to be retrained to compensate.
#
#     Each branch ends in GroupNorm and carries a learnable scalar
#     gain. Without this the two branches see very different input
#     distributions (unit-scale normalised flow vs raw RGB) and
#     nothing forces them to contribute comparably - the RGB branch
#     could be silently suppressed, and ablation (1) would come back
#     flat for an architectural reason rather than a scientific one.
#     Log both gains during training.
#
# 2.2 Two heads.
#       d0, d1 = heads(Xi)
#       m0 = F + d0        velocity as the pixel leaves frame 0
#       m1 = F + d1        velocity as it arrives at frame 1
#     d_i = 0 means constant velocity F throughout, i.e. linear.
#     Both head output convs are zero-initialised, so training
#     begins at exactly the linear baseline.
#
# 2.3 Occlusion gate.
#       alpha = sigmoid(w1 - w2 U),   d_i <- alpha * d_i
#     Where RAFT is unreliable the correction dies and the
#     trajectory reverts to linear.
#
# 2.4 Convert to the evaluation basis (see phase3_evaluate.py):
#       A = -2 d0 - d1,    B = d0 + d1
#
# 2.5 Frame-1 side: the same weights on the swapped pair.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fi_components import LateralBlock

# F (2) + backwarp(F', F) (2) + U (1)
FLOW_BRANCH_CHANNELS = 5
# I0 (3) + backwarp(I1, F) (3)
RGB_BRANCH_CHANNELS = 6


class OcclusionGate(nn.Module):
    """
    alpha = sigmoid(w1 - w2 * U)

    Two learnable scalars, as specified, with two guarantees added that
    preserve the meaning:

    * w2 is held non-negative via softplus. With an unconstrained w2 the
      sign can flip during training and the gate would then *amplify* the
      velocity residual precisely where the flow is least trustworthy.

    * U arrives pre-normalised by the clip's motion magnitude, so one
      learned threshold is meaningful across clips with very different
      motion.
    """

    def __init__(self, init_bias=5.0, init_scale=20.0):
        super().__init__()
        w2_raw = float(init_scale) if init_scale > 20.0 else float(
            torch.log(torch.expm1(torch.tensor(float(init_scale))))
        )
        self.w1 = nn.Parameter(torch.tensor(float(init_bias)))
        self.w2_raw = nn.Parameter(torch.tensor(w2_raw))

    def forward(self, occlusion_norm):
        w2 = F.softplus(self.w2_raw)
        return torch.sigmoid(self.w1 - w2 * occlusion_norm)


class _Branch(nn.Module):
    """
    Stem for one input branch, producing `channels` feature maps.

    Ends in GroupNorm plus a learnable scalar gain so the flow and RGB
    branches are on comparable footing before they are summed. The flow
    branch sees unit-scale normalised input; the RGB branch sees raw
    images. Without normalisation nothing forces the two to contribute
    comparably, and a suppressed RGB branch would make ablation (1) come
    back flat for an architectural reason rather than a scientific one.

    The gain is also a measurement: if the RGB gain grows relative to
    the flow gain, the network is choosing to use image context, which
    is evidence for the RGB-curvature claim independent of the on/off
    ablation.
    """

    def __init__(self, in_channels, channels, groups=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            LateralBlock(channels),
            nn.GroupNorm(groups, channels),
        )
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.gain * self.layers(x)


class CoeffNet(nn.Module):
    """
    Predicts the velocity residuals d0, d1 (and, for the quartic
    ablation, a third residual feeding coefficient C).

    Scale handling: every input in pixel units (F, backwarp(F',F), U) is
    divided by the per-sample motion scale s on the way in, and the
    predicted residuals are multiplied by s on the way out. The network
    therefore sees a scale-free problem while its predictions stay
    dimensionally correct - double the motion, double the residuals.

    Because that output multiplication also multiplies the gradient
    reaching the head convolutions by s (tens to hundreds of pixels),
    the heads see an effective learning rate far above the rest of the
    network. Build the optimizer with `param_groups()`.

    Bounded output. Phase 3 observes the residuals only through

        Phi(t) - t F = beta2(t) [ (t - 1) d0 + t d1 ]

    so the map (d0, d1) -> trajectory is ill-conditioned: over t in
    k/8, the antisymmetric mode d0 = -d1 moves the trajectory ~7.7x
    more than the symmetric mode d0 = +d1, which shifts it by only
    0.043 px per pixel of residual. That flat direction is not free -
    Adam normalises per-parameter gradient magnitude, so a direction
    the loss barely constrains still receives a full ~lr step, and the
    residuals random-walk along it. Observed directly: a run at
    lr 5e-4 reached |d0| = 1183 px while the loss was still 0.026,
    then overflowed. tanh caps |d_i| at `residual_bound * s`, which at
    the default 2.0 sits ~5x above anything the oracle asks for, so it
    binds only when the run is already diverging.
    """

    def __init__(
        self,
        channels=64,
        gate_init_bias=5.0,
        gate_init_scale=20.0,
        use_rgb_branch=True,
        num_residuals=2,
        deep=False,
        residual_bound=2.0,
    ):
        super().__init__()
        self.use_rgb_branch = use_rgb_branch
        self.num_residuals = num_residuals
        self.deep = deep
        self.residual_bound = float(residual_bound)

        c1, c2, c3 = channels, channels * 2, channels * 4

        self.flow_branch = _Branch(FLOW_BRANCH_CHANNELS, c1)
        # Always constructed, so a checkpoint trained with RGB can be
        # evaluated with the branch gated off and vice versa - the
        # ablation is a runtime switch, not a different architecture.
        self.rgb_branch = _Branch(RGB_BRANCH_CHANNELS, c1)

        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            LateralBlock(c2),
            LateralBlock(c2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            LateralBlock(c3),
            LateralBlock(c3),
        )

        # Optional third downsample, bottleneck at H/8 instead of H/4.
        # OFF by default. The hypothesis is that object-scale context is
        # needed for the RGB-curvature claim and H/4 may not cover a fast
        # object at 1080p - but that is speculation, and it costs ~1.6M
        # parameters in a network whose parameter count is already the
        # awkward number next to GIMM's 0.25M motion module. Turn it on
        # only if large-displacement benchmarks (X-TEST, SNU-FILM
        # extreme) underperform. Channels stay at c3 rather than
        # doubling, since down2 already holds most of the parameters.
        if deep:
            self.down3 = nn.Sequential(
                nn.Conv2d(c3, c3, 3, 2, 1),
                nn.LeakyReLU(0.1, inplace=True),
                LateralBlock(c3),
            )
            self.up0 = nn.Sequential(
                nn.Conv2d(c3 + c3, c3, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                LateralBlock(c3),
            )

        self.up1 = nn.Sequential(
            nn.Conv2d(c3 + c2, c2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            LateralBlock(c2),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(c2 + c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            LateralBlock(c1),
        )

        # One head per residual: d0, d1 (+ d2 for the quartic ablation).
        self.heads = nn.ModuleList(
            [
                nn.Conv2d(c1, 2, 3, 1, 1, padding_mode="reflect")
                for _ in range(num_residuals)
            ]
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.gate = OcclusionGate(gate_init_bias, gate_init_scale)

        self.num_active_heads = num_residuals
        self.set_rgb_branch(use_rgb_branch)

    # ---------------- runtime switches ----------------

    def set_rgb_branch(self, enabled):
        """
        Runtime switch for ablation (1). No retraining, no reshaping.

        Also flips requires_grad: a branch that never contributes to the
        output receives no gradient, and DDP with
        find_unused_parameters=False raises on exactly that. Freezing it
        removes it from DDP's bucket and from the optimizer, while the
        weights stay in the state_dict so one checkpoint still serves
        both arms of the ablation.

        Rebuild the optimizer after calling this - a frozen parameter
        left in an AdamW group keeps drifting under weight decay despite
        receiving no gradient.
        """
        self.use_rgb_branch = bool(enabled)
        for param in self.rgb_branch.parameters():
            param.requires_grad = self.use_rgb_branch

    def set_active_heads(self, num_used):
        """
        Keep only the heads the configured degree actually consumes.

          1 head  -> quadratic (B = 0), matches IQ-VFI's single term
          2 heads -> cubic, ours
          3 heads -> quartic, ablation upper end

        The module always builds enough heads for the widest degree so
        that a single checkpoint serves the whole degree ablation, but
        the unused ones must be frozen for the same DDP reason as above.
        Rebuild the optimizer after calling this.
        """
        assert 1 <= num_used <= len(self.heads)
        self.num_active_heads = num_used
        for index, head in enumerate(self.heads):
            for param in head.parameters():
                param.requires_grad = index < num_used

    def param_groups(self, lr, head_lr_divisor=50.0, weight_decay=0.0):
        """
        Optimizer parameter groups.

        The residuals are returned as `head(feat) * alpha * scale`, so
        the gradient reaching the head weights carries a factor of
        `scale` - tens to hundreds of pixels, and varying per sample.
        With a shared learning rate the heads train far faster than the
        trunk, and the instability lands in the first few thousand
        steps, which is exactly when the zero-initialised heads are
        leaving the linear baseline. Adam's per-parameter normalisation
        absorbs some of this, so treat the divisor as insurance; if
        training is stable with a single group, drop it.

        Weight decay is disabled on the heads and on the scalar
        parameters (branch gains, gate) - decaying a zero-initialised
        head just fights the signal you are trying to grow.
        """
        head_params, scalar_params, trunk_params = [], [], []
        for name, param in self.named_parameters():
            if name.startswith("heads."):
                head_params.append(param)
            elif name.endswith(".gain") or name.startswith("gate."):
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
        scale,
    ):
        """
        Every argument is on the SOURCE lattice.

        Args:
            img_src:          (B, 3, H, W)  I0
            img_dst_aligned:  (B, 3, H, W)  backwarp(I1, F)
            flow:             (B, 2, H, W)  F,        pixels
            flow_bwd_aligned: (B, 2, H, W)  backwarp(F', F), pixels
            occlusion:        (B, 1, H, W)  U,        pixels
            scale:            (B, 1, 1, 1)  s,        pixels

        Returns:
            residuals: list of `num_active_heads` tensors (B, 2, H, W),
                       in pixels, occlusion-gated. [d0, d1] (+ d2).
        """
        occlusion_norm = occlusion / scale

        feat = self.flow_branch(
            torch.cat(
                [flow / scale, flow_bwd_aligned / scale, occlusion_norm], dim=1
            )
        )
        if self.use_rgb_branch:
            feat = feat + self.rgb_branch(
                torch.cat([img_src, img_dst_aligned], dim=1)
            )

        f1 = feat
        f2 = self.down1(f1)
        f3 = self.down2(f2)

        if self.deep:
            f4 = self.down3(f3)
            u0 = F.interpolate(
                f4, size=f3.shape[-2:], mode="bilinear", align_corners=False
            )
            f3 = self.up0(torch.cat([u0, f3], dim=1))

        u1 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, f2], dim=1))
        u2 = F.interpolate(u1, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, f1], dim=1))

        alpha = self.gate(occlusion_norm)
        # Saturating output. See the class docstring: the symmetric mode
        # d0 ~ d1 is nearly invisible to the trajectory loss, so nothing
        # in the objective stops it drifting. tanh caps |d_i| at
        # bound * s while staying EXACTLY linear for small arguments -
        # at the magnitudes the oracle actually wants (~0.375 s) this is
        # numerically the identity, so it changes no result, it only
        # removes the escape.
        bound = self.residual_bound
        return [
            alpha * bound * scale * torch.tanh(self.heads[i](u2) / bound)
            for i in range(self.num_active_heads)
        ]


# ---------------- helpers ----------------


def run_both_sides(
    coeff_net,
    img0,
    img1,
    flow_fwd,
    flow_bwd,
    occ_fwd,
    occ_bwd,
    scale,
    align,
):
    """
    Run the same weights on the forward inputs and on the swapped inputs,
    batched into a single forward pass.

        lattice 0:  (I0, backwarp(I1, F ), F , backwarp(F', F ), U )
        lattice 1:  (I1, backwarp(I0, F'), F', backwarp(F , F'), U')

    Note that occ_bwd must be computed ON LATTICE 1, i.e.

        occ_bwd = || F' + backwarp(F, F') ||_1

    and is NOT occ_fwd warped across. Getting this wrong is silent - the
    shapes match and training proceeds, but the gate fires in the wrong
    places on the frame-1 side.

    Args:
        align: callable(tensor_on_other_lattice, flow) -> aligned tensor.

    Returns:
        (residuals_0, residuals_1) - each a list of (B, 2, H, W) tensors
        on lattice 0 and lattice 1 respectively.
    """
    batch = img0.shape[0]

    residuals = coeff_net(
        torch.cat([img0, img1], dim=0),
        torch.cat([align(img1, flow_fwd), align(img0, flow_bwd)], dim=0),
        torch.cat([flow_fwd, flow_bwd], dim=0),
        torch.cat([align(flow_bwd, flow_fwd), align(flow_fwd, flow_bwd)], dim=0),
        torch.cat([occ_fwd, occ_bwd], dim=0),
        torch.cat([scale, scale], dim=0),
    )

    return (
        [r[:batch] for r in residuals],
        [r[batch:] for r in residuals],
    )


@torch.no_grad()
def residual_stats(residuals, scale, prefix="coeff"):
    """
    Diagnostic for the headline claim. Log this from the first few
    thousand steps, not after a full run.

    If the residual magnitudes stay near zero the network has fallen
    back to linear motion and the contribution does not exist yet -
    the failure mode the Time Lens++ ablation predicts, and one that
    zero-init makes easy to slide into.

    Watch the relative figure: absolute pixel magnitude is meaningless
    without the clip's motion scale. Note also that `scale` is one
    number per clip, set by the fastest content, so static background
    will show small residuals for a numerical reason as well as a
    semantic one - read the spatial maps, not just these scalars.
    """
    stats = {}
    for i, residual in enumerate(residuals):
        magnitude = residual.norm(dim=1)
        stats[f"{prefix}/d{i}_px"] = magnitude.mean().item()
        stats[f"{prefix}/d{i}_rel"] = (magnitude.mean() / scale.mean()).item()
        stats[f"{prefix}/d{i}_p99_px"] = magnitude.flatten().quantile(0.99).item()
    return stats


@torch.no_grad()
def branch_gains(coeff_net, prefix="coeff"):
    """
    Second piece of evidence for the RGB claim, independent of the
    on/off ablation. If the RGB gain grows relative to the flow gain,
    the network is choosing to use image context.
    """
    return {
        f"{prefix}/gain_flow": coeff_net.flow_branch.gain.item(),
        f"{prefix}/gain_rgb": coeff_net.rgb_branch.gain.item(),
    }