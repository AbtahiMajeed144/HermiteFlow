# --------------------------------------------------------
# HermiteFlow — Phase 1: Measure
#
#   In:  I0, I1        Out: F, F', U, Z
#
#   F, F' = RAFT(I0, I1)
#   U     = |F  + backwarp(F', F)|_1        occlusion signal
#   Z     = -|I0 - backwarp(I1, F)|_1       splat importance
#
# U: travel forward then back; failure to return means occlusion.
# Z: how well a pixel matches its correspondence - used by Phase 4
#    to resolve collisions during splatting.
#
# RAFT (or FlowFormer) is frozen. This is the only measurement in
# the pipeline; everything downstream is inference about the
# unobserved interval.
#
# This file holds only the pure-math part of Phase 1. The flow
# estimator itself lives in ../raft and ../flowformer and is
# invoked by the model.
# --------------------------------------------------------

import torch

from .fi_utils import warp


def forward_backward_error(flow_fwd, flow_bwd):
    """
    Forward-backward consistency error, Phase 1's `U`.

        U(x) = | F(x) + F'(x + F(x)) |_1

    Go forward along F, then come back along F'; if you do not land
    where you started, that pixel is occluded (or the flow is wrong),
    and Phase 2 uses U to distrust it.

    Args:
        flow_fwd: (B, 2, H, W)  F   = f_{0->1}
        flow_bwd: (B, 2, H, W)  F'  = f_{1->0}

    Returns:
        U: (B, 1, H, W), non-negative, in pixels.
    """
    # backwarp(F', F): sample F' at the position each frame-0 pixel
    # moves to, i.e. F'(x + F(x)).
    flow_bwd_at_dst = warp(flow_bwd, flow_fwd)
    return (flow_fwd + flow_bwd_at_dst).abs().sum(dim=1, keepdim=True)


def brightness_consistency(img_src, img_dst, flow):
    """
    Splat importance `Z`, on the source lattice.

        Z(x) = -| I_src(x) - I_dst(x + F(x)) |_1

    How well a pixel matches the correspondence the flow claims for it.
    Phase 4 splats with weight e^Z, so a pixel whose correspondence is
    photometrically convincing wins collisions against one whose is not -
    which is what puts foreground in front of background.

    Z <= 0 by construction, and for images in [0, 1] it is bounded below
    by -3, so e^Z lands in [0.05, 1] and needs no max-subtraction.

    Args:
        img_src: (B, 3, H, W) in [0, 1], on the source lattice
        img_dst: (B, 3, H, W) in [0, 1], on the destination lattice
        flow:    (B, 2, H, W) source -> destination

    Returns:
        Z: (B, 1, H, W), non-positive.
    """
    matched = warp(img_dst, flow)
    return -(img_src - matched).abs().sum(dim=1, keepdim=True)


def align_to_source(tensor_on_dst, flow):
    """
    Pull a field defined on the destination lattice back onto the source
    lattice: `backwarp(X', F)`.

    Phase 2 encodes `F, backwarp(F', F), U` and `I0, backwarp(I1, F)`,
    i.e. everything on lattice 0. Tensors on different lattices cannot be
    stacked and handed to a convolution as if they were registered; this
    is the function that makes them commensurable.
    """
    return warp(tensor_on_dst, flow)


def flow_scale(flow_fwd, flow_bwd, min_scale=1.0):
    """
    Per-sample motion magnitude s = max(|F|, |F'|) over all pixels and
    both flow components.

    Phase 2 is a CNN and must not have to cope with inputs whose
    numeric range swings between 0.1 px (a static shot) and 200 px (a
    whip pan). Every quantity in pixels (F, F', U, and the predicted
    A, B) is divided by s on the way in and multiplied by s on the way
    out, which makes CoeffNet exactly scale-equivariant: doubling the
    motion doubles A and B, as the physics requires.

    Args:
        flow_fwd, flow_bwd: (B, 2, H, W)
        min_scale: floor, so near-static clips do not amplify noise.

    Returns:
        s: (B, 1, 1, 1)
    """
    both = torch.cat([flow_fwd, flow_bwd], dim=1)
    s = both.abs().amax(dim=(1, 2, 3), keepdim=True)
    return s.clamp_min(min_scale)
