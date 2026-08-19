# --------------------------------------------------------
# HermiteFlow — Phase 4: Reverse, lattice 0 -> lattice t
#
#   In:  Phi(t), Phi'(t), Z      Out: G0, G1
#
# Phi(t) answers "where does frame-0 pixel x go?". Warping needs
# "where does frame-t pixel u come from?" - a different lattice.
#
#                sum_{x : x + Phi(t)(x) in N(u)} w(x) e^Z(x) (-Phi(t)(x))
#   G0~(u)  =  ------------------------------------------------------------
#                              sum_x w(x) e^Z(x)
#
#   N(u)      : bilinear neighbours of u
#   w         : bilinear scatter weight
#   e^Z       : softmax-splat importance, resolves collisions - the
#               pixel whose correspondence is photometrically
#               convincing wins, which is what puts foreground in
#               front of background
#   negation  : the arrow now points from t back to 0
#   G0 = f_{t->0},  G1 = f_{t->1}
#
# G1~ likewise from Phi'(t) on lattice 1, with its own Z'.
#
# Hole filling. Let H_i be the accumulated splat weight; H_i = 0
# marks a pixel nothing landed on (two objects both moved away):
#
#   G0, G1  =  G0~, G1~ + RefineNet(G0~, G1~, H0, H1)
#
# This whole phase is what RIFE argued you should skip. Keeping it
# is the price of having a real trajectory model.
# --------------------------------------------------------

import torch
import torch.nn as nn

from .fi_components import LateralBlock

# A target pixel that receives anything at all accumulates a bilinear
# weight well above this; a pixel that receives nothing accumulates
# exactly 0.
_HOLE_THRESHOLD = 1e-3
# Floor on the importance-weighted denominator, so the division is finite
# even in the pixels that are about to be zeroed out as holes anyway.
_HOLE_EPS = 1e-6


def _splat_torch(value, flow, importance):
    """
    Reference forward (scatter) splat in pure PyTorch. Works on CPU and
    CUDA, and is differentiable with respect to `value`, `flow` (through
    the bilinear weights) and `importance`.

    Accumulates three things in one scatter, by pre-multiplying e^Z into
    the payload and letting the bilinear weight w do the scattering:

        numerator  = sum_x  w(x) e^Z(x) value(x)
        denominator= sum_x  w(x) e^Z(x)
        coverage   = sum_x  w(x)

    `coverage` is kept separate from `denominator` because it is the
    honest hole test: a target pixel is empty when nothing landed on it,
    which is a statement about w alone and must not be confounded with
    how photometrically convincing the arrivals were.
    """
    batch, channels, height, width = value.shape
    dtype, device = value.dtype, value.device

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    dst_x = xx.unsqueeze(0) + flow[:, 0]  # (B, H, W)
    dst_y = yy.unsqueeze(0) + flow[:, 1]

    x0 = torch.floor(dst_x)
    y0 = torch.floor(dst_y)
    fx = dst_x - x0
    fy = dst_y - y0

    ones = value.new_ones(batch, 1, height, width)
    # [ value * e^Z , e^Z , 1 ]
    src = torch.cat([value * importance, importance, ones], dim=1)
    payload = channels + 2
    src = src.reshape(batch, payload, height * width)
    acc = value.new_zeros(batch, payload, height * width)

    for x_int, wx in ((x0, 1.0 - fx), (x0 + 1.0, fx)):
        for y_int, wy in ((y0, 1.0 - fy), (y0 + 1.0, fy)):
            inside = (
                (x_int >= 0) & (x_int <= width - 1) & (y_int >= 0) & (y_int <= height - 1)
            )
            w = wx * wy * inside.to(dtype)  # (B, H, W)
            idx = (
                y_int.clamp(0, height - 1) * width + x_int.clamp(0, width - 1)
            ).long().reshape(batch, 1, height * width).expand(batch, payload, -1)
            acc = acc.scatter_add(2, idx, src * w.reshape(batch, 1, height * width))

    acc = acc.reshape(batch, payload, height, width)
    return acc[:, :channels], acc[:, channels : channels + 1], acc[:, channels + 1 :]


def _splat_cupy(value, flow, importance):
    """Same contract as _splat_torch, via the CUDA softmax-splatting kernel."""
    from .softsplat import softsplat

    # softsplat's "softmax" mode exponentiates the metric internally, so
    # it is handed log(e^Z) = Z.
    numerator, denominator = softsplat(
        tenIn=value,
        tenFlow=flow,
        tenMetric=importance.log(),
        strMode="softmax",
        return_norm=True,
    )
    # A second, unweighted pass for the coverage term; "avg" adds 1e-7 to
    # the normaliser, so remove it and an empty pixel reports exactly 0.
    _unused, coverage = softsplat(
        tenIn=value,
        tenFlow=flow,
        tenMetric=None,
        strMode="avg",
        return_norm=True,
    )
    return numerator, denominator, (coverage - 1e-7).clamp_min(0.0)


def forward_splat(value, flow, importance=None, impl="auto"):
    """
    Scatter `value` from the source lattice onto the target lattice along
    `flow`, with bilinear weights and softmax-splat importance, then
    normalise:

                 sum_x  w(x) e^Z(x) value(x)
        out(u) = ----------------------------
                 sum_x  w(x) e^Z(x)

    Args:
        value:      (B, C, H, W) quantity to scatter
        flow:       (B, 2, H, W) where each source pixel lands
        importance: (B, 1, H, W) Z from Phase 1, or None for uniform
                    importance (equivalent to plain average splatting)
        impl:       "auto" | "torch" | "cupy"

    Returns:
        out:  (B, C, H, W) importance-weighted average of everything that
              landed on each target pixel; exactly 0 where nothing landed.
        hole: (B, 1, H, W) 1.0 where nothing landed, else 0.0
    """
    assert impl in ("auto", "torch", "cupy"), f"unknown splat impl: {impl}"

    use_cupy = impl == "cupy"
    if impl == "auto" and value.is_cuda:
        try:
            import cupy  # noqa: F401

            use_cupy = True
        except Exception:
            use_cupy = False

    # Always accumulate and normalise in fp32. Under autocast the incoming
    # tensors may be fp16, and this is the one place in the pipeline that
    # divides by a sum of small weights - exactly the operation half
    # precision handles worst. The reversed flow is also kept in fp32: it
    # is measured in pixels and feeds a grid_sample, so it is worth more
    # than the memory it costs.
    value = value.float()
    flow = flow.float()
    if importance is None:
        weights = torch.ones_like(value[:, :1])
    else:
        # Z <= 0 for images in [0, 1], so e^Z lands in [0.05, 1] and no
        # max-subtraction is needed; the clamp is belt-and-braces against
        # an unnormalised input.
        weights = importance.float().clamp(-20.0, 20.0).exp()

    splat = _splat_cupy if use_cupy else _splat_torch
    numerator, denominator, coverage = splat(value, flow, weights)

    hole = (coverage < _HOLE_THRESHOLD).to(numerator.dtype)
    out = numerator / denominator.clamp_min(_HOLE_EPS)
    return out * (1.0 - hole), hole


class RefineNet(nn.Module):
    """
    Small CNN that patches the holes left by scattering.

        G0, G1 <- G0, G1 + RefineNet(G0, G1, hole mask)

    Input  : G0 (2) + G1 (2) + hole0 (1) + hole1 (1) = 6 channels.
    Output : a 4-channel residual, added to (G0, G1).

    The output layer is zero-initialised, so at the start of training
    the reversed flows are exactly the raw scatter result and the
    refinement can only add to it.
    """

    def __init__(self, channels=64, num_blocks=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(6, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            *[LateralBlock(channels) for _ in range(num_blocks)],
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.head = nn.Conv2d(channels, 4, 3, 1, 1, padding_mode="reflect")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, flow_t0, flow_t1, hole_0, hole_1, scale):
        """
        Args:
            flow_t0, flow_t1: (B, 2, H, W) G0, G1 in pixels
            hole_0, hole_1:   (B, 1, H, W) hole masks
            scale:            (B, 1, 1, 1) motion scale, as in Phase 2

        Returns:
            refined G0, G1 (B, 2, H, W) each.
        """
        x = torch.cat(
            [flow_t0 / scale, flow_t1 / scale, hole_0, hole_1], dim=1
        )
        residual = self.head(self.body(x)) * scale
        return flow_t0 + residual[:, 0:2], flow_t1 + residual[:, 2:4]


class FlowReversal(nn.Module):
    """Phase 4 end to end: scatter both sides, then refine."""

    def __init__(self, channels=64, num_blocks=3, splat_impl="auto"):
        super().__init__()
        self.splat_impl = splat_impl
        self.refine_net = RefineNet(channels, num_blocks)

    def forward(self, phi_t, psi_t, scale, importance_0=None, importance_1=None):
        """
        Args:
            phi_t: (B, 2, H, W) Phi(t),  on lattice 0
            psi_t: (B, 2, H, W) Phi'(t), on lattice 1
            scale: (B, 1, 1, 1) motion scale
            importance_0: (B, 1, H, W) Z  on lattice 0
            importance_1: (B, 1, H, W) Z' on lattice 1

        Returns:
            flow_t0: (B, 2, H, W) G0 = f_{t->0}, on lattice t
            flow_t1: (B, 2, H, W) G1 = f_{t->1}, on lattice t
            hole_0, hole_1: (B, 1, H, W) pre-refinement hole masks
        """
        flow_t0, hole_0 = forward_splat(
            -phi_t, phi_t, importance=importance_0, impl=self.splat_impl
        )
        flow_t1, hole_1 = forward_splat(
            -psi_t, psi_t, importance=importance_1, impl=self.splat_impl
        )

        flow_t0, flow_t1 = self.refine_net(flow_t0, flow_t1, hole_0, hole_1, scale)
        return flow_t0, flow_t1, hole_0, hole_1
