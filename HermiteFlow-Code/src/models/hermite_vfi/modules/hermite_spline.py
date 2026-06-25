# --------------------------------------------------------
# HermiteFlow-VFI
# Module 3 — "The Engine": Pixel-Space Hermite Flow
#
# Pure-math module with ZERO learnable parameters.
# Computes intermediate bilateral flows via cubic Hermite
# spline interpolation, modulated by learned coefficients.
#
# Physics Mapping (from details.md):
#   P₀ = 0              (start position in relative space)
#   V₀ = F₀→₁           (start velocity = forward flow)
#   P₁ = F₀→₁           (end position = total displacement)
#   V₁ = -F₁→₀          (end velocity, forward-facing)
#
# Standard Cubic Hermite:
#   P(t) = h₀₀(t)·P₀ + h₁₀(t)·V₀ + h₀₁(t)·P₁ + h₁₁(t)·V₁
#
# Since P₀ = 0:
#   P(t) = h₁₀(t)·F₀→₁ + h₀₁(t)·F₀→₁ + h₁₁(t)·(-F₁→₀)
#        = [h₁₀(t) + h₀₁(t)]·F₀→₁ - h₁₁(t)·F₁→₀
#
# With learned coefficient modulation (α,β,γ,δ):
#   P(t) = α·h₀₀(t)·P₀ + β·h₁₀(t)·V₀ + γ·h₀₁(t)·P₁ + δ·h₁₁(t)·V₁
#        = β·h₁₀(t)·F₀→₁ + γ·h₀₁(t)·F₀→₁ - δ·h₁₁(t)·F₁→₀
#
# (α modulates the zero P₀ term — kept for generality as a
#  learned additive residual channel if desired in future work)
#
# Hermite basis functions:
#   h₀₀(t) = 2t³ - 3t² + 1
#   h₁₀(t) = t³ - 2t² + t
#   h₀₁(t) = -2t³ + 3t²
#   h₁₁(t) = t³ - t²
# --------------------------------------------------------

import torch
import torch.nn as nn


class HermiteSplineEngine(nn.Module):
    """
    Pure-math Hermite cubic spline flow interpolation engine.
    Zero learnable parameters — all dynamics come from the
    coefficient maps predicted by CoefficientNet.
    """

    def __init__(self):
        super().__init__()
        # No learnable parameters

    @staticmethod
    def hermite_basis(t):
        """
        Compute the 4 cubic Hermite basis functions.

        Args:
            t: (B, 1, 1, 1) — time scalar broadcastable over spatial dims

        Returns:
            h00, h10, h01, h11: each (B, 1, 1, 1)
        """
        t2 = t * t
        t3 = t2 * t

        h00 = 2 * t3 - 3 * t2 + 1      # value at t=0
        h10 = t3 - 2 * t2 + t           # tangent at t=0
        h01 = -2 * t3 + 3 * t2          # value at t=1
        h11 = t3 - t2                   # tangent at t=1

        return h00, h10, h01, h11

    def forward(self, flow_01, flow_10, coefficients, t):
        """
        Compute intermediate bilateral flows via Hermite spline.

        Args:
            flow_01:      (B, 2, H, W) — optical flow frame 0 → 1
            flow_10:      (B, 2, H, W) — optical flow frame 1 → 0
            coefficients: (B, 4, H, W) — α, β, γ, δ from CoefficientNet
            t:            (B,) or scalar — target timestep in (0, 1)

        Returns:
            flow_t0: (B, 2, H, W) — flow from time t → frame 0
            flow_t1: (B, 2, H, W) — flow from time t → frame 1
        """
        # Reshape t for broadcasting: (B, 1, 1, 1)
        if isinstance(t, (int, float)):
            t = torch.tensor(t, device=flow_01.device, dtype=flow_01.dtype)
        t = t.reshape(-1, 1, 1, 1)

        # Extract coefficient maps: each (B, 1, H, W)
        alpha = coefficients[:, 0:1]    # modulates h00·P0 (= 0, but kept)
        beta = coefficients[:, 1:2]     # modulates h10·V0 = h10·F01
        gamma = coefficients[:, 2:3]    # modulates h01·P1 = h01·F01
        delta = coefficients[:, 3:4]    # modulates h11·V1 = h11·(-F10)

        # Compute Hermite basis functions
        h00, h10, h01, h11 = self.hermite_basis(t)

        # Full coefficient-modulated Hermite interpolation:
        # P(t) = α·h00·P0 + β·h10·V0 + γ·h01·P1 + δ·h11·V1
        # With P0=0, V0=F01, P1=F01, V1=-F10:
        # P(t) = β·h10·F01 + γ·h01·F01 - δ·h11·F10
        # (the α·h00·0 term is zero but included conceptually)
        position_t = (beta * h10 + gamma * h01) * flow_01 \
                     - (delta * h11) * flow_10

        # P(t) is the displaced position at time t (relative to pixel origin)
        # Bilateral flow decomposition:
        #   flow_t→0 = 0 - P(t) = -P(t)         (flow from t back to frame 0)
        #   flow_t→1 = P1 - P(t) = F01 - P(t)   (flow from t to frame 1)
        flow_t0 = -position_t
        flow_t1 = flow_01 - position_t

        return flow_t0, flow_t1
