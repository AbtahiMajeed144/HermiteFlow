# --------------------------------------------------------
# HermiteFlow — Phase 3: Evaluate the trajectory
#
#   In:  F, A, B and t      Out: Phi(t)      no parameters
#
#     beta2(s) = s^2 - s,   beta3(s) = s^3 - s,   beta4(s) = s^4 - s
#
#     Phi(t)  = t F  + beta2(t) A + beta3(t) B            (lattice 0)
#     Phi'(t) = (1-t) F' + beta2(1-t) A' + beta3(1-t) B'  (lattice 1)
#
# Every basis function vanishes at s=0 and s=1, so for ANY network
# output:
#
#     Phi(0) = 0,     Phi(1) = F,     Phi'(0) = F',   Phi'(1) = 0
#
# and the endpoint velocities come back exactly as intended:
#
#     dPhi(0) = F - A - B  = F + d0 = m0
#     dPhi(1) = F + A + 2B = F + d1 = m1
#
# which is what fixes the basis conversion in Phase 2.4:
#
#     A = -2 d0 - d1,      B = d0 + d1
#
# Degree switch (config flag arch.degree):
#     linear     d_i = 0            RIFE-style linear
#     quadratic  B = 0              IQ-VFI
#     cubic      full               ours
#     quartic    add C beta4(t)     ablation upper end
#
# This is the only stage that consumes t. Three tensor multiplies.
#
# NOTE ON SCOPE. The endpoint guarantees above hold for Phi - the
# FLOW field. They do not transfer to the output image: Phases 4-5
# run at every t regardless, so I_hat_0 != I_0. Likewise "one smooth
# curve" describes the K flow estimates, not the K rendered frames,
# because SynthNet runs per t. Both claims must be scoped to the
# trajectory in any write-up.
# --------------------------------------------------------

import torch

DEGREES = ("linear", "quadratic", "cubic", "quartic")

# How many velocity residuals CoeffNet must emit for each degree.
# The quartic ablation needs a third residual to drive C.
RESIDUALS_PER_DEGREE = {
    "linear": 2,
    "quadratic": 2,
    "cubic": 2,
    "quartic": 3,
}


def _as_time_tensor(t, ref):
    """Coerce t to a (B, 1, 1, 1) tensor broadcastable over ref (B, C, H, W)."""
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=ref.device, dtype=ref.dtype)
    return t.to(device=ref.device, dtype=ref.dtype).reshape(-1, 1, 1, 1)


def basis(t):
    """beta2, beta3, beta4 evaluated at t."""
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    return t2 - t, t3 - t, t4 - t


def coefficients_from_residuals(residuals, degree="cubic"):
    """
    Phase 2.4 — convert velocity residuals into evaluation-basis
    coefficients, applying the degree switch.

    The conversion is DEGREE-DEPENDENT. The familiar

        A = -2 d0 - d1,   B = d0 + d1

    is the solution of dPhi(0) = F + d0, dPhi(1) = F + d1 for the CUBIC
    basis specifically. Reusing it verbatim at another degree silently
    changes what the two heads mean, because the basis derivatives at
    the endpoints change:

        beta2'(0) = -1   beta2'(1) = +1
        beta3'(0) = -1   beta3'(1) = +2
        beta4'(0) = -1   beta4'(1) = +3

    QUARTIC. With C != 0 the endpoint velocities are
        dPhi(0) = F - A - B - C,   dPhi(1) = F + A + 2B + 3C
    Solving for the velocity semantics while leaving C free as the extra
    shape parameter gives

        A = -2 d0 - d1 + C,   B = d0 + d1 - 2C,   C = d2

    which collapses to the cubic at C = 0. Using the cubic conversion
    instead would make the realised velocities F + d0 - C and
    F + d1 + 3C.

    QUADRATIC. With B = 0 only ONE degree of freedom remains:
    dPhi(0) = F - A and dPhi(1) = F + A are forced to be symmetric about
    F. Two independent residuals cannot be represented, so the honest
    restriction is the least-squares projection onto that subspace,

        A = (d1 - d0) / 2

    i.e. the antisymmetric part of the residual pair. Setting B = 0 in
    the cubic conversion instead yields A = -2 d0 - d1, whose realised
    residuals are (+2 d0 + d1, -2 d0 - d1) - neither d0 nor d1 - and
    leaves the two heads redundant, since only the combination
    2 d0 + d1 can affect the output. That degeneracy is observable in
    training: the two heads drift to the same magnitude.

    Args:
        residuals: list of (B, 2, H, W); [d0, d1] (+ d2 for quartic)
        degree: one of DEGREES

    Returns:
        A, B, C  — each (B, 2, H, W); C is None unless degree == "quartic".
    """
    assert degree in DEGREES, f"degree must be one of {DEGREES}, got {degree}"
    d0, d1 = residuals[0], residuals[1]

    if degree == "linear":
        # d_i = 0: a straight line at constant speed.
        zero = torch.zeros_like(d0)
        return zero, zero, None

    if degree == "quadratic":
        return 0.5 * (d1 - d0), torch.zeros_like(d0), None

    if degree == "quartic":
        assert len(residuals) >= 3, "quartic degree needs a third residual head"
        coeff_c = residuals[2]
        return (
            -2.0 * d0 - d1 + coeff_c,
            d0 + d1 - 2.0 * coeff_c,
            coeff_c,
        )

    # cubic
    return -2.0 * d0 - d1, d0 + d1, None


def hermite_displacement(flow, coeff_a, coeff_b, t, coeff_c=None):
    """
    Phi(t) = t F + beta2(t) A + beta3(t) B [+ beta4(t) C]

    Args:
        flow:    (B, 2, H, W)  F   (or F' for the lattice-1 side)
        coeff_a: (B, 2, H, W)  A   (or A')
        coeff_b: (B, 2, H, W)  B   (or B')
        t:       scalar, (B,) or (B, 1); use s = 1 - t for the
                 lattice-1 side
        coeff_c: (B, 2, H, W)  C, quartic ablation only

    Returns:
        Phi(t): (B, 2, H, W) displacement, in pixels.
    """
    t = _as_time_tensor(t, flow)
    beta2, beta3, beta4 = basis(t)

    displacement = t * flow + beta2 * coeff_a + beta3 * coeff_b
    if coeff_c is not None:
        displacement = displacement + beta4 * coeff_c
    return displacement


def hermite_velocity(flow, coeff_a, coeff_b, t, coeff_c=None):
    """
    dPhi/dt = F + (2t - 1) A + (3t^2 - 1) B [+ (4t^3 - 1) C]

    Not used by the forward pass, but it is the quantity the endpoint
    identities are stated in: dPhi(0) = m0 and dPhi(1) = m1. The
    verification script checks exactly that.
    """
    t = _as_time_tensor(t, flow)
    velocity = flow + (2.0 * t - 1.0) * coeff_a + (3.0 * t * t - 1.0) * coeff_b
    if coeff_c is not None:
        velocity = velocity + (4.0 * t * t * t - 1.0) * coeff_c
    return velocity


def endpoint_velocities(flow, residuals):
    """m0 = F + d0, m1 = F + d1 — Phase 2.2, stated directly."""
    return flow + residuals[0], flow + residuals[1]


def linear_baseline(flow, t):
    """Phi(t) with d0 = d1 = 0. The model reduces to exactly this at init."""
    return _as_time_tensor(t, flow) * flow
