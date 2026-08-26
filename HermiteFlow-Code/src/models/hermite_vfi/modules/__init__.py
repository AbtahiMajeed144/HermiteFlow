# --------------------------------------------------------
# HermiteFlow — phase modules
#
#   phase1_measure.py     F, F', U, Z, B    (frozen RAFT + pure math)
#   phase2_coeffnet.py    d0, d1 -> m0, m1  (AppNet + CoeffHead, 1/8 res)
#   phase3_evaluate.py    Phi(t)            (formula, no parameters)
#   phase4_reverse.py     G0, G1            (scatter + small CNN)
#   phase5_synthesize.py  I_hat_t           (CNN)
#
# fi_components.py, fi_utils.py, layers.py and softsplat.py are
# shared building blocks inherited from GIMM-VFI.
# --------------------------------------------------------

from .phase1_measure import (
    align_to_source,
    blur_descriptor,
    brightness_consistency,
    flow_scale,
    forward_backward_error,
)
from .phase2_coeffnet import (
    AppNet,
    CoeffNet,
    OcclusionGate,
    convex_upsample,
    run_both_sides,
)
from .phase3_evaluate import (
    DEGREES,
    RESIDUALS_PER_DEGREE,
    basis,
    coefficients_from_residuals,
    endpoint_velocities,
    hermite_displacement,
    hermite_velocity,
    linear_baseline,
)
from .phase4_reverse import FlowReversal, RefineNet, forward_splat
from .phase5_synthesize import FrameSynthesis, SynthNet

__all__ = [
    "align_to_source",
    "blur_descriptor",
    "brightness_consistency",
    "forward_backward_error",
    "flow_scale",
    "AppNet",
    "CoeffNet",
    "OcclusionGate",
    "convex_upsample",
    "run_both_sides",
    "DEGREES",
    "RESIDUALS_PER_DEGREE",
    "basis",
    "coefficients_from_residuals",
    "endpoint_velocities",
    "hermite_displacement",
    "hermite_velocity",
    "linear_baseline",
    "FlowReversal",
    "RefineNet",
    "forward_splat",
    "FrameSynthesis",
    "SynthNet",
]
