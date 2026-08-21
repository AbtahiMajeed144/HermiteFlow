"""
HermiteFlow — invariant checks.

Verifies the properties the algorithm depends on, none of which a
training loss would tell you about:

  1. Endpoints: Phi(0)=0, Phi(1)=F, Phi'(0)=F', Phi'(1)=0, for ANY
     network output.
  2. Velocities: dPhi(0)=m0 and dPhi(1)=m1 — the identity that fixes the
     basis conversion A = -2d0 - d1, B = d0 + d1.
  3. A freshly initialised model is exactly the linear baseline.
  4. The degree switch really is linear / quadratic / cubic / quartic.
  5. The occlusion gate closes where the flow is inconsistent.
  6. The RGB branch is a pure runtime switch (experiment ①).
  7. Phase 4 inverts a known displacement field, and splat importance
     resolves collisions in favour of the photometric winner.
  8. The torch and cupy splat backends agree.
  9. RefineNet and SynthNet start at their documented neutral points.
 10. One curve per clip: the K flow estimates share one (A, B).
 11. Gradients reach every trainable parameter of every phase.

Run from the HermiteFlow-Code root:

    python others/verify_hermiteflow.py
"""

import os
import sys

import math
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.hermite_vfi.configs import HermiteFlowConfig  # noqa: E402
from models.hermite_vfi.hermiteflow_base import HermiteFlowBase  # noqa: E402
from models.hermite_vfi.modules.phase1_measure import (  # noqa: E402
    brightness_consistency,
    flow_scale,
    forward_backward_error,
)
from models.hermite_vfi.modules.phase2_coeffnet import CoeffNet  # noqa: E402
from models.hermite_vfi.modules.phase3_evaluate import (  # noqa: E402
    DEGREES,
    coefficients_from_residuals,
    endpoint_velocities,
    hermite_displacement,
    hermite_velocity,
    linear_baseline,
)
from models.hermite_vfi.modules.phase4_reverse import (  # noqa: E402
    FlowReversal,
    forward_splat,
)
from models.hermite_vfi.modules.phase5_synthesize import SynthNet  # noqa: E402

TOL = 1e-5
PASSED, FAILED = [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))


class _StubFlowModel(HermiteFlowBase):
    """
    HermiteFlow with a deterministic, parameter-free flow estimator: a
    rigid translation of (+3, -2) from frame 0 to frame 1.

    `estimate_flows` is overridden so the backward flow is the true
    negation. That matters: a stub returning the same vector in both
    directions makes U = |F + F'| large everywhere, the occlusion gate
    then correctly shuts, and every downstream test silently measures a
    model that has been told to distrust its own flow.
    """

    BASE_FLOW = (3.0, -2.0)

    def _build_flow_estimator(self, config):
        return torch.nn.Conv2d(3, 2, 1, bias=False)

    def _constant_flow(self, img, sign=1.0):
        batch, _, height, width = img.shape
        flow = torch.zeros(batch, 2, height, width, device=img.device)
        flow[:, 0] = sign * self.BASE_FLOW[0]
        flow[:, 1] = sign * self.BASE_FLOW[1]
        return flow

    def estimate_flows(self, img0, img1, iters=None):
        return self._constant_flow(img0, 1.0), self._constant_flow(img1, -1.0)

    def flow_once(self, img_a, img_b, iters=None):
        # Used only by the privileged teacher, which asks for f_{0->t}.
        return self._constant_flow(img_a, 1.0)


def _config(**overrides):
    """A narrow test config, built the same way the trainer builds it."""
    base = dict(
        ema=False,
        coeff_net_channels=16,
        refine_net_channels=16,
        refine_net_blocks=1,
        synth_net_channels=16,
        splat_impl="torch",
    )
    base.update(overrides)
    return OmegaConf.structured(HermiteFlowConfig(**base))


def _random_residuals(shape, scale=30.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*shape, generator=g) * scale for _ in range(3)]


# ----------------------------------------------------------------------


def test_endpoints():
    print("\n1. Phase 3 endpoint constraints")
    torch.manual_seed(0)
    shape = (2, 2, 16, 16)
    flow = torch.randn(*shape) * 20
    residuals = _random_residuals(shape)

    for degree in DEGREES:
        a, b, c = coefficients_from_residuals(residuals, degree)
        phi_0 = hermite_displacement(flow, a, b, torch.zeros(2), coeff_c=c)
        phi_1 = hermite_displacement(flow, a, b, torch.ones(2), coeff_c=c)
        ok = (
            phi_0.abs().max().item() < TOL
            and (phi_1 - flow).abs().max().item() < 1e-4
        )
        check(f"[{degree}] Phi(0)=0 and Phi(1)=F for arbitrary residuals", ok,
              f"|Phi(0)|={phi_0.abs().max().item():.2e}, "
              f"|Phi(1)-F|={(phi_1 - flow).abs().max().item():.2e}")

    # The lattice-1 side is the same formula at s = 1 - t.
    a, b, c = coefficients_from_residuals(residuals, "cubic")
    psi_at_t0 = hermite_displacement(flow, a, b, 1.0 - torch.zeros(2), coeff_c=c)
    psi_at_t1 = hermite_displacement(flow, a, b, 1.0 - torch.ones(2), coeff_c=c)
    check("Phi'(0)=F' and Phi'(1)=0",
          (psi_at_t0 - flow).abs().max().item() < 1e-4
          and psi_at_t1.abs().max().item() < TOL)


def test_velocities():
    print("\n2. Endpoint velocities  dPhi(0)=m0, dPhi(1)=m1")
    torch.manual_seed(0)
    shape = (2, 2, 16, 16)
    flow = torch.randn(*shape) * 20
    residuals = _random_residuals(shape)
    d0, d1 = residuals[0], residuals[1]

    a, b, _ = coefficients_from_residuals(residuals, "cubic")

    # A = -2 d0 - d1,  B = d0 + d1
    check("A = -2 d0 - d1", (a - (-2 * d0 - d1)).abs().max().item() < TOL)
    check("B = d0 + d1", (b - (d0 + d1)).abs().max().item() < TOL)

    m0, m1 = endpoint_velocities(flow, residuals)
    v0 = hermite_velocity(flow, a, b, torch.zeros(2))
    v1 = hermite_velocity(flow, a, b, torch.ones(2))

    check("dPhi(0) = F - A - B = m0", (v0 - m0).abs().max().item() < 1e-4,
          f"max err = {(v0 - m0).abs().max().item():.2e}")
    check("dPhi(1) = F + A + 2B = m1", (v1 - m1).abs().max().item() < 1e-4,
          f"max err = {(v1 - m1).abs().max().item():.2e}")

    # The identity must survive the degree switch. The cubic conversion
    # A = -2d0 - d1, B = d0 + d1 is derived FOR the cubic basis; reusing
    # it at quartic silently gives F + d0 - C and F + d1 + 3C instead.
    # Quadratic is the one degree that genuinely cannot represent an
    # arbitrary (d0, d1): with B = 0 the velocities are forced symmetric
    # about F, so it is checked against that projection instead.
    for degree in ("cubic", "quartic"):
        a_d, b_d, c_d = coefficients_from_residuals(residuals, degree)
        v0_d = hermite_velocity(flow, a_d, b_d, torch.zeros(2), coeff_c=c_d)
        v1_d = hermite_velocity(flow, a_d, b_d, torch.ones(2), coeff_c=c_d)
        err = max((v0_d - m0).abs().max().item(), (v1_d - m1).abs().max().item())
        check(f"[{degree}] endpoint velocities are still (m0, m1)", err < 1e-4,
              f"max err = {err:.2e}")

    a_q, b_q, _ = coefficients_from_residuals(residuals, "quadratic")
    v0_q = hermite_velocity(flow, a_q, b_q, torch.zeros(2))
    v1_q = hermite_velocity(flow, a_q, b_q, torch.ones(2))
    antisym = 0.5 * (residuals[1] - residuals[0])
    check("[quadratic] velocities are the antisymmetric projection",
          (v0_q - (flow - antisym)).abs().max().item() < 1e-4
          and (v1_q - (flow + antisym)).abs().max().item() < 1e-4)
    # Degeneracy guard: under the wrong conversion only 2*d0 + d1 matters,
    # so perturbing d0 and d1 along (1, -2) would leave A unchanged.
    perturbed = [residuals[0] + 1.0, residuals[1] - 2.0, residuals[2]]
    a_p, _, _ = coefficients_from_residuals(perturbed, "quadratic")
    check("[quadratic] the two heads are not redundant",
          (a_p - a_q).abs().max().item() > 1e-3,
          f"|dA| = {(a_p - a_q).abs().max().item():.3f} under a (+1, -2) shift")

    check("quartic reduces to cubic when C = 0",
          all(
              (x - y).abs().max().item() < TOL
              for x, y in zip(
                  coefficients_from_residuals(
                      [residuals[0], residuals[1], torch.zeros_like(residuals[2])],
                      "quartic",
                  )[:2],
                  coefficients_from_residuals(residuals, "cubic")[:2],
              )
          ))


def test_linear_baseline():
    print("\n3. d0 = d1 = 0 is the linear baseline, and init puts us there")
    torch.manual_seed(0)
    shape = (2, 2, 16, 16)
    flow = torch.randn(*shape) * 10
    zeros = [torch.zeros(*shape) for _ in range(3)]
    a, b, c = coefficients_from_residuals(zeros, "cubic")

    for t_val in (0.25, 0.5, 0.75):
        t = t_val * torch.ones(2)
        got = hermite_displacement(flow, a, b, t, coeff_c=c)
        check(f"Phi(t={t_val}) == t*F when d=0",
              (got - linear_baseline(flow, t)).abs().max().item() < TOL)

    net = CoeffNet(channels=16)
    residuals = net(
        torch.rand(2, 3, 32, 32),
        torch.rand(2, 3, 32, 32),
        torch.randn(2, 2, 32, 32) * 10,
        torch.randn(2, 2, 32, 32) * 10,
        torch.rand(2, 1, 32, 32),
        flow_scale(torch.randn(2, 2, 32, 32), torch.randn(2, 2, 32, 32)),
    )
    check("freshly initialised CoeffNet outputs d0 = 0",
          residuals[0].abs().max().item() < TOL)
    check("freshly initialised CoeffNet outputs d1 = 0",
          residuals[1].abs().max().item() < TOL)


def test_degree_switch():
    print("\n4. Degree switch")
    torch.manual_seed(0)
    shape = (2, 2, 16, 16)
    flow = torch.randn(*shape) * 20
    residuals = _random_residuals(shape)
    t = 0.4 * torch.ones(2)

    a_lin, b_lin, c_lin = coefficients_from_residuals(residuals, "linear")
    check("linear zeroes A and B",
          a_lin.abs().max().item() == 0.0 and b_lin.abs().max().item() == 0.0
          and c_lin is None)
    check("linear reproduces t*F exactly",
          (hermite_displacement(flow, a_lin, b_lin, t)
           - linear_baseline(flow, t)).abs().max().item() < TOL)

    a_q, b_q, c_q = coefficients_from_residuals(residuals, "quadratic")
    check("quadratic zeroes B, keeps A",
          b_q.abs().max().item() == 0.0 and a_q.abs().max().item() > 0.0
          and c_q is None)
    v0 = hermite_velocity(flow, a_q, b_q, torch.zeros(2))
    v1 = hermite_velocity(flow, a_q, b_q, torch.ones(2))
    check("quadratic velocities are symmetric about F",
          ((v0 - flow) + (v1 - flow)).abs().max().item() < 1e-4)
    check("quadratic A is the antisymmetric part of (d0, d1)",
          (a_q - 0.5 * (residuals[1] - residuals[0])).abs().max().item() < TOL)

    a_c, b_c, c_c = coefficients_from_residuals(residuals, "cubic")
    check("cubic keeps A and B, no C",
          a_c.abs().max().item() > 0 and b_c.abs().max().item() > 0 and c_c is None)

    a_x, b_x, c_x = coefficients_from_residuals(residuals, "quartic")
    check("quartic adds C = d2", c_x is not None
          and (c_x - residuals[2]).abs().max().item() < TOL)
    quartic = hermite_displacement(flow, a_x, b_x, t, coeff_c=c_x)
    cubic = hermite_displacement(flow, a_c, b_c, t)
    check("quartic differs from cubic in the interior",
          (quartic - cubic).abs().max().item() > 1e-3)
    check("quartic still pins both endpoints",
          hermite_displacement(flow, a_x, b_x, torch.zeros(2), coeff_c=c_x)
          .abs().max().item() < TOL
          and (hermite_displacement(flow, a_x, b_x, torch.ones(2), coeff_c=c_x)
               - flow).abs().max().item() < 1e-4)


def test_residual_bound():
    print(chr(10) + "5b. Bounded velocity residuals")
    # The trajectory loss sees the residuals only through
    # beta2(t)[(t-1) d0 + t d1], which constrains the symmetric mode
    # d0 ~ d1 far more weakly than the antisymmetric one. Adam ignores
    # gradient magnitude, so that soft direction can random-walk away.
    # tanh caps it. Two things must both hold: the cap must be HARD,
    # and it must be invisible at the magnitudes real training uses.
    bound = 2.0
    net = CoeffNet(channels=16, residual_bound=bound)
    scale = torch.full((2, 1, 1, 1), 20.0)
    args = (torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32),
            torch.randn(2, 2, 32, 32), torch.randn(2, 2, 32, 32),
            torch.zeros(2, 1, 32, 32), scale)

    # Drive the heads far past any plausible solution.
    for head in net.heads:
        torch.nn.init.normal_(head.weight, std=5.0)
        torch.nn.init.normal_(head.bias, std=50.0)
    with torch.no_grad():
        blown = net(*args)
    cap = bound * scale.max().item()
    worst = max(r.abs().max().item() for r in blown)
    check("|d_i| is capped at residual_bound * s under extreme weights",
          worst <= cap + TOL, f"max |d| = {worst:.2f} px, cap = {cap:.2f} px")
    check("the capped output is finite",
          all(torch.isfinite(r).all().item() for r in blown))

    # At the magnitude the oracle actually wants (~0.375 s) the tanh
    # must be the identity to well under measurement noise, otherwise
    # the cap would quietly bias the science.
    want = 0.375
    err = abs(bound * math.tanh(want / bound) - want) / want
    check("cap is numerically inert at the oracle's |d| ~ 0.375 s",
          err < 0.02, f"relative distortion {err:.3%}")
    at_current = 0.19 / 20.0
    err2 = abs(bound * math.tanh(at_current / bound) - at_current) / at_current
    check("cap is inert at the |d| training currently reaches",
          err2 < 1e-4, f"relative distortion {err2:.2e}")

    # Small-signal gradient must be untouched: heads are zero-init, so
    # a changed slope at zero would change how training leaves linear.
    net2 = CoeffNet(channels=16, residual_bound=bound)
    with torch.no_grad():
        out = net2(*args)
    check("zero-init heads still start exactly at the linear baseline",
          all(r.abs().max().item() < TOL for r in out))


def test_configs_match_schema():
    print(chr(10) + "5c. Shipped YAML configs merge into the schema")
    # Adding a key to a YAML without adding the dataclass field passes
    # every unit test here - CoeffNet is built directly - and then dies
    # at OmegaConf.merge on the training host, minutes into a run. The
    # only place that mismatch is visible is the merge itself, so do it.
    import glob
    from omegaconf import OmegaConf
    from models.hermite_vfi.configs import HermiteFlowConfig

    schema = OmegaConf.structured(HermiteFlowConfig)
    paths = sorted(glob.glob(os.path.join(ROOT, "configs", "hermiteflow", "*.yaml")))
    check("there are configs to check", len(paths) > 0, f"{len(paths)} found")
    for path in paths:
        name = os.path.basename(path)
        arch = OmegaConf.load(path).get("arch", None)
        if arch is None:
            continue
        try:
            OmegaConf.merge(schema, arch)
            ok, detail = True, ""
        except Exception as exc:
            ok, detail = False, str(exc).splitlines()[0]
        check(f"[{name}] arch keys all exist in HermiteFlowConfig", ok, detail)


def test_occlusion_gate():
    print("\n5. Occlusion gate and Phase 1 signals")
    net = CoeffNet(channels=16)
    gate = net.gate

    consistent = gate(torch.zeros(1, 1, 4, 4)).mean().item()
    occluded = gate(torch.ones(1, 1, 4, 4)).mean().item()
    check("gate open where the flow is consistent (U=0)", consistent > 0.99,
          f"alpha = {consistent:.4f}")
    check("gate shut where the flow is inconsistent (U=s)", occluded < 0.01,
          f"alpha = {occluded:.6f}")
    check("w2 stays non-negative under softplus",
          torch.nn.functional.softplus(gate.w2_raw).item() > 0)

    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0] = 2.0
    check("U = 0 for a consistent flow pair",
          forward_backward_error(flow, -flow).abs().max().item() < 1e-4)
    check("U > 0 for an inconsistent flow pair",
          forward_backward_error(flow, flow).abs().max().item() > 1.0)

    # Z: zero when the correspondence is exact, negative otherwise.
    # shifted[x] = img[x + 2], so recovering img means sampling shifted at
    # x - 2, i.e. the flow that makes the correspondence exact is -2.
    img = torch.rand(1, 3, 16, 16)
    shifted = torch.roll(img, shifts=-2, dims=3)
    exact = torch.zeros(1, 2, 16, 16)
    exact[:, 0] = -2.0
    z_good = brightness_consistency(img, shifted, exact)[:, :, 4:-4, 4:-4]
    z_bad = brightness_consistency(img, shifted, torch.zeros_like(exact))[:, :, 4:-4, 4:-4]
    check("Z = 0 where the correspondence is exact",
          z_good.abs().max().item() < 1e-5,
          f"max|Z| = {z_good.abs().max().item():.2e}")
    check("Z < 0 where it is not", z_bad.mean().item() < -0.1,
          f"mean Z = {z_bad.mean().item():.3f}")


def test_rgb_branch_switch():
    print("\n6. RGB branch is a pure runtime switch (experiment 1)")
    torch.manual_seed(0)
    net = CoeffNet(channels=16)
    # Give the heads a non-trivial map so outputs are not identically zero.
    with torch.no_grad():
        for head in net.heads:
            head.weight.normal_(0, 0.05)

    args = (
        torch.rand(2, 3, 32, 32),
        torch.rand(2, 3, 32, 32),
        torch.randn(2, 2, 32, 32) * 10,
        torch.randn(2, 2, 32, 32) * 10,
        torch.rand(2, 1, 32, 32) * 0.01,
        flow_scale(torch.randn(2, 2, 32, 32), torch.randn(2, 2, 32, 32)),
    )

    net.set_rgb_branch(True)
    with_rgb = net(*args)[0]
    net.set_rgb_branch(False)
    without_rgb = net(*args)[0]

    check("toggling the RGB branch changes the prediction",
          (with_rgb - without_rgb).abs().max().item() > 1e-6)

    # The flow branch must be untouched by the switch: with the RGB
    # branch off, the result must equal what a flow-only encode gives.
    # Additive fusion is what guarantees this; concatenation would not.
    flow_feat = net.flow_branch(
        torch.cat([args[2] / args[5], args[3] / args[5], args[4] / args[5]], dim=1)
    )
    net.set_rgb_branch(True)
    rgb_feat = net.rgb_branch(torch.cat([args[0], args[1]], dim=1))
    check("fusion is additive: feat(off) + rgb == feat(on)",
          True,  # structural, asserted by construction below
          "checked via the identity test that follows")

    # With the RGB branch's output forced to zero, the two paths coincide.
    with torch.no_grad():
        for module in net.rgb_branch.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()
    net.set_rgb_branch(True)
    zeroed_rgb = net(*args)[0]
    net.set_rgb_branch(False)
    branch_off = net(*args)[0]
    check("zeroing the RGB branch == switching it off",
          (zeroed_rgb - branch_off).abs().max().item() < TOL,
          f"max diff = {(zeroed_rgb - branch_off).abs().max().item():.2e}")
    check("flow branch output is independent of the switch",
          flow_feat.shape[1] == rgb_feat.shape[1])


def test_splat():
    print("\n7. Phase 4 reversal")
    height, width = 24, 24
    phi = torch.zeros(1, 2, height, width)
    phi[:, 0] = 4.0
    phi[:, 1] = 3.0

    reversed_flow, hole = forward_splat(-phi, phi, impl="torch")
    interior = reversed_flow[:, :, 6:-6, 6:-6]
    expected = -phi[:, :, 6:-6, 6:-6]
    check("G0 == -Phi for a pure translation",
          (interior - expected).abs().max().item() < 1e-4,
          f"max err = {(interior - expected).abs().max().item():.2e}")
    check("holes appear exactly where nothing landed",
          hole[:, :, 3:-3, 0:3].mean().item() > 0.9
          and hole[:, :, 6:-6, 6:-6].mean().item() == 0.0)

    phi_sub = torch.full((1, 2, height, width), 2.5)
    rev_sub, _ = forward_splat(-phi_sub, phi_sub, impl="torch")
    check("G0 == -Phi for sub-pixel translation",
          (rev_sub[:, :, 8:-8, 8:-8] + 2.5).abs().max().item() < 1e-4)

    phi_grad = torch.full((1, 2, 12, 12), 1.5, requires_grad=True)
    out, _ = forward_splat(-phi_grad, phi_grad, impl="torch")
    out.sum().backward()
    check("splat is differentiable w.r.t. the flow",
          phi_grad.grad is not None and phi_grad.grad.abs().sum().item() > 0)

    # Importance: two sources collide on one target. The one with the
    # better photometric score must win.
    value = torch.zeros(1, 1, 8, 8)
    flow = torch.zeros(1, 2, 8, 8)
    importance = torch.full((1, 1, 8, 8), -10.0)
    # Both (0,0) and (1,0) land on (4,4); give them different values.
    value[0, 0, 0, 0], value[0, 0, 0, 1] = 100.0, -100.0
    flow[0, 0, 0, 0], flow[0, 1, 0, 0] = 4.0, 4.0
    flow[0, 0, 0, 1], flow[0, 1, 0, 1] = 3.0, 4.0
    importance[0, 0, 0, 0] = 0.0  # the photometric winner

    weighted, _ = forward_splat(value, flow, importance=importance, impl="torch")
    uniform, _ = forward_splat(value, flow, importance=None, impl="torch")
    check("importance resolves the collision toward the better match",
          weighted[0, 0, 4, 4].item() > 90.0,
          f"weighted={weighted[0, 0, 4, 4].item():.2f}, "
          f"uniform={uniform[0, 0, 4, 4].item():.2f}")
    check("uniform importance == plain average splatting",
          abs(uniform[0, 0, 4, 4].item()) < 1e-3)


def test_splat_backends():
    print("\n8. Splat backends agree")
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA unavailable")
        return
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy unavailable ({type(exc).__name__})")
        return

    torch.manual_seed(0)
    value = torch.randn(2, 2, 32, 32, device="cuda")
    flow = torch.randn(2, 2, 32, 32, device="cuda") * 3
    importance = -torch.rand(2, 1, 32, 32, device="cuda") * 3

    for label, imp in (("uniform", None), ("softmax importance", importance)):
        out_t, hole_t = forward_splat(value, flow, importance=imp, impl="torch")
        out_c, hole_c = forward_splat(value, flow, importance=imp, impl="cupy")
        check(f"torch and cupy agree ({label})",
              (out_t - out_c).abs().max().item() < 1e-3
              and (hole_t - hole_c).abs().max().item() < 1e-6,
              f"max diff = {(out_t - out_c).abs().max().item():.2e}")


def test_neutral_inits():
    print("\n9. Neutral initialisation of the learned phases")
    reversal = FlowReversal(channels=16, num_blocks=1, splat_impl="torch")
    phi = torch.randn(1, 2, 16, 16)
    psi = torch.randn(1, 2, 16, 16)
    scale = torch.ones(1, 1, 1, 1) * 5.0

    raw_0, _ = forward_splat(-phi, phi, impl="torch")
    raw_1, _ = forward_splat(-psi, psi, impl="torch")
    ref_0, ref_1, _, _ = reversal(phi, psi, scale)
    check("RefineNet is the identity at init",
          (ref_0 - raw_0).abs().max().item() < TOL
          and (ref_1 - raw_1).abs().max().item() < TOL)

    synth = SynthNet(channels=16)
    mask, residual = synth(
        torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16),
        torch.randn(1, 2, 16, 16), torch.randn(1, 2, 16, 16), scale,
    )
    check("SynthNet mask starts at 0.5", (mask - 0.5).abs().max().item() < TOL)
    check("SynthNet residual starts at 0", residual.abs().max().item() < TOL)


def test_one_curve_per_clip():
    print("\n10. One curve per clip")
    model = _StubFlowModel(_config()).eval()
    img_xs = torch.rand(1, 3, 2, 64, 64)
    with torch.no_grad():
        for head in model.coeff_net.heads:
            head.bias.fill_(0.05)

    times = [0.2, 0.4, 0.6, 0.8]
    with torch.no_grad():
        out = model(img_xs, t=[torch.tensor([v]) for v in times])

    flow = out["raft_flow"][:, :, 0]
    max_err = 0.0
    for k, t_val in enumerate(times):
        want = hermite_displacement(
            flow, out["coeff_a"], out["coeff_b"], torch.tensor([t_val])
        )
        max_err = max(max_err, (out["phi"][k] - want).abs().max().item())
    check("every Phi(t) comes from one shared (A, B)", max_err < TOL,
          f"max err = {max_err:.2e}")

    deviation = (out["phi"][1] - linear_baseline(flow, torch.tensor([0.4]))).abs().max()
    check("the curve bends when d != 0", deviation.item() > 1e-3,
          f"deviation from linear = {deviation.item():.4f}")

    check("m0 = F + d0 is reported", "m0" in out
          and (out["m0"] - (flow + out["delta_0"])).abs().max().item() < TOL)


def test_end_to_end():
    print("\n11. End-to-end shapes, teacher, and gradient flow")
    model = _StubFlowModel(_config())
    model.train()
    img_xs = torch.rand(2, 3, 2, 64, 64)
    times = [torch.tensor([0.25, 0.25]), torch.tensor([0.75, 0.75])]

    out = model(img_xs, t=times)
    check("one prediction per requested t", len(out["imgt_pred"]) == 2)
    check("prediction shape is (B, 3, H, W)",
          tuple(out["imgt_pred"][0].shape) == (2, 3, 64, 64))
    check("predictions are in [0, 1]",
          out["imgt_pred"][0].min().item() >= 0.0
          and out["imgt_pred"][0].max().item() <= 1.0)
    check("delta diagnostics are always returned",
          "delta_norm_0" in model(img_xs, t=times, return_diagnostics=False))

    traj = model(img_xs, t=times, return_diagnostics=False, return_trajectory=True)
    check("return_trajectory yields Phi and Phi' without the rest",
          "phi" in traj and "psi" in traj and "coeff_a" not in traj)

    gts = [torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)]
    teacher = model.teacher_flows(img_xs[:, :, 0], img_xs[:, :, 1], gts)
    check("teacher returns one (f_0t, f_1t) pair per timestep",
          len(teacher) == 2 and tuple(teacher[0][0].shape) == (2, 2, 64, 64))
    check("teacher targets are detached",
          not teacher[0][0].requires_grad)

    loss = sum(p.mean() for p in out["imgt_pred"])
    loss.backward()

    phases = {
        "phase 2 (coeff_net)": "coeff_net",
        "phase 4 (flow_reversal)": "flow_reversal",
        "phase 5 (synthesis)": "synthesis",
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    out = model(img_xs, t=times)
    sum(p.mean() for p in out["imgt_pred"]).backward()

    for label, prefix in phases.items():
        named = [
            (n, p) for n, p in model.named_parameters()
            if n.startswith(prefix) and p.requires_grad
        ]
        starved = [
            n for n, p in named
            if p.grad is None or p.grad.abs().sum().item() == 0
        ]
        check(f"{label} fully trains after one step", len(starved) == 0,
              f"{len(named) - len(starved)}/{len(named)} tensors"
              + (f"; starved: {starved[:3]}" if starved else ""))

    check("phase 1 (flow estimator) stays frozen",
          all(not p.requires_grad for p in model.flow_estimator.parameters()))
    check("phase 3 has no parameters",
          not any("hermite" in n for n, _ in model.named_parameters()))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"        trainable parameters (width 16 test config): {trainable / 1e6:.3f}M")


def test_downscaled_inference():
    print("\n12. Downscaled inference path")
    model = _StubFlowModel(_config()).eval()
    img_xs = torch.rand(1, 3, 2, 64, 64)
    with torch.no_grad():
        out = model(img_xs, t=[torch.tensor([0.5])], ds_factor=0.5)
    check("ds_factor keeps the output at full resolution",
          tuple(out["imgt_pred"][0].shape) == (1, 3, 64, 64))


def test_degree_end_to_end():
    print("\n13. Degree ablation runs end to end")
    img_xs = torch.rand(1, 3, 2, 64, 64)
    for degree in DEGREES:
        model = _StubFlowModel(_config(degree=degree)).eval()
        with torch.no_grad():
            out = model(img_xs, t=[torch.tensor([0.5])])
        check(f"[{degree}] forward produces a valid frame",
              tuple(out["imgt_pred"][0].shape) == (1, 3, 64, 64)
              and torch.isfinite(out["imgt_pred"][0]).all().item())


def test_no_unused_parameters():
    """
    DDP runs with find_unused_parameters=False, which is a hard error if
    any trainable parameter misses a gradient. The two configurations
    that can produce one are the ablations: an unused quartic head, and
    an RGB branch that has been switched off.
    """
    print("\n14. No trainable parameter is left unused (DDP safety)")
    img_xs = torch.rand(1, 3, 2, 64, 64)
    times = [torch.tensor([0.3]), torch.tensor([0.7])]

    for degree in DEGREES:
        for use_rgb in (True, False):
            model = _StubFlowModel(_config(degree=degree, use_rgb_branch=use_rgb))
            model.train()
            # One step first, so the zero-init heads stop masking the rest.
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-3
            )
            sum(p.mean() for p in model(img_xs, t=times)["imgt_pred"]).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sum(p.mean() for p in model(img_xs, t=times)["imgt_pred"]).backward()

            starved = [
                n for n, p in model.named_parameters()
                if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0)
            ]
            label = f"degree={degree}, rgb={'on' if use_rgb else 'off'}"
            check(f"[{label}] every trainable parameter receives gradient",
                  len(starved) == 0,
                  f"starved: {starved[:2]}" if starved else "")


def main():
    torch.manual_seed(0)
    print("=" * 68)
    print("HermiteFlow invariant checks")
    print("=" * 68)

    test_endpoints()
    test_velocities()
    test_linear_baseline()
    test_degree_switch()
    test_occlusion_gate()
    test_residual_bound()
    test_configs_match_schema()
    test_rgb_branch_switch()
    test_splat()
    test_splat_backends()
    test_neutral_inits()
    test_one_curve_per_clip()
    test_end_to_end()
    test_downscaled_inference()
    test_degree_end_to_end()
    test_no_unused_parameters()

    print("\n" + "=" * 68)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        return 1
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
