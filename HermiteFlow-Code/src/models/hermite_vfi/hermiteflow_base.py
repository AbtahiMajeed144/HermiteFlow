# --------------------------------------------------------
# HermiteFlow — the chain
#
#   I0, I1
#     |
#     +--[1]-- RAFT --------------------> F, F', U, Z
#     |                                     |
#     +--[2]-- CoeffNet -----------------> d0, d1 -> m0, m1 -> A, B   (no t)
#     |                                     |
#     +--[3]-- Phi(t) = tF + b2(t)A + b3(t)B                          (per t)
#                                           |
#        [4]-- reversal + hole fill ------> G0, G1
#                                           |
#        [5]-- warp + fuse ---------------> I_hat_t
#
# LATTICE DISCIPLINE. A tensor on lattice i holds one value per pixel
# OF FRAME i. Tensors on different lattices cannot be added. F and Z
# live on lattice 0, F' and Z' on lattice 1; Phase 2 backwarps
# everything onto the lattice it is encoding, and Phase 4 exists
# solely to move a field from lattice 0 to lattice t.
#
# Phase 3 is the only stage that consumes t. Phases 1-2 run once per
# clip; phases 3-5 run once per output frame, so for 16x you get one
# CoeffNet pass and 15 closed-form evaluations - the amortization is
# on the TRAJECTORY MODEL only, not on the renderer.
#
# Reference for the surrounding infrastructure:
#   gimm-vfi: https://github.com/GSeanCDAT/GIMM-VFI
# --------------------------------------------------------

import torch
import torch.nn as nn

from .configs import HermiteFlowConfig
from .modules.phase1_measure import (
    align_to_source,
    brightness_consistency,
    flow_scale,
    forward_backward_error,
)
from .modules.phase2_coeffnet import CoeffNet, run_both_sides
from .modules.phase3_evaluate import (
    RESIDUALS_PER_DEGREE,
    coefficients_from_residuals,
    endpoint_velocities,
    hermite_displacement,
)
from .modules.phase4_reverse import FlowReversal
from .modules.phase5_synthesize import FrameSynthesis
from .modules.fi_utils import resize


class HermiteFlowBase(nn.Module):
    """
    Backbone-agnostic implementation of the five phases.

    Subclasses provide the frozen flow estimator by implementing
    `_build_flow_estimator` and `estimate_flows`.
    """

    Config = HermiteFlowConfig

    def __init__(self, config: HermiteFlowConfig):
        super().__init__()
        self.config = config = config.copy()
        self.raft_iter = config.raft_iter
        self.min_flow_scale = config.min_flow_scale
        self.degree = config.degree
        self.use_splat_importance = config.use_splat_importance
        self.train_stage = int(config.train_stage)
        assert self.train_stage in (1, 2), "arch.train_stage must be 1 or 2"
        assert self.degree in RESIDUALS_PER_DEGREE, (
            f"arch.degree must be one of {tuple(RESIDUALS_PER_DEGREE)}, "
            f"got {self.degree}"
        )

        # ===== Phase 1: measure =====
        self.flow_estimator = self._build_flow_estimator(config)
        self.flow_estimator.eval()
        for param in self.flow_estimator.parameters():
            param.requires_grad = False

        # ===== Phase 2: endpoint velocities =====
        # The head count is fixed by the widest degree so that a single
        # checkpoint serves every entry in the degree ablation - only the
        # basis conversion in Phase 3 changes.
        self.coeff_net = CoeffNet(
            channels=config.coeff_net_channels,
            gate_init_bias=config.gate_init_bias,
            gate_init_scale=config.gate_init_scale,
            use_rgb_branch=config.use_rgb_branch,
            num_residuals=max(RESIDUALS_PER_DEGREE.values()),
            residual_bound=getattr(config, "residual_bound", 2.0),
        )
        # Heads the configured degree does not consume are frozen and not
        # evaluated: an unused parameter is a hard error under DDP with
        # find_unused_parameters=False.
        self.coeff_net.set_active_heads(RESIDUALS_PER_DEGREE[self.degree])

        # The linear ablation has no curvature model at all - d_i = 0 by
        # definition - so Phase 2 is skipped outright rather than run and
        # discarded. Freezing it keeps the parameters in the state_dict
        # (one checkpoint still serves the whole ablation) while removing
        # them from the optimizer and from DDP's gradient buckets.
        if self.degree == "linear":
            for param in self.coeff_net.parameters():
                param.requires_grad = False

        # ===== Phase 3 is a formula - no module =====

        # ===== Phase 4: reverse =====
        self.flow_reversal = FlowReversal(
            channels=config.refine_net_channels,
            num_blocks=config.refine_net_blocks,
            splat_impl=config.splat_impl,
        )

        # ===== Phase 5: synthesize =====
        self.synthesis = FrameSynthesis(channels=config.synth_net_channels)

        # Stage 1 trains the motion model alone. Phases 4-5 are skipped in
        # the forward pass, so their parameters would receive no gradient -
        # a hard error under DDP with find_unused_parameters=False. They
        # stay in the state_dict so stage 2 can resume from the checkpoint.
        if self.train_stage == 1:
            for module in (self.flow_reversal, self.synthesis):
                for param in module.parameters():
                    param.requires_grad = False

    # ------------------------------------------------------------------
    # Backbone hooks
    # ------------------------------------------------------------------

    def _build_flow_estimator(self, config):
        raise NotImplementedError

    def flow_once(self, img_a, img_b, iters=None):
        """
        One directional flow estimate, f_{a->b}, on lattice a.

        Subclasses implement this; `estimate_flows` calls it twice and the
        teacher calls it once per direction it actually needs.
        """
        raise NotImplementedError

    def estimate_flows(self, img0, img1, iters=None):
        """
        Args:
            img0, img1: (B, 3, H, W) in [0, 1]
        Returns:
            f01, f10: (B, 2, H, W) in pixels
        """
        return (
            self.flow_once(img0, img1, iters=iters),
            self.flow_once(img1, img0, iters=iters),
        )

    def train(self, mode=True):
        """Keep the frozen flow estimator in eval mode at all times."""
        super().train(mode)
        self.flow_estimator.eval()
        return self

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def measure(self, img0, img1, iters=None):
        """
        Phase 1. Returns a dict with F, F', U, U', Z, Z' and the motion
        scale s. Nothing here is differentiable - it is the measurement.
        """
        with torch.no_grad():
            flow_fwd, flow_bwd = self.estimate_flows(img0, img1, iters=iters)
            flow_fwd = flow_fwd.detach()
            flow_bwd = flow_bwd.detach()

            return {
                "flow_fwd": flow_fwd,
                "flow_bwd": flow_bwd,
                "occ_fwd": forward_backward_error(flow_fwd, flow_bwd),
                "occ_bwd": forward_backward_error(flow_bwd, flow_fwd),
                "importance_fwd": brightness_consistency(img0, img1, flow_fwd),
                "importance_bwd": brightness_consistency(img1, img0, flow_bwd),
                "scale": flow_scale(
                    flow_fwd, flow_bwd, min_scale=self.min_flow_scale
                ),
            }

    def predict_velocities(self, img0, img1, measured):
        """
        Phase 2. Returns the velocity residuals on each lattice:
        ([d0, d1, ...] on lattice 0, [d0', d1', ...] on lattice 1).
        Runs once per clip - no t involved.
        """
        return run_both_sides(
            self.coeff_net,
            img0,
            img1,
            measured["flow_fwd"],
            measured["flow_bwd"],
            measured["occ_fwd"],
            measured["occ_bwd"],
            measured["scale"],
            align=align_to_source,
        )

    @torch.no_grad()
    def teacher_flows(self, img0, img1, gt_frames, iters=None):
        """
        Privileged teacher for the trajectory, used only during training.

        Photometric loss alone cannot separate d0 from d1: time-to-location
        ambiguity lets the network average over trajectories and drive both
        residuals to zero while the rendered frame still looks right. The
        fix is a teacher that has seen the true intermediate frame.

        Here the teacher is the frozen flow estimator itself, run on the
        ground-truth middle frame:

            f_{0->t} = RAFT(I0, I_t^GT)      on lattice 0, like Phi(t)
            f_{1->t} = RAFT(I1, I_t^GT)      on lattice 1, like Phi'(t)

        No extra parameters, no extra training stage, and the target lives
        on exactly the lattice the prediction does. Supervising Phi at K
        different t pins A and B, and therefore m0 and m1, because the
        basis conversion is invertible: d0 = -A - B, d1 = A + 2B.

        Args:
            img0, img1: (B, 3, H, W) in [0, 1]
            gt_frames:  list of K tensors (B, 3, H, W), the true frames at
                        the supervised timesteps

        Returns:
            list of K (target_phi, target_psi) pairs, each (B, 2, H, W).
        """
        iters = self.raft_iter if iters is None else iters
        return [
            (
                self.flow_once(img0, gt, iters=iters).detach(),
                self.flow_once(img1, gt, iters=iters).detach(),
            )
            for gt in gt_frames
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        img_xs,
        t=None,
        iters=None,
        ds_factor=None,
        return_diagnostics=True,
        return_trajectory=False,
        trajectory_only=None,
    ):
        """
        Args:
            img_xs:    (B, 3, 2, H, W) - the two real frames [I0, I1] in [0, 1]
            t:         list of tensors, each (B,) or (B, 1); one entry per
                       frame to generate. Phases 3-5 run once per entry.
            iters:     flow-estimator iterations (RAFT only)
            ds_factor: run phases 1-4 at this fraction of the input
                       resolution, then synthesise at full resolution.
            return_diagnostics: also return the per-phase intermediates
                       (A, B, U, Phi, Psi, hole masks, ...). Useful for
                       analysis and required by others/verify_hermiteflow.py,
                       but they are tens of megabytes per step and under
                       DataParallel every one of them is copied to GPU 0.
                       The trainer switches them off.
            return_trajectory: additionally return Phi(t) and Phi'(t) only.
                       The trajectory-distillation loss needs these two and
                       nothing else, so it does not pay for the rest.
            trajectory_only: stop after Phase 3 and return no images.
                       Defaults to True in stage 1, where phases 4-5 are
                       frozen and there is no image loss - running them
                       would be pure waste. Pass explicitly to override.

        Returns:
            dict; "imgt_pred" is a list of (B, 3, H, W) frames, one per t.
        """
        assert isinstance(t, (list, tuple)) and len(t) > 0, "t must be a non-empty list"
        if trajectory_only is None:
            trajectory_only = self.train_stage == 1 and self.training

        full_size_img = None
        if ds_factor is not None and abs(ds_factor - 1.0) > 1e-6:
            full_size_img = img_xs
            img_xs = torch.cat(
                [
                    resize(img_xs[:, :, 0], scale_factor=ds_factor).unsqueeze(2),
                    resize(img_xs[:, :, 1], scale_factor=ds_factor).unsqueeze(2),
                ],
                dim=2,
            )

        img0, img1 = img_xs[:, :, 0], img_xs[:, :, 1]

        # ---------------- Phase 1: measure ----------------
        measured = self.measure(
            img0, img1, iters=self.raft_iter if iters is None else iters
        )
        flow_fwd, flow_bwd = measured["flow_fwd"], measured["flow_bwd"]
        scale = measured["scale"]

        # ---------------- Phase 2: endpoint velocities ----------------
        if self.degree == "linear":
            zeros = torch.zeros_like(flow_fwd)
            residuals_0 = residuals_1 = [zeros, zeros]
        else:
            residuals_0, residuals_1 = self.predict_velocities(img0, img1, measured)

        # ---------------- Phase 2.4: basis conversion ----------------
        coeff_a, coeff_b, coeff_c = coefficients_from_residuals(
            residuals_0, self.degree
        )
        coeff_a_sw, coeff_b_sw, coeff_c_sw = coefficients_from_residuals(
            residuals_1, self.degree
        )

        importance_0 = measured["importance_fwd"] if self.use_splat_importance else None
        importance_1 = measured["importance_bwd"] if self.use_splat_importance else None

        synth_img0, synth_img1 = img0, img1
        if full_size_img is not None:
            synth_img0, synth_img1 = full_size_img[:, :, 0], full_size_img[:, :, 1]

        imgt_preds, flowt0_preds, flowt1_preds, all_others = [], [], [], []
        phis, psis, holes = [], [], []

        if trajectory_only:
            phis, psis, phis_lin, psis_lin = [], [], [], []
            for cur_t in t:
                if not torch.is_tensor(cur_t):
                    cur_t = torch.tensor(cur_t, device=img0.device, dtype=img0.dtype)
                cur_t = cur_t.to(device=img0.device, dtype=img0.dtype).reshape(-1, 1, 1, 1)
                phis.append(hermite_displacement(
                    flow_fwd, coeff_a, coeff_b, cur_t, coeff_c=coeff_c))
                psis.append(hermite_displacement(
                    flow_bwd, coeff_a_sw, coeff_b_sw, 1.0 - cur_t, coeff_c=coeff_c_sw))
                # The d = 0 trajectory, i.e. what a purely linear model
                # would have predicted. Free to compute (one multiply) and
                # it is the only thing that makes the flow PSNR
                # interpretable: the gap between the two IS the curvature
                # contribution, which is the claim the paper rests on.
                phis_lin.append(cur_t * flow_fwd)
                psis_lin.append((1.0 - cur_t) * flow_bwd)
            return {
                "phi": phis,
                "psi": psis,
                "phi_linear": phis_lin,
                "psi_linear": psis_lin,
                "flow_scale": scale,
                "delta_norm_0": residuals_0[0].abs().mean().detach(),
                "delta_norm_1": residuals_0[1].abs().mean().detach(),
            }

        for cur_t in t:
            if not torch.is_tensor(cur_t):
                cur_t = torch.tensor(cur_t, device=img0.device, dtype=img0.dtype)
            cur_t = cur_t.to(device=img0.device, dtype=img0.dtype).reshape(-1, 1, 1, 1)

            # ------------ Phase 3: evaluate the trajectory ------------
            phi_t = hermite_displacement(
                flow_fwd, coeff_a, coeff_b, cur_t, coeff_c=coeff_c
            )
            psi_t = hermite_displacement(
                flow_bwd, coeff_a_sw, coeff_b_sw, 1.0 - cur_t, coeff_c=coeff_c_sw
            )

            # ------------ Phase 4: lattice 0 -> lattice t ------------
            flow_t0, flow_t1, hole_0, hole_1 = self.flow_reversal(
                phi_t, psi_t, scale, importance_0, importance_1
            )

            synth_scale = scale
            if full_size_img is not None:
                inv = synth_img0.shape[-2] / flow_t0.shape[-2]
                flow_t0 = inv * resize(flow_t0, scale_factor=inv)
                flow_t1 = inv * resize(flow_t1, scale_factor=inv)
                synth_scale = scale * inv

            # ------------ Phase 5: build the frame ------------
            imgt_pred, warped_0, warped_1, mask, _residual = self.synthesis(
                synth_img0, synth_img1, flow_t0, flow_t1, synth_scale
            )

            # The same blend without the residual. Supervising this as an
            # auxiliary target keeps the residual R from quietly taking over
            # the job of the flows: if R can repaint the frame, phases 2-4
            # stop receiving a useful gradient.
            blend_no_residual = torch.clamp(
                mask * warped_0 + (1.0 - mask) * warped_1, 0.0, 1.0
            )

            imgt_preds.append(imgt_pred)
            flowt0_preds.append(flow_t0)
            flowt1_preds.append(flow_t1)
            all_others.append([blend_no_residual])
            phis.append(phi_t)
            psis.append(psi_t)
            holes.append(torch.cat([hole_0, hole_1], dim=1))

        outputs = {
            "imgt_pred": imgt_preds,
            "other_pred": all_others,
            # The residual magnitudes are cheap scalars and are NOT
            # optional. The spec's diagnostic: if ||d0||, ||d1|| stay near
            # zero over the first few thousand steps, the network has
            # collapsed to the linear baseline, the "curvature from RGB"
            # claim is dead, and you need to know immediately rather than
            # after a full run.
            "delta_norm_0": residuals_0[0].abs().mean().detach(),
            "delta_norm_1": residuals_0[1].abs().mean().detach(),
        }
        if return_trajectory or return_diagnostics:
            outputs["phi"] = phis
            outputs["psi"] = psis
            # Four floats per sample. The trajectory loss divides by this
            # so its magnitude does not depend on how much the scene moves.
            outputs["flow_scale"] = scale
        if not return_diagnostics:
            return outputs

        vel_0, vel_1 = endpoint_velocities(flow_fwd, residuals_0)
        outputs.update({
            "flowt0_pred": flowt0_preds,
            "flowt1_pred": flowt1_preds,
            "raft_flow": torch.cat(
                [flow_fwd.unsqueeze(2), flow_bwd.unsqueeze(2)], dim=2
            ),
            "delta_0": residuals_0[0],
            "delta_1": residuals_0[1],
            "delta_0_swapped": residuals_1[0],
            "delta_1_swapped": residuals_1[1],
            "m0": vel_0,
            "m1": vel_1,
            "coeff_a": coeff_a,
            "coeff_b": coeff_b,
            "coeff_a_swapped": coeff_a_sw,
            "coeff_b_swapped": coeff_b_sw,
            "occlusion": torch.cat(
                [measured["occ_fwd"], measured["occ_bwd"]], dim=1
            ),
            "importance": torch.cat(
                [measured["importance_fwd"], measured["importance_bwd"]], dim=1
            ),
            "hole": holes,
        })
        if coeff_c is not None:
            outputs["coeff_c"] = coeff_c
        return outputs

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_psnr(self, preds, targets, reduction="mean"):
        assert reduction in ["mean", "sum", "none"]
        batch_size = preds.shape[0]
        sample_mses = torch.reshape((preds - targets) ** 2, (batch_size, -1)).mean(
            dim=-1
        )

        if reduction == "mean":
            return (-10 * torch.log10(sample_mses)).mean()
        if reduction == "sum":
            return (-10 * torch.log10(sample_mses)).sum()
        return -10 * torch.log10(sample_mses)
