# --------------------------------------------------------
# HermiteFlow — trainer
#
# Single stage, end to end. Every batch carries K ground-truth
# middle frames for the same pair of endpoints, and the loss is
# averaged over all K. Phases 1-2 run once per batch; phases 3-5
# run K times.
#
# THREE loss terms:
#
# 1. Image loss on I_hat_t = M*I_{t->0} + (1-M)*I_{t->1} + R.
#
# 2. Image loss on the same blend WITHOUT the residual R, at half
#    weight. If R alone can repaint the frame, phases 2-4 stop
#    receiving a useful gradient and the model degenerates into an
#    image-to-image network with a flow-shaped ornament attached.
#
# 3. Trajectory distillation. Photometric loss alone is
#    insufficient: time-to-location ambiguity lets the network
#    average over trajectories, driving d0, d1 -> 0 while the
#    rendered frame still looks right. A teacher that has seen
#    I_t^GT supervises Phi(t) directly - see
#    HermiteFlowBase.teacher_flows. Multiple t per clip is what
#    makes d0 and d1 separable at all; at a single t only
#    beta2*A + beta3*B is observable and the two never split.
#
# Watch loss/delta_0 and loss/delta_1 in TensorBoard. If they sit
# at zero through the first few thousand steps the model has
# collapsed to the linear baseline and the curvature claim is dead.
# --------------------------------------------------------

import logging
from contextlib import nullcontext

import torch
import torchvision
from tqdm import tqdm

from models.hermite_vfi.modules.fi_utils import warp
from utils.accumulator import AccmStageINR
from .trainer import TrainerTemplate

logger = logging.getLogger(__name__)
from utils.loss import LapLoss, Ternary, Charbonnier_L1
from utils.lpips import LPIPS

AUX_LOSS_WEIGHT = 0.5


class Trainer(TrainerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.using_lpips = getattr(self.config.loss, "perceptual_loss", False)
        self.lap = LapLoss()
        self.census = Ternary()
        self.l1 = Charbonnier_L1()
        self.lpips = LPIPS(net="alex", version="0.1").eval()
        for _name, param in self.lpips.named_parameters():
            param.requires_grad = False
        self.grad_accm_steps = max(1, int(self.config.optimizer.grad_accm_steps))
        # Micro-batches dropped for a non-finite loss; see train().
        self.nonfinite_skips = 0
        self.flow_distill_weight = float(
            getattr(self.config.loss, "flow_distill_weight", 0.0)
        )
        # The teacher runs 2*K frozen flow passes per step, which at
        # raft_iter=20 and K=5 costs more than the entire rest of the
        # step (measured 901 ms vs 658 ms). RAFT's refinement converges
        # quickly and this is a target, not the measurement Phase 1
        # depends on, so it runs at a lower iteration count by default.
        self.teacher_raft_iter = int(
            getattr(self.config.loss, "teacher_raft_iter", 12)
        )
        # v2.1's cross-lattice velocity-consistency regulariser (Training,
        # item 3). d = 0 satisfies it trivially, so it prunes inconsistent
        # non-linear solutions between the two passes rather than driving
        # curvature by itself.
        self.velocity_consistency_weight = float(
            getattr(self.config.loss, "velocity_consistency_weight", 0.0)
        )
        # Write scalars every N optimizer steps. Logging once per epoch is
        # useless here: a Vimeo epoch is ~2k optimizer steps and an X4K
        # one with repeat=8 is ~1.1k, so the first point would land hours
        # in, long after you need to know whether d0/d1 are moving.
        self.log_every = int(self.config.experiment.get("log_every", 50))
        # Two-stage recipe, after GIMM-VFI. Stage 1 trains the motion
        # model alone against pseudo-ground-truth flow; stage 2 trains
        # everything on image losses and keeps the flow term as GIMM's
        # L_rec regulariser.
        self.stage = int(getattr(self.config.arch, "train_stage", 2))
        if self.stage == 1 and self.flow_distill_weight <= 0:
            raise ValueError(
                "stage 1 has no objective without loss.flow_distill_weight > 0"
            )

    def get_accm(self):
        return AccmStageINR(
            scalar_metric_names=(
                "loss_total",
                "lap",
                "census",
                "l1",
                "lpips",
                "distill",
                "vel_consistency",
                "delta_0",
                "delta_1",
                "psnr",
                "psnr_lin",
            ),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Batch plumbing
    # ------------------------------------------------------------------

    @staticmethod
    def unpack(batch, device):
        """
        Returns:
            img_xs: (B, 3, 2, H, W)   the two real frames
            gts:    list of K tensors (B, 3, H, W)
            t_list: list of K tensors (B,)
        """
        xs = batch["xs"].to(device, non_blocking=True)  # (B, 3, 2 + K, H, W)
        num_targets = xs.shape[2] - 2
        # A cached-teacher batch carries no middle frames at all: their
        # only consumer in stage 1 is teacher_flows, which has already
        # run offline. See datasets/x4k_cached.py.
        cached = "teacher_phi" in batch
        assert num_targets >= 1 or cached, (
            "each sample needs a ground-truth frame, or a cached teacher target"
        )

        img_xs = xs[:, :, :2]
        gts = [xs[:, :, 2 + k] for k in range(num_targets)]

        if "t" in batch:
            times = batch["t"].to(device, non_blocking=True).float()
            if times.ndim == 1:
                times = times.unsqueeze(1)
        else:
            assert num_targets >= 1, "no timesteps and no ground-truth frames"
            times = 0.5 * torch.ones(
                xs.shape[0], num_targets, device=device, dtype=torch.float
            )

        assert num_targets in (0, times.shape[1]), (
            f"got {num_targets} ground-truth frames but {times.shape[1]} timesteps"
        )
        t_list = [times[:, k] for k in range(times.shape[1])]
        return img_xs, gts, t_list

    @staticmethod
    def cached_teacher(batch, device):
        """
        The offline trajectory targets, in the same shape teacher_flows
        returns: a list of K (f_{0->t}, f_{1->t}) pairs, each (B, 2, H, W).

        Returns None when the batch has no cache, so the caller falls
        back to running the teacher online.
        """
        if "teacher_phi" not in batch:
            return None
        phi = batch["teacher_phi"].to(device, non_blocking=True).float()
        psi = batch["teacher_psi"].to(device, non_blocking=True).float()
        return [(phi[:, k], psi[:, k]) for k in range(phi.shape[1])]

    def image_losses(self, pred, target):
        """Returns (lap, census, l1, lpips) for one prediction/target pair."""
        loss_lap = self.lap(pred, target).mean()
        loss_census = self.census(pred, target)
        loss_l1 = self.l1(pred, target)
        if self.using_lpips:
            loss_lpips = self.lpips(pred, target, normalize=True).mean()
        else:
            loss_lpips = torch.zeros((), device=pred.device, dtype=pred.dtype)
        return loss_lap, loss_census, loss_l1, loss_lpips

    def trajectory_loss(self, outputs, teacher):
        """
        Distil the privileged teacher's f_{0->t} and f_{1->t} into Phi(t)
        and Phi'(t).

        Charbonnier rather than plain L1: the teacher is RAFT run on a
        real intermediate frame, so it is confidently wrong at occlusions
        and around disocclusions, and a robust penalty stops those pixels
        from dominating. The targets are already on the same lattices as
        the predictions - f_{0->t} on lattice 0, f_{1->t} on lattice 1 -
        so no warping is involved.

        Both sides are divided by the per-sample motion scale s. Without
        it this term is measured in PIXELS while every other term is in
        [0, 1] intensity units, so its relative weight would swing with
        the dataset: on Vimeo triplets it lands near 0.07, on X4K with a
        32-frame window near 0.9. Normalising makes
        `loss.flow_distill_weight` mean the same thing everywhere.
        """
        scale = outputs["flow_scale"]
        loss = 0.0
        for k, (target_phi, target_psi) in enumerate(teacher):
            loss = loss + self.l1(outputs["phi"][k] / scale, target_phi / scale)
            loss = loss + self.l1(outputs["psi"][k] / scale, target_psi / scale)
        return loss / (2 * len(teacher))

    @staticmethod
    def velocity_consistency_loss(outputs):
        """
        v2.1 Training, item 3: cross-lattice velocity consistency.

        The particle leaving x arrives at x + F(x) on lattice 1, so the
        two independently-predicted velocity fields must agree there:

            m1(x) = -backwarp(m0', F)(x),   m0(x) = -backwarp(m1', F')(x)

        A regulariser, not a driver - d = 0 satisfies it trivially, so it
        prunes inconsistent non-linear solutions between the forward and
        swapped passes rather than creating curvature by itself. Weighted
        by (1 - alpha_occ), exactly as specified: heaviest where the
        occlusion gate has already decided the flow is unreliable.

        Divided by the per-sample motion scale s, for exactly the reason
        trajectory_loss is: m0/m1 are raw pixels, and without normalising
        this term's magnitude swings with how much the clip moves while
        `loss.velocity_consistency_weight` stays fixed. Measured without
        this on a real run: vel_consistency ~0.56 px against distill's
        ~0.018 (already scale-normalised), so at the documented weight of
        0.1 this unnormalised term supplied ~76% of the total loss instead
        of acting as a minor regulariser - and flow-PSNR gain over the
        linear baseline went NEGATIVE (the model was being pulled towards
        cross-lattice agreement at the expense of matching the teacher).
        """
        scale = outputs["flow_scale"]
        flow_fwd = outputs["raft_flow"][:, :, 0]
        flow_bwd = outputs["raft_flow"][:, :, 1]
        term0 = (1.0 - outputs["alpha_fwd"]) * (
            (outputs["m1"] + warp(outputs["m0_swapped"], flow_fwd)) / scale
        )
        term1 = (1.0 - outputs["alpha_bwd"]) * (
            (outputs["m0"] + warp(outputs["m1_swapped"], flow_bwd)) / scale
        )
        return 0.5 * (term0.abs().mean() + term1.abs().mean())

    @staticmethod
    def _flow_psnr(preds, targets, scale):
        """PSNR of a predicted trajectory against the teacher, on the
        scale-normalised flow so it is comparable across clips."""
        mse = torch.stack(
            [(((p - t) / scale) ** 2).mean() for p, t in zip(preds, targets)]
        ).mean()
        return -10 * torch.log10(mse.clamp_min(1e-12))

    def compute_loss(self, outputs, gts, model_module, teacher=None):
        """
        Average the per-timestep losses. Returns (total, parts dict, psnr).

        Stage 1 has no image term at all: phases 4-5 never ran, so the
        only objective is the trajectory distillation, exactly as GIMM's
        stage 1 optimises an MSE flow loss on the motion module alone.
        """
        if self.stage == 1:
            zero = torch.zeros((), device=outputs["flow_scale"].device)
            distill = self.trajectory_loss(outputs, teacher)
            total = distill
            parts = {k: zero for k in ("lap", "census", "l1", "lpips")}
            parts["distill"] = distill
            if self.velocity_consistency_weight > 0:
                vel_consistency = self.velocity_consistency_loss(outputs)
                total = total + self.velocity_consistency_weight * vel_consistency
                parts["vel_consistency"] = vel_consistency
            else:
                parts["vel_consistency"] = zero
            parts["delta_0"] = outputs["delta_norm_0"].mean()
            parts["delta_1"] = outputs["delta_norm_1"].mean()
            # Flow PSNR, the stage-1 analogue of image PSNR: how well
            # Phi(t) matches the teacher, on the scale-free flow. Reported
            # alongside the SAME metric for the d = 0 linear trajectory,
            # because on its own a flow PSNR is uninterpretable - X4K
            # clips differ enormously in difficulty, so the absolute
            # number swings by 15 dB between batches while saying nothing
            # about whether the model is learning. psnr - psnr_lin is the
            # quantity that matters: dB gained over linear motion.
            with torch.no_grad():
                scale = outputs["flow_scale"]
                targets = [tp for tp, _ in teacher]
                psnr = self._flow_psnr(outputs["phi"], targets, scale)
                parts["psnr_lin"] = self._flow_psnr(
                    outputs["phi_linear"], targets, scale
                )
            return total, parts, psnr

        num_targets = len(gts)
        parts = {"lap": 0.0, "census": 0.0, "l1": 0.0, "lpips": 0.0}
        psnr = 0.0

        for k, target in enumerate(gts):
            pred = outputs["imgt_pred"][k]
            lap, census, l1, lpips = self.image_losses(pred, target)

            for aux in outputs["other_pred"][k]:
                a_lap, a_census, a_l1, a_lpips = self.image_losses(aux, target)
                lap = lap + AUX_LOSS_WEIGHT * a_lap
                census = census + AUX_LOSS_WEIGHT * a_census
                l1 = l1 + AUX_LOSS_WEIGHT * a_l1
                lpips = lpips + AUX_LOSS_WEIGHT * a_lpips

            parts["lap"] = parts["lap"] + lap
            parts["census"] = parts["census"] + census
            parts["l1"] = parts["l1"] + l1
            parts["lpips"] = parts["lpips"] + lpips
            psnr = psnr + model_module.compute_psnr(pred.detach(), target)

        parts = {key: value / num_targets for key, value in parts.items()}
        total = parts["lap"] + parts["census"] + parts["l1"] + parts["lpips"]

        if teacher is not None and self.flow_distill_weight > 0:
            distill = self.trajectory_loss(outputs, teacher)
            total = total + self.flow_distill_weight * distill
            parts["distill"] = distill
        else:
            parts["distill"] = torch.zeros((), device=total.device)

        if self.velocity_consistency_weight > 0:
            vel_consistency = self.velocity_consistency_loss(outputs)
            total = total + self.velocity_consistency_weight * vel_consistency
            parts["vel_consistency"] = vel_consistency
        else:
            parts["vel_consistency"] = torch.zeros((), device=total.device)

        parts["psnr_lin"] = torch.zeros((), device=total.device)
        # Not losses - the collapse diagnostic.
        parts["delta_0"] = outputs["delta_norm_0"].mean()
        parts["delta_1"] = outputs["delta_norm_1"].mean()

        return total, parts, psnr / num_targets

    # ------------------------------------------------------------------

    @torch.no_grad()
    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        model = self.model_ema if ema else self.model
        loader = self.loader_val if valid else self.loader_trn
        n_inst = len(self.dataset_val) if valid else len(self.dataset_trn)

        self.lpips.to(self.device)
        accm = self.get_accm()
        # For the EMA path model.module is the EMA wrapper, which forwards
        # compute_psnr to the wrapped model.
        model_module = model.module

        pbar = tqdm(enumerate(loader), total=len(loader)) if self.distenv.master else enumerate(loader)

        model.eval()
        for _it, batch in pbar:
            img_xs, gts, t_list = self.unpack(batch, self.device)
            teacher = self.cached_teacher(batch, self.device)
            if teacher is None and (self.stage == 1 or self.flow_distill_weight > 0):
                teacher = model_module.teacher_flows(
                    img_xs[:, :, 0], img_xs[:, :, 1], gts,
                    iters=self.teacher_raft_iter,
                )
            outputs = model(
                img_xs,
                t=t_list,
                return_diagnostics=False,
                return_trajectory=teacher is not None,
                trajectory_only=self.stage == 1,
            )

            total, parts, _psnr = self.compute_loss(
                outputs, gts, model_module, teacher=teacher
            )
            count = img_xs.shape[0]
            if self.stage == 1:
                psnr_sum = _psnr * count
            else:
                psnr_sum = sum(
                    model_module.compute_psnr(
                        outputs["imgt_pred"][k], gt, reduction="sum"
                    )
                    for k, gt in enumerate(gts)
                ) / len(gts)

            accm.update(
                dict(
                    loss_total=total * count,
                    lap=parts["lap"] * count,
                    census=parts["census"] * count,
                    l1=parts["l1"] * count,
                    lpips=parts["lpips"] * count,
                    distill=parts["distill"] * count,
                    vel_consistency=parts["vel_consistency"] * count,
                    delta_0=parts["delta_0"] * count,
                    delta_1=parts["delta_1"] * count,
                    psnr=psnr_sum,
                    psnr_lin=parts["psnr_lin"] * count,
                ),
                count=count,
                sync=True,
                distenv=self.distenv,
            )

            if self.distenv.master:
                pbar.set_description(accm.get_summary().print_line())

        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + accm.get_summary(n_inst).print_line())

        summary = accm.get_summary(n_inst)
        summary["xs"] = batch["xs"]
        summary["t"] = batch["t"]
        return summary

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        model = self.model
        model_ema = self.model_ema
        model_module = model.module
        total_step = len(self.loader_trn) * epoch

        self.lpips.to(self.device)
        accm = self.get_accm()
        amp_enabled = bool(self.config.experiment.amp)

        # Fixed 10-char bar. tqdm cannot detect the width of a Kaggle
        # notebook cell, so with a default (elastic) bar the postfix is
        # what gets clipped - and the postfix is the part worth reading.
        pbar = (
            tqdm(
                enumerate(self.loader_trn),
                total=len(self.loader_trn),
                # ncols pins the total width so tqdm neither pads to a
                # guessed terminal size nor lets the bar grow and push
                # the metrics out of view.
                ncols=110,
                bar_format="{desc} {percentage:3.0f}%|{bar:6}|{n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining},{rate_fmt}]{postfix}",
            )
            if self.distenv.master
            else enumerate(self.loader_trn)
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        steps_per_epoch = max(1, len(self.loader_trn) // self.grad_accm_steps)
        opt_step = epoch * steps_per_epoch

        for it, batch in pbar:
            img_xs, gts, t_list = self.unpack(batch, self.device)

            # Gradient accumulation: experiment.total_batch_size is reached by
            # accumulating grad_accm_steps micro-batches, which is also what
            # the LR schedule in main.py was built for. Under DDP, suppress
            # the all-reduce on the micro-batches that are not going to step -
            # syncing them would be correct but pure wasted bandwidth.
            stepping = (it + 1) % self.grad_accm_steps == 0
            sync_context = (
                model.no_sync() if is_ddp and not stepping else nullcontext()
            )

            # The privileged teacher runs outside autocast and without
            # grad: it is the frozen estimator applied to the ground-truth
            # middle frames, so it is a target, not part of the graph.
            teacher = self.cached_teacher(batch, self.device)
            if teacher is None and self.flow_distill_weight > 0:
                teacher = model_module.teacher_flows(
                    img_xs[:, :, 0],
                    img_xs[:, :, 1],
                    gts,
                    iters=self.teacher_raft_iter,
                )

            with sync_context:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(
                        img_xs,
                        t=t_list,
                        return_diagnostics=False,
                        return_trajectory=teacher is not None,
                    )
                    loss, parts, psnr = self.compute_loss(
                        outputs, gts, model_module, teacher=teacher
                    )

                # A non-finite loss must never reach .backward(): it
                # writes NaN into every weight it touches and the run is
                # unrecoverable from that point on - the whole optimizer
                # state is poisoned, not just the step. Skipping the
                # micro-batch costs one sample; not skipping it costs the
                # run. GradScaler already skips on inf GRADIENTS, but it
                # cannot see a loss that was non-finite to begin with.
                # The decision must be COLLECTIVE. Under DDP .backward()
                # runs an all-reduce, so a rank that skips it while its
                # peer does not leaves that peer waiting on a collective
                # that never arrives - the job hangs instead of crashing.
                # All-reduce the flag first so every rank skips together.
                bad = torch.tensor(
                    [0.0 if torch.isfinite(loss) else 1.0], device=loss.device
                )
                if self.distenv.world_size > 1:
                    torch.distributed.all_reduce(bad)
                if bad.item() > 0:
                    self.nonfinite_skips += 1
                    if self.nonfinite_skips <= 10 and self.distenv.master:
                        logger.warning(
                            "non-finite loss at step %d, micro-batch skipped "
                            "(%d so far)", total_step, self.nonfinite_skips
                        )
                        # Which PART is non-finite, on the rank that logs -
                        # a bare "non-finite loss" line does not say whether
                        # it is the (pre-existing) distillation term or the
                        # (new, v2.1) velocity-consistency term, and the two
                        # have very different likely causes.
                        if self.nonfinite_skips <= 3:
                            detail = {
                                k: (float(v.detach()) if torch.is_tensor(v) else v)
                                for k, v in parts.items()
                            }
                            logger.warning("  loss parts: %s", detail)
                            for key in ("m0", "m1", "alpha_fwd", "raft_flow"):
                                if key in outputs and torch.is_tensor(outputs[key]):
                                    t = outputs[key]
                                    logger.warning(
                                        "  outputs[%s]: finite=%s min=%.4g max=%.4g",
                                        key, bool(torch.isfinite(t).all()),
                                        float(t.min()), float(t.max()),
                                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss / self.grad_accm_steps).backward()

            if stepping:
                if self.config.optimizer.max_gn is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.config.optimizer.max_gn
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if model_ema:
                    model_ema.module.update(model.module, total_step)

                opt_step += 1
                if self.distenv.master and opt_step % self.log_every == 0:
                    # Instantaneous, not the epoch running mean. Detach
                    # first: several of these still carry grad, and float()
                    # on a grad-tracking tensor warns and forces a sync.
                    def scalar(v):
                        return float(v.detach() if torch.is_tensor(v) else v)

                    for key in ("lap", "census", "l1", "lpips", "distill", "vel_consistency"):
                        self.writer.add_scalar(
                            f"step/{key}", scalar(parts[key]), "train", opt_step
                        )
                    if self.stage == 1:
                        self.writer.add_scalar(
                            "step/psnr_linear", scalar(parts["psnr_lin"]),
                            "train", opt_step,
                        )
                        self.writer.add_scalar(
                            "step/psnr_gain_over_linear",
                            scalar(psnr) - scalar(parts["psnr_lin"]),
                            "train", opt_step,
                        )
                    for key in ("delta_0", "delta_1"):
                        self.writer.add_scalar(
                            f"step/{key}", scalar(parts[key]), "train", opt_step
                        )
                    self.writer.add_scalar(
                        "step/loss_total", scalar(loss), "train", opt_step
                    )
                    self.writer.add_scalar(
                        "step/psnr", scalar(psnr), "train", opt_step
                    )
                    self.writer.add_scalar(
                        "step/lr", scheduler.get_last_lr()[0], "train", opt_step
                    )
                    # Peak motion of the batch, in pixels. If this is a
                    # large fraction of crop_size the flow has nothing to
                    # match against and every downstream signal is noise.
                    if "flow_scale" in outputs:
                        self.writer.add_scalar(
                            "step/flow_scale_px",
                            scalar(outputs["flow_scale"].mean()),
                            "train", opt_step,
                        )

            accm.update(
                dict(
                    loss_total=loss.detach(),
                    lap=parts["lap"].detach(),
                    census=parts["census"].detach(),
                    l1=parts["l1"].detach(),
                    lpips=parts["lpips"].detach()
                    if torch.is_tensor(parts["lpips"])
                    else parts["lpips"],
                    distill=parts["distill"].detach(),
                    vel_consistency=parts["vel_consistency"].detach(),
                    delta_0=parts["delta_0"].detach(),
                    delta_1=parts["delta_1"].detach(),
                    psnr=psnr,
                    psnr_lin=parts["psnr_lin"].detach(),
                ),
                count=1,
            )
            total_step += 1

            if self.distenv.master:
                # Keep the description short so tqdm still has room for the
                # bar, rate and ETA; metrics go in the postfix. These are
                # INSTANTANEOUS values - the epoch running means are logged
                # at epoch end and to TensorBoard, and over thousands of
                # iterations a running mean lags too badly to be useful
                # here. Everything is detached: several still carry grad,
                # and float() on those warns and forces a device sync.
                # Short keys and no leading zeros: a Kaggle output pane
                # clips horizontally rather than wrapping, so anything
                # past ~110 characters is simply invisible. Everything
                # here is also in TensorBoard (step/*) and in the
                # epoch-end log line, which is where to read exact
                # values; this is a glance, not a record.
                now_loss = float(loss.detach())
                now_psnr = float(psnr.detach() if torch.is_tensor(psnr) else psnr)
                postfix = {"L": f"{now_loss:.4f}", "P": f"{now_psnr:.3f}"}
                if self.stage == 1:
                    # 4 decimals on the gain: early on it is thousandths
                    # of a dB and coarser rounding hides the whole signal.
                    # `lin` is omitted - it is exactly P - g, and it is
                    # logged to TensorBoard as step/psnr_linear.
                    now_lin = float(parts["psnr_lin"].detach())
                    postfix["g"] = f"{now_psnr - now_lin:+.4f}"
                postfix["d"] = f"{float(parts['delta_0'].detach()):.4f}"
                postfix["lr"] = f"{scheduler.get_last_lr()[0]:.0e}"

                pbar.set_description(f"ep{epoch}")
                pbar.set_postfix(postfix, refresh=False)

        summary = accm.get_summary()
        summary["xs"] = batch["xs"]
        summary["t"] = batch["t"]
        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode="train"):
        if epoch % 10 == 1 or epoch % self.config.experiment.test_imlog_freq == 0:
            self.reconstruct(summary, epoch=epoch, mode=mode)

        for key in ("lap", "census", "l1", "psnr", "lpips", "distill", "vel_consistency"):
            self.writer.add_scalar(f"loss/{key}", summary[key], mode, epoch)
        if self.stage == 1:
            self.writer.add_scalar("loss/psnr_lin", summary["psnr_lin"], mode, epoch)

        # The collapse diagnostic. If these flatline at zero the model is
        # the linear baseline wearing a cubic costume.
        for key in ("delta_0", "delta_1"):
            self.writer.add_scalar(f"velocity_residual/{key}", summary[key], mode, epoch)

        if mode == "train":
            self.writer.add_scalar("lr", scheduler.get_last_lr()[0], mode, epoch)

        line = f"""ep:{epoch}, {mode:10s}, """
        line += summary.print_line()
        if scheduler:
            line += f""", lr: {scheduler.get_last_lr()[0]:e}"""
        logger.info(line)

    @torch.no_grad()
    def reconstruct(self, summary, epoch=0, mode="valid"):
        """Log the ground truth and the prediction for every supervised t."""
        if self.stage == 1:
            return  # stage 1 never renders a frame
        model = self.model_ema if "ema" in mode else self.model
        model.eval()

        batch = {"xs": summary["xs"][:4], "t": summary["t"][:4]}
        img_xs, gts, t_list = self.unpack(batch, self.device)

        preds = model(img_xs, t=t_list, return_diagnostics=False)["imgt_pred"]

        rows = []
        for gt, pred in zip(gts, preds):
            rows.append(gt)
            rows.append(torch.clamp(pred, 0, 1))

        grid = torchvision.utils.make_grid(torch.cat(rows, dim=0), nrow=img_xs.shape[0])
        self.writer.add_image("reconstruction", grid, mode, epoch)
