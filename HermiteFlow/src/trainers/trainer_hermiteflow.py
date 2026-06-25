# --------------------------------------------------------
# HermiteFlow-VFI — Trainer
#
# Single-stage end-to-end training. No INR flow reconstruction
# loss — only image-level losses (Laplacian, Census, L1, LPIPS).
# Mirrors trainer_gimmvfi.py structure but simplified for
# HermiteFlow's direct coefficient-to-frame pipeline.
# --------------------------------------------------------

import logging

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from utils.accumulator import AccmStageINR
from .trainer import TrainerTemplate

logger = logging.getLogger(__name__)
from utils.loss import LapLoss, Ternary, Charbonnier_L1
from utils.lpips import LPIPS


class Trainer(TrainerTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.using_lpips = getattr(self.config.loss, "perceptual_loss", False)
        self.lap = LapLoss()
        self.census = Ternary()
        self.l1 = Charbonnier_L1()
        self.lpips = LPIPS(net="alex", version="0.1").eval()
        for name, param in self.lpips.named_parameters():
            param.requires_grad = False

    def get_accm(self):
        accm = AccmStageINR(
            scalar_metric_names=(
                "loss_total",
                "lap",
                "census",
                "l1",
                "lpips",
                "psnr",
            ),
            device=self.device,
        )
        return accm

    @torch.no_grad()
    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        model = self.model_ema if ema else self.model
        loader = self.loader_val if valid else self.loader_trn
        n_inst = len(self.dataset_val) if valid else len(self.dataset_trn)

        accm = self.get_accm()

        if self.distenv.master:
            pbar = tqdm(enumerate(loader), total=len(loader))
        else:
            pbar = enumerate(loader)

        model.eval()
        for it, xs in pbar:
            model.zero_grad()
            timesteps = (
                [xs["t"].to(self.device)]
                if "t" in xs.keys()
                else [
                    0.5 * torch.ones(xs["xs"].shape[0]).to(self.device).to(torch.float)
                ]
            )

            precomputed_flows = xs.get("precomputed_flows", None)
            if precomputed_flows is not None:
                precomputed_flows = precomputed_flows.to(self.device, non_blocking=True)
            xs = xs["xs"]
            xs = xs.to(self.device)  # [B, C, T, H, W]

            img_xs, gt = xs[:, :, :2], xs[:, :, 2]
            batch_size = img_xs.shape[0]

            # HermiteFlow doesn't use coord_inputs — just pass t
            coord_inputs = [
                (
                    model.module.sample_coord_input(
                        batch_size, img_xs.shape[-2:], timesteps[0],
                        device=img_xs.device
                    ),
                    None,
                )
            ]
            all_outputs = model(img_xs, coord_inputs, t=timesteps, precomputed_flows=precomputed_flows)
            targets = gt.detach()

            ######Loss Calculation######
            cur_count = targets.shape[0]
            loss_lap = (self.lap(all_outputs["imgt_pred"][0], targets)).mean()
            loss_census = self.census(all_outputs["imgt_pred"][0], targets)
            loss_l1 = self.l1(all_outputs["imgt_pred"][0], targets)
            loss_lpips = self.lpips(
                all_outputs["imgt_pred"][0], targets, normalize=True
            ).mean()
            psnr = model.module.compute_psnr(
                all_outputs["imgt_pred"][0], targets, reduction="sum"
            )

            metrics = dict(
                lap=loss_lap * cur_count,
                census=loss_census * cur_count,
                l1=loss_l1 * cur_count,
                psnr=psnr,
                lpips=loss_lpips,
            )

            accm.update(metrics, count=cur_count, sync=True, distenv=self.distenv)

            if self.distenv.master:
                line = accm.get_summary().print_line()
                pbar.set_description(line)
        line = accm.get_summary(n_inst).print_line()

        if self.distenv.master and verbose:
            mode = "valid" if valid else "train"
            mode = "%s_ema" % mode if ema else mode
            logger.info(f"""{mode:10s}, """ + line)
            self.reconstruct(xs, epoch=0, mode=mode)

        summary = accm.get_summary(n_inst)
        summary["xs"] = xs
        summary["t"] = timesteps[-1]
        return summary

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        model = self.model
        model_ema = self.model_ema
        total_step = len(self.loader_trn) * epoch

        accm = self.get_accm()

        if self.distenv.master:
            pbar = tqdm(enumerate(self.loader_trn), total=len(self.loader_trn))
        else:
            pbar = enumerate(self.loader_trn)

        self.lpips.to(self.device)
        model.train()
        for it, xs in pbar:
            timesteps = (
                xs["t"].to(self.device, non_blocking=True)
                if "t" in xs.keys()
                else 0.5
                * torch.ones(xs["xs"].shape[0])
                .to(self.device, non_blocking=True)
                .to(torch.float)
            )
            precomputed_flows = xs.get("precomputed_flows", None)
            if precomputed_flows is not None:
                precomputed_flows = precomputed_flows.to(self.device, non_blocking=True)
            xs = xs["xs"]

            model.zero_grad(set_to_none=True)
            xs = xs.to(self.device, non_blocking=True)
            img_xs, gt = xs[:, :, :2], xs[:, :, 2]
            batch_size = img_xs.shape[0]

            # Single-stage: only the target timestep (no flow reconstruction)
            timesteps_list = [timesteps]

            coord_inputs = [
                (
                    model.module.sample_coord_input(
                        batch_size, img_xs.shape[-2:], timesteps,
                        device=xs.device
                    ),
                    None,
                )
            ]

            all_outputs = model(img_xs, coord_inputs, t=timesteps_list, precomputed_flows=precomputed_flows)

            targets = [gt.detach()]
            mid_id = 0
            assert len(all_outputs["imgt_pred"]) == 1
            psnr = model.module.compute_psnr(
                all_outputs["imgt_pred"][mid_id], targets[mid_id]
            )

            ######Loss Calculation######
            ## i. image loss
            loss_lap = 0
            loss_census = 0
            loss_l1 = 0
            loss_lpips = 0

            # Auxiliary loss from intermediate warped images (scale 1/4)
            if all_outputs["other_pred"][0][0] is not None:
                for i in range(len(all_outputs["other_pred"][0])):
                    loss_lap = (
                        loss_lap
                        + 0.5
                        * (
                            self.lap(all_outputs["other_pred"][0][i], targets[mid_id])
                        ).mean()
                    )
                    loss_census = loss_census + 0.5 * self.census(
                        all_outputs["other_pred"][0][i], targets[mid_id]
                    )
                    loss_l1 = loss_l1 + 0.5 * self.l1(
                        all_outputs["other_pred"][0][i], targets[mid_id]
                    )
                    if self.using_lpips:
                        loss_lpips = (
                            loss_lpips
                            + 0.5
                            * self.lpips(
                                all_outputs["other_pred"][0][i],
                                targets[mid_id],
                                normalize=True,
                            ).mean()
                        )

            # Main loss from final prediction
            loss_lap = (
                loss_lap
                + (self.lap(all_outputs["imgt_pred"][0], targets[mid_id])).mean()
            )
            loss_census = loss_census + self.census(
                all_outputs["imgt_pred"][0], targets[mid_id]
            )
            loss_l1 = loss_l1 + self.l1(all_outputs["imgt_pred"][0], targets[mid_id])
            if self.using_lpips:
                loss_lpips = (
                    loss_lpips
                    + self.lpips(
                        all_outputs["imgt_pred"][0], targets[mid_id], normalize=True
                    ).mean()
                )

            # Total loss — no flow reconstruction loss (unlike GIMM-VFI)
            loss = loss_census + loss_l1 + loss_lap + loss_lpips

            loss.backward()
            if self.config.optimizer.max_gn is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.optimizer.max_gn
                )
            optimizer.step()
            scheduler.step()
            if model_ema:
                model_ema.module.update(model.module, total_step)

            metrics = dict(
                loss_total=loss,
                lap=loss_lap,
                census=loss_census,
                l1=loss_l1,
                lpips=loss_lpips,
                psnr=psnr,
            )
            accm.update(metrics, count=1)
            total_step += 1
            if self.distenv.master:
                line = f"""(epoch {epoch} / iter {it}) """
                line += accm.get_summary().print_line()
                line += f""", lr: {scheduler.get_last_lr()[0]:e}"""
                pbar.set_description(line)

        summary = accm.get_summary()
        summary["xs"] = xs
        summary["t"] = timesteps
        return summary

    def logging(self, summary, scheduler=None, epoch=0, mode="train"):
        if epoch % 10 == 1 or epoch % self.config.experiment.test_imlog_freq == 0:
            self.reconstruct(summary, upsample_ratio=1, epoch=epoch, mode=mode)

        self.writer.add_scalar("loss/lap", summary["lap"], mode, epoch)
        self.writer.add_scalar("loss/census", summary["census"], mode, epoch)
        self.writer.add_scalar("loss/l1", summary["l1"], mode, epoch)
        self.writer.add_scalar("loss/psnr", summary["psnr"], mode, epoch)
        self.writer.add_scalar("loss/lpips", summary["lpips"], mode, epoch)

        if mode == "train":
            self.writer.add_scalar("lr", scheduler.get_last_lr()[0], mode, epoch)

        line = f"""ep:{epoch}, {mode:10s}, """
        line += summary.print_line()
        line += f""", """
        if scheduler:
            line += f"""lr: {scheduler.get_last_lr()[0]:e}"""

        logger.info(line)

    @torch.no_grad()
    def reconstruct(self, summary, upsample_ratio=1, epoch=0, mode="valid"):
        xs = summary["xs"]
        timesteps = [summary["t"][:8]]

        def get_recon_imgs(xs_real, xs_recon, upsample_ratio=1):
            xs_real = xs_real
            if not upsample_ratio == 1:
                xs_real = torch.nn.functional.interpolate(
                    xs_real, scale_factor=upsample_ratio
                )
            xs_recon = xs_recon
            xs_recon = torch.clamp(xs_recon, 0, 1)
            return xs_real, xs_recon

        model = self.model_ema if "ema" in mode else self.model
        model.eval()

        xs_real = xs[:8]

        img_xs, xs_real = xs_real[:, :, :2], xs_real[:, :, 2]
        batch_size = img_xs.shape[0]
        coord_inputs = [
            (
                model.module.sample_coord_input(
                    batch_size, img_xs.shape[-2:], timesteps[0],
                    device=img_xs.device
                ),
                None,
            )
        ]

        xs_pred = model(img_xs, coord_inputs, t=timesteps)["imgt_pred"][0]

        xs_real, xs_recon = get_recon_imgs(xs_real, xs_pred, upsample_ratio)
        grid = torch.cat([xs_real, xs_recon], dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=xs_real.shape[0])
        self.writer.add_image(f"reconstruction_x{upsample_ratio}", grid, mode, epoch)
