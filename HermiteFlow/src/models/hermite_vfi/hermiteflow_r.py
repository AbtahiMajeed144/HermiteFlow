# --------------------------------------------------------
# HermiteFlow-VFI (RAFT-based variant)
#
# Module 1 (Head):   RAFT flow estimator (pretrained)
# Module 2 (Brains): CoefficientNet — predicts Hermite coefficients
# Module 3 (Engine): HermiteSplineEngine — pure-math flow interpolation
# Module 4 (Canvas): Backward warping via bilinear grid_sample
# Module 5 (Tail):   AMT-style synthesis decoder (from GIMM-VFI)
#
# References:
# amt: https://github.com/MCG-NKU/AMT
# gimm-vfi: https://github.com/GSeanCDAT/GIMM-VFI
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import HermiteFlowConfig
from .modules.coefficient_net import CoefficientNet
from .modules.hermite_spline import HermiteSplineEngine
from .modules.fi_components import (
    BasicUpdateBlock,
    NewInitDecoder,
    NewMultiFlowDecoder,
    multi_flow_combine,
)
from .raft import initialize_RAFT
from .raft.corr import BidirCorrBlock
from .modules.fi_utils import warp, resize, build_coord


class HermiteFlow_R(nn.Module):
    """
    HermiteFlow-VFI with RAFT backbone.

    Replaces GIMM-VFI's INR/HypoNet flow prediction with Hermite cubic
    spline interpolation, while keeping the AMT-style decoder for synthesis.
    """
    Config = HermiteFlowConfig

    def __init__(self, config: HermiteFlowConfig):
        super().__init__()
        self.config = config = config.copy()
        self.raft_iter = config.raft_iter

        # ============== Module 1: The Head (RAFT) ==============
        self.flow_estimator = initialize_RAFT()

        # Feature projection layers (same as GIMM-VFI-R)
        cur_f_dims = [128, 96]   # RAFT feature dimensions
        f_dims = [256, 128]      # Projected dimensions

        skip_channels = f_dims[-1] // 2
        self.num_flows = 3

        self.amt_last_cproj = nn.Conv2d(cur_f_dims[0], f_dims[0], 1)
        self.amt_second_last_cproj = nn.Conv2d(cur_f_dims[1], f_dims[1], 1)
        self.amt_fproj = nn.Conv2d(f_dims[0], f_dims[0], 1)

        # ============== Module 2: The Brains (Coefficient Net) ==============
        # feat_channels = f_dims[-1] since we use projected features at H/4
        self.coefficient_net = CoefficientNet(
            flow_channels=4,
            feat_channels=f_dims[-1],
            num_coefficients=config.num_coefficients,
            mid_channels=config.coeff_net_channels,
        )

        # ============== Module 3: The Engine (Hermite Spline) ==============
        self.hermite_engine = HermiteSplineEngine()

        # ============== Module 5: The Tail (AMT Decoder) ==============
        # Exactly the same AMT-style decoder as GIMM-VFI-R
        self.amt_init_decoder = NewInitDecoder(f_dims[0], skip_channels)
        self.amt_final_decoder = NewMultiFlowDecoder(f_dims[1], skip_channels)

        self.amt_update4_low = self._get_updateblock(f_dims[0] // 2, 2.0)
        self.amt_update4_high = self._get_updateblock(f_dims[0] // 2, None)

        self.amt_comb_block = nn.Sequential(
            nn.Conv2d(3 * self.num_flows, 6 * self.num_flows, 7, 1, 3),
            nn.PReLU(6 * self.num_flows),
            nn.Conv2d(6 * self.num_flows, 3, 7, 1, 3),
        )

    def _get_updateblock(self, cdim, scale_factor=None):
        return BasicUpdateBlock(
            cdim=cdim,
            hidden_dim=192,
            flow_dim=64,
            corr_dim=256,
            corr_dim2=192,
            fc_dim=188,
            scale_factor=scale_factor,
            corr_levels=4,
            radius=4,
        )

    def cal_bidirection_flow(self, im0, im1, iters=20):
        """
        Module 1 — The Head: extract bidirectional flows + features via RAFT.
        """
        f01, features0, fnet0 = self.flow_estimator(
            im0, im1, return_feat=True, iters=iters
        )
        f10, features1, fnet1 = self.flow_estimator(
            im1, im0, return_feat=True, iters=iters
        )
        corr_fn = BidirCorrBlock(
            self.amt_fproj(fnet0), self.amt_fproj(fnet1), radius=4
        )

        # Project features to the expected dimensions
        features0 = [
            self.amt_second_last_cproj(features0[0]),
            self.amt_last_cproj(features0[1]),
        ]
        features1 = [
            self.amt_second_last_cproj(features1[0]),
            self.amt_last_cproj(features1[1]),
        ]

        return f01, f10, features0, features1, corr_fn

    def warp_w_mask(self, img0, img1, ft0, ft1, mask, scale=1):
        """Backward warping with blending mask at a given scale."""
        ft0 = scale * resize(ft0, scale_factor=scale)
        ft1 = scale * resize(ft1, scale_factor=scale)
        mask = resize(mask, scale_factor=scale).sigmoid()
        img0_warp = warp(img0, ft0)
        img1_warp = warp(img1, ft1)
        img_warp = mask * img0_warp + (1 - mask) * img1_warp
        return img_warp

    def frame_synthesize(
        self, img_xs, flow_t0, flow_t1, features0, features1, corr_fn,
        cur_t, full_img=None
    ):
        """
        Module 5 — The Tail: AMT-style decoder for frame synthesis.

        Takes the Hermite-predicted bilateral flows and refines them through
        correlation-guided updates, then blends via multi-flow combination.
        Identical architecture to GIMM-VFI-R.
        """
        batch_size = img_xs.shape[0]
        img0 = 2 * img_xs[:, :, 0] - 1.0
        img1 = 2 * img_xs[:, :, 1] - 1.0

        # Initialize coordinates for correlation lookup
        lookup_coord = build_coord(img_xs[:, :, 0]).to(img_xs.device)

        # Scale Hermite flows to 1/4 resolution
        inv = 1 / 4
        flow_t0_4 = inv * resize(flow_t0, inv)
        flow_t1_4 = inv * resize(flow_t1, inv)

        ######################### scale 1/4 #########################
        # i. Initialize feature at scale 1/4
        flowt0_4, flowt1_4, ft_4_ = self.amt_init_decoder(
            features0[-1], features1[-1],
            flow_t0_4, flow_t1_4,
            img0=img0, img1=img1,
        )
        features0, features1 = features0[:-1], features1[:-1]

        mask_4_, ft_4_ = ft_4_[:, :1], ft_4_[:, 1:]
        img_warp_4 = self.warp_w_mask(
            img0, img1, flowt0_4, flowt1_4, mask_4_, scale=4
        )
        img_warp_4 = (img_warp_4 + 1.0) / 2
        img_warp_4 = torch.clamp(img_warp_4, 0, 1)

        corr_4, flow_4_lr = self._amt_corr_scale_lookup(
            corr_fn, lookup_coord, flowt0_4, flowt1_4, cur_t, downsample=2
        )

        delta_ft_4_, delta_flow_4 = self.amt_update4_low(ft_4_, flow_4_lr, corr_4)
        delta_flow0_4, delta_flow1_4 = torch.chunk(delta_flow_4, 2, 1)
        flowt0_4 = flowt0_4 + delta_flow0_4
        flowt1_4 = flowt1_4 + delta_flow1_4
        ft_4_ = ft_4_ + delta_ft_4_

        # iii. residue update with lookup corr
        corr_4 = resize(corr_4, scale_factor=2.0)
        flow_4 = torch.cat([flowt0_4, flowt1_4], dim=1)
        delta_ft_4_, delta_flow_4 = self.amt_update4_high(ft_4_, flow_4, corr_4)
        flowt0_4 = flowt0_4 + delta_flow_4[:, :2]
        flowt1_4 = flowt1_4 + delta_flow_4[:, 2:4]
        ft_4_ = ft_4_ + delta_ft_4_

        ######################### scale 1/1 #########################
        flowt0_1, flowt1_1, mask, img_res = self.amt_final_decoder(
            ft_4_, features0[0], features1[0],
            flowt0_4, flowt1_4,
            mask=mask_4_, img0=img0, img1=img1,
        )

        if full_img is not None:
            img0 = 2 * full_img[:, :, 0] - 1.0
            img1 = 2 * full_img[:, :, 1] - 1.0
            inv = img1.shape[2] / flowt0_1.shape[2]
            flowt0_1 = inv * resize(flowt0_1, scale_factor=inv)
            flowt1_1 = inv * resize(flowt1_1, scale_factor=inv)
            mask = resize(mask, scale_factor=inv)
            img_res = resize(img_res, scale_factor=inv)

        imgt_pred = multi_flow_combine(
            self.amt_comb_block, img0, img1,
            flowt0_1, flowt1_1, mask, img_res, None
        )
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        ################################################################
        flowt0_1 = flowt0_1.reshape(
            batch_size, self.num_flows, 2, img0.shape[-2], img0.shape[-1]
        )
        flowt1_1 = flowt1_1.reshape(
            batch_size, self.num_flows, 2, img0.shape[-2], img0.shape[-1]
        )

        flowt0_pred = [flowt0_1, flowt0_4]
        flowt1_pred = [flowt1_1, flowt1_4]
        other_pred = [img_warp_4]
        return imgt_pred, flowt0_pred, flowt1_pred, other_pred

    def forward(self, img_xs, coord=None, t=None, iters=None, ds_factor=None):
        """
        Full forward pass through all 5 modules.

        Args:
            img_xs:   (B, 3, 2, H, W)  — stacked input frames [I0, I1]
            coord:    list — kept for API compatibility (unused by Hermite)
            t:        list of (B,) tensors — target timesteps
            iters:    int — RAFT iterations
            ds_factor: float — downsampling for high-res inference

        Returns:
            dict with keys: imgt_pred, other_pred, flowt0_pred, flowt1_pred,
                           raft_flow, hermite_flow_t0, hermite_flow_t1
        """
        assert isinstance(t, list)
        full_size_img = None
        if ds_factor is not None:
            full_size_img = img_xs.clone()
            img_xs = torch.cat(
                [
                    resize(img_xs[:, :, 0], scale_factor=ds_factor).unsqueeze(2),
                    resize(img_xs[:, :, 1], scale_factor=ds_factor).unsqueeze(2),
                ],
                dim=2,
            )

        iters = self.raft_iter if iters is None else iters

        # ===== Module 1: The Head =====
        f01, f10, features0, features1, corr_fn = self.cal_bidirection_flow(
            255 * img_xs[:, :, 0], 255 * img_xs[:, :, 1], iters=iters
        )

        # ===== Module 2: The Brains =====
        # Use the lower-resolution projected features for coefficient prediction
        coefficients = self.coefficient_net(
            f01.detach(), f10.detach(),
            features0[0], features1[0]
        )

        imgt_preds, flowt0_preds, flowt1_preds, all_others = [], [], [], []

        for idx, cur_t_tensor in enumerate(t):
            cur_t = cur_t_tensor.reshape(-1, 1, 1, 1)

            # ===== Module 3: The Engine =====
            flow_t0, flow_t1 = self.hermite_engine(
                f01, f10, coefficients, cur_t_tensor
            )

            # ===== Module 4: Canvas (warping is embedded in frame_synthesize) =====
            # ===== Module 5: The Tail =====
            imgt_pred, flowt0_pred, flowt1_pred, others = self.frame_synthesize(
                img_xs, flow_t0, flow_t1,
                features0, features1, corr_fn,
                cur_t, full_img=full_size_img,
            )

            imgt_preds.append(imgt_pred)
            flowt0_preds.append(flowt0_pred)
            flowt1_preds.append(flowt1_pred)
            all_others.append(others)

        return {
            "imgt_pred": imgt_preds,
            "other_pred": all_others,
            "flowt0_pred": flowt0_preds,
            "flowt1_pred": flowt1_preds,
            "raft_flow": torch.cat(
                [f01.unsqueeze(2), f10.unsqueeze(2)], dim=2
            ),
        }

    def compute_psnr(self, preds, targets, reduction="mean"):
        assert reduction in ["mean", "sum", "none"]
        batch_size = preds.shape[0]
        sample_mses = torch.reshape(
            (preds - targets) ** 2, (batch_size, -1)
        ).mean(dim=-1)

        if reduction == "mean":
            psnr = (-10 * torch.log10(sample_mses)).mean()
        elif reduction == "sum":
            psnr = (-10 * torch.log10(sample_mses)).sum()
        else:
            psnr = -10 * torch.log10(sample_mses)
        return psnr

    def sample_coord_input(
        self, batch_size, s_shape, t_ids,
        coord_range=None, upsample_ratio=1.0, device=None,
    ):
        """
        Kept for API compatibility with GIMM-VFI evaluation scripts.
        HermiteFlow doesn't use INR coordinates, but eval scripts call this.
        Returns a dummy coordinate tensor.
        """
        assert device is not None
        # Return a simple spatial coordinate grid (unused by Hermite engine)
        H, W = s_shape
        H = int(H * upsample_ratio)
        W = int(W * upsample_ratio)
        coords = torch.zeros(batch_size, 1, H, W, 3, device=device)
        return coords

    def _amt_corr_scale_lookup(self, corr_fn, coord, flow0, flow1, embt, downsample=1):
        """AMT correlation lookup — identical to GIMM-VFI-R."""
        t0_scale = 1.0 / embt
        t1_scale = 1.0 / (1.0 - embt)
        if downsample != 1:
            inv = 1 / downsample
            flow0 = inv * resize(flow0, scale_factor=inv)
            flow1 = inv * resize(flow1, scale_factor=inv)
        corr0, corr1 = corr_fn(
            coord + flow1 * t1_scale, coord + flow0 * t0_scale
        )
        corr = torch.cat([corr0, corr1], dim=1)
        flow = torch.cat([flow0, flow1], dim=1)
        return corr, flow
