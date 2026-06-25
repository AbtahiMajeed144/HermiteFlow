# --------------------------------------------------------
# HermiteFlow-VFI
# Module 2 — "The Brains": Latent Coefficient Predictor
#
# A shallow CNN that takes stacked optical flows and deep
# features from RAFT/FlowFormer, and predicts dense per-pixel
# Hermite polynomial coefficients α, β, γ, δ.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fi_components import LateralBlock


class CoefficientNet(nn.Module):
    """
    Shallow CNN that predicts per-pixel Hermite spline coefficients.

    Input:
        - flow_01: (B, 2, H, W)   optical flow frame 0 → 1
        - flow_10: (B, 2, H, W)   optical flow frame 1 → 0
        - feat_0:  (B, C, H/4, W/4)  deep features from flow estimator
        - feat_1:  (B, C, H/4, W/4)  deep features from flow estimator

    Output:
        - coefficients: (B, 4, H, W)   dense α, β, γ, δ maps
    """

    def __init__(self, flow_channels=4, feat_channels=256, num_coefficients=4,
                 mid_channels=64):
        super().__init__()
        self.num_coefficients = num_coefficients

        # Feature branch: project high-dim features down and upsample
        self.feat_proj = nn.Sequential(
            nn.Conv2d(feat_channels * 2, mid_channels, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Flow branch: lightweight convolution on stacked flows
        self.flow_proj = nn.Sequential(
            nn.Conv2d(flow_channels, mid_channels // 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels // 2, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Fusion + refinement with residual blocks
        fused_channels = mid_channels * 2  # flow branch + feat branch
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            LateralBlock(mid_channels),
            LateralBlock(mid_channels),
            LateralBlock(mid_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, num_coefficients, 3, 1, 1,
                      padding_mode="reflect", bias=True),
        )

        # Initialize output layer bias so that default coefficients
        # produce reasonable linear interpolation (α≈1, β≈1, γ≈1, δ≈1)
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.ones_(self.fusion[-1].bias)

    def forward(self, flow_01, flow_10, feat_0, feat_1):
        B, _, H, W = flow_01.shape

        # Stack flows: (B, 4, H, W)
        stacked_flows = torch.cat([flow_01, flow_10], dim=1)
        flow_feat = self.flow_proj(stacked_flows)  # (B, mid, H, W)

        # Stack and project deep features, upsample to full resolution
        stacked_feats = torch.cat([feat_0, feat_1], dim=1)  # (B, 2C, H/4, W/4)
        feat_proj = self.feat_proj(stacked_feats)  # (B, mid, H/4, W/4)
        feat_proj = F.interpolate(feat_proj, size=(H, W),
                                  mode="bilinear", align_corners=False)

        # Fuse and predict coefficients
        fused = torch.cat([flow_feat, feat_proj], dim=1)  # (B, 2*mid, H, W)
        coefficients = self.fusion(fused)  # (B, 4, H, W)

        return coefficients
