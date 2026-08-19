# --------------------------------------------------------
# HermiteFlow — Phase 5: "Build the frame"
#
#   In:  I0, I1, G0, G1        Out: I_hat_t
#
#   I_{t->0} = backwarp(I0, G0)
#   I_{t->1} = backwarp(I1, G1)
#
#   M, R = SynthNet(I_{t->0}, I_{t->1}, G0, G1)
#
#   I_hat_t = M * I_{t->0} + (1 - M) * I_{t->1} + R
#
# Two candidate frames - one made by pulling pixels from I0, one
# from I1. Both are complete images; they disagree at occlusions.
# M in [0, 1] picks which source to trust per pixel; R is a
# residual for whatever neither source could supply.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fi_components import LateralBlock
from .fi_utils import warp

# I_{t->0} (3) + I_{t->1} (3) + G0 (2) + G1 (2)
SYNTHNET_IN_CHANNELS = 10


class SynthNet(nn.Module):
    """
    UNet-style synthesis decoder producing the blending mask and the
    residual.

    The single output layer is zero-initialised, which puts the model
    at a well-defined starting point: M = sigmoid(0) = 0.5 and R = 0,
    i.e. the plain average of the two warped candidates. Everything the
    network learns is a departure from that.
    """

    def __init__(self, channels=64):
        super().__init__()
        c1, c2, c3 = channels, channels * 2, channels * 4

        self.stem = nn.Sequential(
            nn.Conv2d(SYNTHNET_IN_CHANNELS, c1, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            LateralBlock(c1),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            LateralBlock(c2),
            LateralBlock(c2),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            LateralBlock(c3),
            LateralBlock(c3),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(c3 + c2, c2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            LateralBlock(c2),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(c2 + c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            LateralBlock(c1),
        )
        # 4 channels out: mask logit (1) + residual (3)
        self.head = nn.Conv2d(c1, 4, 3, 1, 1, padding_mode="reflect")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, warped_0, warped_1, flow_t0, flow_t1, scale):
        """
        Args:
            warped_0, warped_1: (B, 3, H, W) I_{t->0}, I_{t->1} in [0, 1]
            flow_t0, flow_t1:   (B, 2, H, W) G0, G1 in pixels
            scale:              (B, 1, 1, 1) motion scale

        Returns:
            mask:     (B, 1, H, W) M in (0, 1)
            residual: (B, 3, H, W) R
        """
        x = torch.cat(
            [
                2.0 * warped_0 - 1.0,
                2.0 * warped_1 - 1.0,
                flow_t0 / scale,
                flow_t1 / scale,
            ],
            dim=1,
        )

        f1 = self.stem(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)

        u1 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, f2], dim=1))

        u2 = F.interpolate(u1, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2(torch.cat([u2, f1], dim=1))

        out = self.head(u2)
        return torch.sigmoid(out[:, 0:1]), out[:, 1:4]


class FrameSynthesis(nn.Module):
    """Phase 5 end to end: warp both sources, then blend."""

    def __init__(self, channels=64):
        super().__init__()
        self.synth_net = SynthNet(channels)

    def forward(self, img0, img1, flow_t0, flow_t1, scale):
        """
        Args:
            img0, img1:       (B, 3, H, W) in [0, 1]
            flow_t0, flow_t1: (B, 2, H, W) G0, G1 on the frame-t grid
            scale:            (B, 1, 1, 1) motion scale

        Returns:
            imgt_pred: (B, 3, H, W) I_hat_t, clamped to [0, 1]
            warped_0, warped_1, mask, residual (for logging / aux loss)
        """
        warped_0 = warp(img0, flow_t0)
        warped_1 = warp(img1, flow_t1)

        mask, residual = self.synth_net(warped_0, warped_1, flow_t0, flow_t1, scale)

        imgt_pred = mask * warped_0 + (1.0 - mask) * warped_1 + residual
        return torch.clamp(imgt_pred, 0.0, 1.0), warped_0, warped_1, mask, residual
