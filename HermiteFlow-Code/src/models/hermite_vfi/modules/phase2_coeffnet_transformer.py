# --------------------------------------------------------
# HermiteFlow - Phase 2, pure-transformer variant
#
# CoeffNet (phase2_coeffnet.py) is a CNN U-Net with ONE attention block
# dropped into its bottleneck (GlobalContext). That was a deliberate,
# minimal hybrid - see GlobalContext's docstring. TransformerCoeffNet is
# the thing actually asked for afterward: encoder AND decoder built from
# attention, not convolution. Same public interface as CoeffNet, so it
# is a drop-in swap in hermiteflow_base.py behind a config switch, not a
# parallel pipeline.
#
# THE ONE CONSTRAINT THAT SHAPES EVERY CHOICE HERE
#
# This model is evaluated at 2K and 4K (src/X4K.py ds_factor 0.5/0.25)
# while it trains at 256x256. Plain pixel-to-pixel self-attention is
# quadratic in token count - at a 4K bottleneck that is not slow, it
# does not run. GlobalContext solved this at ONE point (the bottleneck)
# by pooling keys/values to a fixed grid. A full trunk needs the same
# property at EVERY resolution, so every stage here uses WINDOWED
# attention (Swin-style): each location attends only within a window of
# FIXED PIXEL SIZE, so cost is linear in image area - same asymptotic
# class as a convolution - regardless of whether the input is 256px or
# 4096px. Position is a RELATIVE bias table indexed by in-window offset,
# sized only by window_size, not by H or W, for the same reason
# GlobalContext used sinusoidal-from-normalised-coordinates rather than
# a learned per-cell grid: nothing here is tied to the training
# resolution.
#
# Windows alone would make each window an isolated island - exactly the
# receptive-field ceiling a conv has, just with a bigger tile. Every
# other block shifts the window grid by half a window (torch.roll) so
# consecutive blocks connect across the PREVIOUS grid's boundaries. The
# roll wraps the far edge back to the near one, which would silently
# glue unrelated content together, so shifted blocks add an additive
# attention mask that kills any pair whose members did not neighbour
# each other before the roll (the standard Swin technique). Get this
# mask wrong and windows still look locally sane while quietly reading
# from the wrong side of the image - see
# test_transformer_coeffnet.py's mask test, which checks the actual
# gradient path, not the code that built the mask.
#
# Down/up-sampling between stages is a 2x2-stride 1x1-style projection
# (PatchMerging) and a pixel-shuffle projection (PatchExpanding) - both
# a linear map per non-overlapping patch, mathematically what Swin's own
# "patch merging" is (concat a 2x2 neighbourhood, project with a
# Linear), and what every ViT's own input stem already is
# (Conv2d(kernel=patch, stride=patch) IS a linear-per-patch map). No
# spatial convolution appears anywhere in the feature-REFINEMENT path -
# that work is 100% window attention and per-token MLPs.
#
# Zero-init discipline carries over unchanged: the residual heads are
# zero-initialised, so `head(anything) = 0` regardless of what this
# trunk computes - training starts at the exact linear baseline no
# matter what backbone predicts d0, d1. Each SwinBlock ALSO zero-inits
# its own attention and MLP output projections (ReZero/Fixup-style),
# which is not required for that guarantee but keeps early activations
# from being large and erratic feeding into heads that are about to
# start moving. The tanh residual_bound cap is carried over verbatim -
# it is a fact about the (d0, d1) -> trajectory map (see CoeffNet's
# docstring), not about what predicts d0 and d1, so it applies exactly
# as much here as it does to the CNN.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .phase2_coeffnet import FLOW_BRANCH_CHANNELS, RGB_BRANCH_CHANNELS, GlobalContext, OcclusionGate


def window_partition(x, window_size):
    """(B, C, H, W) -> (B*num_windows, window_size^2, C). H, W must already
    be multiples of window_size - callers pad first."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, c)


def window_reverse(windows, window_size, h, w, b):
    """Inverse of window_partition."""
    c = windows.shape[-1]
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, c)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)


def _shift_attn_mask(h, w, window_size, shift, device):
    """
    The standard Swin cyclic-shift mask: label each pixel of the padded
    (h, w) canvas by which of the 9 regions the shift would cut it into,
    partition those labels the same way the features are partitioned,
    and forbid attention between two positions that carry different
    labels. h, w are the PADDED size (already multiples of window_size).
    """
    img = torch.zeros(1, 1, h, w, device=device)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift), slice(-shift, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift), slice(-shift, None))
    label = 0
    for hs in h_slices:
        for ws in w_slices:
            img[:, :, hs, ws] = label
            label += 1
    windows = window_partition(img, window_size).squeeze(-1)  # (nW, N)
    diff = windows.unsqueeze(1) - windows.unsqueeze(2)  # (nW, N, N)
    return diff.masked_fill(diff != 0, -100.0).masked_fill(diff == 0, 0.0)


class WindowAttention(nn.Module):
    """Multi-head self-attention within one window, with a relative
    position bias sized by window_size alone - resolution-independent."""

    def __init__(self, dim, window_size, heads):
        super().__init__()
        assert dim % heads == 0
        self.window_size = window_size
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        table_size = (2 * window_size - 1) ** 2
        self.bias_table = nn.Parameter(torch.zeros(table_size, heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size), torch.arange(window_size), indexing="ij"
            )
        ).flatten(1)  # (2, N)
        rel = coords[:, :, None] - coords[:, None, :]  # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("bias_index", rel.sum(-1), persistent=False)  # (N, N)

    def forward(self, x, mask=None):
        # x: (B_, N, C), B_ = batch * num_windows
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B_, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.bias_table[self.bias_index.view(-1)].view(n, n, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            num_w = mask.shape[0]
            attn = attn.view(b_ // num_w, num_w, self.heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(b_, self.heads, n, n)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(out)


class SwinBlock(nn.Module):
    """One window-attention block, spatial (B, C, H, W) in and out - a
    drop-in for a conv block in terms of external shape contract."""

    def __init__(self, dim, window_size, heads, shift, mlp_ratio=4):
        super().__init__()
        self.window_size = window_size
        self.shift = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # Masks depend only on (padded H, padded W, device) - cheap to
        # memoise, and training only ever sees one shape.
        self._mask_cache = {}

    def _mask(self, h, w, device):
        if self.shift == 0:
            return None
        key = (h, w, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = _shift_attn_mask(
                h, w, self.window_size, self.shift, device
            )
        return self._mask_cache[key]

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        xp = F.pad(xn, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else xn
        hp, wp = h + pad_h, w + pad_w

        if self.shift:
            xp = torch.roll(xp, shifts=(-self.shift, -self.shift), dims=(2, 3))

        windows = window_partition(xp, self.window_size)
        attended = self.attn(windows, mask=self._mask(hp, wp, x.device))
        xp = window_reverse(attended, self.window_size, hp, wp, b)

        if self.shift:
            xp = torch.roll(xp, shifts=(self.shift, self.shift), dims=(2, 3))
        xp = xp[:, :, :h, :w]

        x = shortcut + xp
        xn2 = self.norm2(x.permute(0, 2, 3, 1))
        return x + self.mlp(xn2).permute(0, 3, 1, 2)


class PatchMerging(nn.Module):
    """H, W -> ceil(H/2), ceil(W/2); a linear projection per 2x2 patch -
    what Swin calls patch merging, parameterised as a strided conv."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.reduce = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(8, dim_out)

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2))
        return self.norm(self.reduce(x))


class PatchExpanding(nn.Module):
    """H, W -> 2H, 2W; a per-pixel linear projection to 4x the target
    channels, then a pixel shuffle (a pure, parameter-free reshape)."""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.expand = nn.Conv2d(dim_in, dim_out * 4, kernel_size=1)
        self.shuffle = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(8, dim_out)

    def forward(self, x):
        return self.norm(self.shuffle(self.expand(x)))


def _stage(dim, window_size, heads, depth, mlp_ratio):
    return nn.ModuleList(
        [
            SwinBlock(dim, window_size, heads, shift=(i % 2 == 1), mlp_ratio=mlp_ratio)
            for i in range(depth)
        ]
    )


class TransformerCoeffNet(nn.Module):
    """
    Same public contract as CoeffNet (phase2_coeffnet.py): the six-arg
    forward, set_rgb_branch, set_active_heads, num_active_heads, .gate,
    .residual_bound. See the module docstring for the design.
    """

    def __init__(
        self,
        channels=64,
        gate_init_bias=5.0,
        gate_init_scale=20.0,
        use_rgb_branch=True,
        num_residuals=2,
        residual_bound=2.0,
        window_size=8,
        depths=(2, 2, 2),
        decoder_depths=(2, 2),
        mlp_ratio=4,
        use_global_context=True,
        global_context_tokens=8,
        global_context_heads=4,
    ):
        super().__init__()
        self.use_rgb_branch = use_rgb_branch
        self.num_residuals = num_residuals
        self.residual_bound = float(residual_bound)
        self.window_size = window_size
        self.use_global_context = bool(use_global_context)

        c1, c2, c3 = channels, channels * 2, channels * 4
        heads1, heads2, heads3 = max(1, c1 // 32), max(1, c2 // 32), max(1, c3 // 32)

        # Stem: identical role and shape contract to CoeffNet's _Branch
        # pair - same input tensors, same fusion-by-addition so the RGB
        # ablation is a pure runtime switch here too. Reusing GroupNorm
        # (not LayerNorm) here specifically because this is the one spot
        # that still looks like a conv stem feeding a spatial tensor,
        # matching the surrounding codebase convention at that seam.
        self.flow_stem = nn.Sequential(
            nn.Conv2d(FLOW_BRANCH_CHANNELS, c1, 3, 1, 1), nn.GroupNorm(8, c1)
        )
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(RGB_BRANCH_CHANNELS, c1, 3, 1, 1), nn.GroupNorm(8, c1)
        )

        self.stage1 = _stage(c1, window_size, heads1, depths[0], mlp_ratio)
        self.down1 = PatchMerging(c1, c2)
        self.stage2 = _stage(c2, window_size, heads2, depths[1], mlp_ratio)
        self.down2 = PatchMerging(c2, c3)
        self.stage3 = _stage(c3, window_size, heads3, depths[2], mlp_ratio)

        # The same pooled global-attention block as the hybrid, at the
        # same insertion point. Windowed attention gives every stage a
        # bounded, resolution-agnostic local reach; this is what still
        # gives the whole-image reach the branch exists for.
        self.global_context = GlobalContext(
            c3, pooled_size=global_context_tokens, heads=global_context_heads
        )
        self.set_global_context(use_global_context)

        self.up1 = PatchExpanding(c3, c2)
        self.fuse1 = nn.Conv2d(c2 * 2, c2, 1)
        self.decode2 = _stage(c2, window_size, heads2, decoder_depths[0], mlp_ratio)

        self.up2 = PatchExpanding(c2, c1)
        self.fuse2 = nn.Conv2d(c1 * 2, c1, 1)
        self.decode1 = _stage(c1, window_size, heads1, decoder_depths[1], mlp_ratio)

        self.heads = nn.ModuleList(
            [
                nn.Conv2d(c1, 2, 3, 1, 1, padding_mode="reflect")
                for _ in range(num_residuals)
            ]
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.gate = OcclusionGate(gate_init_bias, gate_init_scale)
        self.num_active_heads = num_residuals
        self.set_rgb_branch(use_rgb_branch)

    # ---------------- runtime switches (mirrors CoeffNet) ----------------

    def set_rgb_branch(self, enabled):
        self.use_rgb_branch = bool(enabled)
        for param in self.rgb_stem.parameters():
            param.requires_grad = self.use_rgb_branch

    def set_global_context(self, enabled):
        self.use_global_context = bool(enabled)
        for param in self.global_context.parameters():
            param.requires_grad = self.use_global_context

    def set_active_heads(self, num_used):
        assert 1 <= num_used <= len(self.heads)
        self.num_active_heads = num_used
        for index, head in enumerate(self.heads):
            for param in head.parameters():
                param.requires_grad = index < num_used

    @staticmethod
    def _run_stage(x, blocks):
        for block in blocks:
            x = block(x)
        return x

    def forward(self, img_src, img_dst_aligned, flow, flow_bwd_aligned, occlusion, scale):
        occlusion_norm = occlusion / scale
        feat = self.flow_stem(
            torch.cat([flow / scale, flow_bwd_aligned / scale, occlusion_norm], dim=1)
        )
        if self.use_rgb_branch:
            feat = feat + self.rgb_stem(torch.cat([img_src, img_dst_aligned], dim=1))

        f1 = self._run_stage(feat, self.stage1)
        f2 = self._run_stage(self.down1(f1), self.stage2)
        f3 = self._run_stage(self.down2(f2), self.stage3)

        if self.use_global_context:
            f3 = self.global_context(f3)

        u1 = F.interpolate(self.up1(f3), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self._run_stage(self.fuse1(torch.cat([u1, f2], dim=1)), self.decode2)

        u2 = F.interpolate(self.up2(u1), size=f1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self._run_stage(self.fuse2(torch.cat([u2, f1], dim=1)), self.decode1)

        alpha = self.gate(occlusion_norm)
        bound = self.residual_bound
        return [
            alpha * bound * scale * torch.tanh(self.heads[i](u2) / bound)
            for i in range(self.num_active_heads)
        ]
