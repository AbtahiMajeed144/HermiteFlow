# --------------------------------------------------------
# HermiteFlow — RAFT backbone
#
# Phase 1's flow estimator. Frozen: it is the only real
# measurement in the pipeline, and nothing downstream is allowed
# to bend it.
#
# References:
#   raft:     https://github.com/princeton-vl/RAFT
#   gimm-vfi: https://github.com/GSeanCDAT/GIMM-VFI
# --------------------------------------------------------

from .configs import HermiteFlowConfig
from .hermiteflow_base import HermiteFlowBase
from .raft import initialize_RAFT


class HermiteFlow_R(HermiteFlowBase):
    Config = HermiteFlowConfig

    def _build_flow_estimator(self, config):
        return initialize_RAFT(config.pretrained_raft_ckpt)

    def flow_once(self, img_a, img_b, iters=None):
        """
        f_{a->b}. RAFT expects images in [0, 255]; it rescales to [-1, 1]
        internally.

        Args:
            img_a, img_b: (B, 3, H, W) in [0, 1]
        Returns:
            (B, 2, H, W) in pixels
        """
        iters = self.raft_iter if iters is None else iters
        flow, _feats, _fmap = self.flow_estimator(
            255.0 * img_a, 255.0 * img_b, return_feat=True, iters=iters
        )
        return flow

    def flow_with_context(self, img_a, img_b, iters=None):
        """
        f_{a->b} plus the RAFT internals Phase 2 (AppNet + CoeffHead) reads
        directly: context features, final GRU hidden state, and the raw
        convex-upsample mask logits. See RAFT.forward(return_context=True).

        Args:
            img_a, img_b: (B, 3, H, W) in [0, 1]
        Returns:
            flow:    (B, 2, H, W) in pixels
            context: (B, 128, H/8, W/8)  c_i
            hidden:  (B, 128, H/8, W/8)  h_i^(N)
            mask:    (B, 576, H/8, W/8)  W_i, pre-softmax
        """
        iters = self.raft_iter if iters is None else iters
        return self.flow_estimator(
            255.0 * img_a, 255.0 * img_b, return_context=True, iters=iters
        )
