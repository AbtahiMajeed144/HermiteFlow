# --------------------------------------------------------
# HermiteFlow — FlowFormer backbone
#
# Identical to HermiteFlow_R apart from the frozen flow
# estimator used in Phase 1. FlowFormer returns its flow as a
# list; the first entry is the final prediction.
#
# References:
#   flowformer: https://github.com/drinkingcoder/FlowFormer-Official
#   gimm-vfi:   https://github.com/GSeanCDAT/GIMM-VFI
# --------------------------------------------------------

from .configs import HermiteFlowConfig
from .flowformer import initialize_Flowformer
from .hermiteflow_base import HermiteFlowBase


class HermiteFlow_F(HermiteFlowBase):
    Config = HermiteFlowConfig

    def _build_flow_estimator(self, config):
        return initialize_Flowformer(config.pretrained_flowformer_ckpt)

    def flow_once(self, img_a, img_b, iters=None):
        """
        f_{a->b}. FlowFormer expects images in [0, 255] and rescales
        internally; `iters` is ignored - it is not iterative in this sense.
        It returns its flow as a list, whose first entry is the final
        prediction.

        Args:
            img_a, img_b: (B, 3, H, W) in [0, 1]
        Returns:
            (B, 2, H, W) in pixels
        """
        flow, _feats, _fnet = self.flow_estimator(
            255.0 * img_a, 255.0 * img_b, return_feat=True, iters=None
        )
        return flow[0]
