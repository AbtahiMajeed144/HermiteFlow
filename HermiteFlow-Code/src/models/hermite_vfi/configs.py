# --------------------------------------------------------
# HermiteFlow-VFI
# Config dataclasses for HermiteFlow model variants.
# --------------------------------------------------------

from typing import List, Optional
from dataclasses import dataclass

from omegaconf import OmegaConf, MISSING


@dataclass
class HermiteFlowConfig:
    type: str = "hermiteflow_r"
    ema: Optional[bool] = None
    ema_value: Optional[float] = None
    raft_iter: int = 20
    num_coefficients: int = 4       # α, β, γ, δ  Hermite basis modulation
    coeff_net_channels: int = 64    # Width of coefficient predictor CNN
    coord_range: List[float] = MISSING
    pretrained_decoder_ckpt: Optional[str] = None

    @classmethod
    def create(cls, config):
        defaults = OmegaConf.structured(cls(ema=False))
        config = OmegaConf.merge(defaults, config)
        return config
