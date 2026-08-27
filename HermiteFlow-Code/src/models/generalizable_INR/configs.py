# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

from typing import List, Optional
from dataclasses import dataclass, field

from omegaconf import OmegaConf, MISSING
from .modules.module_config import HypoNetConfig


@dataclass
class GIMMConfig:
    type: str = "gimm"
    ema: Optional[bool] = None
    ema_value: Optional[float] = None
    fwarp_type: str = "linear"
    # See module_config.py's HypoNetConfig for why this must be
    # default_factory, not a bare mutable default - upstream's own
    # `= HypoNetConfig()` breaks on Python 3.11+ (Kaggle's 3.12).
    hyponet: HypoNetConfig = field(default_factory=HypoNetConfig)
    coord_range: List[float] = MISSING
    modulated_layer_idxs: Optional[List[int]] = None

    @classmethod
    def create(cls, config):
        # We need to specify the type of the default DataEncoderConfig.
        # Otherwise, data_encoder will be initialized & structured as "unfold" type (which is default value)
        # hence merging with the config with other type would cause config error.
        defaults = OmegaConf.structured(cls(ema=False))
        config = OmegaConf.merge(defaults, config)
        return config


@dataclass
class GIMMVFIConfig:
    type: str = "gimmvfi"
    ema: Optional[bool] = None
    ema_value: Optional[float] = None
    fwarp_type: str = "linear"
    rec_weight: float = 0.1
    hyponet: HypoNetConfig = field(default_factory=HypoNetConfig)
    raft_iter: int = 20
    coord_range: List[float] = MISSING
    modulated_layer_idxs: Optional[List[int]] = None
    # Not in upstream GIMM-VFI (its flow estimator ctors take no
    # arguments and hardcode "pretrained_ckpt/raft-things.pth" /
    # a matching FlowFormer path). Added so this baseline uses the SAME
    # --raft-ckpt / --flowformer-ckpt CLI flags and pretrained/ layout
    # HermiteFlow's own runs use, instead of a second hardcoded path
    # convention that would silently 404 in this repo's directory layout.
    pretrained_raft_ckpt: str = "pretrained/raft-things.pth"
    pretrained_flowformer_ckpt: str = "pretrained/flowformer_sintel.pth"

    @classmethod
    def create(cls, config):
        # We need to specify the type of the default DataEncoderConfig.
        # Otherwise, data_encoder will be initialized & structured as "unfold" type (which is default value)
        # hence merging with the config with other type would cause config error.
        defaults = OmegaConf.structured(cls(ema=False))
        config = OmegaConf.merge(defaults, config)
        return config
