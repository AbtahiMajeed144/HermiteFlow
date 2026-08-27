# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import os
import torch

from .flow_dataset import fast_vimeo_flow, vimeo_rgb_with_flow
from .septuplet_multi_t import VimeoSeptupletMultiT
from .vimeo_arb import Vimeo_Arbitrary
from .x4k_cached import X4KCachedGT
from .x4k_multi_t import X4KMultiT
from .x4k_single_t import X4KGimmFlowCache, X4KSingleT

from utils.env import env_flag

SMOKE_TEST = env_flag("SMOKE_TEST")


def create_dataset(config, is_eval=False, logger=None):
    if config.dataset.type == "x4k_cached":
        # Stage 1 with the frozen teacher precomputed. Training reads the
        # cache; validation stays on the online loader, because it needs
        # real middle frames for image metrics and is small enough that
        # the teacher cost there does not matter.
        common = dict(
            num_timesteps=config.dataset.get("num_timesteps", 5),
            crop_size=config.dataset.get("crop_size", 256),
            frame_gap=config.dataset.get("frame_gap", 32),
            num_divisions=config.dataset.get("num_divisions", 8),
            clip_length=config.dataset.get("clip_length", 65),
            source=config.dataset.get("source", "auto"),
            downsample=config.dataset.get("downsample", 1.0),
        )
        dataset_trn = X4KCachedGT(
            config.dataset.cache_path,
            num_timesteps=config.dataset.get("num_timesteps", 5),
            aug=config.dataset.aug,
            repeat=config.dataset.get("repeat", 1),
            expect={
                "num_divisions": config.dataset.get("num_divisions", 8),
                "frame_gap": config.dataset.get("frame_gap", 32),
                "downsample": config.dataset.get("downsample", 1.0),
            },
        )
        val_path = config.dataset.get("val_path", None) or config.dataset.path
        dataset_val = X4KMultiT("test", val_path, aug=False, **common)
    elif config.dataset.type == "x4k_multi_t":
        # X4K1000FPS. `path` is the training root (mp4 or png); `val_path`
        # points at the already-decoded validation frames. Both are
        # searched recursively, so the doubled directories in the Kaggle
        # mirror (encoded_train/encoded_train/...) resolve either way.
        common = dict(
            num_timesteps=config.dataset.get("num_timesteps", 5),
            crop_size=config.dataset.get("crop_size", 256),
            frame_gap=config.dataset.get("frame_gap", 32),
            num_divisions=config.dataset.get("num_divisions", 8),
            clip_length=config.dataset.get("clip_length", 65),
            source=config.dataset.get("source", "auto"),
            downsample=config.dataset.get("downsample", 1.0),
        )
        repeat = config.dataset.get("repeat", 1)
        dataset_trn = X4KMultiT(
            "train", config.dataset.path, aug=config.dataset.aug,
            repeat=repeat, **common
        )
        val_path = config.dataset.get("val_path", None) or config.dataset.path
        dataset_val = X4KMultiT("test", val_path, aug=False, **common)
    elif config.dataset.type == "x4k_single_t":
        # GIMM stage 2 (gimmvfi_r/_f) baseline: one (I0, I1, GT) triple and
        # one scalar t per item, drawn from the SAME clip/frame-gap/
        # downsample/crop/augmentation pipeline HermiteFlow's own X4K
        # training uses - see x4k_single_t.py.
        common = dict(
            num_timesteps=config.dataset.get("num_timesteps", 5),
            crop_size=config.dataset.get("crop_size", 256),
            frame_gap=config.dataset.get("frame_gap", 32),
            num_divisions=config.dataset.get("num_divisions", 8),
            clip_length=config.dataset.get("clip_length", 65),
            source=config.dataset.get("source", "auto"),
            downsample=config.dataset.get("downsample", 1.0),
        )
        repeat = config.dataset.get("repeat", 1)
        dataset_trn = X4KSingleT(
            "train", config.dataset.path, aug=config.dataset.aug,
            repeat=repeat, **common
        )
        val_path = config.dataset.get("val_path", None) or config.dataset.path
        dataset_val = X4KSingleT("test", val_path, aug=False, **common)
    elif config.dataset.type == "x4k_gimm_flow_cache":
        # GIMM stage 1 (gimm) baseline: reads what
        # scripts/generate_gimm_flow_cache.py wrote. trainer_gimm.py's
        # eval() needs the same {"xs","flow_scaler","ori_flows"} shape as
        # train(), so both splits are cached - point val_cache_path at a
        # separate (smaller) cache generated from the val data.
        expect = {
            "num_divisions": config.dataset.get("num_divisions", 8),
            "frame_gap": config.dataset.get("frame_gap", 32),
            "downsample": config.dataset.get("downsample", 1.0),
        }
        dataset_trn = X4KGimmFlowCache(
            config.dataset.cache_path,
            aug=config.dataset.aug,
            repeat=config.dataset.get("repeat", 1),
            expect=expect,
        )
        val_cache_path = config.dataset.get("val_cache_path", None) or config.dataset.cache_path
        dataset_val = X4KGimmFlowCache(val_cache_path, aug=False, expect=expect)
    elif config.dataset.type == "vimeo_septuplet_multi_t":
        # HermiteFlow's default: K ground-truth middle frames per clip, so
        # the curve fitted once in Phase 2 is supervised at several t.
        num_timesteps = config.dataset.get("num_timesteps", 3)
        crop_size = config.dataset.get("crop_size", 256)
        span_mode = config.dataset.get("span_mode", "full")
        dataset_trn = VimeoSeptupletMultiT(
            "train",
            config.dataset.path,
            num_timesteps=num_timesteps,
            aug=config.dataset.aug,
            crop_size=crop_size,
            span_mode=span_mode,
        )
        dataset_val = VimeoSeptupletMultiT(
            "test",
            config.dataset.path,
            num_timesteps=num_timesteps,
            aug=False,
            crop_size=crop_size,
            span_mode=span_mode,
        )
    elif config.dataset.type == "fast_vimeo_flow":
        dataset_trn = fast_vimeo_flow(
            "train", config.dataset.path, config.dataset.expansion, config.dataset.aug
        )
        dataset_val = fast_vimeo_flow(
            "test", config.dataset.path, config.dataset.expansion, config.dataset.aug
        )
    elif config.dataset.type == "vimeo_arb":
        dataset_trn = Vimeo_Arbitrary("train", config.dataset.path, config.dataset.aug)
        dataset_val = Vimeo_Arbitrary("test", config.dataset.path, config.dataset.aug)
    elif config.dataset.type == "vimeo_rgb_with_flow":
        dataset_trn = vimeo_rgb_with_flow(
            "train", config.dataset.path, config.dataset.expansion, config.dataset.aug
        )
        dataset_val = vimeo_rgb_with_flow(
            "test", config.dataset.path, config.dataset.expansion, config.dataset.aug
        )
    else:
        raise ValueError("%s not supported..." % config.dataset.type)

    if SMOKE_TEST:
        dataset_len = config.experiment.total_batch_size * 2
        dataset_trn = torch.utils.data.Subset(
            dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len]
        )
        dataset_val = torch.utils.data.Subset(
            dataset_val, torch.randperm(len(dataset_val))[:dataset_len]
        )

    if logger is not None:
        logger.info(
            f"#train samples: {len(dataset_trn)}, #valid samples: {len(dataset_val)}"
        )

    return dataset_trn, dataset_val
