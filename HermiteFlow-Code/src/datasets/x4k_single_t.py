# --------------------------------------------------------
# HermiteFlow-VFI — X4K adapters for GIMM-VFI's own (vendored,
# unmodified) trainers
#
# trainer_gimmvfi.py / trainer_gimm.py expect the batch shapes GIMM-VFI's
# own vimeo_arb.py / fast_vimeo_flow.py produce (both vendored verbatim
# in this repo - see those two files). Both classes here wrap
# X4KMultiT so the GIMM baseline draws frames through the EXACT SAME
# clip discovery / frame-gap / downsample / crop / augmentation code
# path HermiteFlow's own X4K training uses - the data pipeline is
# identical by construction, not by parallel maintenance.
#
#   X4KSingleT       stage 2 (gimmvfi_r), vimeo_arb's shape:
#                     one (I0, I1, GT) triple + one scalar t per item,
#                     drawn on the fly - no caching needed, X4KMultiT's
#                     own aug=True path already gives full augmentation.
#
#   X4KGimmFlowCache  stage 1 (gimm), fast_vimeo_flow's shape: reads
#                     what scripts/generate_gimm_flow_cache.py wrote.
# --------------------------------------------------------

import glob
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .x4k_cached import X4KCachedGT
from .x4k_multi_t import X4KMultiT


class X4KSingleT(Dataset):
    """
    vimeo_arb.py's contract: {"xs": (3, 3, H, W), "t": (1,)} - one
    (I0, I1, GT) triple and one scalar t per item.

    GIMM's arbitrary-time model has no reason to need multiple
    simultaneous t the way HermiteFlow's residual heads do (see
    Learned_Hermite_VFI_v2.md's identifiability argument), so training
    it on one random t per step is its own native recipe, not a
    handicap imposed by this adapter. This only collapses X4KMultiT's K
    supervised middles down to one randomly-chosen one per item; every
    frame/crop/augmentation decision is X4KMultiT's own, unchanged.
    """

    def __init__(self, split, path, **kwargs):
        self._inner = X4KMultiT(split, path, **kwargs)

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, index):
        item = self._inner[index]
        xs, t = item["xs"], item["t"]  # (3, 2+K, H, W), (K,)
        k = random.randrange(t.shape[0])
        return {
            "xs": xs[:, [0, 1, 2 + k]],  # (3, 3, H, W)
            "t": t[k : k + 1],  # (1,)
        }


class X4KGimmFlowCache(Dataset):
    """
    fast_vimeo_flow.py's contract for GIMM stage 1 (trainer_gimm.py):
    {"xs", "flow_scaler", "ori_flows"}. Reads what
    scripts/generate_gimm_flow_cache.py wrote - three flow observations
    at X4K's true grid centre (k = num_divisions // 2, t = 0.5), the
    same three quantities upstream's precomputed .flo files hold:

        F  = RAFT(I0, I1)                     upstream's im1_im3
        M  = RAFT(Igt, I1) - RAFT(Igt, I0)     upstream's im2_im3-im2_im1
        F' = RAFT(I1, I0)                      upstream's -(-im3_im1)

    trainer_gimm.py hardcodes its three supervised positions at
    t in {0, 0.5, 1} - it has no continuous-t path - so unlike
    X4KSingleT this must always be the true grid centre, not a random
    k: feeding it a frame whose real t isn't 0.5 while the trainer still
    labels it "t=0.5" would silently mislabel the supervision.

    Only the three flow fields are cached - trainer_gimm.py's train()/
    eval()/reconstruct() never touch raw RGB (stage 1 is motion-only,
    reconstruct() visualises the flow itself via flow_to_image), so
    there is nothing to store or read back for I0/I1/GT.

    Augmentation: horizontal/vertical flip and the three 90-degree
    rotations, replayed on all three flow fields via X4KCachedGT's own
    exact-transform static methods (the ones
    others/verify_hermiteflow.py's test_flow_augmentation checks).
    Time reversal is NOT replayed here (unlike X4KCachedGT) - swapping
    which endpoint is "I0" would need M re-derived under the swap, and
    it is cheap to skip: the cache already covers the full clip list,
    only the dihedral augmentations are needed for read-time diversity.
    """

    def __init__(self, path, aug=True, repeat=1, expect=None):
        self.if_aug = bool(aug)
        self.repeat = max(1, int(repeat))

        path = os.path.expanduser(path)
        self.root = path

        # Recursive glob, not a flat listdir: the generator writes one
        # split_N/ subdirectory per --split-idx, and several splits may
        # be merged under one path - same convention X4KCachedGT uses.
        self.samples = sorted(
            p[: -len("_flow.npz")]
            for p in glob.glob(os.path.join(path, "**", "sample_*_flow.npz"), recursive=True)
        )
        if not self.samples:
            raise FileNotFoundError(
                f"no split_*/sample_*_flow.npz under {path} - run "
                f"scripts/generate_gimm_flow_cache.py first"
            )

        manifests = [
            json.load(open(p))
            for p in sorted(glob.glob(os.path.join(path, "**", "manifest.json"), recursive=True))
        ]
        if manifests:
            for key, want in (expect or {}).items():
                values = {m.get(key) for m in manifests}
                if want is not None and values - {None} - {want}:
                    raise ValueError(
                        f"cache at {path} was generated with {key}={sorted(values)}, "
                        f"but this config asks for {key}={want}. "
                        f"Regenerate the cache or fix the config."
                    )

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, index):
        prefix = self.samples[index % len(self.samples)]  # already a full path
        npz = np.load(prefix + "_flow.npz")
        flow_f = npz["flow_f"].astype(np.float32)  # F  = RAFT(I0, I1)
        flow_m = npz["flow_m"].astype(np.float32)  # M  = RAFT(Igt,I1)-RAFT(Igt,I0)
        flow_b = npz["flow_b"].astype(np.float32)  # F' = RAFT(I1, I0)

        if self.if_aug:
            flows = [flow_f, flow_m, flow_b]
            if random.uniform(0, 1) < 0.3:
                _, flows = X4KCachedGT._flip_y([], flows)
            if random.uniform(0, 1) < 0.5:
                _, flows = X4KCachedGT._flip_x([], flows)
            roll = random.uniform(0, 1)
            if roll < 0.05:
                _, flows = X4KCachedGT._rot([], flows, 3)
            elif roll < 0.10:
                _, flows = X4KCachedGT._rot([], flows, 2)
            elif roll < 0.15:
                _, flows = X4KCachedGT._rot([], flows, 1)
            flow_f, flow_m, flow_b = flows

        # Same normalisation upstream's fast_vimeo_flow.py uses: the
        # scaler comes from the two ENDPOINT flows only (F, F'), not M,
        # then all three are divided by it.
        ori_f = torch.from_numpy(np.ascontiguousarray(flow_f)).float()
        ori_b = torch.from_numpy(np.ascontiguousarray(flow_b)).float()
        raw_m = torch.from_numpy(np.ascontiguousarray(flow_m)).float()

        flow_scaler = torch.max(torch.abs(torch.cat((ori_f, ori_b), dim=0)))
        flow_scaler = flow_scaler.clamp_min(1.0)

        def norm(x):
            return (x / flow_scaler + 1.0) / 2.0

        xs = torch.stack([norm(ori_f), norm(raw_m), norm(ori_b)], dim=1)  # (2,3,H,W)
        ori_flows = torch.stack([ori_f, ori_b], dim=1)  # (2,2,H,W)

        return {
            "xs": xs,
            "flow_scaler": flow_scaler,
            "ori_flows": ori_flows,
        }
