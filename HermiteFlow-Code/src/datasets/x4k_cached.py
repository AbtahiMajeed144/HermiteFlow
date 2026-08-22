# --------------------------------------------------------
# HermiteFlow — cached-teacher X4K loader
#
# Reads what scripts/generate_offline_gt.py wrote: per clip, the two
# endpoint frames and the frozen teacher's f_{0->t} / f_{1->t} on the
# whole k/8 grid. Stage 1's only objective is trajectory distillation
# against those targets, so with them on disk the 2*K RAFT passes per
# step - about 900 ms of a ~1.19 s step - disappear entirely.
#
# The cache stores ONE unaugmented entry per clip. Everything except the
# random temporal window is reapplied here, on the images and the flows
# together, so the model still sees fresh augmentation every epoch:
#
#   channel reverse   images only; RGB->BGR moves nothing
#   horizontal flip   reverse columns, negate u
#   vertical flip     reverse rows,    negate v
#   rot90 cw          rot90(k=-1),     (u, v) -> (-v,  u)
#   rot180            rot90(k= 2),     (u, v) -> (-u, -v)
#   rot90 ccw         rot90(k= 1),     (u, v) -> ( v, -u)
#   time reversal     swap endpoints, reverse k, t -> 1-t, and swap
#                     flow_0_t <-> flow_1_t (both directions are cached,
#                     so this costs nothing)
#   crop              a no-op at downsample 3.0 (768/3 = 256 = crop_size)
#
# `repeat` is applied HERE rather than at generation time. The cache has
# one entry per clip, so without it an epoch would be 138 optimizer
# steps instead of ~1.1k; each redraw of the same clip gets a different
# augmentation and a different K-subset of the cached grid.
# --------------------------------------------------------

import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class X4KCachedGT(Dataset):
    """
    Args:
        path: directory holding split_*/ from generate_offline_gt.py
        num_timesteps: K to supervise, chosen from the cached grid.
        aug: apply the dihedral + time-reversal augmentation above.
        repeat: draws per clip per epoch (see the module docstring).
    """

    def __init__(self, path, num_timesteps=5, aug=True, repeat=1,
                 expect=None):
        self.num_timesteps = int(num_timesteps)
        self.if_aug = bool(aug)
        self.repeat = max(1, int(repeat))

        path = os.path.expanduser(path)
        self.samples = sorted(
            glob.glob(os.path.join(path, "**", "sample_*_flow.npz"), recursive=True)
        )
        if not self.samples:
            raise FileNotFoundError(
                f"no cached samples under {path}. Expected "
                f"split_*/sample_*_flow.npz from scripts/generate_offline_gt.py."
            )

        # Globbing rather than assuming a contiguous range: the generator
        # drops any sample whose teacher flow came back non-finite, and
        # several splits may be merged into one directory.
        self.manifests = [
            json.load(open(p)) for p in
            sorted(glob.glob(os.path.join(path, "**", "manifest.json"), recursive=True))
        ]
        self._check(expect)

    def _check(self, expect):
        """Fail loudly if the cache was built for a different protocol."""
        if not self.manifests:
            return
        for key in ("num_divisions", "frame_gap", "downsample"):
            values = {m.get(key) for m in self.manifests}
            if len(values) > 1:
                raise ValueError(
                    f"cached splits disagree on {key}: {sorted(values)}. They were "
                    f"not generated from the same config; regenerate or separate them."
                )
        manifest = self.manifests[0]
        cached_k = manifest.get("num_timesteps")
        if cached_k is not None and self.num_timesteps > cached_k:
            raise ValueError(
                f"dataset.num_timesteps={self.num_timesteps} but the cache holds "
                f"only {cached_k} timesteps per sample. Lower num_timesteps or "
                f"regenerate with a larger num_divisions."
            )
        for key, want in (expect or {}).items():
            got = manifest.get(key)
            if got is not None and want is not None and got != want:
                raise ValueError(
                    f"cache was generated with {key}={got} but the config says "
                    f"{want}. Training against a cache built for a different "
                    f"protocol would be silently wrong; regenerate it."
                )

    def __len__(self):
        return len(self.samples) * self.repeat

    # ------------------------------------------------------------------
    # Timestep selection - mirrors X4KMultiT._grid_steps, anchored at
    # both extremes because recovering (A, B) from interior-only samples
    # is far worse conditioned (26.8x vs 26.5x at K=5, 173x at K=2).
    # ------------------------------------------------------------------

    def _pick(self, cached):
        if self.num_timesteps >= cached:
            return list(range(cached))
        if self.num_timesteps == 1:
            return [random.randrange(cached)]
        if self.num_timesteps == 2:
            return [0, cached - 1]
        interior = list(range(1, cached - 1))
        chosen = random.sample(interior, self.num_timesteps - 2)
        return sorted([0, cached - 1] + chosen)

    def _eval_pick(self, cached):
        if self.num_timesteps >= cached:
            return list(range(cached))
        picks = np.linspace(0, cached - 1, self.num_timesteps)
        return sorted({int(round(v)) for v in picks})

    # ------------------------------------------------------------------
    # Augmentation, applied identically to images and flows
    # ------------------------------------------------------------------

    @staticmethod
    def _flip_x(images, flows):
        """Reverse columns. A rightward displacement becomes leftward."""
        images = [im[:, ::-1] for im in images]
        flows = [f[:, :, ::-1].copy() for f in flows]
        for f in flows:
            f[0] = -f[0]
        return images, flows

    @staticmethod
    def _flip_y(images, flows):
        """Reverse rows. A downward displacement becomes upward."""
        images = [im[::-1] for im in images]
        flows = [f[:, ::-1, :].copy() for f in flows]
        for f in flows:
            f[1] = -f[1]
        return images, flows

    @staticmethod
    def _rot(images, flows, quarters):
        """
        Rotate by `quarters` * 90 degrees counter-clockwise on screen.

        np.rot90 with k>0 is CCW in array terms. The vector has to turn
        with the scene: with y pointing DOWN, one CCW quarter-turn sends
        a rightward motion to an upward one, i.e. (u, v) -> (v, -u).
        """
        quarters %= 4
        if quarters == 0:
            return images, flows
        images = [np.rot90(im, quarters, axes=(0, 1)) for im in images]
        out = []
        for f in flows:
            r = np.rot90(f, quarters, axes=(1, 2)).copy()
            u, v = r[0].copy(), r[1].copy()
            for _ in range(quarters):
                u, v = v.copy(), (-u).copy()
            r[0], r[1] = u, v
            out.append(r)
        return images, out

    def _augment(self, img0, img1, phi, psi, times):
        images = [img0, img1]
        # One list so a single transform touches every field at once and
        # they cannot drift out of step.
        flows = list(phi) + list(psi)

        if random.uniform(0, 1) < 0.5:  # RGB -> BGR; the flow is unaffected
            images = [im[:, :, ::-1] for im in images]
        if random.uniform(0, 1) < 0.3:
            images, flows = self._flip_y(images, flows)
        if random.uniform(0, 1) < 0.5:
            images, flows = self._flip_x(images, flows)

        roll = random.uniform(0, 1)
        if roll < 0.05:
            images, flows = self._rot(images, flows, 3)   # 90 cw
        elif roll < 0.10:
            images, flows = self._rot(images, flows, 2)
        elif roll < 0.15:
            images, flows = self._rot(images, flows, 1)   # 90 ccw

        k = len(phi)
        phi, psi = flows[:k], flows[k:]

        if random.uniform(0, 1) < 0.5:
            # Endpoints swap, so f_{0->t} and f_{1->t} swap roles, the
            # grid runs backwards, and t -> 1-t keeps it ascending.
            images = [images[1], images[0]]
            phi, psi = psi[::-1], phi[::-1]
            times = [1.0 - v for v in reversed(times)]

        return images[0], images[1], phi, psi, times

    # ------------------------------------------------------------------

    def __getitem__(self, index):
        path = self.samples[index % len(self.samples)]
        prefix = path[: -len("_flow.npz")]

        with np.load(path) as data:
            phi_all = data["flow_0_t"].astype(np.float32)  # (Kc, 2, H, W)
            psi_all = data["flow_1_t"].astype(np.float32)
            t_all = data["t"].astype(np.float32)

        keep = self._pick(len(t_all)) if self.if_aug else self._eval_pick(len(t_all))
        phi = [phi_all[i] for i in keep]
        psi = [psi_all[i] for i in keep]
        times = [float(t_all[i]) for i in keep]

        # cv2 reads BGR; the cache was written from RGB, so reverse back.
        img0 = cv2.imread(prefix + "_img0.png")[:, :, ::-1]
        img1 = cv2.imread(prefix + "_img1.png")[:, :, ::-1]

        if self.if_aug:
            img0, img1, phi, psi, times = self._augment(img0, img1, phi, psi, times)

        tensors = [
            torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0
            for im in (img0, img1)
        ]
        return {
            "xs": torch.stack(tensors, dim=1),  # (3, 2, H, W) - no gt middles
            "t": torch.tensor(times, dtype=torch.float32),
            "teacher_phi": torch.from_numpy(
                np.ascontiguousarray(np.stack(phi))
            ).float(),  # (K, 2, H, W)
            "teacher_psi": torch.from_numpy(
                np.ascontiguousarray(np.stack(psi))
            ).float(),
        }
