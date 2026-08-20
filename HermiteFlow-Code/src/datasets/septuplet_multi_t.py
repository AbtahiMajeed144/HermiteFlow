# --------------------------------------------------------
# HermiteFlow — multi-timestep Vimeo-90K septuplet loader
#
# Phase 2 fits ONE curve per clip and Phase 3 evaluates it at many
# t. That separation is only meaningful if training actually
# observes more than one t per fitted curve:
#
#     Phi(t) = t*F + (t^2 - t)*A + (t^3 - t)*B
#
# has two unknowns per pixel, so a single supervised t gives one
# equation in two unknowns and the curvature is unidentifiable -
# any (A, B) on a line through the solution fits equally well and
# only one of them is right at every other t.
#
# This loader therefore returns K ground-truth middle frames per
# sample, all lying on the same trajectory between the same pair of
# endpoints, together with their K time values.
#
# Item layout:
#   xs: (3, 2 + K, H, W)   [ I0, I1, gt_1, ..., gt_K ]
#   t:  (K,)               t_k in (0, 1)
#
# K = 1 reproduces the conventional single-timestep setup, so the
# same trainer handles this dataset and the plain triplet loaders.
# --------------------------------------------------------

import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# A Vimeo septuplet has 7 frames, so the widest possible endpoint span
# leaves 5 usable intermediate frames.
NUM_FRAMES = 7
MAX_TIMESTEPS = NUM_FRAMES - 2


class VimeoSeptupletMultiT(Dataset):
    """
    Args:
        split: "train" or "test"
        path:  root of vimeo_septuplet (holds sequences/ and sep_*list.txt)
        num_timesteps: K, how many middle frames to supervise per sample
        aug: enable training augmentation
        crop_size: square crop taken during training
    """

    def __init__(
        self,
        split,
        path,
        num_timesteps=3,
        aug=True,
        crop_size=256,
        span_mode="full",
    ):
        assert split in ("train", "test"), split
        assert span_mode in ("full", "random"), span_mode
        assert 1 <= num_timesteps <= MAX_TIMESTEPS, (
            f"num_timesteps must be in [1, {MAX_TIMESTEPS}] for {NUM_FRAMES}-frame "
            f"septuplets, got {num_timesteps}"
        )

        self.span_mode = span_mode
        self.split = split
        self.data_root = path
        self.image_root = os.path.join(path, "sequences")
        self.num_timesteps = num_timesteps
        self.if_aug = aug and split == "train"
        self.crop_size = crop_size

        list_name = "sep_trainlist.txt" if split == "train" else "sep_testlist.txt"
        list_path = os.path.join(path, list_name)
        if not os.path.isfile(list_path):
            raise FileNotFoundError(
                f"{list_path} not found. Point dataset.path (or --data-path) at the "
                f"vimeo_septuplet root containing sequences/ and {list_name}."
            )
        with open(list_path, "r") as handle:
            self.meta_data = [line for line in handle.read().splitlines() if line]

    def __len__(self):
        return len(self.meta_data)

    # ------------------------------------------------------------------
    # Timestep selection
    # ------------------------------------------------------------------

    def _spread_middles(self):
        """
        Choose K middle frames, always anchored at both extremes.

        Recovering (A, B) from samples of Phi(t) is an ill-conditioned
        inverse problem: beta3(t)/beta2(t) = t+1 exactly, so the two
        basis functions are near-collinear and their ratio only spans
        1.17 to 1.83 across the whole interval. How badly conditioned
        depends sharply on WHICH t are sampled. Measured amplification
        of a 1 px error in Phi into error in (A, B), over the 5 middle
        frames of a septuplet:

            K=3 {1,2,5}/6   24.8x     <- best
            K=3 {1,3,5}/6   27.7x
            K=3 {2,3,4}/6   34.5x
            K=3 {1,2,3}/6   39.6x
            K=3 {3,4,5}/6   43.8x     <- worst, and uniform sampling
                                         picks it 1 time in 10
            K=5 all         21.6x

        Sampling uniformly averages 31.2x with a worst case of 43.8x.
        Anchoring both extremes keeps every draw in 24.8-27.7x, at no
        cost. Use num_timesteps=5 to reach 21.6x.
        """
        first, last = 1, NUM_FRAMES - 2
        if self.num_timesteps == 1:
            return [random.randint(first, last)]
        if self.num_timesteps == 2:
            return [first, last]

        interior = list(range(first + 1, last))
        chosen = random.sample(interior, self.num_timesteps - 2)
        return sorted([first, last] + chosen)

    def _sample_indices(self):
        """
        Pick endpoint indices (i, j) and K intermediate indices strictly
        between them. Returns (i, j, [k_1 < ... < k_K]).
        """
        if self.split == "train" and self.span_mode == "random":
            # Any span wide enough to hold K distinct intermediate frames.
            # Wider t coverage and more varied motion magnitudes, at the
            # cost of no longer matching the paper's stated protocol.
            spans = [
                (i, j)
                for i in range(NUM_FRAMES)
                for j in range(i + self.num_timesteps + 1, NUM_FRAMES)
            ]
            start, end = random.choice(spans)
            middles = sorted(random.sample(range(start + 1, end), self.num_timesteps))
        elif self.split == "train":
            # span_mode "full": endpoints im1 and im7, so the supervised
            # timesteps are drawn from t in {1/6, ..., 5/6} exactly as the
            # algorithm document specifies. This is also the widest
            # available motion, which is where curvature actually shows.
            start, end = 0, NUM_FRAMES - 1
            middles = self._spread_middles()
        else:
            # Deterministic: widest span, evenly spaced middles.
            start, end = 0, NUM_FRAMES - 1
            middles = sorted(
                set(
                    int(round(v))
                    for v in np.linspace(1, NUM_FRAMES - 2, self.num_timesteps)
                )
            )
            # np.linspace can collide after rounding; fill from the left.
            candidates = [k for k in range(1, NUM_FRAMES - 1) if k not in middles]
            while len(middles) < self.num_timesteps:
                middles.append(candidates.pop(0))
            middles = sorted(middles)

        return start, end, middles

    def _load(self, index, frame_ids):
        seq_dir = os.path.join(self.image_root, self.meta_data[index])
        return [
            np.array(Image.open(os.path.join(seq_dir, f"im{f + 1}.png")).convert("RGB"))
            for f in frame_ids
        ]

    # ------------------------------------------------------------------
    # Augmentation (applied identically to every frame of the sample)
    # ------------------------------------------------------------------

    def _augment(self, frames, times):
        height, width = frames[0].shape[:2]
        # Multiple of 8: RAFT's H//8 grid and its ceil-strided conv stack
        # disagree otherwise, and grid_sample fails with a batch mismatch.
        crop = ((min(self.crop_size, height, width)) // 8) * 8
        top = random.randint(0, height - crop)
        left = random.randint(0, width - crop)
        frames = [f[top : top + crop, left : left + crop, :] for f in frames]

        if random.uniform(0, 1) < 0.5:  # channel order
            frames = [f[:, :, ::-1] for f in frames]
        if random.uniform(0, 1) < 0.3:  # vertical flip
            frames = [f[::-1] for f in frames]
        if random.uniform(0, 1) < 0.5:  # horizontal flip
            frames = [f[:, ::-1] for f in frames]

        rotation = random.uniform(0, 1)
        if rotation < 0.05:
            frames = [cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) for f in frames]
        elif rotation < 0.10:
            frames = [cv2.rotate(f, cv2.ROTATE_180) for f in frames]
        elif rotation < 0.15:
            frames = [cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE) for f in frames]

        if random.uniform(0, 1) < 0.5:  # reverse time
            img0, img1, middles = frames[0], frames[1], frames[2:]
            frames = [img1, img0] + middles[::-1]
            times = [1.0 - v for v in reversed(times)]

        return frames, times

    # ------------------------------------------------------------------

    def __getitem__(self, index):
        start, end, middles = self._sample_indices()
        frames = self._load(index, [start, end] + middles)
        times = [(k - start) / (end - start) for k in middles]

        if self.if_aug:
            frames, times = self._augment(frames, times)

        tensors = [
            torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float() / 255.0
            for f in frames
        ]

        return {
            "xs": torch.stack(tensors, dim=1),  # (3, 2 + K, H, W)
            "t": torch.tensor(times, dtype=torch.float32),  # (K,)
        }
