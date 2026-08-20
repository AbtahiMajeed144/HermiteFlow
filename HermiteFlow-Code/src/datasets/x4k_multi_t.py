# --------------------------------------------------------
# HermiteFlow — X4K1000FPS multi-timestep loader
#
# X4K is the right dataset for this model. Vimeo triplets are
# near-linear: F alone explains them, so d0 and d1 have nothing to
# learn and the curvature claim cannot be demonstrated. X4K is
# captured at 1000 fps, so a wide temporal window contains genuinely
# non-linear motion - which is the regime the whole trajectory model
# exists for.
#
# Temporal protocol follows X-TEST, as implemented in src/X4K.py's
# getXVFI: endpoints `frame_gap` apart (32), intermediate frames on
# the t = k/8 grid. Training therefore sees exactly the structure the
# benchmark evaluates.
#
# Two sources, auto-detected:
#
#   mp4 : encoded_train/**/<scene>/<clip>.mp4  - decoded on the fly
#   png : val/**/<Type>/<clip>/%04d.png        - read directly
#
# The mp4 path matters. Decoding X-TRAIN to PNG costs ~240 GB, which
# does not fit on a Kaggle working disk, and the input mount is
# read-only. Decoding the handful of frames a sample needs takes
# ~27 ms per clip (measured, 65 frames at 768x768), which disappears
# behind the dataloader workers.
#
# Item layout, identical to the septuplet loader:
#   xs: (3, 2 + K, H, W)   [ I0, I1, gt_1, ..., gt_K ]
#   t:  (K,)               t_k = k/num_divisions
# --------------------------------------------------------

import os
import glob
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# X-TRAIN clips are 65 consecutive 1000 fps frames at 768x768.
DEFAULT_CLIP_LENGTH = 65
# X-TEST evaluates with endpoints 32 apart on the k/8 grid.
DEFAULT_FRAME_GAP = 32
DEFAULT_DIVISIONS = 8


def _numeric_stem(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return (0, int(stem)) if stem.isdigit() else (1, stem)


class X4KMultiT(Dataset):
    """
    Args:
        split: "train" or "test"
        path:  a directory anywhere above the clips; the tree is searched
               recursively, so both `.../encoded_train` and the doubled
               `.../encoded_train/encoded_train` work.
        num_timesteps: K, ground-truth middle frames supervised per clip.
               Max is num_divisions - 1 (7 by default).
        frame_gap: temporal distance between the two endpoints, in frames.
        num_divisions: t grid denominator; middles land on k/num_divisions.
        source: "auto" | "mp4" | "png"
    """

    def __init__(
        self,
        split,
        path,
        num_timesteps=5,
        aug=True,
        crop_size=256,
        frame_gap=DEFAULT_FRAME_GAP,
        num_divisions=DEFAULT_DIVISIONS,
        clip_length=DEFAULT_CLIP_LENGTH,
        source="auto",
        repeat=1,
        downsample=1.0,
    ):
        assert split in ("train", "test"), split
        assert source in ("auto", "mp4", "png"), source
        assert 1 <= num_timesteps <= num_divisions - 1, (
            f"num_timesteps must be in [1, {num_divisions - 1}] for a "
            f"{num_divisions}-way grid, got {num_timesteps}"
        )
        assert frame_gap % num_divisions == 0, (
            f"frame_gap ({frame_gap}) must be divisible by num_divisions "
            f"({num_divisions}) so the middles land on real frames"
        )

        self.split = split
        self.num_timesteps = num_timesteps
        self.num_divisions = num_divisions
        self.frame_gap = frame_gap
        self.clip_length = clip_length
        self.if_aug = aug and split == "train"
        self.crop_size = crop_size
        # Resize each frame by 1/downsample BEFORE cropping.
        #
        # X4K's difficulty is temporal, not spatial: a 32-frame gap at
        # 1000 fps displaces objects ~60 px, which a 256 crop of a 768
        # clip cannot hold - forward-backward error measured 0.272 at
        # gap 32 versus 0.011 at gap 16, i.e. RAFT losing the
        # correspondence entirely. Downsampling divides displacement by
        # the same factor while keeping the WHOLE field of view, so
        # nothing moves out of frame.
        #
        # Curvature - the thing this model exists to learn - is a ratio
        # of flow energies and is therefore scale-invariant: dividing
        # every flow by 3 leaves it untouched. Tracking quality is not.
        # So this trades spatial detail, which the trajectory model does
        # not need, for tracking reliability, which it does.
        #
        # It also matches how the X4K benchmark actually runs the model:
        # src/X4K.py evaluates with ds_factor 0.5 (2K) and 0.25 (4K).
        self.downsample = float(downsample)
        assert self.downsample >= 1.0, "downsample is a shrink factor, >= 1"
        # X-TRAIN has only ~4.4k clips, but each one yields many distinct
        # samples: a random 32-frame window out of 65, a random crop, and
        # random flips. One pass over the clip list is therefore a very
        # small epoch - at batch 4 and total_batch 32 it is about 137
        # optimizer steps, so a nominal 60 epochs would be ~8k updates,
        # far too few to train three networks from scratch. `repeat`
        # draws each clip this many times per epoch so that "epoch"
        # remains a useful unit for checkpointing and LR scheduling.
        self.repeat = max(1, int(repeat)) if split == "train" else 1

        self.source, self.clips = self._index(path, source)
        if not self.clips:
            raise FileNotFoundError(
                f"no X4K clips found under {path}. Expected .mp4 files "
                f"(encoded_train / encoded_test) or directories of .png "
                f"frames (train / val / test)."
            )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _index(path, source):
        path = os.path.expanduser(path)
        if source in ("auto", "mp4"):
            clips = sorted(glob.glob(os.path.join(path, "**", "*.mp4"), recursive=True))
            if clips:
                return "mp4", clips
            if source == "mp4":
                return "mp4", []

        # PNG mode: every directory that directly contains .png frames.
        frames = glob.glob(os.path.join(path, "**", "*.png"), recursive=True)
        dirs = sorted({os.path.dirname(f) for f in frames})
        return "png", dirs

    def __len__(self):
        return len(self.clips) * self.repeat

    # ------------------------------------------------------------------
    # Timestep selection
    # ------------------------------------------------------------------

    def _grid_steps(self):
        """
        Which k of the t = k/num_divisions grid to supervise.

        Anchored at both extremes. Recovering (A, B) from samples of
        Phi(t) is ill-conditioned - beta3/beta2 = t+1 exactly, so the
        basis is near-collinear - and how bad it is depends sharply on
        which t are drawn. On this k/8 grid, measured amplification of a
        1 px error in Phi into error in (A, B):

            K=2  unanchored worst 173.1x   anchored 31.5x
            K=3  unanchored worst  71.4x   anchored 31.3x
            K=5  unanchored worst  26.8x   anchored 26.5x
            K=7  (all)                     18.6x

        Decoding cost is set by the LAST frame needed, not by how many
        are kept, so raising K to 7 costs no extra I/O - only the
        per-t compute of phases 3-5. K=7 is the best-conditioned
        setting this dataset offers.
        """
        first, last = 1, self.num_divisions - 1
        if self.num_timesteps == 1:
            return [random.randint(first, last)]
        if self.num_timesteps == 2:
            return [first, last]
        interior = list(range(first + 1, last))
        chosen = random.sample(interior, self.num_timesteps - 2)
        return sorted([first, last] + chosen)

    def _plan(self, num_frames):
        """Returns (start, end, [middle frame indices], [t values])."""
        span = self.frame_gap
        if num_frames < span + 1:
            # Short clip: fall back to the widest window it can hold,
            # rounded down to keep the middles on real frames.
            span = ((num_frames - 1) // self.num_divisions) * self.num_divisions
        if span < self.num_divisions:
            return None

        max_start = num_frames - span - 1
        start = random.randint(0, max_start) if self.split == "train" else max_start // 2

        step = span // self.num_divisions
        steps = self._grid_steps() if self.split == "train" else self._eval_steps()
        middles = [start + step * k for k in steps]
        times = [k / self.num_divisions for k in steps]
        return start, start + span, middles, times

    def _eval_steps(self):
        """Deterministic, evenly spread, anchored — for the val split."""
        first, last = 1, self.num_divisions - 1
        if self.num_timesteps == 1:
            return [self.num_divisions // 2]
        picks = np.linspace(first, last, self.num_timesteps)
        steps = sorted({int(round(v)) for v in picks})
        pool = [k for k in range(first, last + 1) if k not in steps]
        while len(steps) < self.num_timesteps and pool:
            steps.append(pool.pop(0))
        return sorted(steps)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _clip_length(self, clip):
        if self.source == "png":
            return len(self._png_frames(clip))
        return self.clip_length

    def _png_frames(self, clip_dir):
        return sorted(glob.glob(os.path.join(clip_dir, "*.png")), key=_numeric_stem)

    def _read_png(self, clip_dir, indices):
        files = self._png_frames(clip_dir)
        return [np.array(Image.open(files[i]).convert("RGB")) for i in indices]

    def _read_mp4(self, path, indices):
        """
        Sequential decode with grab()/retrieve(): frames that are not
        needed are advanced past without the colour conversion and copy,
        which measured ~2.6x faster than reading every frame. Seeking
        with CAP_PROP_POS_FRAMES is avoided - it is not frame-accurate
        on every OpenCV/H.264 build, and a silent off-by-one here would
        corrupt the timestep labels.
        """
        wanted = set(indices)
        last = max(wanted)
        found = {}

        capture = cv2.VideoCapture(path)
        try:
            index = 0
            while index <= last:
                if index in wanted:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    found[index] = frame[:, :, ::-1]  # BGR -> RGB
                else:
                    if not capture.grab():
                        break
                index += 1
        finally:
            capture.release()

        if len(found) != len(wanted):
            return None
        return [found[i] for i in indices]

    def _load(self, clip, indices):
        frames = (
            self._read_png(clip, indices) if self.source == "png"
            else self._read_mp4(clip, indices)
        )
        if frames is None or self.downsample == 1.0:
            return frames
        height, width = frames[0].shape[:2]
        size = (
            max(8, int(round(width / self.downsample))),
            max(8, int(round(height / self.downsample))),
        )
        # INTER_AREA is the correct (area-averaging) filter for shrinking;
        # bilinear would alias the high frequencies straight into the
        # flow estimate.
        return [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]

    # ------------------------------------------------------------------
    # Augmentation (applied identically to every frame of the sample)
    # ------------------------------------------------------------------

    def _augment(self, frames, times):
        height, width = frames[0].shape[:2]
        crop = min(self.crop_size, height, width)
        top = random.randint(0, height - crop)
        left = random.randint(0, width - crop)
        frames = [f[top : top + crop, left : left + crop, :] for f in frames]

        if random.uniform(0, 1) < 0.5:
            frames = [f[:, :, ::-1] for f in frames]
        if random.uniform(0, 1) < 0.3:
            frames = [f[::-1] for f in frames]
        if random.uniform(0, 1) < 0.5:
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

    def _center_crop(self, frames):
        height, width = frames[0].shape[:2]
        crop = min(self.crop_size, height, width)
        top, left = (height - crop) // 2, (width - crop) // 2
        return [f[top : top + crop, left : left + crop, :] for f in frames]

    # ------------------------------------------------------------------

    def __getitem__(self, index):
        # A handful of clips can be shorter than the nominal 65 frames or
        # fail to decode; resample rather than crash a multi-hour run.
        index %= len(self.clips)
        for attempt in range(8):
            clip = self.clips[index]
            plan = self._plan(self._clip_length(clip))
            if plan is not None:
                start, end, middles, times = plan
                frames = self._load(clip, [start, end] + middles)
                if frames is not None:
                    break
            index = random.randrange(len(self.clips))
        else:
            raise RuntimeError(
                f"could not read a usable clip after 8 attempts (last: {clip})"
            )

        if self.if_aug:
            frames, times = self._augment(frames, times)
        else:
            frames = self._center_crop(frames)

        tensors = [
            torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float() / 255.0
            for f in frames
        ]

        return {
            "xs": torch.stack(tensors, dim=1),  # (3, 2 + K, H, W)
            "t": torch.tensor(times, dtype=torch.float32),  # (K,)
        }
