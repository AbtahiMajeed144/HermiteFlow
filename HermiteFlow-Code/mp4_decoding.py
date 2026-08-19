"""
X4K1000FPS mp4 -> png decoder (XVFI's original utility, made
argument-driven).

    python mp4_decoding.py <encoded-root> <output-root> [--limit N] [--dry-run]

    # the official test set, ~6 GB
    python mp4_decoding.py ./encoded_test ./test

YOU PROBABLY DO NOT NEED THIS FOR TRAINING. Decoding X-TRAIN costs
about 240 GB, which does not fit on a Kaggle working disk and cannot
be written to the read-only input mount anyway. The training loader
(`src/datasets/x4k_multi_t.py`, `dataset.type: x4k_multi_t`) reads the
mp4 files directly and decodes only the frames each sample needs -
about 27 ms per clip, which disappears behind the dataloader workers.

Use this script when you want frames on disk: for the test/val splits,
for inspecting clips, or when disk is plentiful and you would rather
pay once.

ffmpeg 4+ supports '-pred mixed', which produces smaller lossless PNGs.
Older versions work without it; the script detects and adapts.

    conda install -c conda-forge ffmpeg
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode X4K1000FPS .mp4 clips into directories of .png frames."
    )
    parser.add_argument("encoded_root", help="directory containing .mp4 clips")
    parser.add_argument("output_root", help="where to write <clip>/%%04d.png")
    parser.add_argument(
        "--limit", type=int, default=None, help="decode at most N clips"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands and exit"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-decode clips whose output directory already has frames",
    )
    return parser.parse_args()


def supports_pred_mixed():
    """'-pred mixed' is an ffmpeg 4+ png encoder option."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None  # ffmpeg missing
    first = out.splitlines()[0] if out else ""
    for token in first.split():
        major = token.split(".")[0]
        if major.isdigit():
            return int(major) >= 4
    return False


def main():
    args = parse_args()

    pred_mixed = supports_pred_mixed()
    if pred_mixed is None:
        sys.exit(
            "ffmpeg not found on PATH. Install it (conda install -c conda-forge "
            "ffmpeg), or skip decoding entirely and train straight from the mp4 "
            "files with dataset.type: x4k_multi_t."
        )

    clips = sorted(
        glob.glob(os.path.join(args.encoded_root, "**", "*.mp4"), recursive=True)
    )
    if not clips:
        sys.exit(f"no .mp4 files found under {args.encoded_root}")
    if args.limit is not None:
        clips = clips[: args.limit]

    print(
        f"{len(clips)} clips | ffmpeg {'>=4 (-pred mixed)' if pred_mixed else '<4'} "
        f"| {args.encoded_root} -> {args.output_root}"
    )

    for index, clip in enumerate(clips, 1):
        relative = os.path.relpath(clip, args.encoded_root)
        target = os.path.join(args.output_root, os.path.splitext(relative)[0])

        if not args.overwrite and glob.glob(os.path.join(target, "*.png")):
            print(f"[{index}/{len(clips)}] skip (already decoded) {relative}")
            continue

        command = ["ffmpeg", "-loglevel", "error", "-y", "-i", clip]
        if pred_mixed:
            command += ["-pred", "mixed"]
        command += ["-start_number", "0", os.path.join(target, "%04d.png")]

        print(f"[{index}/{len(clips)}] {relative}")
        if args.dry_run:
            print("   ", " ".join(command))
            continue

        os.makedirs(target, exist_ok=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            # Leave no half-decoded directory behind to be silently
            # picked up as a valid clip by the dataloader.
            shutil.rmtree(target, ignore_errors=True)
            sys.exit(f"ffmpeg failed on {clip} (exit {result.returncode})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
