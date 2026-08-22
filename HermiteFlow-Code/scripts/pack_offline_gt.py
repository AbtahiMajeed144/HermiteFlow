"""
Repack the offline GT cache into flat, memory-mappable arrays.

generate_offline_gt.py writes one zip container and two PNGs per sample,
which is the right shape for a resumable job but the wrong shape for
training: at `repeat: 8` an epoch opens 35k zips and decodes 70k PNGs,
and none of it can be held as raw pages.

This rewrites the same data as three .npy files that np.load can mmap:

    flows.npy   (N, 2, K, 2, H, W)  float16   [:, 0] = f_{0->t}
                                              [:, 1] = f_{1->t}
    images.npy  (N, 2, H, W, 3)     uint8     endpoints, RGB
    times.npy   (N, K)              float32
    packed.json                               provenance + shapes

WHY MMAP AND NOT AN EXPLICIT RAM LOAD

Training runs under DDP, so `nproc_per_node` GPUs means that many
processes. An explicit load costs one full copy per rank - 2 x 16.7 GiB
against a 30 GiB box - and OOMs before the first step. The page cache is
shared by every process and every forked worker, so a mapping is one
physical copy no matter how many ranks read it, needs no RAM budget, and
falls back to disk reads rather than dying when memory is short. After
the first epoch it is resident either way.

    python scripts/pack_offline_gt.py \
        --cache /kaggle/input/<cache-dataset>/offline_gt \
        --out   /kaggle/working/offline_gt_packed

Read the cache from a mounted input and write the pack to working, so
only one copy occupies the writable disk.
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Pack offline GT for mmap")
    parser.add_argument("--cache", required=True, help="dir holding split_*/")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--verify", type=int, default=8,
        help="re-read this many packed samples and compare against the source",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    samples = sorted(
        glob.glob(os.path.join(args.cache, "**", "sample_*_flow.npz"), recursive=True)
    )
    if not samples:
        raise SystemExit(f"no sample_*_flow.npz under {args.cache}")

    with np.load(samples[0]) as probe:
        num_t, _, height, width = probe["flow_0_t"].shape
    count = len(samples)
    os.makedirs(args.out, exist_ok=True)

    flow_bytes = count * 2 * num_t * 2 * height * width * 2
    image_bytes = count * 2 * height * width * 3
    print(
        f"{count} samples, K={num_t}, {height}x{width}\n"
        f"  flows.npy  {flow_bytes / 2**30:6.2f} GiB\n"
        f"  images.npy {image_bytes / 2**30:6.2f} GiB"
    )

    # open_memmap writes the header up front and streams the body, so
    # peak RSS stays at one sample rather than the whole 15 GiB.
    from numpy.lib.format import open_memmap

    flows = open_memmap(
        os.path.join(args.out, "flows.npy"), mode="w+",
        dtype=np.float16, shape=(count, 2, num_t, 2, height, width),
    )
    images = open_memmap(
        os.path.join(args.out, "images.npy"), mode="w+",
        dtype=np.uint8, shape=(count, 2, height, width, 3),
    )
    times = np.zeros((count, num_t), dtype=np.float32)

    for i, path in enumerate(tqdm(samples, desc="packing")):
        prefix = path[: -len("_flow.npz")]
        with np.load(path) as data:
            flows[i, 0] = data["flow_0_t"]
            flows[i, 1] = data["flow_1_t"]
            times[i] = data["t"]
        for j, name in enumerate(("_img0.png", "_img1.png")):
            bgr = cv2.imread(prefix + name)
            if bgr is None:
                raise SystemExit(f"could not read {prefix + name}")
            images[i, j] = bgr[:, :, ::-1]

    flows.flush()
    images.flush()
    np.save(os.path.join(args.out, "times.npy"), times)

    manifests = sorted(
        glob.glob(os.path.join(args.cache, "**", "manifest.json"), recursive=True)
    )
    packed = {
        "num_samples": count,
        "num_timesteps": num_t,
        "height": height,
        "width": width,
        "source_manifests": [json.load(open(p)) for p in manifests],
    }
    with open(os.path.join(args.out, "packed.json"), "w") as handle:
        json.dump(packed, handle, indent=2)

    # A silent transcription error here would train against shuffled
    # targets and look exactly like a model that will not converge, so
    # read a sample of the pack back and compare it to the source.
    if args.verify:
        flows_r = np.load(os.path.join(args.out, "flows.npy"), mmap_mode="r")
        images_r = np.load(os.path.join(args.out, "images.npy"), mmap_mode="r")
        times_r = np.load(os.path.join(args.out, "times.npy"))
        step = max(1, count // args.verify)
        bad = 0
        for i in range(0, count, step):
            prefix = samples[i][: -len("_flow.npz")]
            with np.load(samples[i]) as data:
                same = (
                    np.array_equal(flows_r[i, 0], data["flow_0_t"])
                    and np.array_equal(flows_r[i, 1], data["flow_1_t"])
                    and np.array_equal(times_r[i], data["t"])
                )
            for j, name in enumerate(("_img0.png", "_img1.png")):
                same = same and np.array_equal(
                    images_r[i, j], cv2.imread(prefix + name)[:, :, ::-1]
                )
            bad += not same
        print(f"verified {len(range(0, count, step))} samples, {bad} mismatched")
        if bad:
            raise SystemExit("pack does not match the source cache")

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
