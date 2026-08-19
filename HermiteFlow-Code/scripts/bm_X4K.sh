#!/usr/bin/env bash
# X4K1000FPS (XTEST-2k / XTEST-4k) benchmark.
#   ./scripts/bm_X4K.sh <checkpoint> [x4k-test-root]
set -euo pipefail
CKPT=${1:?usage: bm_X4K.sh <checkpoint> [x4k-test-root]}
DATA_ROOT=${2:-./data/x4k/test}

python src/X4K.py \
  -m configs/hermiteflow/hermiteflow_f.yaml \
  -l "${CKPT}" \
  --eval \
  --data-root "${DATA_ROOT}" \
  -p ./eval_output/x4k
