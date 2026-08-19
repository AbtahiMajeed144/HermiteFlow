#!/usr/bin/env bash
# Nx video interpolation.
#   ./scripts/video_Nx.sh <checkpoint> <input-frames> <output-dir> [N] [ds-factor]
set -euo pipefail
CKPT=${1:?usage: video_Nx.sh <checkpoint> <input-frames> <output-dir> [N] [ds-factor]}
SOURCE=${2:?missing input frame directory}
OUTPUT=${3:?missing output directory}
N=${4:-8}
DS=${5:-1.0}

python src/video_Nx.py \
  -m configs/hermiteflow/hermiteflow_r.yaml \
  -l "${CKPT}" \
  --eval \
  --source-path "${SOURCE}" \
  --output-path "${OUTPUT}" \
  --N "${N}" \
  --ds-factor "${DS}"
