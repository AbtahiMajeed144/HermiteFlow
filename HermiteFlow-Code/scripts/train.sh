#!/usr/bin/env bash
# HermiteFlow training launcher.
#
#   ./scripts/train.sh <config> <output-dir> <num-gpus> <data-path> [extra dotlist...]
#
# Example (Kaggle, 2x T4):
#   ./scripts/train.sh configs/hermiteflow/hermiteflow_r.yaml \
#                      /kaggle/working/runs 2 \
#                      /kaggle/input/vimeo90k/vimeo_septuplet
#
# Any trailing arguments are passed through as OmegaConf dotlist
# overrides, e.g. experiment.batch_size=2 arch.coeff_net_channels=48

set -euo pipefail

MODEL_CONFIG=${1:?usage: train.sh <config> <output-dir> <num-gpus> <data-path> [overrides...]}
OUTPUT=${2:?missing output dir}
NPROC_PER_NODE=${3:?missing gpu count}
DATA_PATH=${4:?missing data path}
shift 4

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
         --nnodes=1 \
         --node_rank=0 \
         --master_port=16890 \
         src/main.py \
         --model-config="${MODEL_CONFIG}" \
         --result-path="${OUTPUT}" \
         --data-path="${DATA_PATH}" \
         --seed=0 \
         "$@"
