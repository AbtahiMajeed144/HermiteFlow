#!/usr/bin/env bash
# SNU-FILM arbitrary-timestep benchmark.
#   ./scripts/bm_SNU_FILM_arb.sh <checkpoint> [snu-film-root]
set -euo pipefail
CKPT=${1:?usage: bm_SNU_FILM_arb.sh <checkpoint> [snu-film-root]}
DATA_ROOT=${2:-./data/SNU-FILM}

python src/SNU_FILM_arb.py \
  -m configs/hermiteflow/hermiteflow_r.yaml \
  -l "${CKPT}" \
  --eval \
  --data-root "${DATA_ROOT}" \
  -p ./eval_output/snu_film_arb
