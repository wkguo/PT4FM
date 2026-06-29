#!/bin/bash
# PT4FM stage 3 — advantages = RECAP value inference (3a) + PT4FM augmentation (3b).
#
#   3a) RLinf compute_advantages.py: A = normalize(sum r) + gamma^N V(o_{t+N}) - V(o_t),
#       quantile-thresholded bool label  -> meta/advantages_{tag}.parquet
#   3b) pt4fm.process.postprocess_advantages: Cal-QL calibration of the bootstrap +
#       AWR `advantage_weight` column + `is_demo` column (overwrites/adds out-tag).
#
# Usage:
#   bash scripts/3_compute_advantages.sh \
#       --value-ckpt /path/to/value_ckpt --tag fail300_N10_q30 --returns-tag fail300 \
#       --dataset /data/libero_sft:sft [--dataset /data/libero_rollout:rollout] \
#       --lookahead 10 --gamma 1.0 --positive-quantile 0.3 \
#       --awr-beta 0.7 --awr-wmax 20 --calql \
#       [--recap-config compute_advantages] [--nproc 1] [--skip-recap]
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

VALUE_CKPT=""; TAG=""; RETURNS_TAG=""; LOOKAHEAD=10; GAMMA=1.0; PQ=0.3
AWR_BETA=0.7; AWR_WMAX=20; CALQL=""; RECAP_CONFIG="compute_advantages"; NPROC=1
SKIP_RECAP=""; DATASETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --value-ckpt) VALUE_CKPT="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --returns-tag) RETURNS_TAG="$2"; shift 2;;
    --dataset) DATASETS+=("$2"); shift 2;;
    --lookahead) LOOKAHEAD="$2"; shift 2;;
    --gamma) GAMMA="$2"; shift 2;;
    --positive-quantile) PQ="$2"; shift 2;;
    --awr-beta) AWR_BETA="$2"; shift 2;;
    --awr-wmax) AWR_WMAX="$2"; shift 2;;
    --calql) CALQL="--calql"; shift 1;;
    --recap-config) RECAP_CONFIG="$2"; shift 2;;
    --nproc) NPROC="$2"; shift 2;;
    --skip-recap) SKIP_RECAP="1"; shift 1;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# ---- 3a: RECAP distributed value inference (skip if parquet already exists) ----
if [[ -z "${SKIP_RECAP}" ]]; then
  echo "[pt4fm] stage 3a: RECAP compute_advantages (tag=${TAG})"
  ( cd "${RLINF_PATH}/examples/recap/process"
    if [[ "${NPROC}" -gt 1 ]]; then
      torchrun --nproc_per_node="${NPROC}" compute_advantages.py --config-name "${RECAP_CONFIG}" \
        advantage.value_checkpoint="${VALUE_CKPT}" advantage.tag="${TAG}" advantage.returns_tag="${RETURNS_TAG}"
    else
      ${PT4FM_PYTHON} compute_advantages.py --config-name "${RECAP_CONFIG}" \
        advantage.value_checkpoint="${VALUE_CKPT}" advantage.tag="${TAG}" advantage.returns_tag="${RETURNS_TAG}"
    fi )
fi

# ---- 3b: PT4FM Cal-QL + AWR weight + is_demo augmentation ----
echo "[pt4fm] stage 3b: postprocess_advantages (calql=${CALQL:-off}, beta=${AWR_BETA})"
DS_ARGS=()
for d in "${DATASETS[@]}"; do DS_ARGS+=(--dataset "$d"); done
${PT4FM_PYTHON} -m pt4fm.process.postprocess_advantages \
    "${DS_ARGS[@]}" --tag "${TAG}" \
    --lookahead "${LOOKAHEAD}" --gamma "${GAMMA}" --positive-quantile "${PQ}" \
    --awr-beta "${AWR_BETA}" --awr-wmax "${AWR_WMAX}" ${CALQL}
echo "[pt4fm] stage 3 done -> meta/advantages_${TAG}.parquet (+advantage_weight,is_demo)"
