#!/bin/bash
# PT4FM stage 1 — compute discounted returns (sidecar parquet).
#
# This stage is method-agnostic and reuses RLinf's RECAP script unchanged. It
# writes meta/returns_{tag}.parquet + return stats into meta/stats.json.
#
# Usage:
#   bash scripts/1_compute_returns.sh [CONFIG_NAME] [HYDRA_OVERRIDES...]
# Example:
#   bash scripts/1_compute_returns.sh compute_returns \
#       data.train_data_paths='[{dataset_path:/data/libero_sft,type:sft}]' \
#       data.tag=fail300 data.failure_reward=-300
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CONFIG_NAME=${1:-compute_returns}
shift 1 2>/dev/null || true

cd "${RLINF_PATH}/examples/recap/process"
echo "[pt4fm] stage 1: compute_returns ($CONFIG_NAME)"
${PT4FM_PYTHON} compute_returns.py --config-name "${CONFIG_NAME}" "$@"
echo "[pt4fm] stage 1 done -> meta/returns_{tag}.parquet"
