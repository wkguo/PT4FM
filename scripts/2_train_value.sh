#!/bin/bash
# PT4FM stage 2 — value-model SFT (RECAP categorical value + optional expectile).
#
# Usage:
#   bash scripts/2_train_value.sh [CONFIG_NAME] [HYDRA_OVERRIDES...]
# Example:
#   bash scripts/2_train_value.sh libero_value \
#       data.tag=fail300 actor.model.expectile.enabled=true actor.model.expectile.tau=0.7
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CONFIG_NAME=${1:-libero_value}
shift 1 2>/dev/null || true

LOG_DIR="${PT4FM_PATH}/logs/value/${CONFIG_NAME}-$(date +'%Y%m%d-%H%M%S')"
mkdir -p "${LOG_DIR}"

echo "[pt4fm] stage 2: value SFT ($CONFIG_NAME) -> ${LOG_DIR}"
${PT4FM_PYTHON} -m pt4fm.train_value \
    --config-name "${CONFIG_NAME}" \
    runner.logger.log_path="${LOG_DIR}" "$@" \
    2>&1 | tee "${LOG_DIR}/run.log"
echo "[pt4fm] stage 2 done -> checkpoints under ${LOG_DIR}"
