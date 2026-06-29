#!/bin/bash
# PT4FM stage 4 — AWR-CFG + SFT-aux policy post-training for pi0.5.
#
# Usage:
#   bash scripts/4_train_policy.sh [CONFIG_NAME] [HYDRA_OVERRIDES...]
# Example (full AWR + Cal-QL value + SFT anchor):
#   bash scripts/4_train_policy.sh libero_policy \
#       data.advantage_tag=fail300_N10_q30 \
#       actor.model.openpi.awr.enabled=true actor.model.openpi.awr.beta=0.7 \
#       actor.model.openpi.sft_aux.mode=separate_forward actor.model.openpi.sft_aux.weight=0.5
# Example (reproduce RECAP exactly):
#   bash scripts/4_train_policy.sh libero_policy \
#       actor.model.openpi.awr.enabled=false \
#       actor.model.openpi.sft_aux.mode=reuse_unconditional actor.model.openpi.sft_aux.weight=1.0
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CONFIG_NAME=${1:-libero_policy}
shift 1 2>/dev/null || true

LOG_DIR="${PT4FM_PATH}/logs/policy/${CONFIG_NAME}-$(date +'%Y%m%d-%H%M%S')"
mkdir -p "${LOG_DIR}"

echo "[pt4fm] stage 4: policy post-training ($CONFIG_NAME) -> ${LOG_DIR}"
${PT4FM_PYTHON} -m pt4fm.train_policy \
    --config-name "${CONFIG_NAME}" \
    runner.logger.log_path="${LOG_DIR}" "$@" \
    2>&1 | tee "${LOG_DIR}/run.log"
echo "[pt4fm] stage 4 done -> checkpoints under ${LOG_DIR}"
