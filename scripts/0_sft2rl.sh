#!/bin/bash
# PT4FM stage 0 — make a LeRobot SFT/demo dataset RL-ready (adds is_success, etc.).
#
# Run this once per incoming dataset BEFORE stage 1 (compute_returns). It writes a
# new self-contained LeRobot dataset (videos symlinked) with the RL contract the
# PT4FM/RECAP pipeline needs. Designed for batches of hundreds of hours.
#
# Usage:
#   bash scripts/0_sft2rl.sh --src SRC --dst DST [options passed to pt4fm.data.sft2rl]
# Examples:
#   # clean demos (all success):
#   bash scripts/0_sft2rl.sh --src /data/lerobot_demos --dst /data/lerobot_demos_rlready \
#       --default-success true --add-done --num-workers 16
#   # self-collected rollouts with a success label map (episode_index -> 0/1):
#   bash scripts/0_sft2rl.sh --src /data/rollouts --dst /data/rollouts_rlready \
#       --success-map /data/rollouts/success.json --add-done --add-reward --num-workers 16
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

echo "[pt4fm] stage 0: sft2rl"
${PT4FM_PYTHON} -m pt4fm.data.sft2rl "$@"
echo "[pt4fm] stage 0 done. Next: declare the output as type sft|rollout in the data config and run scripts/1_compute_returns.sh"
