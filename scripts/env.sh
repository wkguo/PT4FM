#!/bin/bash
# PT4FM shared environment. Source this from every stage script.
#
# Resolves the three packages PT4FM depends on and puts them on PYTHONPATH:
#   - rlinf   (RLinf RECAP infrastructure)         -> $RLINF_PATH
#   - openpi  (pi0.5 model + LeRobot data)         -> $OPENPI_PATH
#   - pt4fm   (this framework)                     -> $PT4FM_PATH
#
# Override any of RLINF_PATH / OPENPI_PATH / PT4FM_PYTHON in your shell if your
# layout differs from the default sibling-directory layout.

PT4FM_SCRIPTS_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export PT4FM_PATH="$( cd "${PT4FM_SCRIPTS_DIR}/.." && pwd )"
FPT_DIR="$( cd "${PT4FM_PATH}/.." && pwd )"            # foundry_post_training

# RLinf root (provides `rlinf` package + hydra group configs via REPO_PATH).
export RLINF_PATH="${RLINF_PATH:-${FPT_DIR}/RLinf}"
# RLinf scripts expect REPO_PATH to point at the RLinf root.
export REPO_PATH="${RLINF_PATH}"
# openpi source root (the `openpi` package lives under src/).
export OPENPI_PATH="${OPENPI_PATH:-${FPT_DIR}/../VLA-R3L/openpi/src}"

export PYTHONPATH="${PT4FM_PATH}:${RLINF_PATH}:${OPENPI_PATH}:${LIBERO_REPO_PATH}:${PYTHONPATH}"

# Use $PT4FM_PYTHON if set, else whatever `python` is active (e.g. after
# `source switch_env openpi` in the RLinf docker image).
export PT4FM_PYTHON="${PT4FM_PYTHON:-python}"

# Quiet noisy AV/GL backends (same as RLinf).
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export OPENCV_LOG_LEVEL=off

# Optionally switch into the RLinf openpi venv if available (no-op otherwise).
source switch_env openpi 2>/dev/null || true

echo "[pt4fm/env] PT4FM_PATH=${PT4FM_PATH}"
echo "[pt4fm/env] RLINF_PATH=${RLINF_PATH}"
echo "[pt4fm/env] OPENPI_PATH=${OPENPI_PATH}"
echo "[pt4fm/env] python=$(${PT4FM_PYTHON} -c 'import sys; print(sys.executable)' 2>/dev/null || echo ${PT4FM_PYTHON})"
