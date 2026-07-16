#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash verify_instruction/run_verify_instruction.sh [verify_instruction.py options]

Examples:
  # Default: one visible route for each of the three required basic-track scenes.
  bash verify_instruction/run_verify_instruction.sh

  # List the English model instructions and display-only Chinese translations.
  bash verify_instruction/run_verify_instruction.sh --list

  # Run one command on its recommended route.
  bash verify_instruction/run_verify_instruction.sh --instruction-id S1-03

  # Run core commands from the complex-obstacle scene.
  bash verify_instruction/run_verify_instruction.sh --scene S2 --core-only

  # Run all 30 English instructions.
  bash verify_instruction/run_verify_instruction.sh --preset all

  # Validate route placement and evaluator commands without starting CARLA.
  bash verify_instruction/run_verify_instruction.sh --preset three-scenes --dry-run

Environment overrides:
  CARLA_ROOT=~/carla/carla0915
  CONDA_ENV=simlingo
  CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
  CHECKPOINT=/path/to/pytorch_model.pt
  HF_ENDPOINT=https://hf-mirror.com
  GPU_RANK=0
  PORT=10000
  TM_PORT=30000
  CARLA_WINDOW_WIDTH=1280
  CARLA_WINDOW_HEIGHT=720
  SIMLINGO_VIZ_WIDTH=1280
  SIMLINGO_VIZ_HEIGHT=800
  SIMLINGO_CJK_FONT=/path/to/NotoSansCJK-Regular.ttc

The evaluator starts and stops a visible CARLA server for every selected test.
The Pygame window shows the camera, predicted paths, controls, English prompt,
model answer, and (when a CJK font is available) the display-only translation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

NEEDS_RUNTIME=1
for arg in "$@"; do
  if [[ "$arg" == "--list" || "$arg" == "--dry-run" ]]; then
    NEEDS_RUNTIME=0
  fi
done

if [[ "$NEEDS_RUNTIME" == "1" && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "No graphical desktop detected. DISPLAY or WAYLAND_DISPLAY is required." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Temporary mirror setting: it exists only for this shell and child processes.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

CONDA_ENV="${CONDA_ENV:-simlingo}"
CONDA_SH="${CONDA_SH:-}"
if [[ -z "$CONDA_SH" ]]; then
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    CONDA_BASE="$("$CONDA_EXE" info --base)"
    CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
  elif [[ -f "${HOME:?}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
fi
if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "Unable to locate conda.sh. Set CONDA_SH explicitly." >&2
  exit 2
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CARLA_ROOT="${CARLA_ROOT:-$HOME/carla/carla0915}"
export CHECKPOINT="${CHECKPOINT:-$REPO_ROOT/pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}"
export GPU_RANK="${GPU_RANK:-0}"
export CARLA_RENDER_OFFSCREEN=0
export SIMLINGO_VIZ=1
export CARLA_WINDOW_WIDTH="${CARLA_WINDOW_WIDTH:-1280}"
export CARLA_WINDOW_HEIGHT="${CARLA_WINDOW_HEIGHT:-720}"
export SIMLINGO_VIZ_WIDTH="${SIMLINGO_VIZ_WIDTH:-1280}"
export SIMLINGO_VIZ_HEIGHT="${SIMLINGO_VIZ_HEIGHT:-800}"

if [[ "$NEEDS_RUNTIME" == "1" && ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
  echo "CARLA launcher not found or not executable: $CARLA_ROOT/CarlaUE4.sh" >&2
  exit 2
fi
if [[ "$NEEDS_RUNTIME" == "1" && ! -f "$CHECKPOINT" ]]; then
  echo "SimLingo checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

# A second CARLA instance was previously enough to exhaust a 24 GB RTX 3090.
# Refuse to continue and leave process ownership/cleanup to the user.
if [[ "$NEEDS_RUNTIME" == "1" ]] && pgrep -af '[C]arlaUE4' >/dev/null 2>&1; then
  echo "A CARLA process is already running:" >&2
  pgrep -af '[C]arlaUE4' >&2 || true
  echo "Stop the stale/other CARLA instance before running this verification." >&2
  exit 2
fi

export WORK_DIR="$REPO_ROOT"
export SCENARIO_RUNNER_ROOT="$REPO_ROOT/Bench2Drive/scenario_runner"
export LEADERBOARD_ROOT="$REPO_ROOT/Bench2Drive/leaderboard"
export ROUTES="${ROUTES:-$REPO_ROOT/leaderboard/data/bench2drive220.xml}"

PYTHON_TAG="$($PYTHON_BIN -c 'import sys; print(f"py{sys.version_info.major}.{sys.version_info.minor}")')"
CARLA_EGG="${CARLA_EGG:-$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-${PYTHON_TAG}-linux-x86_64.egg}"
export PYTHONPATH="$REPO_ROOT:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$SCENARIO_RUNNER_ROOT:$LEADERBOARD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -e "$CARLA_EGG" ]]; then
  export PYTHONPATH="$CARLA_EGG:$PYTHONPATH"
else
  echo "Warning: matching CARLA egg not found at $CARLA_EGG; using installed carla package." >&2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_RANK}"

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "Conda environment=$CONDA_ENV"
echo "CARLA_ROOT=$CARLA_ROOT"
echo "Checkpoint=$CHECKPOINT"

exec "$PYTHON_BIN" "$SCRIPT_DIR/verify_instruction.py" --routes "$ROUTES" "$@"
