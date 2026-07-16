#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_bench2drive_simlingo.sh

Runs all 220 routes from leaderboard/data/bench2drive220.xml.
Default CARLA_ROOT: ~/carla/carla0915
Default checkpoint: pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt

Useful overrides:
  CARLA_ROOT=/path/to/carla   Override the default CARLA installation.
  CHECKPOINT=/path/to/model   Override the default SimLingo checkpoint.
  CONDA_ENV=simlingo         Conda environment created from environment.yaml.
  CONDA_SH=/path/conda.sh    Conda shell hook if it cannot be auto-detected.
  GPU_RANK=0                 CUDA device and CARLA graphics adapter.
  PORT=10000                 Starting CARLA RPC port.
  TM_PORT=30000              Starting Traffic Manager port.
  RESULT_DIR=/path/to/output Evaluation output directory.
  ROUTES=/path/to/routes.xml Override the default 220-route XML.
  ROUTES_SUBSET=1711        Run only selected route IDs (for example 1711).
  RESUME=1                   Resume from CHECKPOINT_ENDPOINT.
  CARLA_RENDER_OFFSCREEN=0   Show the CARLA window (default: 1).
  HF_ENDPOINT=https://...    Override the temporary Hugging Face mirror.

The Bench2Drive evaluator starts and stops CARLA automatically.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo "This script does not accept positional arguments." >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# This export only affects this script and its child processes. It disappears
# when the script exits, unless HF_ENDPOINT was already exported by the caller.
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
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
fi
if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "Unable to locate conda.sh. Set CONDA_SH=/path/to/conda/etc/profile.d/conda.sh." >&2
  exit 2
fi

# environment.yaml defines the simlingo environment used by the README.
source "$CONDA_SH"
conda activate "$CONDA_ENV"
PYTHON_BIN="${PYTHON_BIN:-python}"

CARLA_ROOT="${CARLA_ROOT:-$HOME/carla/carla0915}"
CHECKPOINT="${CHECKPOINT:-${SIMLINGO_CHECKPOINT:-$REPO_ROOT/pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt}}"
GPU_RANK="${GPU_RANK:-0}"
export CARLA_RENDER_OFFSCREEN="${CARLA_RENDER_OFFSCREEN:-1}"
PORT="${PORT:-10000}"
TM_PORT="${TM_PORT:-30000}"
TRAFFIC_MANAGER_SEED="${TRAFFIC_MANAGER_SEED:-1}"
TIMEOUT="${TIMEOUT:-600}"
RESUME="${RESUME:-0}"

ROUTES="${ROUTES:-$REPO_ROOT/leaderboard/data/bench2drive220.xml}"
ROUTES_SUBSET="${ROUTES_SUBSET:-}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/eval_results/Bench2Drive}"
CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT:-$RESULT_DIR/full_res.json}"
DEBUG_CHECKPOINT_ENDPOINT="${DEBUG_CHECKPOINT_ENDPOINT:-$RESULT_DIR/full_live.txt}"
SAVE_PATH="${SAVE_PATH:-$RESULT_DIR/viz/}"
SAVE_PATH="${SAVE_PATH%/}/"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
  echo "CARLA launcher not found or not executable: $CARLA_ROOT/CarlaUE4.sh" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "SimLingo checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$ROUTES" ]]; then
  echo "Route XML not found: $ROUTES" >&2
  exit 2
fi

MODEL_ROOT="$(dirname "$(dirname "$(dirname "$CHECKPOINT")")")"
MODEL_CONFIG="$MODEL_ROOT/.hydra/config.yaml"
if [[ ! -f "$MODEL_CONFIG" ]]; then
  echo "Hydra model config not found: $MODEL_CONFIG" >&2
  echo "Keep the downloaded simlingo/.hydra and simlingo/checkpoints directory structure intact." >&2
  exit 2
fi

export WORK_DIR="$REPO_ROOT"
export CARLA_ROOT
export SCENARIO_RUNNER_ROOT="$REPO_ROOT/Bench2Drive/scenario_runner"
export LEADERBOARD_ROOT="$REPO_ROOT/Bench2Drive/leaderboard"
export ROUTES
export SAVE_PATH

PYTHON_TAG="$($PYTHON_BIN -c 'import sys; print(f"py{sys.version_info.major}.{sys.version_info.minor}")')"
CARLA_EGG="${CARLA_EGG:-$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-${PYTHON_TAG}-linux-x86_64.egg}"
export PYTHONPATH="$REPO_ROOT:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$SCENARIO_RUNNER_ROOT:$LEADERBOARD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -e "$CARLA_EGG" ]]; then
  export PYTHONPATH="$CARLA_EGG:$PYTHONPATH"
else
  echo "Warning: matching CARLA egg not found at $CARLA_EGG; using the installed carla package if available." >&2
fi

mkdir -p "$RESULT_DIR" "$SAVE_PATH"

EVALUATOR_ARGS=(
  "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py"
  --routes="$ROUTES"
  --repetitions=1
  --track=SENSORS
  --checkpoint="$CHECKPOINT_ENDPOINT"
  --debug-checkpoint="$DEBUG_CHECKPOINT_ENDPOINT"
  --timeout="$TIMEOUT"
  --agent="$REPO_ROOT/team_code/agent_simlingo.py"
  --agent-config="$CHECKPOINT"
  --traffic-manager-seed="$TRAFFIC_MANAGER_SEED"
  --port="$PORT"
  --traffic-manager-port="$TM_PORT"
  --gpu-rank="$GPU_RANK"
)
if [[ -n "$ROUTES_SUBSET" ]]; then
  EVALUATOR_ARGS+=(--routes-subset="$ROUTES_SUBSET")
fi
if [[ "$RESUME" == "1" ]]; then
  EVALUATOR_ARGS+=(--resume=True)
fi

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "Conda environment=$CONDA_ENV"
echo "Routes=$ROUTES"
echo "Checkpoint=$CHECKPOINT"
echo "Results=$CHECKPOINT_ENDPOINT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_RANK}"
"$PYTHON_BIN" "${EVALUATOR_ARGS[@]}"
