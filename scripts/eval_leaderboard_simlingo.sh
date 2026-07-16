#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_leaderboard_simlingo.sh

Runs all routes from leaderboard/data/routes_validation.xml.
Default CARLA_ROOT: ~/carla/carla0915
Default checkpoint: pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt

Useful overrides:
  CARLA_ROOT=/path/to/carla   Override the default CARLA installation.
  CHECKPOINT=/path/to/model   Override the default SimLingo checkpoint.
  CONDA_ENV=simlingo         Conda environment created from environment.yaml.
  CONDA_SH=/path/conda.sh    Conda shell hook if it cannot be auto-detected.
  GPU_RANK=0                 CUDA device used by SimLingo.
  CARLA_GRAPHICS_ADAPTER=0   Vulkan adapter used by CARLA.
  PORT=2000                  CARLA RPC port.
  TM_PORT=8000               Traffic Manager port.
  RESULT_DIR=/path/to/output Evaluation output directory.
  ROUTES=/path/to/routes.xml Override the default full route XML.
  ROUTES_SUBSET=0           Run only selected route IDs (for example 0 or 0,2-4).
  RESUME=1                   Resume from CHECKPOINT_ENDPOINT.
  START_CARLA=0              Connect to an already running CARLA server.
  CARLA_RENDER_OFFSCREEN=0   Show the CARLA window (default: 1).
  CARLA_WINDOW_WIDTH=1280    Visible CARLA window width.
  CARLA_WINDOW_HEIGHT=720    Visible CARLA window height.
  HF_ENDPOINT=https://...    Override the temporary Hugging Face mirror.
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
CARLA_GRAPHICS_ADAPTER="${CARLA_GRAPHICS_ADAPTER:-$GPU_RANK}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
TRAFFIC_MANAGER_SEED="${TRAFFIC_MANAGER_SEED:-1}"
TIMEOUT="${TIMEOUT:-600}"
RESUME="${RESUME:-0}"
START_CARLA="${START_CARLA:-1}"
CARLA_RENDER_OFFSCREEN="${CARLA_RENDER_OFFSCREEN:-1}"
CARLA_STARTUP_TIMEOUT="${CARLA_STARTUP_TIMEOUT:-120}"

ROUTES="${ROUTES:-$REPO_ROOT/leaderboard/data/routes_validation.xml}"
ROUTES_SUBSET="${ROUTES_SUBSET:-}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/eval_results/leaderboard}"
CHECKPOINT_ENDPOINT="${CHECKPOINT_ENDPOINT:-$RESULT_DIR/full_res.json}"
DEBUG_CHECKPOINT_ENDPOINT="${DEBUG_CHECKPOINT_ENDPOINT:-$RESULT_DIR/full_live.txt}"
SAVE_PATH="${SAVE_PATH:-$RESULT_DIR/viz/}"
SAVE_PATH="${SAVE_PATH%/}/"
RUN_TAG="${RUN_TAG:-leaderboard_full}"
CARLA_LOG="${CARLA_LOG:-$RESULT_DIR/carla.log}"

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
export SCENARIO_RUNNER_ROOT="$REPO_ROOT/scenario_runner"
export LEADERBOARD_ROOT="$REPO_ROOT/leaderboard"
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

CARLA_PID=""
CARLA_HAS_OWN_GROUP=0
cleanup() {
  if [[ -z "$CARLA_PID" ]] || ! kill -0 "$CARLA_PID" >/dev/null 2>&1; then
    return
  fi

  echo "Stopping CARLA (PID $CARLA_PID)..."
  if [[ "$CARLA_HAS_OWN_GROUP" == "1" ]]; then
    kill -TERM -- "-$CARLA_PID" >/dev/null 2>&1 || true
  else
    kill -TERM "$CARLA_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$START_CARLA" == "1" ]]; then
  echo "Starting CARLA on port $PORT (graphics adapter $CARLA_GRAPHICS_ADAPTER)..."
  CARLA_ARGS=(
    -nosound
    -carla-rpc-port="$PORT"
    -graphicsadapter="$CARLA_GRAPHICS_ADAPTER"
  )
  if [[ "$CARLA_RENDER_OFFSCREEN" == "1" ]]; then
    CARLA_ARGS=(-RenderOffScreen "${CARLA_ARGS[@]}")
  else
    CARLA_ARGS=(
      -windowed
      -ResX="${CARLA_WINDOW_WIDTH:-1280}"
      -ResY="${CARLA_WINDOW_HEIGHT:-720}"
      "${CARLA_ARGS[@]}"
    )
  fi
  if command -v setsid >/dev/null 2>&1; then
    setsid "$CARLA_ROOT/CarlaUE4.sh" "${CARLA_ARGS[@]}" \
      >"$CARLA_LOG" 2>&1 &
    CARLA_HAS_OWN_GROUP=1
  else
    "$CARLA_ROOT/CarlaUE4.sh" "${CARLA_ARGS[@]}" \
      >"$CARLA_LOG" 2>&1 &
  fi
  CARLA_PID=$!

  CARLA_READY=0
  for ((elapsed = 0; elapsed < CARLA_STARTUP_TIMEOUT; elapsed += 2)); do
    if ! kill -0 "$CARLA_PID" >/dev/null 2>&1; then
      echo "CARLA exited before becoming ready. Last log lines:" >&2
      tail -n 40 "$CARLA_LOG" >&2 || true
      exit 1
    fi

    if "$PYTHON_BIN" -c "import carla; c=carla.Client('localhost', $PORT); c.set_timeout(2.0); c.get_world()" >/dev/null 2>&1; then
      CARLA_READY=1
      break
    fi
    sleep 2
  done

  if [[ "$CARLA_READY" != "1" ]]; then
    echo "CARLA did not become ready within ${CARLA_STARTUP_TIMEOUT}s. Last log lines:" >&2
    tail -n 40 "$CARLA_LOG" >&2 || true
    exit 1
  fi
fi

EVALUATOR_ARGS=(
  "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py"
  --routes="$ROUTES"
  --repetitions=1
  --track=SENSORS
  --checkpoint="$CHECKPOINT_ENDPOINT"
  --debug-checkpoint="$DEBUG_CHECKPOINT_ENDPOINT"
  --agent="$REPO_ROOT/team_code/agent_simlingo.py"
  --agent-config="${CHECKPOINT}+${RUN_TAG}"
  --traffic-manager-seed="$TRAFFIC_MANAGER_SEED"
  --port="$PORT"
  --traffic-manager-port="$TM_PORT"
  --timeout="$TIMEOUT"
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
