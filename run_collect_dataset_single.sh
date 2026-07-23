#!/usr/bin/env bash
set -e

# Usage:
#   bash run_collect_dataset_single.sh            # choose a route with ROUTE_SEED
#   bash run_collect_dataset_single.sh route.xml  # run a specific route

ROOT="$(cd "$(dirname "$0")" && pwd)"
CARLA_ROOT="${CARLA_ROOT:-$HOME/carla/carla0915}"
ROUTE_SEED="${ROUTE_SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/database/four_view_single}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
GPU_RANK="${GPU_RANK:-0}"
START_CARLA="${START_CARLA:-1}"

# Activate the project environment unless it is already active.
if [[ "${SKIP_CONDA:-0}" != "1" && "${CONDA_DEFAULT_ENV:-}" != "simlingo" ]]; then
    CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    if [[ ! -f "$CONDA_SH" && -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    if [[ ! -f "$CONDA_SH" ]]; then
        echo "Cannot find conda.sh. Set CONDA_SH or use SKIP_CONDA=1." >&2
        exit 1
    fi
    source "$CONDA_SH"
    conda activate simlingo
fi

PYTHON="${PYTHON_BIN:-python}"
export CARLA_ROOT
export WORK_DIR="$ROOT"
export LEADERBOARD_ROOT="$ROOT/leaderboard_autopilot"
export SCENARIO_RUNNER_ROOT="$ROOT/scenario_runner_autopilot"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_RANK}"

PY_TAG="$($PYTHON -c 'import sys; print(f"py{sys.version_info.major}.{sys.version_info.minor}")')"
CARLA_EGG="$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-${PY_TAG}-linux-x86_64.egg"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -e "$CARLA_EGG" ]]; then
    export PYTHONPATH="$CARLA_EGG:$PYTHONPATH"
fi

mkdir -p "$OUTPUT_ROOT/logs"
CARLA_PID=""

cleanup() {
    if [[ -n "$CARLA_PID" ]]; then
        kill "$CARLA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$START_CARLA" == "1" ]]; then
    "$CARLA_ROOT/CarlaUE4.sh" \
        -RenderOffScreen -nosound \
        -carla-rpc-port="$PORT" \
        -graphicsadapter="${CARLA_GRAPHICS_ADAPTER:-$GPU_RANK}" \
        >"$OUTPUT_ROOT/logs/carla.log" 2>&1 &
    CARLA_PID=$!

    echo "Waiting for CARLA on port $PORT..."
    CARLA_READY=0
    for _ in $(seq 1 60); do
        if "$PYTHON" -c "import carla; c=carla.Client('localhost', $PORT); c.set_timeout(2); c.get_world()" 2>/dev/null; then
            CARLA_READY=1
            break
        fi
        sleep 2
    done
    if [[ "$CARLA_READY" != "1" ]]; then
        echo "CARLA did not become ready on port $PORT. See $OUTPUT_ROOT/logs/carla.log" >&2
        exit 1
    fi
fi

ARGS=(
    --route-root "$ROOT/data/simlingo"
    --seed "$ROUTE_SEED"
    --output "$OUTPUT_ROOT"
    --port "$PORT"
    --tm-port "$TM_PORT"
    --tm-seed "${TRAFFIC_MANAGER_SEED:-100}"
)

if [[ $# -eq 1 ]]; then
    ARGS+=(--route "$1")
elif [[ $# -gt 1 ]]; then
    echo "Usage: bash run_collect_dataset_single.sh [route.xml]" >&2
    exit 1
fi

"$PYTHON" "$ROOT/collect_dataset_single.py" "${ARGS[@]}"
