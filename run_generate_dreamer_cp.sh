#!/usr/bin/env bash
set -euo pipefail

# Generate four-view Dreamer labels from a collected CP dataset.
# Dreamer preserves the original output convention: DATASET_ROOT/data is
# replaced by DATASET_ROOT/<DREAMER_SAVE_FOLDER_NAME>.
#
# Common overrides:
#   DATASET_ROOT=/path/to/dataset
#   DREAMER_SAVE_FOLDER_NAME=dreamer
#   WORKERS=1
#   RANDOM_SUBSET_COUNT=-1
#   OVERWRITE=0
#   FILTER_ROUTES_BY_RESULT=0
#   SAVE_INSTRUCTIONS=1
#   CONDA_ENV=simlingo
#   PYTHON_BIN=python

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/database/four_view_single}"
SAVE_FOLDER_NAME="${DREAMER_SAVE_FOLDER_NAME:-dreamer}"
VIZ_SAVE_PATH="${DREAMER_VIZ_SAVE_PATH:-$ROOT/viz/dreamer_cp}"
WORKERS="${WORKERS:-1}"
RANDOM_SUBSET_COUNT="${RANDOM_SUBSET_COUNT:--1}"
OVERWRITE="${OVERWRITE:-0}"
FILTER_ROUTES_BY_RESULT="${FILTER_ROUTES_BY_RESULT:-0}"
SAVE_INSTRUCTIONS="${SAVE_INSTRUCTIONS:-1}"
CONDA_ENV="${CONDA_ENV:-simlingo}"

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Dataset root does not exist: $DATASET_ROOT" >&2
    exit 1
fi

if [[ "${SKIP_CONDA:-0}" != "1" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    CONDA_SH="${CONDA_SH:-}"
    if [[ -z "$CONDA_SH" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -z "$CONDA_SH" && -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
        echo "Cannot find conda.sh. Set CONDA_SH or use SKIP_CONDA=1." >&2
        exit 1
    fi
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/simlingo_matplotlib}"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT"

ARGS=(
    --data-directory "$DATASET_ROOT"
    --save-folder-name "$SAVE_FOLDER_NAME"
    --viz-save-path "$VIZ_SAVE_PATH"
    --random-subset-count "$RANDOM_SUBSET_COUNT"
    --workers "$WORKERS"
)

if [[ "$OVERWRITE" == "1" ]]; then
    ARGS+=(--overwrite)
else
    ARGS+=(--no-overwrite)
fi

if [[ "$FILTER_ROUTES_BY_RESULT" == "1" ]]; then
    ARGS+=(--filter-routes-by-result)
else
    ARGS+=(--no-filter-routes-by-result)
fi

if [[ "$SAVE_INSTRUCTIONS" == "1" ]]; then
    ARGS+=(--save-instructions)
else
    ARGS+=(--no-save-instructions)
fi

echo "Generating Dreamer labels"
echo "  dataset: $DATASET_ROOT"
echo "  output:  $DATASET_ROOT/$SAVE_FOLDER_NAME"

exec "$PYTHON_BIN" \
    dataset_generation/dreamer_data/dreamer_generator_main_cp.py \
    "${ARGS[@]}" "$@"
