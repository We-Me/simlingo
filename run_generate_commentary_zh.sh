#!/usr/bin/env bash
set -euo pipefail

# Generate four-view Commentary labels from a collected CP dataset.
#
# Common overrides:
#   DATASET_ROOT=/path/to/dataset
#   COMMENTARY_OUTPUT_DIRECTORY=/path/to/output
#   WORKERS=1
#   RANDOM_SUBSET_COUNT=-1
#   SKIP_EXISTING=1
#   FILTER_ROUTES_BY_RESULT=0
#   CONDA_ENV=simlingo
#   PYTHON_BIN=python

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/database/four_view_single}"
OUTPUT_DIRECTORY="${COMMENTARY_OUTPUT_DIRECTORY:-$DATASET_ROOT/commentary}"
OUTPUT_EXAMPLES_DIRECTORY="${COMMENTARY_EXAMPLES_DIRECTORY:-$DATASET_ROOT/commentary_examples}"
WORKERS="${WORKERS:-1}"
RANDOM_SUBSET_COUNT="${RANDOM_SUBSET_COUNT:--1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FILTER_ROUTES_BY_RESULT="${FILTER_ROUTES_BY_RESULT:-0}"
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
    --output-directory "$OUTPUT_DIRECTORY"
    --output-examples-directory "$OUTPUT_EXAMPLES_DIRECTORY"
    --random-subset-count "$RANDOM_SUBSET_COUNT"
    --workers "$WORKERS"
)

if [[ "$SKIP_EXISTING" == "1" ]]; then
    ARGS+=(--skip-existing)
else
    ARGS+=(--no-skip-existing)
fi

if [[ "$FILTER_ROUTES_BY_RESULT" == "1" ]]; then
    ARGS+=(--filter-routes-by-result)
else
    ARGS+=(--no-filter-routes-by-result)
fi

echo "Generating Commentary labels"
echo "  dataset: $DATASET_ROOT"
echo "  output:  $OUTPUT_DIRECTORY"

exec "$PYTHON_BIN" \
    dataset_generation/language_labels/commentary/carla_commentary_generator_main_cp.py \
    "${ARGS[@]}" "$@"
