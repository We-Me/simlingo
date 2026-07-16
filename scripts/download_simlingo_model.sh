#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_simlingo_model.sh

Downloads only the files required for closed-loop evaluation:
  pretrained/simlingo/.hydra/config.yaml
  pretrained/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt

Useful overrides:
  CONDA_ENV=simlingo          Conda environment created from environment.yaml.
  CONDA_SH=/path/conda.sh     Conda shell hook if it cannot be auto-detected.
  DOWNLOAD_DIR=/path/to/root  Download root; defaults to <repo>/pretrained.
  HF_ENDPOINT=https://...     Override the temporary Hugging Face mirror.
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

# Temporary for this script and its child download process.
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

source "$CONDA_SH"
conda activate "$CONDA_ENV"

if command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=(huggingface-cli download)
elif command -v hf >/dev/null 2>&1; then
  HF_CLI=(hf download)
else
  echo "Hugging Face CLI was not found in Conda environment '$CONDA_ENV'." >&2
  echo "Install it with: pip install huggingface-hub" >&2
  exit 2
fi

DOWNLOAD_DIR="${DOWNLOAD_DIR:-$REPO_ROOT/pretrained}"
MODEL_CONFIG="$DOWNLOAD_DIR/simlingo/.hydra/config.yaml"
CHECKPOINT="$DOWNLOAD_DIR/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"

mkdir -p "$DOWNLOAD_DIR"

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "Conda environment=$CONDA_ENV"
echo "Download directory=$DOWNLOAD_DIR"
echo "Downloading the SimLingo evaluation config and checkpoint..."

"${HF_CLI[@]}" RenzKa/simlingo \
  simlingo/.hydra/config.yaml \
  simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt \
  --local-dir "$DOWNLOAD_DIR"

if [[ ! -f "$MODEL_CONFIG" || ! -f "$CHECKPOINT" ]]; then
  echo "Download finished, but one or more required files are missing:" >&2
  echo "  $MODEL_CONFIG" >&2
  echo "  $CHECKPOINT" >&2
  exit 1
fi

echo "SimLingo model is ready."
echo "Checkpoint: $CHECKPOINT"

