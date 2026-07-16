#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/demo_bench2drive_simlingo.sh

Runs one Bench2Drive route with a visible CARLA window and a SimLingo Pygame
window. The Pygame view shows the camera, predicted paths, controls, prompt,
and language answer.

Useful overrides:
  ROUTE_ID=1711              Route ID from bench2drive220.xml.
  ROUTES=/path/to/routes.xml Route XML to use.
  GPU_RANK=0                 CUDA device and CARLA graphics adapter.
  PORT=10000                 Starting CARLA RPC port.
  RESULT_DIR=/path/to/output Demo output directory.
  CARLA_WINDOW_WIDTH=1280    Initial CARLA window width.
  CARLA_WINDOW_HEIGHT=720    Initial CARLA window height.
  SIMLINGO_VIZ_WIDTH=1280    Initial Pygame window width.
  SIMLINGO_VIZ_HEIGHT=800    Initial Pygame window height.

CARLA_ROOT, CHECKPOINT, CONDA_ENV, CONDA_SH, and HF_ENDPOINT have the same
overrides as scripts/eval_bench2drive_simlingo.sh.
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

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "No graphical display detected. Run this script from the Linux desktop session" >&2
  echo "where CARLA and Pygame windows should be shown." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROUTE_ID="${ROUTE_ID:-1711}"

export ROUTES_SUBSET="$ROUTE_ID"
export RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/eval_results/demo_bench2drive/route_${ROUTE_ID}}"
export CARLA_RENDER_OFFSCREEN=0
export SIMLINGO_VIZ=1
export SIMLINGO_VIZ_TITLE="${SIMLINGO_VIZ_TITLE:-SimLingo - Bench2Drive route ${ROUTE_ID}}"

exec bash "$SCRIPT_DIR/eval_bench2drive_simlingo.sh"
