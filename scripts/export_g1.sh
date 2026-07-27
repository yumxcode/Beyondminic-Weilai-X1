#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MOTION_NPZ [additional play.py arguments]" >&2
  exit 2
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MOTION_FILE="$1"
shift

python "${PROJECT_DIR}/training/whole_body_tracking/scripts/rsl_rl/play.py" \
  --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
  --motion_file "${MOTION_FILE}" \
  --num_envs 1 \
  --export_only \
  "$@"
