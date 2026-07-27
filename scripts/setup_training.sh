#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="${PROJECT_DIR}/training/whole_body_tracking"

python -m pip install -e "${TRAINING_DIR}/source/whole_body_tracking"

echo "BeyondMimic training extension installed."
echo "Expected base environment: Isaac Lab v2.1.0 with its RSL-RL dependencies."
echo "G1 assets are resolved from: ${PROJECT_DIR}/unitree_description"
