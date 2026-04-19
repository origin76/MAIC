#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ENV_CONFIG="${1:-sc2_5m_vs_6m_local}"

for SEED in 1 2 3; do
  echo "============================================================"
  echo "Running vanilla_mappo_sc2_5m6m_microcomm_v4_targeted_fusion_softuse with seed=${SEED} env=${ENV_CONFIG}"
  echo "============================================================"
  python src/main.py --config=vanilla_mappo_sc2_5m6m_microcomm_v4_targeted_fusion_softuse --env-config="${ENV_CONFIG}" with seed="${SEED}"
done
