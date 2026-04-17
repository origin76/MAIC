#!/bin/zsh

set -euo pipefail

SEED="${1:-1}"
ENV_CONFIG="${2:-sc2_5m_vs_6m_local}"

CONFIGS=(
  "vanilla_mappo_sc2_5m6m_microcomm_v5_dualstream_top2move_softplus"
  "vanilla_mappo_sc2_5m6m_microcomm_v5_dualstream_top2move_softplus_movegain"
  "vanilla_mappo_sc2_5m6m_microcomm_v5_dualstream_top2move_softplus_distpen"
)

for CONFIG in "${CONFIGS[@]}"; do
  echo "=== Running ${CONFIG} with seed=${SEED} env=${ENV_CONFIG} ==="
  python src/main.py --config="${CONFIG}" --env-config="${ENV_CONFIG}" with seed="${SEED}"
done
