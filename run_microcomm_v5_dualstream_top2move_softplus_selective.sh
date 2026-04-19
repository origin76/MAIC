#!/bin/zsh

set -euo pipefail

SEED="${1:-1}"
ENV_CONFIG="${2:-sc2_5m_vs_6m_local}"

python src/main.py --config=vanilla_mappo_sc2_5m6m_microcomm_v5_dualstream_top2move_softplus_selective --env-config="${ENV_CONFIG}" with seed="${SEED}"
