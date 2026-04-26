#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
ENV_CONFIG="${2:-sc2_5m_vs_6m_local}"

python src/main.py --config=vanilla_mappo_sc2_5m6m_microcomm_v7_igg_move_selective_softgain_robust_g3b_attack_sharper_move_lighter --env-config="${ENV_CONFIG}" with seed="${SEED}"
