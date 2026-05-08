#!/bin/zsh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
ENV_CONFIG="${1:-sc2_5m_vs_6m_local}"
SEED="${2:-1}"
python src/main.py --config=vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_align_msginterp --env-config="${ENV_CONFIG}" with seed="${SEED}"
