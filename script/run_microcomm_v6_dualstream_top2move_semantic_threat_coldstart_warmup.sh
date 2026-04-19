#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
ENV_CONFIG="${2:-sc2_5m_vs_6m_local}"

python src/main.py --config=vanilla_mappo_sc2_5m6m_microcomm_v6_dualstream_top2move_semantic_threat_coldstart_warmup --env-config="${ENV_CONFIG}" with seed="${SEED}"
