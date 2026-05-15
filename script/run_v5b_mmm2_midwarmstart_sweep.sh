#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ENV_CONFIG="${1:-sc2}"
SEED="${2:-1}"
BACKBONE_DIR="${3:-results/models/2026-05-12_22-04-48_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4_sc2_MMM2}"

for STEP in 476836 526897 551924; do
  echo ""
  echo "######## MMM2 MID-WARMSTART @ ${STEP} ########"
  zsh "$ROOT_DIR/script/run_v5b_mmm2_midwarmstart_probe.sh" \
    "$ENV_CONFIG" "$STEP" "$SEED" "$BACKBONE_DIR"
  echo "######## MMM2 MID-WARMSTART @ ${STEP} DONE ########"
done

