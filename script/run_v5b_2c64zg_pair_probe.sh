#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

STEP="${1:?Usage: $0 <init_load_step> [seed] [backbone_dir]}"
SEED="${2:-1}"
BACKBONE_DIR="${3:-results/models/2026-05-13_23-49-03_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4_2c_vs_64zg_sc2_2c_vs_64zg}"

echo "=============================================="
echo " 2c_vs_64zg Pair Probe"
echo " Seed:          $SEED"
echo " Init load step:$STEP"
echo " Backbone dir:  $BACKBONE_DIR"
echo "=============================================="

zsh "$ROOT_DIR/script/run_comm_vs_backbone_sc2_pair.sh" \
  2c_vs_64zg \
  "$STEP" \
  "$SEED" \
  "$BACKBONE_DIR" \
  sc2
