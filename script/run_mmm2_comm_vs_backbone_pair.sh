#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ENV_CONFIG="${1:-sc2}"
STEP="${2:?Usage: $0 [env_config] <init_load_step> [seed] [backbone_dir]}"
SEED="${3:-1}"
BACKBONE_DIR="${4:-results/models/2026-05-13_16-07-33_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4_mmm2_equalbudget_476836_sc2_MMM2}"

echo "=============================================="
echo " MMM2 Strong-Backbone Pair Run"
echo " Map:           MMM2"
echo " Env config:    $ENV_CONFIG"
echo " Seed:          $SEED"
echo " Backbone dir:  $BACKBONE_DIR"
echo " Init load step:$STEP"
echo " Pair:          v5b communication vs backbone"
echo "=============================================="

echo ""
echo "######## [1/2] v5b communication continuation ########"
zsh "$ROOT_DIR/script/run_v5b_mmm2_midwarmstart_probe.sh" \
  "$ENV_CONFIG" "$STEP" "$SEED" "$BACKBONE_DIR"
echo "######## [1/2] DONE ########"

echo ""
echo "######## [2/2] equal-budget backbone continuation ########"
zsh "$ROOT_DIR/script/run_backbone_mmm2_equalbudget_probe.sh" \
  "$ENV_CONFIG" "$STEP" "$SEED" "$BACKBONE_DIR"
echo "######## [2/2] DONE ########"
