#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4"
ENV_CONFIG="${1:-sc2}"
STEP="${2:?Usage: $0 [env_config] <init_load_step> [seed] [backbone_dir]}"
SEED="${3:-1}"
BACKBONE_DIR="${4:-results/models/2026-05-12_22-04-48_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4_sc2_MMM2}"

RUN_NAME="${CONFIG}_mmm2_equalbudget_${STEP}"

echo "=============================================="
echo " MMM2 Equal-Budget Backbone Probe"
echo " Config:        $CONFIG"
echo " Env config:    $ENV_CONFIG"
echo " Map:           MMM2"
echo " Seed:          $SEED"
echo " Init dir:      $BACKBONE_DIR"
echo " Init load step:$STEP"
echo " Extra budget:  500000 env steps"
echo " Run name:      $RUN_NAME"
echo "=============================================="

python src/main.py \
  --config="$CONFIG" \
  --env-config="$ENV_CONFIG" \
  with \
    env_args.map_name="MMM2" \
    seed="$SEED" \
    t_max=500000 \
    init_checkpoint_path="$BACKBONE_DIR" \
    init_load_step="$STEP" \
    init_test_nepisode=0 \
    name="$RUN_NAME"
