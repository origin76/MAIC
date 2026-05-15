#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

MAP_NAME="${1:?Usage: $0 <map_name> <init_load_step> [seed] [backbone_dir] [env_config] [comm_config] [backbone_config]}"
STEP="${2:?Usage: $0 <map_name> <init_load_step> [seed] [backbone_dir] [env_config] [comm_config] [backbone_config]}"
SEED="${3:-1}"
BACKBONE_DIR="${4:?Usage: $0 <map_name> <init_load_step> [seed] [backbone_dir] [env_config] [comm_config] [backbone_config]}"
ENV_CONFIG="${5:-sc2}"
COMM_CONFIG="${6:-vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04}"
BACKBONE_CONFIG="${7:-vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4}"

COMM_RUN_NAME="${COMM_CONFIG}_${MAP_NAME}_midwarm_${STEP}"
BACKBONE_RUN_NAME="${BACKBONE_CONFIG}_${MAP_NAME}_equalbudget_${STEP}"

echo "=============================================="
echo " SC2 Comm-vs-Backbone Pair Run"
echo " Map:             $MAP_NAME"
echo " Seed:            $SEED"
echo " Backbone dir:    $BACKBONE_DIR"
echo " Init load step:  $STEP"
echo " Env config:      $ENV_CONFIG"
echo " Comm config:     $COMM_CONFIG"
echo " Backbone config: $BACKBONE_CONFIG"
echo "=============================================="

echo ""
echo "######## [1/2] communication continuation ########"
python src/main.py \
  --config="$COMM_CONFIG" \
  --env-config="$ENV_CONFIG" \
  with \
    env_args.map_name="$MAP_NAME" \
    seed="$SEED" \
    init_checkpoint_path="$BACKBONE_DIR" \
    init_load_step="$STEP" \
    init_test_nepisode=0 \
    name="$COMM_RUN_NAME"
echo "######## [1/2] DONE ########"

echo ""
echo "######## [2/2] equal-budget backbone continuation ########"
python src/main.py \
  --config="$BACKBONE_CONFIG" \
  --env-config="$ENV_CONFIG" \
  with \
    env_args.map_name="$MAP_NAME" \
    seed="$SEED" \
    t_max=500000 \
    init_checkpoint_path="$BACKBONE_DIR" \
    init_load_step="$STEP" \
    init_test_nepisode=0 \
    name="$BACKBONE_RUN_NAME"
echo "######## [2/2] DONE ########"
