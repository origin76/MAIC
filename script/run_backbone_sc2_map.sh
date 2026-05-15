#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4}"
MAP_NAME="${2:?Usage: $0 [config] <map_name> [seed] [env_config]}"
SEED="${3:-1}"
ENV_CONFIG="${4:-sc2}"

RUN_NAME="${CONFIG}_${MAP_NAME}"

echo "=============================================="
echo " SC2 Backbone Run"
echo " Config:        $CONFIG"
echo " Map:           $MAP_NAME"
echo " Seed:          $SEED"
echo " Env config:    $ENV_CONFIG"
echo " Run name:      $RUN_NAME"
echo "=============================================="

python src/main.py \
  --config="$CONFIG" \
  --env-config="$ENV_CONFIG" \
  with \
    env_args.map_name="$MAP_NAME" \
    seed="$SEED" \
    name="$RUN_NAME"
