#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_2p5m_entropy03_critic5e4"
ENV_CONFIG="${1:-sc2_5m_vs_6m_local}"

echo "========================================"
echo "Backbone 2.5M × 3 seeds"
echo "Config:  $CONFIG"
echo "Env:     $ENV_CONFIG"
echo "========================================"

for SEED in 1 2 3; do
    echo ""
    echo "######## SEED $SEED ########"
    python src/main.py \
        --config="$CONFIG" \
        --env-config="$ENV_CONFIG" \
        with seed="$SEED"
    echo "######## SEED $SEED DONE ########"
done

echo ""
echo "All 3 seeds completed."
