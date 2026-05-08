#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:?Usage: $0 <config> <checkpoint_dir> <load_step> [env] [test_nepisode] [seed]}"
CKPT="${2:?Usage: $0 <config> <checkpoint_dir> <load_step> [env] [test_nepisode] [seed]}"
STEP="${3:-0}"
ENV="${4:-sc2_5m_vs_6m_local}"
TEST_NEPISODE="${5:-128}"
SEED="${6:-1}"

echo "=============================================="
echo " Peer-vs-Local Causal Diagnostic"
echo " Config:        $CONFIG"
echo " Checkpoint:    $CKPT"
echo " Load step:     $STEP"
echo " Env:           $ENV"
echo " Test episodes: $TEST_NEPISODE"
echo " Seed:          $SEED"
echo "=============================================="

python src/main.py \
  --config="$CONFIG" \
  --env-config="$ENV" \
  with seed="$SEED" checkpoint_path="$CKPT" load_step="$STEP" evaluate=True \
       test_nepisode="$TEST_NEPISODE" lr=0.0 critic_lr=0.0 \
       eval_peer_local_diagnostic=True \
  2>&1 | grep -E "Recent Stats|test_return_mean|test_battle_won_mean|test_peer_local_|test_ep_length_mean|test_return_std"

echo ""
echo "Done."
