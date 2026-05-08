#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_signsplit_correction"
ENV="${1:-sc2_5m_vs_6m_local}"
TEST_NEPISODE="${2:-128}"

# Peak-nearest saved checkpoints from the 3-seed sign-split run:
# seed 1: peak test win 0.8125 @ 260902, nearest saved step 250991
# seed 2: peak test win 0.8750 @ 281097, nearest saved step 276282
# seed 3: peak test win 0.8125 @ 421903, nearest saved step 426683
CKPT_1="results/models/2026-05-06_18-59-02_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_signsplit_correction_sc2_5m_vs_6m"
CKPT_2="results/models/2026-05-06_20-23-25_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_signsplit_correction_sc2_5m_vs_6m"
CKPT_3="results/models/2026-05-06_21-47-13_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_signsplit_correction_sc2_5m_vs_6m"

echo "=============================================="
echo " Attack Comm Isolation Eval"
echo " Sign-split peak-nearest checkpoints"
echo " 3 seeds x 4 modes = 12 evaluations"
echo " Config:        $CONFIG"
echo " Env:           $ENV"
echo " Test episodes: $TEST_NEPISODE"
echo "=============================================="

COUNT=0
for SEED in 1 2 3; do
  if [ "$SEED" -eq 1 ]; then CKPT="$CKPT_1"; STEP=250991; PEAK=260902; fi
  if [ "$SEED" -eq 2 ]; then CKPT="$CKPT_2"; STEP=276282; PEAK=281097; fi
  if [ "$SEED" -eq 3 ]; then CKPT="$CKPT_3"; STEP=426683; PEAK=421903; fi

  for MODE in normal gate_open gate_closed no_attack; do
    FLAG=""
    if [ "$MODE" = "gate_open" ]; then FLAG="eval_force_comm_gate_open=True"; fi
    if [ "$MODE" = "gate_closed" ]; then FLAG="eval_force_comm_gate_closed=True"; fi
    if [ "$MODE" = "no_attack" ]; then FLAG="eval_disable_attack_comm=True"; fi

    COUNT=$((COUNT + 1))
    echo ""
    echo ">>> [$COUNT/12] Seed $SEED | $MODE | peak=$PEAK eval_step=$STEP | $(date +%H:%M:%S)"

    TMAX=$((STEP + 2000))
    python src/main.py \
      --config="$CONFIG" \
      --env-config="$ENV" \
      with seed=${SEED} checkpoint_path="${CKPT}" load_step=${STEP} \
           test_interval=100000 test_nepisode=${TEST_NEPISODE} t_max=${TMAX} \
           lr=0.0 critic_lr=0.0 ${FLAG} \
      2>&1 | grep -E "Recent Stats|test_battle_won_mean|test_return_mean|test_ep_length_mean|test_return_std"

    sleep 5
  done
done

echo ""
echo "Done."
