#!/bin/zsh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber"
ENV="${1:-sc2_5m_vs_6m_local}"

CKPT_1="results/models/2026-05-05_22-42-32_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber_sc2_5m_vs_6m"
CKPT_2="results/models/2026-05-06_00-06-33_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber_sc2_5m_vs_6m"
CKPT_3="results/models/2026-05-06_01-23-50_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_adv_leverage_huber_sc2_5m_vs_6m"

echo "=============================================="
echo " Attack Comm Isolation Eval"
echo " 3 seeds x 4 modes = 12 evaluations"
echo " Config: $CONFIG"
echo " Env:    $ENV"
echo "=============================================="

COUNT=0
for SEED in 1 2 3; do
  if [ $SEED -eq 1 ]; then CKPT="$CKPT_1"; STEP=476785; fi
  if [ $SEED -eq 2 ]; then CKPT="$CKPT_2"; STEP=476926; fi
  if [ $SEED -eq 3 ]; then CKPT="$CKPT_3"; STEP=476667; fi

  for MODE in normal gate_open gate_closed no_attack; do
    FLAG=""
    if [ "$MODE" = "gate_open" ];   then FLAG="eval_force_comm_gate_open=True"; fi
    if [ "$MODE" = "gate_closed" ]; then FLAG="eval_force_comm_gate_closed=True"; fi
    if [ "$MODE" = "no_attack" ];   then FLAG="eval_disable_attack_comm=True"; fi

    COUNT=$((COUNT + 1))
    echo ""
    echo ">>> [$COUNT/12] Seed $SEED | $MODE | $(date +%H:%M:%S)"

    TMAX=$((STEP + 2000))
    python src/main.py \
      --config="$CONFIG" \
      --env-config="$ENV" \
      with seed=${SEED} checkpoint_path="${CKPT}" load_step=${STEP} \
           test_interval=100000 test_nepisode=128 t_max=${TMAX} lr=0.0 critic_lr=0.0 ${FLAG} \
      2>&1 | grep -E "test_battle_won_mean|test_return_mean|test_ep_length_mean|Recent Stats"

    sleep 5
  done
done

echo ""
echo "Done."
