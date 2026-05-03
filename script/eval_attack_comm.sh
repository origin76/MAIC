#!/bin/zsh
# Usage: bash script/eval_attack_comm.sh <seed> <mode>
# seed: 1 2 3
# mode: normal | gate_open | gate_closed | no_attack

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:?Usage: eval_attack_comm.sh <seed> <mode>}"
MODE="${2:?Usage: eval_attack_comm.sh <seed> <mode>}"

case $SEED in
  1) CKPT="results/models/2026-05-03_00-35-24_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_sc2_5m_vs_6m/477095" ;;
  2) CKPT="results/models/2026-05-03_01-42-38_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_sc2_5m_vs_6m/476786" ;;
  3) CKPT="results/models/2026-05-03_10-26-50_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_sc2_5m_vs_6m/477198" ;;
  *) echo "Unknown seed: $SEED"; exit 1 ;;
esac

OVERRIDES="checkpoint_path=\"${CKPT}\" load_step=0"
case $MODE in
  normal)      ;;
  gate_open)   OVERRIDES="$OVERRIDES eval_force_comm_gate_open=True" ;;
  gate_closed) OVERRIDES="$OVERRIDES eval_force_comm_gate_closed=True" ;;
  no_attack)   OVERRIDES="$OVERRIDES eval_disable_attack_comm=True" ;;
  *) echo "Unknown mode: $MODE"; exit 1 ;;
esac

echo "Seed $SEED | Mode $MODE | $CKPT"
python src/main.py \
  --config=vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_eval \
  --env-config=sc2_5m_vs_6m_local \
  with $OVERRIDES
