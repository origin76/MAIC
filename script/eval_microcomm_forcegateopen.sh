#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Usage:
#   zsh script/eval_microcomm_forcegateopen.sh <alg_config> <checkpoint_dir> [seed] [test_nepisode] [load_step] [env_config]
#
# Example (g2c latest checkpoint):
#   zsh script/eval_microcomm_forcegateopen.sh \
#     vanilla_mappo_sc2_5m6m_microcomm_v7_igg_move_selective_softgain_robust_g2c \
#     results/models/2026-04-22_16-44-05_vanilla_mappo_sc2_5m6m_microcomm_v7_igg_move_selective_softgain_robust_g2c_sc2_5m_vs_6m \
#     1 1024 0 sc2_5m_vs_6m_local

ALG_CONFIG="${1:?alg_config required}"
CKPT_DIR="${2:?checkpoint_dir required}"
SEED="${3:-1}"
TEST_NEPISODE="${4:-1024}"
LOAD_STEP="${5:-0}"
ENV_CONFIG="${6:-sc2_5m_vs_6m_local}"

python src/main.py --config="${ALG_CONFIG}" --env-config="${ENV_CONFIG}" with \
  seed="${SEED}" \
  checkpoint_path="${CKPT_DIR}" \
  load_step="${LOAD_STEP}" \
  evaluate=True \
  test_nepisode="${TEST_NEPISODE}" \
  eval_force_comm_gate_open=True

