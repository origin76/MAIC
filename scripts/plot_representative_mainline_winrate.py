#!/usr/bin/env python3
"""Plot the stitched representative mainline win-rate trajectory."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = Path("/tmp/maic_matplotlib_cache")
XDG_CACHE_DIR = Path("/tmp/maic_xdg_cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpl_font_utils import apply_paper_font_rcparams


BACKBONE_JSON = ROOT / (
    "results/sc2/5m_vs_6m/"
    "vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_1p5m_lrdecay_klstop_relaxactor/"
    "2026-04-13_10-42-26_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_1p5m_lrdecay_klstop_relaxactor_sc2_5m_vs_6m.json"
)
MAINLINE_JSON = ROOT / (
    "results/sc2/5m_vs_6m/"
    "vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04/"
    "2026-05-08_23-17-04_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04_sc2_5m_vs_6m.json"
)
NO_COMM_JSON = ROOT / (
    "results/sc2/5m_vs_6m/"
    "vanilla_mappo_sc2_5m6m_finetune_control_485k_from_relaxactor1404757_seed2/"
    "2026-05-17_18-40-43_vanilla_mappo_sc2_5m6m_finetune_control_485k_from_relaxactor1404757_seed2_sc2_5m_vs_6m.json"
)
BRANCH_STEP = 1_404_757.0
OUTPUT_PATH = ROOT / "paper/figures/generated/representative_mainline_winrate.png"


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    x = np.asarray(obj["test_battle_won_mean_T"], dtype=float)
    y = np.asarray(obj["test_battle_won_mean"], dtype=float)
    return x, y


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def prepend_anchor(x: np.ndarray, y: np.ndarray, anchor_x: float, anchor_y: float) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.asarray([anchor_x], dtype=float), np.asarray([anchor_y], dtype=float)
    if np.isclose(x[0], anchor_x) and np.isclose(y[0], anchor_y):
        return x, y
    return np.concatenate(([anchor_x], x)), np.concatenate(([anchor_y], y))


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    apply_paper_font_rcparams(
        plt,
        font_size=11,
        figure_dpi=150,
        savefig_dpi=220,
        extra={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        },
    )

    prefix_x_raw, prefix_y_raw = load_curve(BACKBONE_JSON)
    prefix_mask = prefix_x_raw <= BRANCH_STEP
    prefix_x = prefix_x_raw[prefix_mask]
    prefix_y = prefix_y_raw[prefix_mask]

    cont_x_raw, cont_y_raw = load_curve(MAINLINE_JSON)
    cont_x = cont_x_raw + BRANCH_STEP
    cont_y = cont_y_raw

    no_comm_x_raw, no_comm_y_raw = load_curve(NO_COMM_JSON)
    no_comm_x = no_comm_x_raw + BRANCH_STEP
    no_comm_y = no_comm_y_raw

    prefix_y_smooth = smooth(prefix_y, window=5)
    cont_y_smooth = smooth(cont_y, window=5)
    no_comm_y_smooth = smooth(no_comm_y, window=5)

    anchor_x = prefix_x[-1]
    anchor_y = prefix_y_smooth[-1]
    cont_x_plot, cont_y_plot = prepend_anchor(cont_x, cont_y_smooth, anchor_x, anchor_y)
    no_comm_x_plot, no_comm_y_plot = prepend_anchor(no_comm_x, no_comm_y_smooth, anchor_x, anchor_y)

    anchor_x_m = anchor_x / 1e6

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(prefix_x / 1e6, prefix_y_smooth, color="#8f8f8f", linewidth=2.3, label="主干轨迹")
    ax.plot(cont_x_plot / 1e6, cont_y_plot, color="#e45756", linewidth=2.4, label="通信续跑")
    ax.plot(
        no_comm_x_plot / 1e6,
        no_comm_y_plot,
        color="#4c78a8",
        linewidth=2.1,
        linestyle="--",
        label="无通信续跑",
    )

    ax.scatter([anchor_x_m], [anchor_y], color="#222222", s=24, zorder=5)
    ax.axvline(anchor_x_m, color="#222222", linestyle="--", linewidth=1.1, alpha=0.75)
    ax.text(anchor_x_m, 0.98, "热启动分叉点", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, color="#222222")

    ax.set_xlabel("环境步数（百万）")
    ax.set_ylabel("测试胜率")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(left=0.0)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
