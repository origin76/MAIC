#!/usr/bin/env python3
"""Plot full training trajectories for the representative mainline and MAIC."""

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


OURS_BACKBONE_JSON = ROOT / (
    "results/sc2/5m_vs_6m/"
    "vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_1p5m_lrdecay_klstop_relaxactor/"
    "2026-04-13_10-42-26_vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_1p5m_lrdecay_klstop_relaxactor_sc2_5m_vs_6m.json"
)
OURS_CONT_JSON = ROOT / (
    "results/sc2/5m_vs_6m/"
    "vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04/"
    "2026-05-08_23-17-04_vanilla_mappo_sc2_5m6m_microcomm_v6_attack_top2_selective_silencebudget_peer_conflict_margin_leverage_v5b_margin_near_exposure_minw04_sc2_5m_vs_6m.json"
)
OURS_BRANCH_STEP = 1_404_757.0

MAIC_BASE_JSON = ROOT / (
    "results/sc2/5m_vs_6m/"
    "maic_sc2_5m6m_parallel_local_safe_1p5m_relaxactor_compare/"
    "2026-04-16_15-37-15_maic_sc2_5m6m_parallel_local_safe_1p5m_relaxactor_compare_sc2_5m_vs_6m.json"
)
MAIC_CONT_JSON = ROOT / (
    "results/sc2/5m_vs_6m/"
    "maic_sc2_5m6m_parallel_local_safe_2m_relaxactor_compare_continue/"
    "2026-05-17_13-14-24_maic_sc2_5m6m_parallel_local_safe_2m_relaxactor_compare_continue_sc2_5m_vs_6m.json"
)
MAIC_BRANCH_STEP = 1_477_802.0

OUTPUT_PATH = ROOT / "paper/figures/generated/mainline_vs_maic_comparison.png"


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


def build_stitched_curve(
    prefix_json: Path,
    cont_json: Path,
    branch_step: float,
    cont_x_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    prefix_x_raw, prefix_y_raw = load_curve(prefix_json)
    prefix_mask = prefix_x_raw <= branch_step
    prefix_x = prefix_x_raw[prefix_mask]
    prefix_y = prefix_y_raw[prefix_mask]

    cont_x_raw, cont_y = load_curve(cont_json)
    cont_x = cont_x_raw + cont_x_offset

    prefix_y_smooth = smooth(prefix_y, window=5)
    cont_y_smooth = smooth(cont_y, window=5)

    anchor_x = prefix_x[-1]
    anchor_y = prefix_y_smooth[-1]
    cont_x_plot, cont_y_plot = prepend_anchor(cont_x, cont_y_smooth, anchor_x, anchor_y)

    x_all = np.concatenate([prefix_x, cont_x_plot])
    y_all = np.concatenate([prefix_y_smooth, cont_y_plot])
    order = np.argsort(x_all)
    x_all = x_all[order]
    y_all = y_all[order]

    unique_x, unique_indices = np.unique(x_all, return_index=True)
    unique_y = y_all[unique_indices]
    return unique_x, unique_y


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "font.size": 11,
        }
    )

    ours_x, ours_y = build_stitched_curve(
        OURS_BACKBONE_JSON,
        OURS_CONT_JSON,
        OURS_BRANCH_STEP,
        cont_x_offset=OURS_BRANCH_STEP,
    )
    maic_x, maic_y = build_stitched_curve(
        MAIC_BASE_JSON,
        MAIC_CONT_JSON,
        MAIC_BRANCH_STEP,
        cont_x_offset=0.0,
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.plot(ours_x / 1e6, ours_y, color="#e45756", linewidth=2.5, label="Representative two-phase mainline")
    ax.plot(maic_x / 1e6, maic_y, color="#4c78a8", linewidth=2.3, label="MAIC continued to 2.0M")

    ax.axvline(OURS_BRANCH_STEP / 1e6, color="#e45756", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.text(
        OURS_BRANCH_STEP / 1e6,
        0.98,
        "phase-2 start",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.5,
        color="#b23b3a",
    )

    ax.set_title("Representative mainline vs. MAIC on SMAC 5m_vs_6m", fontsize=13, fontweight="bold")
    ax.set_xlabel("Env steps (M)")
    ax.set_ylabel("Test win rate")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
