#!/usr/bin/env python3
"""Plot backbone trajectory with communication/no-communication branching on supplementary maps."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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


STATS_RE = re.compile(r"Recent Stats \| t_env:\s*([0-9]+)")
WIN_RE = re.compile(r"test_battle_won_mean:\s*([0-9.]+)")


@dataclass(frozen=True)
class RunCurve:
    log_path: Path
    start_offset: float = 0.0
    take_until: Optional[float] = None
    take_from: Optional[float] = None


@dataclass(frozen=True)
class PanelSpec:
    title: str
    branch_step_abs: float
    backbone_prefix: List[RunCurve]
    comm_curve: RunCurve
    no_comm_curve: RunCurve


PANELS = [
    PanelSpec(
        title="MMM2",
        branch_step_abs=476836.0 + 476449.0,
        backbone_prefix=[
            RunCurve(log_path=ROOT / "results/sacred/365/cout.txt", take_until=476836.0),
            RunCurve(
                log_path=ROOT / "results/sacred/368/cout.txt",
                start_offset=476836.0,
                take_until=476449.0,
            ),
        ],
        comm_curve=RunCurve(
            log_path=ROOT / "results/sacred/369/cout.txt",
            start_offset=476836.0 + 476449.0,
        ),
        no_comm_curve=RunCurve(
            log_path=ROOT / "results/sacred/370/cout.txt",
            start_offset=476836.0 + 476449.0,
        ),
    ),
    PanelSpec(
        title="2c_vs_64zg",
        branch_step_abs=2463279.0,
        backbone_prefix=[
            RunCurve(log_path=ROOT / "results/sacred/371/cout.txt", take_until=2463279.0),
        ],
        comm_curve=RunCurve(log_path=ROOT / "results/sacred/374/cout.txt", start_offset=2463279.0),
        no_comm_curve=RunCurve(log_path=ROOT / "results/sacred/375/cout.txt", start_offset=2463279.0),
    ),
]


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def parse_recent_stats(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    x_vals: list[float] = []
    win_vals: list[float] = []

    for idx, line in enumerate(lines):
        stats_match = STATS_RE.search(line)
        if not stats_match:
            continue
        block = " ".join(lines[idx : idx + 40])
        win_match = WIN_RE.search(block)
        if not win_match:
            continue
        x_vals.append(float(stats_match.group(1)))
        win_vals.append(float(win_match.group(1)))

    if not x_vals:
        raise ValueError(f"No parsed curve from {log_path}")

    return np.asarray(x_vals, dtype=float), np.asarray(win_vals, dtype=float)


def slice_curve(x: np.ndarray, y: np.ndarray, take_from: Optional[float], take_until: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones_like(x, dtype=bool)
    if take_from is not None:
        mask &= x >= take_from
    if take_until is not None:
        mask &= x <= take_until
    return x[mask], y[mask]


def build_curve(spec: RunCurve) -> tuple[np.ndarray, np.ndarray]:
    x, y = parse_recent_stats(spec.log_path)
    x, y = slice_curve(x, y, spec.take_from, spec.take_until)
    x = x + spec.start_offset
    return x, y


def build_prefix(curves: List[RunCurve]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for curve in curves:
        x, y = build_curve(curve)
        xs.append(x)
        ys.append(y)
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    order = np.argsort(x_all)
    return x_all[order], y_all[order]


def prepend_anchor(x: np.ndarray, y: np.ndarray, anchor_x: float, anchor_y: float) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.asarray([anchor_x], dtype=float), np.asarray([anchor_y], dtype=float)
    if np.isclose(x[0], anchor_x) and np.isclose(y[0], anchor_y):
        return x, y
    return np.concatenate(([anchor_x], x)), np.concatenate(([anchor_y], y))


def plot_panel(ax: plt.Axes, spec: PanelSpec) -> None:
    prefix_x, prefix_y = build_prefix(spec.backbone_prefix)
    comm_x, comm_y = build_curve(spec.comm_curve)
    no_comm_x, no_comm_y = build_curve(spec.no_comm_curve)

    branch_x = spec.branch_step_abs / 1e6
    prefix_y_smooth = smooth(prefix_y)
    comm_y_smooth = smooth(comm_y)
    no_comm_y_smooth = smooth(no_comm_y)

    anchor_x = prefix_x[-1]
    anchor_y = prefix_y_smooth[-1]
    comm_x_plot, comm_y_plot = prepend_anchor(comm_x, comm_y_smooth, anchor_x, anchor_y)
    no_comm_x_plot, no_comm_y_plot = prepend_anchor(no_comm_x, no_comm_y_smooth, anchor_x, anchor_y)

    ax.plot(prefix_x / 1e6, prefix_y_smooth, color="#888888", linewidth=2.3, label="共同主干轨迹")
    ax.plot(comm_x_plot / 1e6, comm_y_plot, color="#e45756", linewidth=2.2, label="通信续跑")
    ax.plot(no_comm_x_plot / 1e6, no_comm_y_plot, color="#4c78a8", linewidth=2.2, label="无通信续跑")

    ax.scatter([branch_x], [anchor_y], color="#222222", s=22, zorder=5)
    ax.axvline(branch_x, color="#222222", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.text(branch_x, ax.get_ylim()[1] * 0.98, "分叉点", fontsize=8.5, ha="center", va="top", color="#222222")

    ax.set_title(spec.title, fontsize=12, fontweight="bold")
    ax.set_ylabel("测试胜率")
    ax.set_xlabel("环境步数（百万）")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_ylim(bottom=0.0)


def main() -> None:
    output_path = ROOT / "paper/figures/generated/cross_map_equalbudget_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "PingFang SC",
                "Hiragino Sans GB",
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=False)

    for ax, spec in zip(axes, PANELS):
        plot_panel(ax, spec)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "跨地图等预算分叉对照：共同主干轨迹与后续续跑结果",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main()
