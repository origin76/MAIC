#!/usr/bin/env python3
"""Batch plotting for the 2026-04-15 stage report figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "current_stage_report_2026_04_15"
MPL_CACHE_DIR = Path("/tmp/maic_matplotlib_cache")
XDG_CACHE_DIR = Path("/tmp/maic_xdg_cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FileCurveSpec:
    label: str
    path: Path
    family: str = ""


@dataclass(frozen=True)
class DirGroupSpec:
    label: str
    directory: Path
    family: str


@dataclass(frozen=True)
class Join1Curve:
    spec: FileCurveSpec
    x: np.ndarray
    y: np.ndarray


JOIN1_SPECS = [
    FileCurveSpec(
        label="QMIX",
        path=ROOT
        / "results/join1/qmix_join1_easy_0p5m/2026-05-16_00-56-40_qmix_join1_easy_0p5m_join1.json",
        family="join1",
    ),
    FileCurveSpec(
        label="Sparse MAPPO (Budgeted)",
        path=ROOT
        / "results/join1/budgeted_sparse_mappo/2026-04-02_16-39-47_budgeted_sparse_mappo_join1.json",
        family="join1",
    ),
    FileCurveSpec(
        label="MAIC Parallel",
        path=ROOT
        / "results/join1/maic_parallel_join1_easy_0p5m/2026-05-16_01-21-58_maic_parallel_join1_easy_0p5m_join1.json",
        family="join1",
    ),
    FileCurveSpec(
        label="MAIC Parallel Tuned",
        path=ROOT
        / "results/join1/maic_parallel_join1_tuned_join1_easy_0p5m/2026-05-16_01-31-17_maic_parallel_join1_tuned_join1_easy_0p5m_join1.json",
        family="join1",
    ),
]


BASELINE_GROUPS = [
    DirGroupSpec(
        label="Officialish RelaxActor",
        directory=ROOT
        / "results/sc2/5m_vs_6m/"
        / "vanilla_mappo_sc2_5m6m_agentwise_centralized_semistable_officialish_1p5m_lrdecay_klstop_relaxactor",
        family="baseline",
    ),
    DirGroupSpec(
        label="Warm-start Control 300k",
        directory=ROOT
        / "results/sc2/5m_vs_6m/"
        / "vanilla_mappo_sc2_5m6m_finetune_control_300k_from_relaxactor1404757",
        family="baseline",
    ),
]


BACKBONE_WARMSTART_INIT_STEP = 1_404_757.0


VERSION_GROUPS = [
    DirGroupSpec(
        label="v1_detach",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_adapter_v1_detach",
        family="v1",
    ),
    DirGroupSpec(
        label="v1_end2end",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_adapter_v1_end2end",
        family="v1",
    ),
    DirGroupSpec(
        label="v2_sharp",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_adapter_v2_sharp",
        family="v2",
    ),
    DirGroupSpec(
        label="v2_fastcomm",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_adapter_v2_sharp_fastcomm",
        family="v2",
    ),
    DirGroupSpec(
        label="v2_sharp_soft",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_adapter_v2_sharp_soft",
        family="v2",
    ),
    DirGroupSpec(
        label="v3_detach",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_adapter_v3_action_intent_detach",
        family="v3",
    ),
    DirGroupSpec(
        label="v3_end2end",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_adapter_v3_action_intent_end2end",
        family="v3",
    ),
    DirGroupSpec(
        label="v3_gatefloor",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_adapter_v3_action_intent_end2end_gatefloor",
        family="v3",
    ),
    DirGroupSpec(
        label="v3_gain",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_adapter_v3_action_intent_end2end_gain",
        family="v3",
    ),
    DirGroupSpec(
        label="v3_fusionboost",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_adapter_v3_action_intent_end2end_fusionboost",
        family="v3",
    ),
    DirGroupSpec(
        label="v4_base",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_v4_targeted_fusion_base",
        family="v4",
    ),
    DirGroupSpec(
        label="v4_softuse",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_v4_targeted_fusion_softuse",
        family="v4",
    ),
    DirGroupSpec(
        label="v5_base",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_v5_dualstream_targeted_fusion_base",
        family="v5",
    ),
    DirGroupSpec(
        label="v5_top1move",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_v5_dualstream_top1move",
        family="v5",
    ),
    DirGroupSpec(
        label="v5_top1move_softplus",
        directory=ROOT
        / "results/sc2/5m_vs_6m/vanilla_mappo_sc2_5m6m_microcomm_v5_dualstream_top1move_softplus",
        family="v5",
    ),
]


REPRESENTATIVE_CURVES = [
    ("v1_detach", 1),
    ("v2_sharp_soft", 2),
    ("v3_gatefloor", 2),
    ("v4_base", 1),
    ("v5_top1move", 1),
]


FAMILY_COLORS = {
    "baseline": "#4c78a8",
    "join1": "#72b7b2",
    "v1": "#9c755f",
    "v2": "#f58518",
    "v3": "#54a24b",
    "v4": "#e45756",
    "v5": "#b279a2",
}


FAMILY_DISPLAY_NAMES = {
    "v1": "Minimal Communication",
    "v2": "Sharpened-Routing Communication",
    "v3": "Action-Intention Sharing",
    "v4": "Attack-Subspace Fusion",
    "v5": "Attack-Move Dual-Stream Communication",
}


CONFIG_DISPLAY_NAMES = {
    "v1_detach": "Minimal Communication (Detach)",
    "v1_end2end": "Minimal Communication (End-to-End)",
    "v2_sharp": "Sharpened Routing Communication",
    "v2_fastcomm": "Sharpened Routing + FastComm",
    "v2_sharp_soft": "Soft Sharpened Routing",
    "v3_detach": "Action Intention Sharing (Detach)",
    "v3_end2end": "Action Intention Sharing (End-to-End)",
    "v3_gatefloor": "Action Intention Sharing + Gate Floor",
    "v3_gain": "Action Intention Sharing + Gain Boost",
    "v3_fusionboost": "Action Intention Sharing + Fusion Boost",
    "v4_base": "Attack-Subspace Fusion",
    "v4_softuse": "Attack-Subspace Fusion (Soft Use)",
    "v5_base": "Attack-Move Dual-Stream Communication",
    "v5_top1move": "Dual-Stream + Top-1 Move Routing",
    "v5_top1move_softplus": "Dual-Stream + Top-1 Move + Softplus",
}


JOIN1_MAX_T_ENV = 500_000.0
JOIN1_SMOOTH_WINDOW = 5


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titleweight": "bold",
    }
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scalar_curve(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    data = load_json(path)
    x_key = f"{key}_T"
    if x_key not in data or key not in data:
        raise KeyError(f"Missing curve '{key}' in {path}")
    x = np.asarray(data[x_key], dtype=float)
    y = np.asarray(data[key], dtype=float)
    size = min(len(x), len(y))
    x = x[:size]
    y = y[:size]
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def truncate_curve(x: np.ndarray, y: np.ndarray, max_t: float) -> tuple[np.ndarray, np.ndarray]:
    mask = x <= max_t
    return x[mask], y[mask]


def smooth_curve(y: np.ndarray, window: int) -> np.ndarray:
    if y.size == 0 or window <= 1:
        return y
    window = min(window, y.size)
    kernel = np.ones(window, dtype=float)
    padded = np.convolve(y, kernel, mode="same")
    counts = np.convolve(np.ones_like(y, dtype=float), kernel, mode="same")
    return padded / np.maximum(counts, 1.0)


def prepare_join1_curve(spec: FileCurveSpec) -> Join1Curve:
    x, y = load_scalar_curve(spec.path, "test_battle_won_mean")
    x, y = truncate_curve(x, y, JOIN1_MAX_T_ENV)
    y = smooth_curve(y, JOIN1_SMOOTH_WINDOW)
    return Join1Curve(spec=spec, x=x, y=y)


def load_join1_curves() -> list[Join1Curve]:
    return [prepare_join1_curve(spec) for spec in JOIN1_SPECS]


def annotate_curve_endpoint(
    ax: plt.Axes,
    x_scaled: np.ndarray,
    y: np.ndarray,
    y_offset_points: float,
) -> None:
    if x_scaled.size == 0 or y.size == 0:
        return
    ax.scatter([x_scaled[-1]], [y[-1]], s=28, zorder=3)
    ax.annotate(
        f"{y[-1]:.3f}",
        (x_scaled[-1], y[-1]),
        textcoords="offset points",
        xytext=(6, y_offset_points),
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
    )


def compute_endpoint_label_offsets(
    curves: list[Join1Curve],
    cluster_gap: float = 0.035,
    spacing_points: float = 12.0,
) -> dict[int, float]:
    endpoints = [(idx, float(curve.y[-1])) for idx, curve in enumerate(curves) if curve.y.size > 0]
    endpoints.sort(key=lambda item: item[1])
    offsets: dict[int, float] = {idx: 0.0 for idx, _ in endpoints}
    cluster: list[tuple[int, float]] = []

    def assign_cluster(current_cluster: list[tuple[int, float]]) -> None:
        if not current_cluster:
            return
        start = -0.5 * spacing_points * (len(current_cluster) - 1)
        for pos, (curve_idx, _) in enumerate(current_cluster):
            offsets[curve_idx] = start + pos * spacing_points

    for item in endpoints:
        if not cluster:
            cluster = [item]
            continue
        if abs(item[1] - cluster[-1][1]) <= cluster_gap:
            cluster.append(item)
        else:
            assign_cluster(cluster)
            cluster = [item]
    assign_cluster(cluster)
    return offsets


def draw_join1_curves(ax: plt.Axes, curves: list[Join1Curve]) -> None:
    all_x = [t for curve in curves for t in curve.x.tolist()]
    format_time_axis(ax, all_x)
    scale = getattr(ax, "_codex_scale", 1.0)
    label_offsets = compute_endpoint_label_offsets(curves)
    for idx, curve in enumerate(curves):
        x_scaled = curve.x / scale
        ax.plot(x_scaled, curve.y, linewidth=2.4, label=curve.spec.label)
        annotate_curve_endpoint(ax, x_scaled, curve.y, label_offsets.get(idx, 0.0))
    ax.set_title("Join1 Prototype Environment")
    ax.set_ylabel("Test win rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.0, JOIN1_MAX_T_ENV / scale)
    ax.legend(frameon=False, ncol=2, title="Method")


def load_seed(path: Path) -> int | None:
    data = load_json(path)
    seed = data.get("seed")
    return int(seed) if seed is not None else None


def iter_json_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.json") if p.is_file())


def curve_stats(y: np.ndarray) -> dict[str, float]:
    if y.size == 0:
        return {"peak": math.nan, "final": math.nan, "last5": math.nan}
    tail = y[-min(5, y.size) :]
    return {
        "peak": float(np.max(y)),
        "final": float(y[-1]),
        "last5": float(np.mean(tail)),
    }


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_time_axis(ax: plt.Axes, values: Iterable[float]) -> None:
    max_x = max(values) if values else 0.0
    if max_x >= 1_000_000:
        ax.set_xlabel("t_env (millions)")
        scale = 1_000_000.0
        ax._codex_scale = scale  # type: ignore[attr-defined]
    elif max_x >= 1_000:
        ax.set_xlabel("t_env (thousands)")
        scale = 1_000.0
        ax._codex_scale = scale  # type: ignore[attr-defined]
    else:
        ax.set_xlabel("t_env")
        ax._codex_scale = 1.0  # type: ignore[attr-defined]


def scaled_x(ax: plt.Axes, x: np.ndarray) -> np.ndarray:
    scale = getattr(ax, "_codex_scale", 1.0)
    return x / scale


def interpolate_mean_std(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not curves:
        return np.asarray([]), np.asarray([]), np.asarray([])
    xs = np.unique(np.concatenate([x for x, _ in curves]))
    ys = []
    for x, y in curves:
        interp = np.interp(xs, x, y, left=np.nan, right=np.nan)
        outside = (xs < x.min()) | (xs > x.max())
        interp[outside] = np.nan
        ys.append(interp)
    arr = np.asarray(ys, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return xs, mean, std


def interpolate_at_x(x: np.ndarray, y: np.ndarray, target_x: float) -> float:
    if x.size == 0:
        return math.nan
    if target_x <= x[0]:
        return float(y[0])
    if target_x >= x[-1]:
        return float(y[-1])
    return float(np.interp(target_x, x, y))


def stitch_warmstart_curve(
    prefix_curve: tuple[np.ndarray, np.ndarray],
    suffix_curve: tuple[np.ndarray, np.ndarray],
    init_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    prefix_x, prefix_y = prefix_curve
    suffix_x, suffix_y = suffix_curve

    prefix_mask = prefix_x < init_step
    stitched_x = prefix_x[prefix_mask]
    stitched_y = prefix_y[prefix_mask]

    bridge_y = interpolate_at_x(prefix_x, prefix_y, init_step)
    stitched_x = np.concatenate([stitched_x, np.asarray([init_step], dtype=float)])
    stitched_y = np.concatenate([stitched_y, np.asarray([bridge_y], dtype=float)])

    shifted_suffix_x = suffix_x + init_step
    suffix_keep = shifted_suffix_x > init_step
    stitched_x = np.concatenate([stitched_x, shifted_suffix_x[suffix_keep]])
    stitched_y = np.concatenate([stitched_y, suffix_y[suffix_keep]])
    return stitched_x, stitched_y


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def family_display_name(family: str) -> str:
    return FAMILY_DISPLAY_NAMES.get(family, family)


def config_display_name(label: str) -> str:
    return CONFIG_DISPLAY_NAMES.get(label, label)


def plot_join1(output_dir: Path) -> None:
    curves = load_join1_curves()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    draw_join1_curves(ax, curves)
    save_figure(fig, output_dir / "join1_test_winrate_comparison.png")


def plot_baseline(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
    for ax, spec in zip(axes, BASELINE_GROUPS):
        json_paths = iter_json_files(spec.directory)
        curves = []
        all_x = []
        for path in json_paths:
            seed = load_seed(path)
            x, y = load_scalar_curve(path, "test_battle_won_mean")
            curves.append((x, y))
            all_x.extend(x.tolist())
            ax.plot(x, y, linewidth=1.8, alpha=0.7, label=f"seed {seed}")
        format_time_axis(ax, all_x)
        for line in ax.lines:
            line.set_xdata(scaled_x(ax, np.asarray(line.get_xdata())))
        ax.relim()
        ax.autoscale_view()
        mean_x, mean_y, mean_std = interpolate_mean_std(curves)
        ax.plot(
            scaled_x(ax, mean_x),
            mean_y,
            color="black",
            linewidth=2.6,
            label="mean",
        )
        ax.fill_between(
            scaled_x(ax, mean_x),
            mean_y - mean_std,
            mean_y + mean_std,
            color="black",
            alpha=0.12,
            linewidth=0.0,
        )
        ax.set_title(spec.label)
        ax.set_ylim(-0.02, 0.8)
        ax.legend(frameon=False)
    axes[0].set_ylabel("test_battle_won_mean")
    fig.suptitle("Vanilla MAPPO Backbone: Officialish vs Warm-start", y=1.02, fontsize=14, fontweight="bold")
    save_figure(fig, output_dir / "baseline_officialish_vs_warmstart.png")


def plot_mappo_backbone(output_dir: Path) -> None:
    prefix_spec = BASELINE_GROUPS[0]
    suffix_spec = BASELINE_GROUPS[1]
    prefix_by_seed = {load_seed(path): path for path in iter_json_files(prefix_spec.directory)}
    suffix_paths = iter_json_files(suffix_spec.directory)
    curves = []
    all_x = []

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    for path in suffix_paths:
        seed = load_seed(path)
        prefix_path = prefix_by_seed.get(seed)
        if prefix_path is None:
            raise KeyError(f"Missing source backbone curve for seed {seed} in {prefix_spec.directory}")
        prefix_curve = load_scalar_curve(prefix_path, "test_battle_won_mean")
        suffix_curve = load_scalar_curve(path, "test_battle_won_mean")
        x, y = stitch_warmstart_curve(prefix_curve, suffix_curve, BACKBONE_WARMSTART_INIT_STEP)
        curves.append((x, y))
        all_x.extend(x.tolist())
        ax.plot(x, y, linewidth=1.9, alpha=0.72, label=f"Seed {seed}")

    format_time_axis(ax, all_x)
    for line in ax.lines:
        line.set_xdata(scaled_x(ax, np.asarray(line.get_xdata())))
    ax.relim()
    ax.autoscale_view()

    mean_x, mean_y, mean_std = interpolate_mean_std(curves)
    ax.plot(
        scaled_x(ax, mean_x),
        mean_y,
        color="black",
        linewidth=2.8,
        label="Mean",
    )
    ax.fill_between(
        scaled_x(ax, mean_x),
        mean_y - mean_std,
        mean_y + mean_std,
        color="black",
        alpha=0.12,
        linewidth=0.0,
    )
    ax.set_title("No-Communication Warm-Start Control on SMAC 5m_vs_6m")
    ax.set_ylabel("Test win rate")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(right=0.76)
    save_figure(fig, output_dir / "mappo_backbone_winrate.png")


def collect_run_rows(groups: list[DirGroupSpec]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in groups:
        for path in iter_json_files(group.directory):
            x, y = load_scalar_curve(path, "test_battle_won_mean")
            stats = curve_stats(y)
            seed = load_seed(path)
            rows.append(
                {
                    "family": group.family,
                    "family_display": family_display_name(group.family),
                    "config_label": group.label,
                    "config_display": config_display_name(group.label),
                    "seed": seed,
                    "json_path": str(path.relative_to(ROOT)),
                    "peak": stats["peak"],
                    "final": stats["final"],
                    "last5": stats["last5"],
                    "num_points": len(y),
                    "best_t_env": float(x[int(np.argmax(y))]) if len(x) else math.nan,
                }
            )
    return rows


def aggregate_rows(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["family"]), str(row["config_label"]))].append(row)

    aggregated = []
    for (family, config_label), rows in grouped.items():
        peaks = np.asarray([float(row["peak"]) for row in rows], dtype=float)
        finals = np.asarray([float(row["final"]) for row in rows], dtype=float)
        last5s = np.asarray([float(row["last5"]) for row in rows], dtype=float)
        aggregated.append(
            {
                "family": family,
                "family_display": family_display_name(family),
                "config_label": config_label,
                "config_display": config_display_name(config_label),
                "num_runs": len(rows),
                "peak_mean": float(np.mean(peaks)),
                "peak_std": float(np.std(peaks)),
                "final_mean": float(np.mean(finals)),
                "final_std": float(np.std(finals)),
                "last5_mean": float(np.mean(last5s)),
                "last5_std": float(np.std(last5s)),
            }
        )
    family_order = {"v1": 1, "v2": 2, "v3": 3, "v4": 4, "v5": 5}
    aggregated.sort(key=lambda row: (family_order.get(str(row["family"]), 999), str(row["config_label"])))
    return aggregated


def plot_v1_v5_variant_summary(output_dir: Path, agg_rows: list[dict[str, object]]) -> None:
    labels = [str(row["config_display"]) for row in agg_rows]
    families = [str(row["family"]) for row in agg_rows]
    colors = [FAMILY_COLORS.get(family, "#7f7f7f") for family in families]
    y_pos = np.arange(len(labels))
    metric_keys = [("peak_mean", "peak_std", "Peak"), ("final_mean", "final_std", "Final"), ("last5_mean", "last5_std", "Last5")]

    fig, axes = plt.subplots(1, 3, figsize=(19, 9), sharey=True)
    for ax, (mean_key, std_key, title) in zip(axes, metric_keys):
        values = np.asarray([float(row[mean_key]) for row in agg_rows], dtype=float)
        errors = np.asarray([float(row[std_key]) for row in agg_rows], dtype=float)
        ax.barh(y_pos, values, xerr=errors, color=colors, alpha=0.88, error_kw={"elinewidth": 1.0, "capsize": 3})
        ax.set_title(title)
        ax.set_xlim(0.0, 0.8)
        ax.set_xlabel("test_battle_won_mean")
        for idx, value in enumerate(values):
            ax.text(min(value + 0.012, 0.79), idx, f"{value:.3f}", va="center", fontsize=9)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    fig.suptitle("Summary of Five Lightweight Communication Algorithms", y=1.02, fontsize=14, fontweight="bold")
    save_figure(fig, output_dir / "lightweight_comm_algorithm_summary.png")


def plot_v1_v5_family_best(output_dir: Path, agg_rows: list[dict[str, object]]) -> None:
    best_rows: list[dict[str, object]] = []
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in agg_rows:
        by_family[str(row["family"])].append(row)
    for family in ["v1", "v2", "v3", "v4", "v5"]:
        rows = by_family.get(family, [])
        if not rows:
            continue
        best_rows.append(max(rows, key=lambda row: float(row["last5_mean"])))

    labels = [f"{row['family_display']}\n({row['config_display']})" for row in best_rows]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 5.8))
    metrics = [
        ("peak_mean", "Peak", -width),
        ("final_mean", "Final", 0.0),
        ("last5_mean", "Last5", width),
    ]
    for key, label, offset in metrics:
        values = np.asarray([float(row[key]) for row in best_rows], dtype=float)
        ax.bar(x + offset, values, width=width, label=label)
        for idx, value in enumerate(values):
            ax.text(x[idx] + offset, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 0.82)
    ax.set_ylabel("test_battle_won_mean")
    ax.set_title("Best Configuration of Each Communication Algorithm")
    ax.legend(frameon=False)
    save_figure(fig, output_dir / "lightweight_comm_family_best_summary.png")


def build_version_map() -> dict[str, list[Path]]:
    return {group.label: iter_json_files(group.directory) for group in VERSION_GROUPS}


def select_seed_path(paths: list[Path], seed: int) -> Path:
    for path in paths:
        if load_seed(path) == seed:
            return path
    raise FileNotFoundError(f"Missing seed {seed} in {[str(p) for p in paths]}")


def plot_representative_curves(output_dir: Path, version_map: dict[str, list[Path]]) -> None:
    curves = []
    all_x = []
    for label, seed in REPRESENTATIVE_CURVES:
        path = select_seed_path(version_map[label], seed)
        x, y = load_scalar_curve(path, "test_battle_won_mean")
        family = next(group.family for group in VERSION_GROUPS if group.label == label)
        curves.append((label, seed, family, x, y))
        all_x.extend(x.tolist())

    fig, ax = plt.subplots(figsize=(11, 5.8))
    format_time_axis(ax, all_x)
    scale = getattr(ax, "_codex_scale", 1.0)
    for label, seed, family, x, y in curves:
        x_scaled = x / scale
        ax.plot(
            x_scaled,
            y,
            linewidth=2.5,
            label=f"{config_display_name(label)} (seed {seed})",
            color=FAMILY_COLORS.get(family),
        )
        ax.scatter([x_scaled[-1]], [y[-1]], s=26)
    ax.set_title("Representative Runs of Five Lightweight Communication Algorithms")
    ax.set_ylabel("test_battle_won_mean")
    ax.set_ylim(0.0, 0.8)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, output_dir / "lightweight_comm_representative_curves.png")


def plot_diagnostics(output_dir: Path, version_map: dict[str, list[Path]]) -> None:
    diag_specs = [
        ("v4_base", "seed 1", select_seed_path(version_map["v4_base"], 1)),
        ("v5_base", "seed 1", select_seed_path(version_map["v5_base"], 1)),
        ("v5_top1move", "seed 1", select_seed_path(version_map["v5_top1move"], 1)),
        ("v5_top1move_softplus", "mean of seeds 1/2/3", None),
    ]

    attack_keys = [
        ("targeted_mean_attn_entropy", "Mean Attention Entropy"),
        ("targeted_no_comm_prob", "No-Comm Probability"),
        ("targeted_attack_gate_mean", "Attack Gate Mean"),
        ("targeted_message_norm", "Message Norm"),
    ]
    move_keys = [
        ("targeted_move_mean_attn_entropy", "Move Attention Entropy"),
        ("targeted_move_no_comm_prob", "Move No-Comm Probability"),
        ("targeted_move_gate_mean", "Move Gate Mean"),
        ("targeted_move_message_norm", "Move Message Norm"),
    ]

    def get_curves_for_diag(label: str, path: Path | None, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if path is not None:
            try:
                x, y = load_scalar_curve(path, key)
            except KeyError:
                return None
            return x, y, np.zeros_like(y)
        curves = []
        for softplus_path in version_map["v5_top1move_softplus"]:
            try:
                curves.append(load_scalar_curve(softplus_path, key))
            except KeyError:
                continue
        if not curves:
            return None
        return interpolate_mean_std(curves)

    def draw_grid(keys: list[tuple[str, str]], title: str, filename: str) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
        axes = axes.reshape(-1)
        all_x = []
        cached_curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for label, _, path in diag_specs:
            for key, _ in keys:
                result = get_curves_for_diag(label, path, key)
                if result is not None:
                    cached_curves[(label, key)] = result
                    all_x.extend(result[0].tolist())
        for ax, (key, subtitle) in zip(axes, keys):
            format_time_axis(ax, all_x)
            for label, legend_suffix, _ in diag_specs:
                curve = cached_curves.get((label, key))
                if curve is None:
                    continue
                x, y, std = curve
                family = next(group.family for group in VERSION_GROUPS if group.label == label)
                color = FAMILY_COLORS.get(family, "#7f7f7f")
                display_label = config_display_name(label)
                if label == "v5_top1move_softplus":
                    ax.plot(scaled_x(ax, x), y, linewidth=2.4, color=color, label=f"{display_label} ({legend_suffix})")
                    ax.fill_between(scaled_x(ax, x), y - std, y + std, color=color, alpha=0.18, linewidth=0.0)
                else:
                    ax.plot(scaled_x(ax, x), y, linewidth=2.0, color=color, label=f"{display_label} ({legend_suffix})")
            ax.set_title(subtitle)
            ax.set_ylabel("value")
        axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle(title, y=1.01, fontsize=14, fontweight="bold")
        save_figure(fig, output_dir / filename)

    draw_grid(
        attack_keys,
        "Communication Diagnostics: Attack-Side Signals of Attack-Subspace Fusion and Dual-Stream Models",
        "communication_diagnostics_attack.png",
    )
    draw_grid(
        move_keys,
        "Communication Diagnostics: Move-Side Signals of the Dual-Stream Model",
        "communication_diagnostics_move.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stage-report figures from exported TensorBoard JSON files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated figures and CSV summaries (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    mkdir(output_dir)

    plot_join1(output_dir)
    plot_mappo_backbone(output_dir)
    plot_baseline(output_dir)

    version_map = build_version_map()
    run_rows = collect_run_rows(VERSION_GROUPS)
    agg_rows = aggregate_rows(run_rows)

    write_csv(
        output_dir / "lightweight_comm_run_summary.csv",
        run_rows,
        [
            "family",
            "family_display",
            "config_label",
            "config_display",
            "seed",
            "json_path",
            "peak",
            "final",
            "last5",
            "num_points",
            "best_t_env",
        ],
    )
    write_csv(
        output_dir / "lightweight_comm_config_summary.csv",
        agg_rows,
        [
            "family",
            "family_display",
            "config_label",
            "config_display",
            "num_runs",
            "peak_mean",
            "peak_std",
            "final_mean",
            "final_std",
            "last5_mean",
            "last5_std",
        ],
    )

    plot_v1_v5_variant_summary(output_dir, agg_rows)
    plot_v1_v5_family_best(output_dir, agg_rows)
    plot_representative_curves(output_dir, version_map)
    plot_diagnostics(output_dir, version_map)

    generated = sorted(path.relative_to(ROOT) for path in output_dir.iterdir() if path.is_file())
    print("Generated files:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
