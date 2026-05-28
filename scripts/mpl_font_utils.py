#!/usr/bin/env python3
"""Shared matplotlib font helpers for paper figures."""

from __future__ import annotations

from matplotlib import font_manager as fm


CN_FONT_CANDIDATES = [
    "Songti SC",
    "STSong",
    "PingFang SC",
    "Hiragino Sans GB",
    "Microsoft YaHei",
    "Arial Unicode MS",
]

EN_SERIF_CANDIDATES = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "DejaVu Serif",
]

EN_SANS_CANDIDATES = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
]


def _pick_first_available_font(candidates: list[str]) -> str | None:
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return None


def build_paper_rcparams(
    *,
    font_size: float = 11,
    figure_dpi: int | None = None,
    savefig_dpi: int | None = None,
) -> dict[str, object]:
    cn_font = _pick_first_available_font(CN_FONT_CANDIDATES) or "DejaVu Sans"
    en_serif = _pick_first_available_font(EN_SERIF_CANDIDATES) or "DejaVu Serif"
    en_sans = _pick_first_available_font(EN_SANS_CANDIDATES) or "DejaVu Sans"

    rcparams: dict[str, object] = {
        "font.size": font_size,
        "font.family": [cn_font, en_serif, en_sans],
        "font.serif": [cn_font, en_serif, "DejaVu Serif"],
        "font.sans-serif": [cn_font, en_sans, "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
    if figure_dpi is not None:
        rcparams["figure.dpi"] = figure_dpi
    if savefig_dpi is not None:
        rcparams["savefig.dpi"] = savefig_dpi
    return rcparams


def apply_paper_font_rcparams(
    plt_module,
    *,
    font_size: float = 11,
    figure_dpi: int | None = None,
    savefig_dpi: int | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    rcparams = build_paper_rcparams(
        font_size=font_size,
        figure_dpi=figure_dpi,
        savefig_dpi=savefig_dpi,
    )
    if extra:
        rcparams.update(extra)
    plt_module.rcParams.update(rcparams)
