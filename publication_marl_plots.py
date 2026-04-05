"""
Publication-style figures for multi-agent RL convergence (throughput, rewards, energy).
Serif-friendly defaults, colorblind-safe palette; PNG export only.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Okabe–Ito (colorblind-friendly); extended with neutral for many curves
_COLORS: List[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "DejaVu Serif",
                "Nimbus Roman",
                "Bitstream Charter",
            ],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def training_episode_xticks(num_episodes: int) -> List[int]:
    """X-axis tick positions for episode index 1..N (always includes 1 and N)."""
    n = int(num_episodes)
    if n <= 1:
        return [1]
    if n <= 10:
        return list(range(1, n + 1))
    ticks = {1, n}
    for step in (500, 1000, 1500):
        if step < n:
            ticks.add(step)
    for frac in (0.25, 0.5, 0.75):
        t = max(1, min(n, round(n * frac)))
        ticks.add(t)
    return sorted(t for t in ticks if 1 <= t <= n)


def finalize_training_episodes_xaxis(ax, num_episodes: int) -> None:
    ax.set_xlim(1, max(2, int(num_episodes)))
    ax.set_xlabel("Training episodes")
    ax.set_xticks(training_episode_xticks(num_episodes))


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y.astype(np.float64)
    pad = window // 2
    yp = np.pad(y.astype(np.float64), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(yp, kernel, mode="valid")[: len(y)]


def _series_for_results(
    results: Dict[str, Dict[str, List[float]]],
    name: str,
    key: str,
    alt_keys: Tuple[str, ...] = (),
) -> Optional[np.ndarray]:
    d = results.get(name) or {}
    k = key
    if k not in d:
        for a in alt_keys:
            if a in d:
                k = a
                break
        else:
            return None
    arr = np.asarray(d[k], dtype=np.float64)
    return arr if arr.size else None


def _plot_one_metric(
    ax,
    ordered_names: Sequence[str],
    results: Dict[str, Dict[str, List[float]]],
    value_key: str,
    alt_keys: Tuple[str, ...],
    y_transform: Callable[[np.ndarray], np.ndarray],
    ylabel: str,
    title: str,
    proposed_names: Sequence[str],
    episodes: np.ndarray,
    num_episodes: int,
    smoothing_window: int,
    colors: Dict[str, str],
    n_curves: int,
    faint_alpha: float = 0.12,
) -> None:
    ylabel_final = ylabel
    for name in ordered_names:
        raw = _series_for_results(results, name, value_key, alt_keys)
        if raw is None:
            continue
        y_plot = y_transform(raw)
        color = colors[name]
        is_prop = name in proposed_names
        lw = 2.4 if is_prop else 1.6
        z = 3 if is_prop else 2
        ls = "-" if is_prop else "-"
        ax.plot(episodes, y_plot, color=color, alpha=faint_alpha, linewidth=0.35, zorder=1)
        sm = _smooth(y_plot, smoothing_window)
        label = name.replace("_", " ")
        ax.plot(
            episodes,
            sm,
            color=color,
            linestyle=ls,
            linewidth=lw,
            label=label,
            zorder=z,
        )
        ylabel_final = ylabel
    finalize_training_episodes_xaxis(ax, num_episodes)
    ax.set_ylabel(ylabel_final)
    ax.set_title(title)
    ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="0.4",
        loc="best",
        ncol=1 if n_curves <= 6 else 2,
    )


def _all_have_energy(
    results: Dict[str, Dict[str, List[float]]], ordered_names: Sequence[str]
) -> bool:
    any_plotted = False
    for n in ordered_names:
        if n not in results:
            continue
        any_plotted = True
        s = _series_for_results(
            results, n, "energy_efficiencies", ("energy_efficiencies_mbit_per_j",)
        )
        if s is None:
            return False
    return any_plotted


def plot_convergence_publication(
    results: Dict[str, Dict[str, List[float]]],
    ordered_names: Sequence[str],
    num_episodes: int,
    proposed_names: Sequence[str],
    out_prefix: str,
    smoothing_window: Optional[int] = None,
) -> None:
    """
    Throughput + rewards (2×1) ``convergence_combined``; optional energy row if all curves
    include ``energy_efficiencies`` (or checkpoint alias ``energy_efficiencies_mbit_per_j``).

    X-axis uses episode indices **1 … N** so the final tick shows **N** (e.g. 2000).
    Writes PNG only (no PDF).
    """
    apply_publication_style()
    if smoothing_window is None:
        smoothing_window = max(11, min(151, max(5, num_episodes // 25)))

    # Episode count on axis: 1 .. N (not 0 .. N-1)
    episodes = np.arange(1, num_episodes + 1, dtype=np.float64)
    present = [n for n in ordered_names if n in results]
    n_curves = len(present)
    colors = {name: _COLORS[i % len(_COLORS)] for i, name in enumerate(ordered_names)}

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    def save_fig(fig: plt.Figure, base: str) -> None:
        path = f"{base}.png"
        fig.savefig(path, format="png")
        print(f"Saved {path}")
        plt.close(fig)

    # --- 2×1: throughput + rewards (primary combined figure) ---
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    _plot_one_metric(
        ax1,
        ordered_names,
        results,
        "throughputs",
        (),
        lambda r: r / 1e9,
        "Episode aggregate throughput (Gbit/s)",
        "(a) Network throughput convergence",
        proposed_names,
        episodes,
        num_episodes,
        smoothing_window,
        colors,
        n_curves,
    )
    _plot_one_metric(
        ax2,
        ordered_names,
        results,
        "rewards",
        (),
        lambda r: r,
        "Rewards",
        "(b) Rewards convergence",
        proposed_names,
        episodes,
        num_episodes,
        smoothing_window,
        colors,
        n_curves,
    )
    save_fig(fig2, f"{out_prefix}_combined")

    has_energy = _all_have_energy(results, ordered_names)
    if has_energy:
        fig_e, (axe1, axe2) = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
        _plot_one_metric(
            axe1,
            ordered_names,
            results,
            "energy_efficiencies",
            ("energy_efficiencies_mbit_per_j",),
            lambda r: r,
            "Energy efficiency (Mbit/J)",
            "(c) Energy efficiency convergence",
            proposed_names,
            episodes,
            num_episodes,
            smoothing_window,
            colors,
            n_curves,
        )
        _plot_one_metric(
            axe2,
            ordered_names,
            results,
            "episode_energies_j",
            (),
            lambda r: r / 1e6,
            "Energy used per Episode (MJ)",
            "(d) Energy used per Episode",
            proposed_names,
            episodes,
            num_episodes,
            smoothing_window,
            colors,
            n_curves,
        )
        save_fig(fig_e, f"{out_prefix}_energy_metrics")

        fig4, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
        ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        _plot_one_metric(
            ax_a,
            ordered_names,
            results,
            "throughputs",
            (),
            lambda r: r / 1e9,
            "Episode aggregate throughput (Gbit/s)",
            "(a) Throughput",
            proposed_names,
            episodes,
            num_episodes,
            smoothing_window,
            colors,
            n_curves,
        )
        _plot_one_metric(
            ax_b,
            ordered_names,
            results,
            "rewards",
            (),
            lambda r: r,
            "Rewards",
            "(b) Rewards",
            proposed_names,
            episodes,
            num_episodes,
            smoothing_window,
            colors,
            n_curves,
        )
        _plot_one_metric(
            ax_c,
            ordered_names,
            results,
            "energy_efficiencies",
            ("energy_efficiencies_mbit_per_j",),
            lambda r: r,
            "Energy efficiency (Mbit/J)",
            "(c) Energy efficiency",
            proposed_names,
            episodes,
            num_episodes,
            smoothing_window,
            colors,
            n_curves,
        )
        _plot_one_metric(
            ax_d,
            ordered_names,
            results,
            "episode_energies_j",
            (),
            lambda r: r / 1e6,
            "Energy used per Episode (MJ)",
            "(d) Energy used per Episode",
            proposed_names,
            episodes,
            num_episodes,
            smoothing_window,
            colors,
            n_curves,
        )
        save_fig(fig4, f"{out_prefix}_four_panel")

    singles: List[Tuple[str, Tuple[str, ...], Callable[[np.ndarray], np.ndarray], str, str, str]] = [
        ("throughputs", (), lambda r: r / 1e9, "Episode aggregate throughput (Gbit/s)", "throughput", "Throughput convergence"),
        ("rewards", (), lambda r: r, "Rewards", "rewards", "Rewards convergence"),
    ]
    if has_energy:
        singles.extend(
            [
                (
                    "energy_efficiencies",
                    ("energy_efficiencies_mbit_per_j",),
                    lambda r: r,
                    "Energy efficiency (Mbit/J)",
                    "energy_efficiency",
                    "Energy efficiency convergence",
                ),
                (
                    "episode_energies_j",
                    (),
                    lambda r: r / 1e6,
                    "Energy used per Episode (MJ)",
                    "energy_used_per_episode",
                    "Energy used per Episode convergence",
                ),
            ]
        )

    for key, alts, transform, ylab, suffix, ttl in singles:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
        _plot_one_metric(
            ax,
            ordered_names,
            results,
            key,
            alts,
            transform,
            ylab,
            ttl,
            proposed_names,
            episodes,
            num_episodes,
            smoothing_window,
            colors,
            n_curves,
        )
        save_fig(fig, f"{out_prefix}_{suffix}")
