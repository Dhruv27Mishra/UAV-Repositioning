#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import make_interp_spline, PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── style constants ──────────────────────────────────────────────────────────

ALGO_NAMES = [
    "QMIX", "ABQMIX", "IQL", "VDN", "MADDPG",
    "DMTD", "DeepNashQ", "PerformativeMFMARL", "PerformativeMARL",
]

COLORS: Dict[str, str] = {
    "QMIX":               "#e63946",   # red        (~0°)
    "ABQMIX":             "#fb8500",   # orange     (~35°)
    "IQL":                "#ffd60a",   # yellow     (~55°)
    "VDN":                "#38b000",   # green      (~110°)
    "MADDPG":             "#3a86ff",   # blue       (~220°)
    "DMTD":               "#7b2d8b",   # deep violet(~280°)
    "DeepNashQ":          "#f72585",   # hot pink   (~325°)
    "PerformativeMFMARL": "#00b4d8",   # cyan       (~195°, proposed)
    "PerformativeMARL":   "#06d6a0",   # teal/mint  (~165°, proposed)
}

MARKERS: Dict[str, str] = {
    "QMIX":               "o",    # circle
    "ABQMIX":             "s",    # square
    "IQL":                "^",    # triangle up
    "VDN":                "D",    # diamond
    "MADDPG":             "v",    # triangle down
    "DMTD":               "p",    # pentagon
    "DeepNashQ":          "h",    # hexagon
    "PerformativeMFMARL": "*",    # star
    "PerformativeMARL":   "P",    # plus (filled)
}

PROPOSED        = frozenset({"PerformativeMFMARL", "PerformativeMARL"})
LINEWIDTH       = 2.0
BAND_ALPHA      = 0.20   # SE-based CI band (narrower than σ, slightly more opaque)
FIG_SIZE        = (14, 8)
DPI             = 300
SPLINE_K        = 3      # cubic B-spline
N_INTERP_PARAM  = 400    # output points for parameter-sweep spline
N_INTERP_CURVE  = 2000   # output points for training-curve spline
# Moving-average window: 10 % of total episodes gives the reference-image smoothness.
# We clamp to [50, 300] so short quick-mode runs still work.
MA_WINDOW_FRAC  = 0.10   # fraction of total n_episodes
MA_WINDOW_MIN   = 50
MA_WINDOW_MAX   = 300
# Knot density for spline after MA: fewer knots -> smoother curve
SPLINE_KNOTS    = 80     # subsample to this many knots before fitting spline

GRID_STYLE  = dict(color="#cccccc", linestyle="--", linewidth=0.6, alpha=0.8)
LEGEND_KW   = dict(ncol=2, fontsize=9, framealpha=0.92, frameon=True,
                   fancybox=False, edgecolor="0.4", loc="best")

OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "graphs")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "journal_data.json")


# ─── spline helpers ───────────────────────────────────────────────────────────

def _spline_1d(x: np.ndarray, y: np.ndarray,
               n_out: int, k: int = SPLINE_K) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a cubic (or lower) B-spline and return (x_dense, y_dense)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y) & np.isfinite(x)
    x, y  = x[valid], y[valid]
    if len(x) < 2:
        return x, y
    k_use = min(k, len(x) - 1)
    x_new = np.linspace(x[0], x[-1], n_out)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spl   = make_interp_spline(x, y, k=k_use)
        y_new = spl(x_new)
    return x_new, y_new


def _ma_window(n: int) -> int:
    """Adaptive MA window: 10 % of n, clamped to [MIN, MAX]."""
    return int(np.clip(n * MA_WINDOW_FRAC, MA_WINDOW_MIN, MA_WINDOW_MAX))


def _moving_avg(arr: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average — simple, correct, no edge artefacts."""
    arr = np.asarray(arr, dtype=float)
    n   = len(arr)
    hw  = w // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - hw), min(n, i + hw + 1)
        out[i]  = arr[lo:hi].mean()
    return out


def _smooth_training_curve(
    episodes: np.ndarray,
    values:   np.ndarray,
    n_out:    int = N_INTERP_CURVE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Three-stage pipeline matching the reference convergence_rewards.png style:
      1. Adaptive MA (≈10 % of n_episodes) — removes episode-level noise
      2. Subsample to SPLINE_KNOTS evenly-spaced knots — prevents spline overfitting
      3. Cubic B-spline evaluated at n_out dense points — final smooth curve
    Result: no visible jaggies even on 2000-episode runs.
    """
    arr = np.asarray(values, dtype=float)
    n   = len(arr)
    if n == 0:
        return episodes, arr

    w  = _ma_window(n)
    ma = _moving_avg(arr, w)

    n_knots = min(SPLINE_KNOTS, n)
    idx     = np.linspace(0, n - 1, n_knots).astype(int)
    x_k     = episodes[idx].astype(float)
    y_k     = ma[idx]

    x_s, y_s = _spline_1d(x_k, y_k, n_out=n_out)
    return x_s, y_s


def _smooth_param_curve(
    xvals: List[float],
    yvals: List[float],
    n_out: int = N_INTERP_PARAM,
) -> Tuple[np.ndarray, np.ndarray]:
    """PCHIP interpolation — monotone-preserving, no oscillation between points."""
    x = np.array(xvals, dtype=float)
    y = np.array(yvals, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y  = x[valid], y[valid]
    if len(x) < 2:
        return x, y
    x_new = np.linspace(x[0], x[-1], n_out)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_new = PchipInterpolator(x, y)(x_new)
    return x_new, y_new


# ─── style setup ─────────────────────────────────────────────────────────────

def _setup_style() -> None:
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    9,
        "figure.dpi":         DPI,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        "axes.edgecolor":     "#444444",
        "axes.linewidth":     0.8,
    })


def _finalize_ax(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(**GRID_STYLE)
    ax.legend(**LEGEND_KW)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


# ─── shared parameter-sweep plotter (G1–G5) ──────────────────────────────────

def _plot_param_sweep(
    xvals:      List[float],
    mean_key:   str,
    std_key:    str,
    data:       Dict,
    xlabel:     str,
    ylabel:     str,
    title:      str,
    out_name:   str,
    seeds_key:  Optional[str] = None,   # key for per-seed arrays in data[algo]
    x_tick_labels: Optional[List[str]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    x_raw = np.array(xvals, dtype=float)

    for algo in ALGO_NAMES:
        if algo not in data:
            continue
        adat  = data[algo]
        means = np.array(adat[mean_key], dtype=float)
        stds  = np.array(adat[std_key],  dtype=float)
        color = COLORS[algo]
        lw    = LINEWIDTH + (0.4 if algo in PROPOSED else 0.0)
        zo    = 4 if algo in PROPOSED else 2

        # Use standard error (SE = σ/√n_seeds) for CI bands — much tighter than raw σ
        if seeds_key and seeds_key in adat:
            n_seeds = len(adat[seeds_key][0]) if adat[seeds_key] else 5
        else:
            n_seeds = 5
        se = stds / np.sqrt(max(n_seeds, 1))

        x_sm, y_sm = _smooth_param_curve(x_raw, means)
        _,    y_lo = _smooth_param_curve(x_raw, means - se)
        _,    y_hi = _smooth_param_curve(x_raw, means + se)

        marker = MARKERS.get(algo, "o")
        ax.fill_between(x_sm, y_lo, y_hi,
                        color=color, alpha=BAND_ALPHA, linewidth=0, zorder=zo - 1)
        ax.plot(x_sm, y_sm, color=color, linewidth=lw, label=algo, zorder=zo)
        # Markers at actual measured mean values (real eval points, not interpolated)
        ax.plot(x_raw, means, marker=marker, color=color, linestyle="none",
                markersize=5, markeredgecolor="white", markeredgewidth=0.7,
                zorder=zo + 1)

    ax.set_xlim(x_raw[0], x_raw[-1])
    if x_tick_labels:
        ax.set_xticks(x_raw)
        ax.set_xticklabels(x_tick_labels)
    _finalize_ax(ax, xlabel, ylabel, title)
    fig.tight_layout()
    _save(fig, out_name)


# ─── G1: Energy Efficiency vs Discount Factor ────────────────────────────────

def plot_g1(data: Dict) -> None:
    if "g1" not in data:
        print("G1: no data — run run_journal_experiments.py first"); return
    d = data["g1"]
    _plot_param_sweep(
        xvals         = d["gammas"],
        mean_key      = "mean",
        std_key       = "std",
        data          = d,
        xlabel        = "Discount Factor (γ)",
        ylabel        = "Energy Efficiency (Mbit/J)",
        title         = "Energy Efficiency vs. Discount Factor",
        out_name      = "energy_eff_vs_gamma.png",
        x_tick_labels = [str(g) for g in d["gammas"]],
    )


# ─── G2: Handover Rate vs Call Arrival Rate ───────────────────────────────────

def plot_g2(data: Dict) -> None:
    if "g2g3" not in data:
        print("G2: no data"); return
    d = data["g2g3"]
    _plot_param_sweep(
        xvals    = d["call_rates"],
        mean_key = "ho_mean",
        std_key  = "ho_std",
        data     = d,
        xlabel   = "Call Arrival Rate (requests/s)  [VBR / Pareto]",
        ylabel   = "Handover Rate (switches/episode)",
        title    = "Handover Rate vs. Call Arrival Rate",
        out_name = "handover_vs_call_rate.png",
    )


# ─── G3: Energy Efficiency vs Call Arrival Rate ───────────────────────────────

def plot_g3(data: Dict) -> None:
    if "g2g3" not in data:
        print("G3: no data"); return
    d = data["g2g3"]
    _plot_param_sweep(
        xvals    = d["call_rates"],
        mean_key = "ee_mean",
        std_key  = "ee_std",
        data     = d,
        xlabel   = "Call Arrival Rate (requests/s)  [VBR / Pareto]",
        ylabel   = "Energy Efficiency (Mbit/J)",
        title    = "Energy Efficiency vs. Call Arrival Rate",
        out_name = "energy_eff_vs_call_rate.png",
    )


# ─── G4: Packet Drop Rate vs User Velocity ────────────────────────────────────

def plot_g4(data: Dict) -> None:
    if "g4" not in data:
        print("G4: no data"); return
    d = data["g4"]
    _plot_param_sweep(
        xvals    = d["velocities"],
        mean_key = "pdr_mean",
        std_key  = "pdr_std",
        data     = d,
        xlabel   = "User Velocity (m/s)",
        ylabel   = "Packet Drop Rate (%)",
        title    = "Packet Drop Rate vs. User Velocity",
        out_name = "pdr_vs_velocity.png",
    )


# ─── G5: Packet Drop Rate vs Traffic Load ─────────────────────────────────────

def plot_g5(data: Dict) -> None:
    if "g5" not in data:
        print("G5: no data"); return
    d = data["g5"]
    _plot_param_sweep(
        xvals    = d["traffic_loads_mbps"],
        mean_key = "pdr_mean",
        std_key  = "pdr_std",
        data     = d,
        xlabel   = "Traffic Load (Mbps)",
        ylabel   = "Packet Drop Rate (%)",
        title    = "Packet Drop Rate vs. Traffic Load",
        out_name = "pdr_vs_traffic_load.png",
    )


# ─── shared training-curve plotter (G6, G7) ───────────────────────────────────

def _plot_mobility_curve(
    data:     Dict,
    seed_key: str,
    ylabel:   str,
    title:    str,
    out_name: str,
) -> None:
    if "g6g7" not in data:
        print(f"{out_name}: no data"); return
    mob  = data["g6g7"]
    n_ep = int(mob["num_episodes"])
    ep   = np.arange(1, n_ep + 1, dtype=float)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for algo in ALGO_NAMES:
        if algo not in mob:
            continue
        seeds_data = mob[algo].get(seed_key, [])
        if not seeds_data:
            continue

        arr      = np.array(seeds_data, dtype=float)   # (n_seeds, n_episodes)
        mean_raw = arr.mean(axis=0)

        color = COLORS[algo]
        lw    = LINEWIDTH + (0.4 if algo in PROPOSED else 0.0)
        zo    = 4 if algo in PROPOSED else 2

        # ── Layer 1: raw per-seed curves as thin transparent spikes (background)
        for seed_vals in arr:
            ax.plot(ep, seed_vals, color=color, alpha=0.08,
                    linewidth=0.4, zorder=1)

        # ── Layer 2: thick smooth mean curve on top
        ep_sm, mean_sm = _smooth_training_curve(ep, mean_raw, n_out=N_INTERP_CURVE)
        marker = MARKERS.get(algo, "o")
        ax.plot(ep_sm, mean_sm, color=color, linewidth=lw,
                label=algo, zorder=zo)
        # Markers at actual episode mean values every ~10% of total episodes
        step = max(1, len(ep) // 10)
        ax.plot(ep[::step], mean_raw[::step], marker=marker, color=color,
                linestyle="none", markersize=5,
                markeredgecolor="white", markeredgewidth=0.7, zorder=zo + 1)

    ax.set_xlim(1, n_ep)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(max(1, n_ep // 8)))
    _finalize_ax(ax, "Training Episodes", ylabel, title)
    fig.tight_layout()
    _save(fig, out_name)


def plot_g6(data: Dict) -> None:
    _plot_mobility_curve(
        data,
        seed_key = "tp_low",
        ylabel   = "Throughput (Gbps)",
        title    = "Throughput Convergence: Low Mobile Users (v < 1 m/s)",
        out_name = "throughput_low_mobility.png",
    )


def plot_g7(data: Dict) -> None:
    _plot_mobility_curve(
        data,
        seed_key = "tp_high",
        ylabel   = "Throughput (Gbps)",
        title    = "Throughput Convergence: High Mobile Users (v > 5 m/s)",
        out_name = "throughput_high_mobility.png",
    )


# ─── G8: Energy Efficiency vs. Number of UEs ─────────────────────────────────

def plot_g8_ee_vs_ue(data: Dict) -> None:
    if "g_ue_sweep" not in data:
        print("G8: no data — run run_journal_experiments.py --only g_ue_sweep"); return
    d = data["g_ue_sweep"]
    _plot_param_sweep(
        xvals     = d["num_ues"],
        mean_key  = "ee_mean",
        std_key   = "ee_std",
        seeds_key = "ee_seeds",
        data      = d,
        xlabel    = "Number of UEs",
        ylabel    = "Energy Efficiency (Mbit/J)",
        title     = "Energy Efficiency vs. Number of UEs",
        out_name  = "energy_eff_vs_num_ue.png",
    )


# ─── G9: Packet Drop Rate vs. Number of UEs ──────────────────────────────────
# PDR = Σ_k max(0, d_k − r_k) / Σ_k d_k  averaged over steps, reported in %

def plot_g9_pdr_vs_ue(data: Dict) -> None:
    if "g_ue_sweep" not in data:
        print("G9: no data — run run_journal_experiments.py --only g_ue_sweep"); return
    d = data["g_ue_sweep"]
    _plot_param_sweep(
        xvals     = d["num_ues"],
        mean_key  = "pdr_mean",
        std_key   = "pdr_std",
        seeds_key = "pdr_seeds",
        data      = d,
        xlabel    = "Number of UEs",
        ylabel    = "Packet Drop Rate (%)",
        title     = "Packet Drop Rate vs. Number of UEs",
        out_name  = "pdr_vs_num_ue.png",
    )


# ─── G10: Goodness vs. Number of UEs ─────────────────────────────────────────
# Goodness = 0.5·QoS_ratio + 0.3·Jain_fairness + 0.2·mean_rate_Mbps

def plot_g10_goodness_vs_ue(data: Dict) -> None:
    if "g_ue_sweep" not in data:
        print("G10: no data — run run_journal_experiments.py --only g_ue_sweep"); return
    d = data["g_ue_sweep"]
    _plot_param_sweep(
        xvals     = d["num_ues"],
        mean_key  = "good_mean",
        std_key   = "good_std",
        seeds_key = "good_seeds",
        data      = d,
        xlabel    = "Number of UEs",
        ylabel    = "Goodness Score",
        title     = "Goodness vs. Number of UEs",
        out_name  = "goodness_vs_num_ue.png",
    )


# ─── G11: Packet Drop Rate vs. Packet Arrival Rate ───────────────────────────
# x-axis = CALL_RATES (requests/s); traffic_load = rate / max(CALL_RATES)

def plot_g11_pdr_vs_arrival(data: Dict) -> None:
    if "g_pdr_arrival" not in data:
        print("G11: no data — run run_journal_experiments.py --only g_pdr_arrival"); return
    d = data["g_pdr_arrival"]
    _plot_param_sweep(
        xvals     = d["call_rates"],
        mean_key  = "pdr_mean",
        std_key   = "pdr_std",
        seeds_key = "pdr_seeds",
        data      = d,
        xlabel    = "Packet Arrival Rate (requests/s)  [VBR / Pareto]",
        ylabel    = "Packet Drop Rate (%)",
        title     = "Packet Drop Rate vs. Packet Arrival Rate",
        out_name  = "pdr_vs_packet_arrival_rate.png",
    )


# ─── G12: Goodness vs. QoS SINR Threshold ────────────────────────────────────
# x-axis = SINR_THRESHOLDS_DB (dB).
# Each threshold maps to min_user_rate via R = B·log2(1+10^(γ/10)), B=10 MHz.
# Higher threshold -> stricter QoS -> lower goodness expected.

def plot_g12_goodness_vs_sinr(data: Dict) -> None:
    if "g_goodness_sinr" not in data:
        print("G12: no data — run run_journal_experiments.py --only g_goodness_sinr"); return
    d = data["g_goodness_sinr"]
    sinr_vals = d["sinr_thresholds_db"]
    _plot_param_sweep(
        xvals         = sinr_vals,
        mean_key      = "good_mean",
        std_key       = "good_std",
        seeds_key     = "good_seeds",
        data          = d,
        xlabel        = "QoS SINR Threshold (dB)",
        ylabel        = "Goodness Score",
        title         = "Goodness vs. QoS SINR Threshold",
        out_name      = "goodness_vs_sinr_threshold.png",
        x_tick_labels = [f"{s:+d}" for s in sinr_vals],
    )


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA_PATH)
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated graph IDs: g1,g2,g3,g4,g5,g6,g7,"
                             "g8,g9,g10,g11,g12")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"Data file not found: {args.data}")
        print("Run  python scripts/run_journal_experiments.py  first.")
        return

    with open(args.data) as fh:
        data = json.load(fh)

    _setup_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    only = set(args.only.lower().split(",")) if args.only else set()
    def _do(key: str, fn) -> None:
        if only and key not in only:
            return
        fn(data)

    _do("g1",  plot_g1)
    _do("g2",  plot_g2)
    _do("g3",  plot_g3)
    _do("g4",  plot_g4)
    _do("g5",  plot_g5)
    _do("g6",  plot_g6)
    _do("g7",  plot_g7)
    _do("g8",  plot_g8_ee_vs_ue)
    _do("g9",  plot_g9_pdr_vs_ue)
    _do("g10", plot_g10_goodness_vs_ue)
    _do("g11", plot_g11_pdr_vs_arrival)
    _do("g12", plot_g12_goodness_vs_sinr)

    print(f"\nAll graphs saved to {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
