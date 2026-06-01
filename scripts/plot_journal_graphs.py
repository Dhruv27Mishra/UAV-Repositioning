#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

GAMMAS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

COLORS: Dict[str, str] = {
    "QMIX":               "#1f77b4",   # blue
    "ABQMIX":             "#ff7f0e",   # orange
    "IQL":                "#2ca02c",   # green
    "VDN":                "#d62728",   # red
    "MADDPG":             "#9467bd",   # purple
    "DMTD":               "#8c564b",   # brown
    "DeepNashQ":          "#e377c2",   # pink
    "PerformativeMFMARL": "#17becf",   # cyan
    "PerformativeMARL":   "#bcbd22",   # yellow-green
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
LINEWIDTH       = 1.2
BAND_ALPHA      = 0.12
FIG_SIZE        = (3.5, 2.8)
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

GRID_STYLE  = dict(color="#cccccc", linestyle="--", linewidth=0.5, alpha=0.3)
LEGEND_KW   = dict(ncol=3, fontsize=6, framealpha=0.92,
                   frameon=True, fancybox=False, edgecolor="0.4",
                   loc="upper center", bbox_to_anchor=(0.5, -0.18),
                   handlelength=1.5, handletextpad=0.4,
                   columnspacing=0.8, labelspacing=0.3)

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
        "figure.figsize":           FIG_SIZE,
        "font.family":              "serif",
        "font.serif":               ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":                8,
        "axes.titlesize":           9,
        "axes.titleweight":         "bold",
        "axes.labelsize":           8,
        "axes.labelweight":         "bold",
        "xtick.labelsize":          7,
        "ytick.labelsize":          7,
        "legend.fontsize":          6,
        "legend.handlelength":      1.5,
        "legend.handletextpad":     0.4,
        "legend.columnspacing":     0.8,
        "legend.labelspacing":      0.3,
        "lines.linewidth":          LINEWIDTH,
        "lines.markersize":         0,
        "figure.dpi":               DPI,
        "savefig.dpi":              DPI,
        "savefig.bbox":             "tight",
        "savefig.pad_inches":       0.02,
        "grid.alpha":               0.3,
        "grid.linewidth":           0.5,
        "grid.linestyle":           "--",
        "axes.linewidth":           0.6,
        "xtick.major.width":        0.5,
        "ytick.major.width":        0.5,
        "xtick.major.size":         3,
        "ytick.major.size":         3,
        "axes.facecolor":           "white",
        "figure.facecolor":         "white",
    })


def _finalize_ax(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=8, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8, fontweight="bold")
    ax.tick_params(axis="both", labelsize=7, pad=2)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="minor", length=1.5, width=0.4)
    ax.grid(**GRID_STYLE)
    ax.legend(**LEGEND_KW)
    ax.spines[["top", "right"]].set_visible(False)
    fig = ax.get_figure()
    fig.subplots_adjust(bottom=0.32)


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
        x_se_lo, y_se_lo = _smooth_param_curve(x_raw, means - se)
        x_se_hi, y_se_hi = _smooth_param_curve(x_raw, means + se)

        ax.plot(x_sm, y_sm, color=color, linewidth=lw, label=algo, zorder=zo)

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
        xlabel   = "Call Arrival Rate (requests/s)  [VBR]",
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
        ax.plot(ep_sm, mean_sm, color=color, linewidth=lw,
                label=algo, zorder=zo)

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
    d       = data["g_ue_sweep"]
    num_ues = d["num_ues"]
    _plot_param_sweep(
        xvals     = num_ues,
        mean_key  = "ee_mean",
        std_key   = "ee_std",
        seeds_key = "ee_seeds",
        data      = d,
        xlabel    = "Number of UEs",
        ylabel    = "Energy Efficiency (Mbit/J)",
        title     = "Energy Efficiency vs. Number of UEs",
        out_name  = "energy_eff_vs_num_ue.png",
    )


# ─── G9: Packet Drop Rate vs. Number of UEs — grouped bar chart ──────────────

def plot_g9_pdr_vs_ue(data: Dict) -> None:
    if "g_ue_sweep" not in data:
        print("G9: no data — run run_journal_experiments.py --only g_ue_sweep"); return
    d       = data["g_ue_sweep"]
    num_ues = d["num_ues"]
    algos   = [a for a in ALGO_NAMES if a in d]

    n_groups = len(num_ues)
    n_bars   = len(algos)
    bar_w    = 0.8 / n_bars
    x        = np.arange(n_groups)

    # Two-column width for bar chart — too many bars for single column
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for i, algo in enumerate(algos):
        means = np.array(d[algo]["pdr_mean"])
        offset = (i - n_bars / 2 + 0.5) * bar_w
        ax.bar(x + offset, means, width=bar_w,
               color=COLORS[algo], label=algo,
               edgecolor="white", linewidth=0.4,
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in num_ues])
    _finalize_ax(ax, "Number of UEs", "Packet Drop Rate (%)",
                 "Packet Drop Rate vs. Number of UEs")
    ax.grid(axis="y", **GRID_STYLE)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "pdr_vs_num_ue.png")


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
        xlabel    = "Packet Arrival Rate (requests/s)  [VBR]",
        ylabel    = "Packet Drop Rate (%)",
        title     = "Packet Drop Rate vs. Packet Arrival Rate",
        out_name  = "pdr_vs_packet_arrival_rate.png",
    )


# ─── G12: Goodness vs. QoS SINR Threshold — TABLE ───────────────────────────
# Outputs a CSV and a LaTeX table instead of a plot.
# Rows = SINR thresholds; columns = algorithms (mean ± std).

def plot_g12_goodness_vs_sinr(data: Dict) -> None:
    if "g_goodness_sinr" not in data:
        print("G12: no data — run run_journal_experiments.py --only g_goodness_sinr"); return
    d         = data["g_goodness_sinr"]
    sinr_vals = d["sinr_thresholds_db"]
    mr_vals   = d.get("min_user_rates_mbps", [""] * len(sinr_vals))
    algos     = [a for a in ALGO_NAMES if a in d]

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "goodness_vs_sinr_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SINR_dB", "R_min_Mbps"] + algos)
        for i, (s, mr) in enumerate(zip(sinr_vals, mr_vals)):
            row = [f"{s:+d}", f"{mr:.2f}" if mr else ""]
            for algo in algos:
                adat = d[algo]
                mu  = adat["good_mean"][i]
                std = adat["good_std"][i]
                row.append(f"{mu:.4f} ± {std:.4f}")
            w.writerow(row)
    print(f"Saved -> {csv_path}")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    tex_path = os.path.join(OUT_DIR, "goodness_vs_sinr_table.tex")
    with open(tex_path, "w") as f:
        col_spec = "r r " + " ".join(["c"] * len(algos))
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Goodness Score vs.\\ QoS SINR Threshold "
                "(mean $\\pm$ std over 5 seeds)}\n")
        f.write("\\label{tab:goodness_sinr}\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")
        header = ["SINR (dB)", "$R_{\\min}$ (Mbps)"] + \
                 [a.replace("_", "\\_") for a in algos]
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for i, (s, mr) in enumerate(zip(sinr_vals, mr_vals)):
            row = [f"${s:+d}$", f"{mr:.1f}" if mr else ""]
            for algo in algos:
                adat = d[algo]
                mu  = adat["good_mean"][i]
                std = adat["good_std"][i]
                row.append(f"${mu:.3f}\\pm{std:.3f}$")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved -> {tex_path}")


# ─── Average Throughput vs. Number of UEs ────────────────────────────────────

def plot_g_tp_vs_ue(data: Dict) -> None:
    if "g_ue_sweep" not in data:
        print("TP_UE: no data — run run_journal_experiments.py --only g_ue_sweep"); return
    d = data["g_ue_sweep"]
    if "tp_mean" not in next((v for _, v in d.items() if isinstance(v, dict)), {}):
        print("TP_UE: tp_mean not in data — delete g_ue_sweep key and re-run"); return
    num_ues = d["num_ues"]
    _plot_param_sweep(
        xvals     = num_ues,
        mean_key  = "tp_mean",
        std_key   = "tp_std",
        seeds_key = "tp_seeds",
        data      = d,
        xlabel    = "Number of UEs",
        ylabel    = "Average Throughput (Mbps)",
        title     = "Average Throughput vs. Number of UEs",
        out_name  = "throughput_vs_num_ue.png",
    )


# ─── Energy Efficiency vs. Number of UAVs ────────────────────────────────────

def plot_g_ee_vs_uav(data: Dict) -> None:
    if "g_uav_sweep" not in data:
        print("EE_UAV: no data — run run_journal_experiments.py --only g_uav_sweep"); return
    d = data["g_uav_sweep"]
    _plot_param_sweep(
        xvals     = d["num_uavs"],
        mean_key  = "ee_mean",
        std_key   = "ee_std",
        seeds_key = "ee_seeds",
        data      = d,
        xlabel    = "Number of UAVs",
        ylabel    = "Energy Efficiency (Mbit/J)",
        title     = "Energy Efficiency vs. Number of UAVs",
        out_name  = "energy_eff_vs_num_uav.png",
    )


# ─── EE (RF+Movement only) vs. Number of UEs ─────────────────────────────────
# EE_rf_move = Throughput (Mbit) / (P_tx·Δt + ξ·d)
#   P_tx = RF transmit power per UAV (1 W at 30 dBm × 3 UAVs)
#   ξ    = propulsion energy coefficient (6 J/m)
#   d    = total UAV displacement per step (m)
# This excludes the constant hover power and isolates controllable energy costs.

def plot_g_ee_rf_vs_ue(data: Dict) -> None:
    if "g_ue_sweep" not in data:
        print("EE_RF: no data — run run_journal_experiments.py --only g_ue_sweep"); return
    d = data["g_ue_sweep"]
    if "ee_rf_mean" not in next(iter(v for k, v in d.items() if isinstance(v, dict)), {}):
        print("EE_RF: ee_rf_mean not in data — re-run g_ue_sweep to collect this metric"); return
    _plot_param_sweep(
        xvals     = d["num_ues"],
        mean_key  = "ee_rf_mean",
        std_key   = "ee_rf_std",
        seeds_key = "ee_rf_seeds",
        data      = d,
        xlabel    = "Number of UEs",
        ylabel    = "Energy Efficiency (Mbit/J)",
        title     = "Energy Efficiency (RF + Movement) vs. Number of UEs\n"
                    r"$\eta = \frac{\text{Throughput (Mbit)}}{P_{tx}\Delta t + \xi d}$",
        out_name  = "energy_eff_rf_move_vs_num_ue.png",
    )


# ─── Reward vs. Discount Factor ───────────────────────────────────────────────

def plot_g_reward_vs_gamma(data: Dict) -> None:
    if "g_reward_vs_gamma" not in data:
        print("Reward vs gamma: no data — run run_journal_experiments.py --only g_reward_vs_gamma")
        return
    d = data["g_reward_vs_gamma"]
    _plot_param_sweep(
        xvals         = d["gammas"],
        mean_key      = "mean",
        std_key       = "std",
        data          = d,
        xlabel        = "Discount Factor (γ)",
        ylabel        = "Average Episodic Reward",
        title         = "Reward vs. Discount Factor",
        out_name      = "reward_vs_gamma.png",
        x_tick_labels = [str(g) for g in d["gammas"]],
    )


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA_PATH)
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated graph IDs: g8,g9,g10,g11,g12,"
                             "g_ee_rf,g_reward_gamma")
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

    _do("g8",           plot_g8_ee_vs_ue)
    _do("g9",           plot_g9_pdr_vs_ue)
    _do("g11",          plot_g11_pdr_vs_arrival)
    _do("g_ee_rf",      plot_g_ee_rf_vs_ue)
    _do("g_tp_ue",      plot_g_tp_vs_ue)
    _do("g_ee_uav",     plot_g_ee_vs_uav)
    _do("g_reward_gamma", plot_g_reward_vs_gamma)

    print(f"\nAll graphs saved to {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
