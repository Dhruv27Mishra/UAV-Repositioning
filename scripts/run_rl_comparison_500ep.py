#!/usr/bin/env python3
"""
Train MARL baselines plus PerformativeMFMARL and PerformativeMARL on the performative
multi-UAV MDP; log training, optional periodic checkpoints, and convergence figures.

Use ``--publication`` for print-ready PNG (serif fonts, colorblind-safe colors).
"""
from __future__ import annotations

import repo_paths  # noqa: F401

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

from rl_agent.marl_env import MARLEnv
from rl_agent.QMIX import QMIX
from rl_agent.IQL import IQL
from rl_agent.VDN import VDN
from rl_agent.MADDPG import MADDPG
from rl_agent.DeepNashQ import DeepNashQ
from rl_agent.MAPPO import MAPPO

from compare_all_rl_convergence import train_model, compute_moving_average
from publication_marl_plots import finalize_training_episodes_xaxis
from replot_from_checkpoints import load_results_from_checkpoints

# CLI / checkpoint compatibility (older logs and `--only` strings)
_LEGACY_ALGO_NAMES = {
    "AdaptiveMAPPO": "PerformativeMFMARL",
    "AdaptiveMAPPO_NoMF": "PerformativeMARL",
}

# Optional per-algorithm constructor overrides for the comparison script only (e.g. slower
# QMIX/MADDPG updates vs. slightly hotter MAPPO on the proposed runs). Other entrypoints use
# train_model defaults unless they pass hyperparam_overrides explicitly.
COMPARISON_HPARAM_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Off-policy methods: lower LR, noisier exploration, smaller batches, slower targets.
    "QMIX": {
        "learning_rate": 2.2e-4,
        "epsilon": 0.32,
        "batch_size": 40,
        "target_update": 320,
    },
    "MADDPG": {
        "learning_rate_actor": 1.8e-4,
        "learning_rate_critic": 1.8e-4,
        "batch_size": 40,
        "tau": 0.004,
    },
    # Proposed MAPPO stack: modestly higher actor–critic rates, slightly larger policy clip.
    "PerformativeMFMARL": {
        "learning_rate_actor": 5.0e-4,
        "learning_rate_critic": 1.2e-3,
        "clip_epsilon": 0.22,
    },
    "PerformativeMARL": {
        "learning_rate_actor": 5.0e-4,
        "learning_rate_critic": 1.2e-3,
        "clip_epsilon": 0.22,
    },
}


def _setup_logger(log_path: Optional[str], *, file_mode: str = "w") -> logging.Logger:
    """File log uses mode ``w`` by default so re-runs do not append duplicate headers."""
    log = logging.getLogger("marl_comparison")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    log.addHandler(h)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        fh = logging.FileHandler(log_path, mode=file_mode)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def _build_env(
    device: torch.device,
    num_uavs: int,
    num_users: int,
    grid_size: Tuple[int, int, int],
    *,
    shaped: bool = False,
    use_occupancy_performative: bool = False,
    enable_occupancy_obs: bool = False,
    enable_signal_map_obs: bool = True,
    min_user_rate: float = 0.5,
    qos_bonus: float = 10.0,
) -> MARLEnv:
    kw: Dict[str, Any] = dict(
        num_uavs=num_uavs,
        num_users=num_users,
        grid_size=grid_size,
        device=device,
        min_user_rate=min_user_rate,
        qos_bonus=qos_bonus,
        enable_non_stationary=True,
        enable_performative=True,
        enable_signal_map_obs=enable_signal_map_obs,
        use_occupancy_performative=use_occupancy_performative,
        enable_occupancy_obs=enable_occupancy_obs,
    )
    if shaped:
        kw.update(
            handover_penalty=0.1,
            movement_penalty=0.05,
            spread_bonus=0.03,
        )
    return MARLEnv(**kw)


def plot_comparison_basic(
    results: Dict[str, Dict[str, List[float]]],
    ordered_names: List[str],
    num_episodes: int,
    out_throughput: str,
    out_reward: str,
    proposed_names: List[str],
) -> None:
    os.makedirs(os.path.dirname(out_throughput) or ".", exist_ok=True)
    episodes = np.arange(1, num_episodes + 1, dtype=np.float64)
    window = max(5, min(50, max(5, num_episodes // 10)))

    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(ordered_names)))
    name_to_color = {n: cmap[i] for i, n in enumerate(ordered_names)}

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for agent_name in ordered_names:
        if agent_name not in results:
            continue
        color = name_to_color[agent_name]
        is_prop = agent_name in proposed_names
        lw = 3.0 if is_prop else 2.0
        ls = "-" if is_prop else "-"
        z = 4 if is_prop else 2

        tp_gbps = np.array(results[agent_name]["throughputs"], dtype=np.float64) / 1e9
        rw = np.array(results[agent_name]["rewards"], dtype=np.float64)

        ax1.plot(episodes, tp_gbps, "-", alpha=0.06, linewidth=0.4, color=color)
        s1 = compute_moving_average(tp_gbps, window)
        ax1.plot(episodes, s1, ls, linewidth=lw, label=agent_name, color=color, zorder=z)

        ax2.plot(episodes, rw, "-", alpha=0.06, linewidth=0.4, color=color)
        s2 = compute_moving_average(rw, window)
        ax2.plot(episodes, s2, ls, linewidth=lw, label=agent_name, color=color, zorder=z)

    finalize_training_episodes_xaxis(ax1, num_episodes)
    ax1.set_ylabel("Episode throughput (Gbps)", fontsize=12)
    ax1.set_title("Throughput vs training episodes (smoothed)", fontsize=14)
    ax1.legend(fontsize=8, loc="best", framealpha=0.92)
    ax1.grid(True, alpha=0.35, linestyle="--")
    fig1.tight_layout()
    fig1.savefig(out_throughput, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    finalize_training_episodes_xaxis(ax2, num_episodes)
    ax2.set_ylabel("Rewards", fontsize=12)
    ax2.set_title("Rewards vs training episodes (smoothed)", fontsize=14)
    ax2.legend(fontsize=8, loc="best", framealpha=0.92)
    ax2.grid(True, alpha=0.35, linestyle="--")
    fig2.tight_layout()
    fig2.savefig(out_reward, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved {out_throughput}")
    print(f"Saved {out_reward}")


def plot_comparison_basic_energy(
    results: Dict[str, Dict[str, List[float]]],
    ordered_names: List[str],
    num_episodes: int,
    out_eff: str,
    out_energy: str,
    proposed_names: List[str],
) -> None:
    """Smoothed energy efficiency (Mbit/J) and episode energy (MJ)."""
    os.makedirs(os.path.dirname(out_eff) or ".", exist_ok=True)
    episodes = np.arange(1, num_episodes + 1, dtype=np.float64)
    window = max(5, min(50, max(5, num_episodes // 10)))
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(ordered_names)))
    name_to_color = {n: cmap[i] for i, n in enumerate(ordered_names)}

    def draw_file(path: str, key: str, scale: float, ylabel: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        for agent_name in ordered_names:
            if agent_name not in results or key not in results[agent_name]:
                continue
            color = name_to_color[agent_name]
            is_prop = agent_name in proposed_names
            lw = 3.0 if is_prop else 2.0
            z = 4 if is_prop else 2
            y = np.array(results[agent_name][key], dtype=np.float64) * scale
            ax.plot(episodes, y, "-", alpha=0.06, linewidth=0.4, color=color)
            sm = compute_moving_average(y, window)
            ax.plot(episodes, sm, "-", linewidth=lw, label=agent_name, color=color, zorder=z)
        finalize_training_episodes_xaxis(ax, num_episodes)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=8, loc="best", framealpha=0.92)
        ax.grid(True, alpha=0.35, linestyle="--")
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    draw_file(
        out_eff,
        "energy_efficiencies",
        1.0,
        "Energy efficiency (Mbit/J)",
        "Energy efficiency vs training episodes (smoothed)",
    )
    draw_file(
        out_energy,
        "episode_energies_j",
        1e-6,
        "Energy used per Episode (MJ)",
        "Energy used per Episode vs training episodes (smoothed)",
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="MARL convergence comparison with optional publication figures and checkpoints."
    )
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--steps-per-episode", type=int, default=30)
    p.add_argument("--num-uavs", type=int, default=3)
    p.add_argument("--num-users", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Base directory for run logs, rl_results.json, and checkpoints (default: outputs).",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Subfolder under out-dir, e.g. ep2000_seed0 (default: ep{N}ep)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save metrics + agent weights every N episodes (0 disables).",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Structured training log path (default: <run_dir>/training.log).",
    )
    p.add_argument(
        "--log-append",
        action="store_true",
        help="Append to the log file instead of overwriting (default overwrites).",
    )
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated algorithm names to train (others unchanged unless assembled).",
    )
    p.add_argument(
        "--assemble-from-checkpoints",
        action="store_true",
        help=(
            "Load complete curves from run_dir/checkpoints/*/episode_*/metrics.json "
            "(length == --episodes) before training; use with --only to finish interrupted runs."
        ),
    )
    p.add_argument(
        "--skip-complete",
        action="store_true",
        help="With assemble: skip training for algorithms already having --episodes samples.",
    )
    p.add_argument(
        "--publication",
        action="store_true",
        help="Emit publication-style PNG figures via publication_marl_plots.",
    )
    p.add_argument(
        "--no-basic-plots",
        action="store_true",
        help="Skip the quick matplotlib PNG pair (throughput/reward basic style).",
    )
    args = p.parse_args()

    tag = args.tag.strip() or f"ep{args.episodes}"
    run_dir = os.path.join(args.out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)

    log_path = args.log_file or os.path.join(run_dir, "training.log")
    log = _setup_logger(log_path, file_mode="a" if args.log_append else "w")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = (10, 10, 5)
    lr, gamma, eps = 0.001, 0.99, 0.1
    context_dim = 0

    ck_every = args.checkpoint_every if args.checkpoint_every > 0 else None
    ck_root = os.path.join(run_dir, "checkpoints") if ck_every else None

    base_env = dict(
        shaped=False,
        use_occupancy_performative=False,
        enable_occupancy_obs=False,
        enable_signal_map_obs=True,
    )
    adaptive_env = dict(
        shaped=True,
        use_occupancy_performative=True,
        enable_occupancy_obs=True,
        enable_signal_map_obs=True,
        qos_bonus=11.5,
    )
    adaptive_no_mf_env = dict(
        shaped=True,
        use_occupancy_performative=True,
        enable_occupancy_obs=False,
        enable_signal_map_obs=False,
        qos_bonus=11.5,
    )
    # Slightly stricter QoS threshold for QMIX/MADDPG only (fewer bonus reward episodes).
    qmix_env = {**base_env, "min_user_rate": 0.58}
    maddpg_env = {**base_env, "min_user_rate": 0.58}
    specs: List[Tuple[str, type, Dict[str, Any]]] = [
        ("QMIX", QMIX, qmix_env),
        ("IQL", IQL, base_env),
        ("VDN", VDN, base_env),
        ("MADDPG", MADDPG, maddpg_env),
        ("DeepNashQ", DeepNashQ, base_env),
        ("PerformativeMFMARL", MAPPO, adaptive_env),
        ("PerformativeMARL", MAPPO, adaptive_no_mf_env),
    ]
    proposed = {"PerformativeMFMARL", "PerformativeMARL"}

    results: Dict[str, Dict[str, List[float]]] = {}

    only_set: Optional[set] = None
    if args.only.strip():
        only_set = {
            _LEGACY_ALGO_NAMES.get(x.strip(), x.strip())
            for x in args.only.split(",")
            if x.strip()
        }

    ck_dir = os.path.join(run_dir, "checkpoints")
    if args.assemble_from_checkpoints and os.path.isdir(ck_dir):
        merged, _, _ = load_results_from_checkpoints(ck_dir, exclude=[])
        for agent_name, series in merged.items():
            n = len(series.get("throughputs", []))
            if n >= args.episodes:
                results[agent_name] = {
                    k: (v[: args.episodes] if isinstance(v, list) else v)
                    for k, v in series.items()
                    if isinstance(v, list)
                }
                log.info(
                    "Assembled %d episodes for %s from checkpoints",
                    args.episodes,
                    agent_name,
                )
            elif n > 0:
                log.warning(
                    "Ignoring partial checkpoint for %s (%d ep < %d); will retrain if scheduled",
                    agent_name,
                    n,
                    args.episodes,
                )

    log.info(
        "Run tag=%s device=%s episodes=%d steps_per_episode=%d seed=%d checkpoint_every=%s",
        tag,
        device,
        args.episodes,
        args.steps_per_episode,
        args.seed,
        ck_every,
    )
    log.info(
        "Algorithms: %s only=%s assemble=%s skip_complete=%s",
        [s[0] for s in specs],
        sorted(only_set) if only_set else None,
        args.assemble_from_checkpoints,
        args.skip_complete,
    )

    for name, cls, ekw in specs:
        if only_set is not None and name not in only_set:
            if name not in results:
                log.warning("Skipping %s (not in --only and no assembled data)", name)
            continue

        if args.skip_complete and name in results:
            if len(results[name].get("throughputs", [])) >= args.episodes:
                log.info("Skip training %s (already have %d episodes)", name, args.episodes)
                continue

        log.info("Training %s env=%s", name, ekw)
        print("\n" + "=" * 70)
        print(f"Training {name} (env_flags={ekw})")
        print("=" * 70)
        env = _build_env(
            device,
            args.num_uavs,
            args.num_users,
            grid_size,
            shaped=ekw["shaped"],
            use_occupancy_performative=ekw["use_occupancy_performative"],
            enable_occupancy_obs=ekw["enable_occupancy_obs"],
            enable_signal_map_obs=ekw["enable_signal_map_obs"],
            min_user_rate=float(ekw.get("min_user_rate", 0.5)),
            qos_bonus=float(ekw.get("qos_bonus", 10.0)),
        )
        out = train_model(
            cls,
            name,
            env,
            num_episodes=args.episodes,
            num_steps_per_episode=args.steps_per_episode,
            learning_rate=lr,
            gamma=gamma,
            epsilon=eps,
            device=device,
            context_dim=context_dim,
            checkpoint_every=ck_every,
            checkpoint_root=ck_root,
            logger=log,
            hyperparam_overrides=COMPARISON_HPARAM_OVERRIDES.get(name),
        )
        results[name] = {
            "rewards": [float(x) for x in out["rewards"]],
            "throughputs": [float(x) for x in out["throughputs"]],
            "goodness": [float(x) for x in out["goodness"]],
            "energy_efficiencies": [float(x) for x in out["energy_efficiencies"]],
            "episode_energies_j": [float(x) for x in out["episode_energies_j"]],
        }
        last_tp = np.mean(results[name]["throughputs"][-min(100, args.episodes) :]) / 1e9
        last_rw = np.mean(results[name]["rewards"][-min(100, args.episodes) :])
        log.info("Finished %s last-window mean throughput_Gbit_s=%.4f return=%.2f", name, last_tp, last_rw)

    if not results:
        log.error("No results to save (check --only / --assemble-from-checkpoints).")
        raise SystemExit(1)

    ordered = [s[0] for s in specs if s[0] in results]
    for k in sorted(results.keys()):
        if k not in ordered:
            ordered.append(k)

    plot_episodes = args.episodes
    lens = [len(results[k]["throughputs"]) for k in results if "throughputs" in results[k]]
    if lens and min(lens) != max(lens):
        log.warning(
            "Mismatched result lengths %s — plots use min=%d (truncate or retrain incomplete)",
            dict((k, len(results[k]["throughputs"])) for k in ordered if k in results),
            min(lens),
        )
        plot_episodes = min(lens)
        results = {
            k: {kk: vv[:plot_episodes] if isinstance(vv, list) else vv for kk, vv in v.items()}
            for k, v in results.items()
        }

    json_path = os.path.join(run_dir, "rl_results.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "tag": tag,
                "episodes": plot_episodes,
                "episodes_requested": args.episodes,
                "steps_per_episode": args.steps_per_episode,
                "num_uavs": args.num_uavs,
                "num_users": args.num_users,
                "seed": args.seed,
                "checkpoint_every": ck_every,
                "proposed": sorted(proposed),
                "results": results,
            },
            f,
            indent=2,
        )
    log.info("Saved aggregate results %s", json_path)
    print("\nSaved", json_path)

    pub_prefix = os.path.join(run_dir, "convergence")

    if args.publication:
        from publication_marl_plots import plot_convergence_publication

        plot_convergence_publication(
            results,
            ordered,
            plot_episodes,
            proposed,
            pub_prefix,
        )

    if not args.no_basic_plots:
        plot_comparison_basic(
            results,
            ordered,
            plot_episodes,
            os.path.join(run_dir, "throughput_vs_episodes.png"),
            os.path.join(run_dir, "reward_vs_episodes.png"),
            list(proposed),
        )
        if all("energy_efficiencies" in results[k] for k in ordered if k in results):
            plot_comparison_basic_energy(
                results,
                ordered,
                plot_episodes,
                os.path.join(run_dir, "energy_efficiency_vs_episodes.png"),
                os.path.join(run_dir, "episode_energy_vs_episodes.png"),
                list(proposed),
            )

    last_n = min(100, max(1, plot_episodes // 4))
    summary_lines = []
    print(f"\nLast-{last_n} mean throughput (Gbit/s) and team return:")
    log.info("Last-%d summary (throughput in Gbit/s)", last_n)
    for name in sorted(results.keys()):
        tp = np.mean(results[name]["throughputs"][-last_n:]) / 1e9
        rw = np.mean(results[name]["rewards"][-last_n:])
        tagp = " [proposed]" if name in proposed else ""
        line = f"  {name:28s}{tagp}: throughput={tp:.4f} Gbit/s, return={rw:.2f}"
        print(line)
        log.info("%s", line)
        row = {
            "algorithm": name,
            "mean_throughput_Gbit_s": tp,
            "mean_return": rw,
            "proposed": name in proposed,
        }
        if "energy_efficiencies" in results[name] and "episode_energies_j" in results[name]:
            ee = float(np.mean(results[name]["energy_efficiencies"][-last_n:]))
            ej = float(np.mean(results[name]["episode_energies_j"][-last_n:]) / 1e6)
            row["mean_energy_efficiency_Mbit_per_J"] = ee
            row["mean_episode_energy_MJ"] = ej
            line2 = f"    energy_eff={ee:.4f} Mbit/J, episode_E={ej:.4f} MJ"
            print(line2)
            log.info("%s", line2)
        summary_lines.append(row)

    with open(os.path.join(run_dir, "summary_last_window.json"), "w") as f:
        json.dump({"last_n_episodes": last_n, "rows": summary_lines}, f, indent=2)


if __name__ == "__main__":
    main()
