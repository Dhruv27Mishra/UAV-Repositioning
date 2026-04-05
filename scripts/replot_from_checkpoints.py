#!/usr/bin/env python3
"""
Rebuild convergence figures from the latest ``metrics.json`` in each
``checkpoints/<Algorithm>/episode_*/`` tree (no retraining).

Use after moving or archiving runs; drops baseline MAPPO from plots by default.
"""
from __future__ import annotations

import repo_paths  # noqa: F401

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Default curve order (excludes MAPPO)
DEFAULT_ORDER = [
    "QMIX",
    "IQL",
    "VDN",
    "MADDPG",
    "DeepNashQ",
    "PerformativeMFMARL",
    "PerformativeMARL",
]

# Checkpoint / JSON keys from older runs
_LEGACY_AGENT_NAMES = {
    "AdaptiveMAPPO": "PerformativeMFMARL",
    "AdaptiveMAPPO_NoMF": "PerformativeMARL",
}


def _latest_metrics_for_agent(checkpoint_root: str, agent_folder: str) -> Optional[Tuple[int, Dict[str, Any]]]:
    adir = os.path.join(checkpoint_root, agent_folder)
    if not os.path.isdir(adir):
        return None
    best_ep = -1
    best: Optional[Dict[str, Any]] = None
    for sub in os.listdir(adir):
        m = re.match(r"episode_(\d+)$", sub)
        if not m:
            continue
        ep = int(m.group(1))
        path = os.path.join(adir, sub, "metrics.json")
        if not os.path.isfile(path):
            continue
        if ep > best_ep:
            best_ep = ep
            with open(path, "r") as f:
                best = json.load(f)
    if best is None:
        return None
    return best_ep, best


def normalize_metrics(data: Dict[str, Any]) -> Dict[str, List[float]]:
    """Map checkpoint JSON keys to keys expected by publication plots."""
    name = data.get("agent_name", "unknown")
    out: Dict[str, List[float]] = {
        "rewards": [float(x) for x in data.get("rewards", [])],
        "throughputs": [float(x) for x in data.get("throughputs", [])],
        "goodness": [float(x) for x in data.get("goodness", [])],
    }
    if "energy_efficiencies_mbit_per_j" in data:
        out["energy_efficiencies"] = [float(x) for x in data["energy_efficiencies_mbit_per_j"]]
    elif "energy_efficiencies" in data:
        out["energy_efficiencies"] = [float(x) for x in data["energy_efficiencies"]]
    if "episode_energies_j" in data:
        out["episode_energies_j"] = [float(x) for x in data["episode_energies_j"]]
    n = len(out["throughputs"])
    if not n:
        raise ValueError(f"Empty throughputs for {name}")
    for ek in ("energy_efficiencies", "episode_energies_j", "goodness"):
        if ek in out and len(out[ek]) != n:
            del out[ek]
    return out


def load_results_from_checkpoints(
    checkpoint_root: str,
    exclude: List[str],
) -> Tuple[Dict[str, Dict[str, List[float]]], List[str], int]:
    results: Dict[str, Dict[str, List[float]]] = {}
    max_ep = 0
    for folder in sorted(os.listdir(checkpoint_root)):
        loaded = _latest_metrics_for_agent(checkpoint_root, folder)
        if not loaded:
            continue
        ep, raw = loaded
        agent_name = raw.get("agent_name", folder)
        agent_name = _LEGACY_AGENT_NAMES.get(agent_name, agent_name)
        if agent_name in exclude or folder in exclude:
            continue
        results[agent_name] = normalize_metrics(raw)
        max_ep = max(max_ep, ep)
    num_episodes = max(
        (len(results[k]["throughputs"]) for k in results),
        default=0,
    )
    ordered = [a for a in DEFAULT_ORDER if a in results]
    for k in sorted(results.keys()):
        if k not in ordered:
            ordered.append(k)
    return results, ordered, num_episodes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-root",
        type=str,
        required=True,
        help="e.g. outputs/ep2000/checkpoints",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Where to write figures (default: parent of checkpoint-root)",
    )
    p.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=["MAPPO"],
        help="Algorithm names to omit (default: MAPPO)",
    )
    p.add_argument("--publication", action="store_true", help="PNG publication-style figures")
    p.add_argument("--no-basic", action="store_true", help="Skip basic throughput/return/energy PNGs")
    p.add_argument(
        "--proposed",
        type=str,
        nargs="*",
        default=["PerformativeMFMARL", "PerformativeMARL"],
    )
    args = p.parse_args()

    ck = os.path.abspath(args.checkpoint_root)
    out_dir = args.out_dir.strip() or os.path.dirname(ck.rstrip(os.sep))
    os.makedirs(out_dir, exist_ok=True)

    results, ordered, num_episodes = load_results_from_checkpoints(ck, list(args.exclude))
    proposed = set(args.proposed)

    meta_path = os.path.join(out_dir, "replot_from_checkpoints_meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "checkpoint_root": ck,
                "out_dir": out_dir,
                "exclude": args.exclude,
                "ordered_algorithms": ordered,
                "num_episodes": num_episodes,
            },
            f,
            indent=2,
        )
    print("Wrote", meta_path)
    print("Algorithms:", ordered, "episodes:", num_episodes)

    pub_prefix = os.path.join(out_dir, "convergence")

    if args.publication:
        from publication_marl_plots import plot_convergence_publication

        plot_convergence_publication(results, ordered, num_episodes, proposed, pub_prefix)

    if not args.no_basic:
        import numpy as np
        import matplotlib.pyplot as plt

        from compare_all_rl_convergence import compute_moving_average
        from publication_marl_plots import finalize_training_episodes_xaxis

        episodes = np.arange(1, num_episodes + 1, dtype=np.float64)
        window = max(5, min(50, max(5, num_episodes // 10)))
        cmap = plt.cm.tab10(np.linspace(0, 0.9, len(ordered)))

        def one_plot(
            ykey: str,
            yscale: float,
            ylabel: str,
            title: str,
            fname: str,
        ) -> None:
            fig, ax = plt.subplots(figsize=(12, 6))
            for i, name in enumerate(ordered):
                if name not in results or ykey not in results[name]:
                    continue
                color = cmap[i % len(cmap)]
                y = np.array(results[name][ykey], dtype=np.float64) * yscale
                ax.plot(episodes, y, "-", alpha=0.06, linewidth=0.4, color=color)
                sm = compute_moving_average(y, window)
                lw = 3.0 if name in proposed else 2.0
                ax.plot(episodes, sm, "-", linewidth=lw, label=name, color=color)
            finalize_training_episodes_xaxis(ax, num_episodes)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.35, linestyle="--")
            fig.tight_layout()
            path = os.path.join(out_dir, fname)
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print("Saved", path)

        one_plot("throughputs", 1e-9, "Throughput (Gbit/s)", "Throughput (smoothed)", "throughput_vs_episodes.png")
        one_plot("rewards", 1.0, "Rewards", "Rewards (smoothed)", "reward_vs_episodes.png")
        if all("energy_efficiencies" in results[n] for n in ordered if n in results):
            one_plot(
                "energy_efficiencies",
                1.0,
                "Energy efficiency (Mbit/J)",
                "Energy efficiency (smoothed)",
                "energy_efficiency_vs_episodes.png",
            )
            one_plot(
                "episode_energies_j",
                1e-6,
                "Energy used per Episode (MJ)",
                "Energy used per Episode (smoothed)",
                "episode_energy_vs_episodes.png",
            )

    agg = os.path.join(out_dir, "rl_results_replot.json")
    with open(agg, "w") as f:
        json.dump({"num_episodes": num_episodes, "results": results}, f, indent=2)
    print("Saved", agg)


if __name__ == "__main__":
    main()
