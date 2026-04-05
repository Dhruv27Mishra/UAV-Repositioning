#!/usr/bin/env python3
"""
Build publication PNG figures (default under ``assets/figures/``) from a saved ``rl_results.json``.

Example (from repository root):

  MPLBACKEND=Agg python3 scripts/make_final_figures.py \\
    --results outputs/ep2000/rl_results.json
"""
from __future__ import annotations

import repo_paths  # noqa: F401

import argparse
import json
import os
import sys

# Default curve order (matches comparison runner; excludes MAPPO)
DEFAULT_ORDER = [
    "QMIX",
    "IQL",
    "VDN",
    "MADDPG",
    "DeepNashQ",
    "PerformativeMFMARL",
    "PerformativeMARL",
]

_LEGACY_RESULT_KEYS = {
    "AdaptiveMAPPO": "PerformativeMFMARL",
    "AdaptiveMAPPO_NoMF": "PerformativeMARL",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results",
        type=str,
        default="outputs/rl_results.json",
        help="Path to rl_results.json",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="assets/figures",
        help="Output directory for PNGs",
    )
    p.add_argument(
        "--proposed",
        type=str,
        nargs="*",
        default=["PerformativeMFMARL", "PerformativeMARL"],
    )
    args = p.parse_args()

    path = args.results
    if not os.path.isfile(path):
        print(f"Missing results file: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r") as f:
        payload = json.load(f)

    raw_results = payload["results"]
    results = {}
    for k, v in raw_results.items():
        nk = _LEGACY_RESULT_KEYS.get(k, k)
        results[nk] = v
    num_episodes = int(payload.get("episodes") or len(next(iter(results.values()))["throughputs"]))

    ordered = [a for a in DEFAULT_ORDER if a in results]
    for k in sorted(results.keys()):
        if k not in ordered:
            ordered.append(k)

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, "convergence")

    from publication_marl_plots import plot_convergence_publication

    plot_convergence_publication(
        results,
        ordered,
        num_episodes,
        set(args.proposed),
        prefix,
    )
    print("Wrote figures under", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
