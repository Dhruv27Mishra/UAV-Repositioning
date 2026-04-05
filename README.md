# 🚁 UAV-Assisted Wi-Fi Coverage Optimization using Multi-Agent Reinforcement Learning

> **Multi-agent reinforcement learning (MARL) meets UAV-based Wi-Fi coverage.** This repository implements Deep Nash Q-learning, QMIX, and VDN algorithms to optimize UAV positioning and user association, outperforming deterministic models under mobility and interference.
---

## 📄 Paper Summary

**📝 Title:** Optimizing Throughput in Wi-Fi enabled UAV Network using Multi-agent Reinforcement Learning  
**👥 Authors:** Parshav Pagaria, Dhruv Mishra, Souvik Deb, Santosh Nagraj, Mahasweta Sarkar, Shankar K. Ghosh  
**🏛️ Affiliations:**  
- San Diego State University, USA  
- Shiv Nadar Institution of Eminence, India  

**🔍 Abstract:**  
This paper introduces a **decentralized MARL framework** to optimize UAV positions and user association in **Wi-Fi-enabled aerial networks**. The environment includes UAV collisions, handover frequency, and transmission failures. Three MARL models—**VDN**, **QMIX**, and **DeepNashQ**—were compared with two deterministic baselines. Results show that **QMIX consistently achieves superior throughput, fairness, and minimum user rate** under user mobility and interference.

📌 **Best Performer:** QMIX  
📌 **Key Metrics:** Throughput, LBT, Goodness  
📌 **IEEE Standard:** 802.11ac (Wi-Fi 5)  
---

## Repository layout

Run all Python entry points **from the repository root** (e.g. `python3 scripts/train_iql.py`). `scripts/repo_paths.py` is imported first so `rl_agent` resolves correctly.

```
├── docs/                    # Technical notes (performative MARL, signal-map MDP)
├── scripts/                 # Training, comparisons, benchmarks, plotting utilities
│   ├── repo_paths.py        # Sys.path bootstrap (import before rl_agent)
│   ├── train*.py            # Per-algorithm trainers
│   ├── compare_all_marl.py
│   ├── compare_all_rl_convergence.py
│   ├── run_rl_comparison_500ep.py
│   ├── make_final_figures.py
│   ├── replot_from_checkpoints.py
│   └── publication_marl_plots.py
├── tests/
│   └── test_performance.py
├── rl_agent/                # Environment + MARL implementations
├── assets/figures/          # Checked-in publication PNGs (optional to regenerate)
├── requirements.txt
├── LICENSE, Makefile, .gitignore
```

**Documentation:** [docs/README.md](docs/README.md) indexes `docs/performative_marl.md` and `docs/signal_map_marl.md`.

Generated runs, logs, checkpoints, and large caches go under **`outputs/`** (default for benchmarks), **`cache_runs/`**, **`models/`**, etc.; these paths are **gitignored**. Regenerate with the scripts above.


---

## 🧠 Problem Statement

Positioning UAVs for **Wi-Fi coverage** and user association is challenging due to:
- 🌀 User mobility
- 📶 Inter-UAV interference
- 📡 Real-time throughput and fairness constraints
- 📏 **Height optimization** (LOS probability vs distance-based path loss trade-off)

We aim to:
- Maximize **system throughput**
- Optimize **worst-case user experience**
- **Learn optimal UAV heights** balancing LOS probability and path loss
- Evaluate learning-based vs rule-based UAV coordination

---

## 🔍 Algorithms Compared

| Model              | Type              | Description |
|-------------------|-------------------|-------------|
| **IQL**            | Independent MARL  | Each agent learns independently (baseline) |
| **VDN**            | Value-based MARL  | Linear decomposition of Q-values per agent |
| **QMIX**           | Value-based MARL  | Nonlinear value mixing for complex agent interactions |
| **Deep Nash Q**    | Game-Theoretic MARL | Learns Nash equilibrium strategies for cooperative behavior |
| **MADDPG**         | Actor-Critic MARL | Multi-agent deep deterministic policy gradient |
| **Deterministic I**| SINR-based        | Greedy placement with static strategy |
| **Deterministic II**| Geometry-based    | Circle-packing trajectory baseline |

---

## 📊 Evaluation Metrics

| Metric             | Description |
|--------------------|-------------|
| 📶 **Total Throughput** | Sum of all user data rates |
| ⬇️ **Minimum User Rate (LBT)** | Worst-case UE experience |
| 💡 **Goodness** | Fraction of UEs above throughput threshold |
| ❌ **Collision Count** | UAV physical overlap incidents |

---

## 🖼️ Visual Results

### Learning curves (publication-style)

Example convergence figures are committed under **`assets/figures/`** (for example `assets/figures/convergence_combined.png`). Regenerate from a saved `rl_results.json`:

```bash
MPLBACKEND=Agg python3 scripts/make_final_figures.py \
  --results outputs/ep500/rl_results.json \
  --out-dir assets/figures
```

---

## Getting started

### Environment

```bash
pip install -r requirements.txt
```

### Train RL agents (from repository root)

```bash
python3 scripts/train_iql.py
python3 scripts/train_vdn.py
python3 scripts/train_qmix.py
python3 scripts/train.py              # Deep Nash Q
python3 scripts/train_maddpg.py
python3 scripts/train_mappo.py
python3 scripts/train_adaptive_nonstationary.py
```

### Compare algorithms

```bash
python3 scripts/compare_all_marl.py           # Value-based MARL suite
python3 scripts/run_rl_comparison_500ep.py    # Performative benchmark + baselines
```

### Other utilities

```bash
python3 scripts/compare_versions.py
python3 scripts/check_gpu.py
python3 tests/test_performance.py
```

### Dependencies

Python 3.8+, PyTorch ≥ 1.9, Gymnasium ≥ 0.26, NumPy, Matplotlib, SciPy, tqdm (see `requirements.txt`).

### Simulation notes

3D grid 10×10×5 m³; UAV heights roughly 5–50 m with height-dependent LoS; IEEE 802.11–style rate modeling; dynamic UE mobility; evaluation over many episodes.

---

## 🆕 Variable Height Enhancement

### Key Improvements

This repository now includes **variable height optimization** with height-dependent LOS (Line-of-Sight) probability modeling:

- **Height Range**: Expanded from fixed [10, 15]m to variable [5, 50]m
- **LOS Probability Model**: Height-dependent function that:
  - Increases with altitude (reduces obstacle blockage)
  - Peaks at optimal height (~20m)
  - Accounts for distance-based path loss at very high altitudes
- **Path Loss Differentiation**: Separate models for LOS (exp=2.0) and NLOS (exp=3.5)
- **Adaptive Shadowing**: Different variance for LOS (3.0 dB) vs NLOS (8.0 dB)

### Performance Results

**UE Density Test Results** (comparing old fixed-height vs new variable-height):

| UE Count | Old Throughput | New Throughput | Improvement |
|----------|----------------|----------------|-------------|
| 5 UEs    | 3.447          | 4.737          | **+37.43%** |
| 10 UEs   | 6.829          | 11.161         | **+63.43%** |
| 15 UEs   | 9.734          | 11.929         | **+22.56%** |
| 20 UEs   | 10.020         | 15.895         | **+58.63%** |
| 25 UEs   | 14.574         | 23.673         | **+62.43%** |
| 30 UEs   | 16.061         | 17.121         | **+6.60%** |
| 40 UEs   | 20.361         | 34.878         | **+71.29%** |
| 50 UEs   | 32.778         | 37.856         | **+15.49%** |

**Summary Statistics:**
- **Average Improvement**: **+42.23%** across all UE densities
- **Per-User Throughput**: 0.850 (new) vs 0.601 (old) - **+41.4% improvement**
- **Best Performance**: **+71.29%** improvement at 40 UEs
- **Height Optimization**: UAVs learn optimal heights (~18-24m) based on conditions

### Testing Variable Height

```bash
# Version / height comparison (writes PNGs to current working directory)
python3 scripts/compare_versions.py
```

Optional: `python3 tests/test_performance.py` for environment smoke tests. Generated plots from these scripts are ignored by git unless renamed.

---

## References

- IEEE 802.11ac Wi-Fi standard  
- Jain’s fairness index (e.g. DEC TR-301)  
- QMIX (ICML 2018), VDN (AAMAS 2018), Deep Nash Q-learning (JMLR 2003)  
- Bianchi’s model for 802.11 throughput analysis  

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

We thank the research advisors and collaborators from SDSU and Shiv Nadar Institution of Eminence for their guidance.

## Citation

If you use this code or paper in your work, please cite:

@article{pagaria2025uav,
  title={Optimizing Throughput in Wi-Fi enabled UAV Network using Multi-agent Reinforcement Learning},
  author={Pagaria, Parshav and Mishra, Dhruv and Deb, Souvik and Nagraj, Santosh and Sarkar, Mahasweta and Ghosh, Shankar K.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},  
  year={2025}
}
