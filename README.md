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

## 📁 Project Structure
```
├── train.py                    # Deep Nash Q-learning agent
├── train_iql.py               # Independent Q-Learning
├── train_qmix.py              # QMIX algorithm
├── train_vdn.py               # VDN algorithm
├── train_maddpg.py            # MADDPG algorithm
├── compare_all_marl.py        # Compare all MARL algorithms
├── model.py                   # SINR-based deterministic strategy
├── abid_model.py              # Custom baseline (Deterministic II)
├── compare_models.py         # Generate performance plots
├── compare_versions.py        # Compare old vs new (fixed vs variable height)
├── test_ue_density.py         # UE density performance test
├── test_performance.py        # Performance validation tests
├── rl_agent/
│   ├── marl_env.py            # MARL environment with variable height
│   ├── IQL.py                 # Independent Q-Learning
│   ├── VDN.py                 # Value Decomposition Networks
│   ├── QMIX.py                # QMIX algorithm
│   ├── DeepNashQ.py           # Deep Nash Q-learning
│   └── MADDPG.py              # Multi-Agent DDPG
├── results/                   # Raw data and plot images (.npy, .png)
├── requirements.txt           # Python dependencies
├── LICENSE, Makefile, .gitignore
```


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

### Learning Curves and Performance Summary
![Results](comparison_results.png)

---

## 🚀 Getting Started

### 🔧 Setup Environment
```bash
pip install -r requirements.txt

🏋️ Train RL Models

python train_iql.py    # Independent Q-Learning
python train_vdn.py    # VDN
python train_qmix.py   # QMIX
python train.py        # Deep Nash Q
python train_maddpg.py # MADDPG

📊 Compare All Algorithms

python compare_all_marl.py  # Comprehensive comparison of all MARL methods

🔬 Run Baseline Models

python model.py        # Deterministic I
python abid_model.py   # Deterministic II

📈 Compare Models
python compare_models.py
📁 Plots and data saved in /results

📦 Dependencies

Python 3.8+
PyTorch ≥ 1.9.0
Gymnasium ≥ 0.26.0
NumPy, Matplotlib, Pandas, SciPy, tqdm
pip install -r requirements.txt

🔬 Simulation Environment

3D grid: 10×10×5 m³
UAVs fly at variable altitudes (5-50m) with height-dependent LOS probability
Wi-Fi standard: IEEE 802.11ac (CSMA/CA)
Dynamic UE mobility (random waypoint)
TDMA-based user scheduling
Evaluation over 1000+ episodes with statistical significance

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
# Run UE density comparison test
python test_ue_density.py

# Run general version comparison
python compare_versions.py
```

Generated plots:
- `ue_density_comparison.png`: Comprehensive UE density analysis
- `ue_density_detailed_analysis.png`: Detailed scaling analysis
```
📚 References

IEEE 802.11ac Wi-Fi standard
Jain’s Index for Fairness [DEC TR-301]
QMIX [ICML 2018], VDN [AAMAS 2018], Deep Nash Q-learning [JMLR 2003]
Bianchi’s model for 802.11 throughput analysis

🪪 License

This repository is licensed under the MIT License. See LICENSE for details.

🙌 Acknowledgements

We thank the research advisors and collaborators from SDSU and Shiv Nadar Institution of Eminence for their guidance.

🔗 Citation

If you use this code or paper in your work, please cite:

@article{pagaria2025uav,
  title={Optimizing Throughput in Wi-Fi enabled UAV Network using Multi-agent Reinforcement Learning},
  author={Pagaria, Parshav and Mishra, Dhruv and Deb, Souvik and Nagraj, Santosh and Sarkar, Mahasweta and Ghosh, Shankar K.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},  
  year={2025}
}
