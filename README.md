# UAV-Assisted Wi-Fi Coverage: Fairness & Throughput Optimization

This repository provides a full simulation and training framework for coordinating UAVs (drones) acting as Wi-Fi base stations to serve mobile ground users. The framework evaluates both reinforcement learning (RL)-based strategies and deterministic algorithms in terms of total throughput, fairness, and minimum user service guarantees.

---

## 📁 Repository Structure

```text
├── marl_env.py              # Multi-agent UAV environment using OpenAI Gym API
├── train.py                 # Deep NashQ training and evaluation
├── train_qmix.py            # QMIX algorithm
├── train_vdn.py             # VDN algorithm
├── model.py                 # Deterministic SINR-based UAV placement
├── abid_model.py            # Custom trajectory-based UAV baseline
├── compare_models.py        # Fairness and min-rate comparison plots
├── results/                 # Stores .npy data and generated plot images



## 🎯 Problem Statement

Multiple tethered UAVs (powered and stationary at discrete altitudes) are deployed to provide Wi-Fi coverage (IEEE 802.11ac standard) to mobile users on the ground. Each UAV dynamically adjusts its 3D position and association strategy to serve ground users with minimal interference and maximum efficiency.

---

## 🧪 Research Objectives

- ✅ Maximize **total network throughput**
- ✅ Ensure **fair distribution of resources** across all users
- ✅ Maximize **worst-case (minimum) user throughput**
- ✅ Compare RL models vs deterministic baselines under different user densities

---

## 📊 Evaluation Metrics

| Metric                    | Description |
|---------------------------|-------------|
| **Total Throughput**      | Sum of user data rates |
| **Jain’s Fairness Index** | Measures equity of throughput distribution |
| **Minimum User Throughput** | Reflects worst-case user experience (max-min fairness) |
| **Fairness vs. Density**  | Jain's Index as user count increases |
| **Min Rate vs. Density**  | Tracks minimum UE rate under load |

---

## 📈 Experiments Included

- Fairness vs UE Density  
- Minimum UE Throughput vs UE Density  
- Multi-model performance comparison

Models compared:
- 🧠 Deep NashQ  
- 🔄 QMIX  
- 🔁 VDN  
- 🟩 Deterministic SINR-based  
- 🟦 Abid’s trajectory-based model

---

## 🚀 How to Run

### 🔹 Train RL Models
```bash
python train.py           # Deep NashQ
python train_qmix.py      # QMIX
python train_vdn.py       # VDN
Run Deterministic / Abid Models
python model.py           # SINR-based greedy UAV association
python abid_model.py      # Custom baseline strategy
Plot Comparison (Fairness + Min-Rate)

python compare_models.py
🖼 Output
All plots and data are saved in the /results/ folder:

*_fairness_vs_density.png

*_min_rate_vs_density.png

fairness_vs_density_all_models.png

min_rate_vs_density_all_models.png

Corresponding .npy files for raw data

📚 References
IEEE 802.11ac standard (Wi-Fi 5)

Jain, R. et al., “A Quantitative Measure of Fairness”, DEC Research Report TR-301, 1984

Max-Min Fairness for wireless QoS optimization

UAV resource allocation and trajectory planning literature
