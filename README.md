UAV-Assisted Wi-Fi Networks: Fairness and Coverage Optimization
This repository provides a simulation framework and reinforcement learning environment to evaluate different UAV coordination algorithms for Wi-Fi coverage in rural or dense environments. It focuses on throughput optimization, Jain’s fairness, and worst-case user experience (max-min fairness).

📦 Contents
graphql
Copy
Edit
├── marl_env.py                 # Multi-agent UAV environment (RL)
├── train.py                   # Deep NashQ training and fairness logging
├── train_qmix.py              # QMIX training and evaluation
├── train_vdn.py               # VDN training and evaluation
├── model.py                   # Deterministic UAV baseline
├── abid_model.py              # Custom alternative UAV positioning model
├── compare_models.py          # Plots comparison of fairness, min-rate, etc.
├── results/                   # Stores .npy results and plots
🚁 Problem Overview
We simulate multiple tethered UAVs acting as Wi-Fi access points (IEEE 802.11ac), each serving mobile ground users. The UAVs dynamically reposition to maximize network performance under various constraints.

🎯 Goals
✅ Maximize total network throughput

✅ Ensure fair access via Jain’s Fairness Index

✅ Improve worst-case user experience (min throughput)

✅ Compare RL-based strategies with deterministic baselines

📊 Metrics Tracked
Metric	Description
Throughput	Total user data rate across the network
Jain's Fairness Index	Measures equity of rate distribution
Minimum UE Throughput	Max-min fairness: how the worst user performs
Gini Index (optional)	Measures rate inequality (like in economics)

🧪 Experiments
1. Fairness vs. UE Density
Plots how Jain’s fairness changes as the number of users increases.

2. Minimum Throughput vs. UE Density
Plots how the worst-off user is treated as network load increases.

3. Multi-model Comparison
Compare:

Deep NashQ

QMIX

VDN

Deterministic SINR-based policy

Abid’s custom baseline

📈 Results
Plots are automatically saved in /results:

*_fairness_vs_density.png

*_min_rate_vs_density.png

fairness_vs_density_all_models.png

min_rate_vs_density_all_models.png

🛠 How to Run
Train and log fairness:
python train.py          # Deep NashQ
python train_qmix.py     # QMIX
python train_vdn.py      # VDN
Run deterministic baseline:
python model.py
Run Abid's baseline:
python abid_model.py
Compare all models:
python compare_models.py
📚 Citation
If you use this code for academic purposes, please cite the repository and include credit to the original authors.

🤝 Acknowledgements
IEEE 802.11ac reference for Wi-Fi modeling

Jain et al. (1984) for fairness metric