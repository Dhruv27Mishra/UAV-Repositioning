# UAV-MARL: Multi-Agent RL for UAV-Assisted Wireless Networks

This repository implements a simulation framework for optimizing UAV-assisted wireless communication networks using **multi-agent reinforcement learning (MARL)**. The goal is to manage UAV mobility, user association, and trajectory planning under realistic wireless channel models and CSMA-based access constraints.

# Project Overview

- **Architecture**: Multiple UAVs act as mobile Wi-Fi APs (IEEE 802.11ac)
- **Objective**: Maximize user throughput under SINR-based interference modeling
- **Control**: UAVs are coordinated using centralized RL agents
- **MAC Layer**: CSMA/CA behavior is abstracted via simplified SINR models

# Modules

## 1. `initialization/`
- Initializes UAV and user positions
- Sets parameters like altitude, power, coverage area

## 2. `mobility_model/`
- Simulates UAV movement across time steps
- Optionally integrates mobility constraints (e.g., max speed)

## 3. `channel_model/`
- Computes path loss, Rayleigh fading, and interference
- Outputs SINR values for all UAV-user pairs

## 4. `rl_agent/`
- Contains the multi-agent RL logic 
- Agents learn UAV positions and association policies over episodes

## 5. `trajectory_planner/`
- Uses RL outputs to compute UAV 2D/3D movement plans

## 6. `association_policy/`
- Determines which UAV serves each user at time \( t \)
- Enforces single association constraint and QoS

## 7. `handover_control/`
- Tracks user-UAV reassociation and applies handover suppression logic

## 8. `evaluation/`
- Logs system metrics: total throughput, SINR heatmaps, user satisfaction
- Plots per-episode training performance

# 🧪 Requirements

- Python 3.8+
- 

Install dependencies:

```bash
pip install -r requirements.txt
