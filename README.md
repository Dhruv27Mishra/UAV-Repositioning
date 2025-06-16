# UAV-MARL: Multi-Agent RL for UAV-Assisted Wireless Networks

This repository implements a simulation framework for optimizing UAV-assisted wireless communication networks using **multi-agent reinforcement learning (MARL)**. The goal is to manage UAV mobility, user association, and trajectory planning under realistic wireless channel models and CSMA-based access constraints.

## Project Overview

- **Architecture**: Multiple UAVs act as mobile Wi-Fi APs (IEEE 802.11ac).
- **Objective**: Maximize user throughput under SINR-based interference modeling.
- **Control**: UAVs are coordinated using centralized RL agents.
- **MAC Layer**: CSMA/CA behavior is abstracted via simplified SINR models.

## Modules

### 1. `initialization/`
- Initializes UAV and user positions.
- Sets parameters like altitude, power, coverage area.

### 2. `mobility_model/`
- Simulates UAV movement across time steps.
- Optionally integrates mobility constraints (e.g., max speed).

### 3. `channel_model/`
- Computes path loss, Rayleigh fading, and interference.
- Outputs SINR values for all UAV-user pairs.

### 4. `rl_agent/`
- Contains the multi-agent RL logic.
- Agents learn UAV positions and association policies over episodes.

### 5. `trajectory_planner/`
- Uses RL outputs to compute UAV 2D/3D movement plans.

### 6. `association_policy/`
- Determines which UAV serves each user at time \( t \).
- Enforces single association constraint and QoS.

### 7. `handover_control/`
- Tracks user-UAV reassociation and applies handover suppression logic.

### 8. `evaluation/`
- Logs system metrics: total throughput, SINR heatmaps, user satisfaction.
- Plots per-episode training performance.

## Problem Statement

In a wireless network, UAVs (Unmanned Aerial Vehicles) are deployed to provide coverage and connectivity to ground users. The goal is to dynamically reposition the UAVs to maximize network performance (e.g., throughput, QoS) while satisfying constraints such as collision avoidance, boundary limits, and altitude restrictions.

## Formulations

### Environment
- **Grid Size**: 3D grid (default: 10x10x5).
- **UAVs**: Each UAV is an agent that can move in 6 directions (up, down, left, right, forward, backward).
- **Users**: Ground users with minimal random movement (simulating movement inside homes).
- **Initial Positions**: UAVs and users have fixed random initial positions for the entire training run.

### Constraints
- **Collision Avoidance**: UAVs must maintain a minimum distance from each other.
- **Boundary Limits**: UAVs must stay within the grid boundaries.
- **Altitude Constraints**: UAVs must maintain an altitude between 10 and 15 meters.

### Reward Structure
- **Positive Reward**: Bonus for each user meeting the QoS requirement (default: +10.0 per user per step).
- **Negative Penalties**:
  - Collision penalty (default: -100.0).
  - Boundary violation penalty (default: -50.0).
  - Altitude violation penalty (default: -50.0).
- **Distance-based Penalty**: Negative reward based on the distance between each UAV and its target position.

### User Association
- Each user is associated with its nearest UAV (minimum Euclidean distance).
- This association is used for throughput calculation and visualization.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Gymnasium
- Matplotlib
- tqdm

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd UAV-Repositioning-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt

```

## Makefile Usage

A `Makefile` is provided for convenience. Common targets:

- `make install` – Install all Python dependencies
- `make train` – Run the main training script
- `make clean` – Remove Python bytecode and model checkpoints
- `make help` – Show all available Makefile commands

Example:
```bash
make train
```
