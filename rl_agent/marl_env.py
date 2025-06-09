"""
Multi-Agent Reinforcement Learning Environment for UAV coordination.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt

class MARLEnv(gym.Env):
    def __init__(self, num_uavs: int = 3, num_users: int = 20, grid_size: Tuple[int, int, int] = (10, 10, 5), device: torch.device = None, min_user_rate: float = 0.5, qos_bonus: float = 10.0):
        super(MARLEnv, self).__init__()
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.grid_size = grid_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_user_rate = min_user_rate
        self.qos_bonus = qos_bonus
        
        # Define action space (6 possible movements: up, down, left, right, forward, backward)
        self.action_space = spaces.MultiDiscrete([6] * num_uavs)
        
        # Define observation space (x, y, z coordinates for each UAV)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0] * num_uavs),
            high=np.array([grid_size[0], grid_size[1], grid_size[2]] * num_uavs),
            dtype=np.float32
        )
        
        # Initialize UAV positions
        self.uav_positions = None
        self.target_positions = None
        self.collision_penalty = -100.0  # Strong penalty for collisions
        self.violation_penalty = -50.0   # Penalty for boundary violations
        # Initialize user positions
        self.user_positions = None
        # Channel parameters
        self.B = 1.0  # Bandwidth (arbitrary units)
        self.noise_power = 1e-3
        self.tx_power = 1.0
        self.path_loss_exp = 2.5
        # Fixed initial UAV and user positions (random, but fixed for the run)
        self.init_uav_positions = np.random.rand(self.num_uavs, 3)
        self.init_uav_positions[:, 0] *= self.grid_size[0]
        self.init_uav_positions[:, 1] *= self.grid_size[1]
        self.init_uav_positions[:, 2] *= self.grid_size[2]
        self.init_user_positions = np.random.rand(self.num_users, 3)
        self.init_user_positions[:, 0] *= self.grid_size[0]
        self.init_user_positions[:, 1] *= self.grid_size[1]
        self.init_user_positions[:, 2] = 0

        #additions for handover,cij
        self.association = np.zeros(self.num_users, dtype=int)
        self.prev_association = np.zeros(self.num_users, dtype=int)
        self.handover_log = []
        self.window_size = 5
        self.handover_penalty = 2.0  

        
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        # Set UAV and user positions to fixed initial positions
        self.uav_positions = self.init_uav_positions.copy()
        # Target positions can still be random per episode
        self.target_positions = np.random.rand(self.num_uavs, 3)
        self.target_positions[:, 0] *= self.grid_size[0]
        self.target_positions[:, 1] *= self.grid_size[1]
        self.target_positions[:, 2] *= self.grid_size[2]
        self.user_positions = self.init_user_positions.copy()
        self.association = np.zeros(self.num_users, dtype=int)
        self.prev_association = np.zeros(self.num_users, dtype=int)
        self.handover_log = []

        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return self.uav_positions.flatten()
    
    def _check_collisions(self) -> bool:
        """Check for collisions between UAVs."""
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                distance = np.linalg.norm(self.uav_positions[i] - self.uav_positions[j])
                if distance < 1.0:  # Collision threshold
                    return True
        return False
    
    def _check_boundary_violations(self) -> bool:
        """Check if any UAV is outside the grid boundaries (horizontal position constraint)."""
        x = self.uav_positions[:, 0]
        y = self.uav_positions[:, 1]
        return np.any(x < 0) or np.any(x >= self.grid_size[0]) or np.any(y < 0) or np.any(y >= self.grid_size[1])
    
    def _check_altitude_violations(self) -> bool:
        """Check if any UAV is outside the allowed altitude range [10, 15] meters."""
        z = self.uav_positions[:, 2]
        return np.any(z < 10) or np.any(z > 15)
    
    def _check_qos_violations(self, user_rates: np.ndarray) -> bool:
        """Check if any user rate is below the minimum required rate (QoS)."""
        return np.any(user_rates < self.min_user_rate)
    
    def _calculate_reward(self, user_rates=None) -> float:
        reward = 0.0
        if self._check_collisions():
            reward += self.collision_penalty
        if self._check_boundary_violations():
            reward += self.violation_penalty
        if self._check_altitude_violations():
            reward += self.violation_penalty
        # QoS: add positive bonus per user meeting QoS, no penalty for violation
        if user_rates is not None:
            reward += np.sum(user_rates >= self.min_user_rate) * self.qos_bonus
        for i in range(self.num_uavs):
            distance = np.linalg.norm(self.uav_positions[i] - self.target_positions[i])
            reward -= distance
        #  Handover penalty
        handover_count = np.sum(self.association != self.prev_association)
        reward -= handover_count * self.handover_penalty
        return reward
    
    def _compute_throughput(self) -> (float, np.ndarray):
        """Compute total network throughput (sum-rate) and per-user rates, associating each user to its nearest UAV."""
        total_rate = 0.0
        user_rates = np.zeros(self.num_users)
        fading = np.random.exponential(1.0, size=(self.num_uavs, self.num_users))
        new_association = np.zeros(self.num_users, dtype=int)
        # Find nearest UAV for each user
        for user_idx in range(self.num_users):
            user_pos = self.user_positions[user_idx]
            distances = np.linalg.norm(self.uav_positions - user_pos, axis=1)
            new_association[user_idx] = np.argmin(distances)

    # Track handovers (keep existing variable names)
        self.prev_association = self.association.copy()
        self.association = new_association

        # Update handover log
        handovers = (self.association != self.prev_association).astype(int)
        self.handover_log.append(handovers)
        if len(self.handover_log) > self.window_size:
            self.handover_log.pop(0)

        # Compute SINR-based rate
        for user_idx in range(self.num_users):
            uav_idx = self.association[user_idx]
            user_pos = self.user_positions[user_idx]
            uav_pos = self.uav_positions[uav_idx]
            d_ij = np.linalg.norm(user_pos - uav_pos) + 1e-3
            signal = self.tx_power * fading[uav_idx, user_idx] / (d_ij ** self.path_loss_exp)

            interference = 0.0
            for other_uav in range(self.num_uavs):
                if other_uav != uav_idx:
                    d_kj = np.linalg.norm(self.uav_positions[other_uav] - user_pos) + 1e-3
                    interference += self.tx_power * fading[other_uav, user_idx] / (d_kj ** self.path_loss_exp)

            sinr = signal / (self.noise_power + interference)
            rate = self.B * np.log2(1 + sinr)
            user_rates[user_idx] = rate
            total_rate += rate

        return total_rate, user_rates
    
    def _move_users(self):
        """Move users minimally (simulate movement inside homes)."""
        # Only move x and y, not z
        movement = np.random.uniform(-0.2, 0.2, size=(self.num_users, 2))
        self.user_positions[:, :2] += movement
        # Clip to grid bounds
        self.user_positions[:, 0] = np.clip(self.user_positions[:, 0], 0, self.grid_size[0])
        self.user_positions[:, 1] = np.clip(self.user_positions[:, 1], 0, self.grid_size[1])
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        # Convert actions to movements
        movements = {
            0: np.array([0, 0, 1]),   # up
            1: np.array([0, 0, -1]),  # down
            2: np.array([0, 1, 0]),   # right
            3: np.array([0, -1, 0]),  # left
            4: np.array([1, 0, 0]),   # forward
            5: np.array([-1, 0, 0])   # backward
        }
        
        # Update UAV positions
        for i, action in enumerate(actions):
            self.uav_positions[i] += movements[action]
        
        # Move users minimally before reward/throughput calculation
        self._move_users()
        # Calculate throughput and per-user rates
        throughput, user_rates = self._compute_throughput()
        # Calculate reward with QoS info
        reward = self._calculate_reward(user_rates)
        # Check violations
        qos_violation = self._check_qos_violations(user_rates)
        # Additional info
        info = {
            'collisions': self._check_collisions(),
            'violations': self._check_boundary_violations(),
            'altitude_violation': self._check_altitude_violations(),
            'qos_violation': qos_violation,
            'throughput': throughput,
            'user_rates': user_rates,
            'handover_count': int(np.sum(self.association != self.prev_association))
        }
        done = self._check_collisions() or self._check_boundary_violations() or self._check_altitude_violations() or qos_violation
        return self._get_observation(), reward, done, False, info 

    def render(self, mode='human', show=True):
        """Visualize UAVs, users, and associations in 2D (nearest UAV association)."""
        plt.figure(figsize=(7, 7))
        # Plot users
        user_xy = self.user_positions[:, :2]
        plt.scatter(user_xy[:, 0], user_xy[:, 1], c='green', label='Users')
        # Plot UAVs
        uav_xy = self.uav_positions[:, :2]
        boundary_viol = self._check_boundary_violations()
        altitude_viol = self._check_altitude_violations()
        uav_color = 'red' if boundary_viol or altitude_viol else 'blue'
        plt.scatter(uav_xy[:, 0], uav_xy[:, 1], c=uav_color, s=100, marker='o', label='UAVs')
        # Draw associations (user to nearest UAV)
        for user_idx in range(self.num_users):
            user = user_xy[user_idx]
            min_dist = float('inf')
            nearest_uav = 0
            for uav_idx in range(self.num_uavs):
                uav = self.uav_positions[uav_idx]
                d = np.linalg.norm(self.user_positions[user_idx] - uav)
                if d < min_dist:
                    min_dist = d
                    nearest_uav = uav_idx
            uav = uav_xy[nearest_uav]
            plt.plot([user[0], uav[0]], [user[1], uav[1]], 'gray', alpha=0.5)
        plt.xlim(0, self.grid_size[0])
        plt.ylim(0, self.grid_size[1])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('UAV and User Positions (Nearest UAV Association)')
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        else:
            plt.close()

    # Optional: log positions for episode playback
    def log_step(self, log_dict):
        log_dict.setdefault('uav_positions', []).append(self.uav_positions.copy())
        log_dict.setdefault('user_positions', []).append(self.user_positions.copy()) 