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
    def __init__(self,
                 num_uavs: int = 3,
                 num_users: int = 20,
                 grid_size: Tuple[int, int, int] = (10, 10, 5),
                 device: torch.device = None,
                 min_user_rate: float = 0.5,
                 qos_bonus: float = 10.0,
                 ue_height_range: Tuple[float, float] = (0.0, 5.0)):
        super(MARLEnv, self).__init__()
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.grid_size = grid_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_user_rate = min_user_rate
        self.qos_bonus = qos_bonus
        # NEW: UE heights in [0, 5] m (configurable)
        self.ue_height_range = ue_height_range
        
        # Define action space (6 possible movements: up, down, left, right, forward, backward)
        self.action_space = spaces.MultiDiscrete([6] * num_uavs)
        
        # Define observation space (x, y, z coordinates for each UAV)
        # Note: z (height) can now vary in expanded range [5, 50] meters
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 5] * num_uavs),
            high=np.array([grid_size[0], grid_size[1], 50] * num_uavs),
            dtype=np.float32
        )
        
        # Initialize UAV positions
        self.uav_positions = None
        self.target_positions = None
        self.collision_penalty = -100.0  # Strong penalty for collisions
        self.violation_penalty = -50.0   # Penalty for boundary violations
        # Initialize user positions
        self.user_positions = None
        self.uav_loads = None
        self.movement_load_penalty = 0.5  # Weight for load-aware movement penalty
        # UE velocity tracking (POINT 2: UE velocity considerations)
        self.user_velocities = None
        self.user_velocity_categories = None  # LOW_MOBILITY, MEDIUM_MOBILITY, HIGH_MOBILITY
        
        # Height-dependent LOS probability model (POINT 1: Height-dependent LOS probability)
        self.optimal_height = 20.0  # Optimal height in meters for maximizing LOS probability
        self.height_spread = 10.0   # Spread parameter for the Gaussian-like height dependency
        
        # Path loss parameters (distance- and height-dependent)
        self.path_loss_exponent_los = 2.0
        self.path_loss_exponent_nlos = 3.5
        self.shadowing_std_los = 2.0
        self.shadowing_std_nlos = 8.0
        
        # Capacity calculation parameters
        self.bandwidth = 10e6  # 10 MHz
        self.noise_figure = 9  # in dB
        self.noise_power_dbm = -174 + 10 * np.log10(self.bandwidth) + self.noise_figure
        self.transmit_power_dbm = 30  # 1 Watt
        
        # Maximum steps per episode
        self.max_steps = 50
        self.current_step = 0
        
        # For plotting and debugging
        self.enable_plotting = False
        self.fig = None
        self.ax = None

    def height_dependent_los_probability(self, height: float) -> float:
        """
        Calculate LOS probability as a function of UAV height.
        Uses a Gaussian-like function centered at an optimal height.
        """
        exponent = -((height - self.optimal_height) ** 2) / (2 * self.height_spread ** 2)
        los_prob = np.exp(exponent)
        return float(np.clip(los_prob, 0.0, 1.0))

    def calculate_path_loss(self, distance: float, height: float) -> Tuple[float, bool]:
        """
        Calculate path loss (in dB) based on distance and height-dependent LOS probability.
        Returns (path_loss_db, is_los).
        """
        los_prob = self.height_dependent_los_probability(height)
        is_los = np.random.rand() < los_prob
        
        # Free-space path loss at reference distance 1m
        frequency = 2.4e9  # 2.4 GHz
        c = 3e8  # Speed of light
        lambda_c = c / frequency
        fspl_db_1m = 20 * np.log10(4 * np.pi / lambda_c)
        
        if is_los:
            path_loss_exponent = self.path_loss_exponent_los
            shadowing_std = self.shadowing_std_los
        else:
            path_loss_exponent = self.path_loss_exponent_nlos
            shadowing_std = self.shadowing_std_nlos
        
        # Distance-dependent path loss
        path_loss_db = fspl_db_1m + 10 * path_loss_exponent * np.log10(max(distance, 1.0))
        
        # Shadowing
        shadowing_db = np.random.normal(0, shadowing_std)
        path_loss_db += shadowing_db
        
        return float(path_loss_db), is_los

    def reset(self, seed=None, options=None):
        """
        Reset the environment with updated initializations for UAV and users.
        """
        super().reset(seed=seed)
        
        # Initialize UAV positions with diversity in height
        self.init_uav_positions = np.random.rand(self.num_uavs, 3)
        self.init_uav_positions[:, 0] *= self.grid_size[0]
        self.init_uav_positions[:, 1] *= self.grid_size[1]
        # Initialize heights in range [10, 30] meters to start near optimal region
        self.init_uav_positions[:, 2] = 10 + np.random.rand(self.num_uavs) * 20
        
        # Initialize user positions with heights between ue_height_range[0] and ue_height_range[1]
        self.init_user_positions = np.random.rand(self.num_users, 3)
        self.init_user_positions[:, 0] *= self.grid_size[0]
        self.init_user_positions[:, 1] *= self.grid_size[1]
        # NEW: UE heights from specified range (default 0–5 m)
        self.init_user_positions[:, 2] = np.random.uniform(
            self.ue_height_range[0],
            self.ue_height_range[1],
            self.num_users
        )
        
        # Initialize user velocities and categories (POINT 2: UE velocity considerations)
        self.init_user_velocities = np.random.uniform(0, 30, self.num_users)  # 0-30 m/s
        self.init_user_velocity_categories = []
        for v in self.init_user_velocities:
            if v < 5:
                self.init_user_velocity_categories.append('LOW_MOBILITY')
            elif v < 15:
                self.init_user_velocity_categories.append('MEDIUM_MOBILITY')
            else:
                self.init_user_velocity_categories.append('HIGH_MOBILITY')
        
        # Copy initial positions to current positions
        self.uav_positions = self.init_uav_positions.copy()
        self.user_positions = self.init_user_positions.copy()
        self.user_velocities = self.init_user_velocities.copy()
        self.user_velocity_categories = self.init_user_velocity_categories.copy()
        
        # Initialize UAV loads
        self.uav_loads = np.zeros(self.num_uavs)
        
        # Reset step counter
        self.current_step = 0
        
        # Initialize target positions (for potential trajectory planning)
        self.target_positions = self.uav_positions.copy()
        
        # Optional plotting setup
        if self.enable_plotting:
            self._setup_plot()
        
        obs = self._get_obs()
        return obs, {}

    def _setup_plot(self):
        """Setup 3D plot for visualization."""
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(0, self.grid_size[0])
        self.ax.set_ylim(0, self.grid_size[1])
        self.ax.set_zlim(0, 50)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Height (m)')
        self.ax.set_title('UAV and User Positions')
    
    def _update_plot(self):
        """Update 3D plot with current positions."""
        if self.fig is None or self.ax is None:
            self._setup_plot()
        
        self.ax.cla()
        self.ax.set_xlim(0, self.grid_size[0])
        self.ax.set_ylim(0, self.grid_size[1])
        self.ax.set_zlim(0, 50)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Height (m)')
        self.ax.set_title('UAV and User Positions')
        
        # Plot UAVs
        self.ax.scatter(self.uav_positions[:, 0],
                        self.uav_positions[:, 1],
                        self.uav_positions[:, 2],
                        c='r', marker='^', s=80, label='UAVs')
        
        # Plot Users
        self.ax.scatter(self.user_positions[:, 0],
                        self.user_positions[:, 1],
                        self.user_positions[:, 2],
                        c='b', marker='o', s=40, alpha=0.5, label='Users')
        
        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

    def _get_obs(self) -> np.ndarray:
        """Get the observation (UAV positions)."""
        return self.uav_positions.flatten().astype(np.float32)

    def step(self, actions: List[int]):
        """
        Take a step in the environment.
        Actions: list of actions for each UAV (0: up, 1: down, 2: left, 3: right, 4: forward, 5: backward)
        """
        self.current_step += 1
        
        # Apply actions to UAV positions
        for i, action in enumerate(actions):
            if action == 0:   # up
                self.uav_positions[i, 2] += 1
            elif action == 1: # down
                self.uav_positions[i, 2] -= 1
            elif action == 2: # left
                self.uav_positions[i, 0] -= 1
            elif action == 3: # right
                self.uav_positions[i, 0] += 1
            elif action == 4: # forward
                self.uav_positions[i, 1] += 1
            elif action == 5: # backward
                self.uav_positions[i, 1] -= 1
        
        # Clip positions to grid boundaries and enforce height limits [5, 50]
        self.uav_positions[:, 0] = np.clip(self.uav_positions[:, 0], 0, self.grid_size[0])
        self.uav_positions[:, 1] = np.clip(self.uav_positions[:, 1], 0, self.grid_size[1])
        self.uav_positions[:, 2] = np.clip(self.uav_positions[:, 2], 5, 50)
        
        # Calculate reward, throughput, fairness, collisions
        reward, info = self._calculate_reward_and_metrics()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Update plot if enabled
        if self.enable_plotting:
            self._update_plot()
        
        obs = self._get_obs()
        return obs, reward, done, False, info

    def _calculate_reward_and_metrics(self) -> Tuple[float, Dict]:
        """
        Calculate reward and performance metrics including throughput and fairness.
        """
        distances = np.zeros((self.num_uavs, self.num_users))
        heights = np.zeros((self.num_uavs, self.num_users))
        gains_db = np.zeros((self.num_uavs, self.num_users))
        is_los_matrix = np.zeros((self.num_uavs, self.num_users), dtype=bool)
        
        # Calculate distances, heights, and channel gains
        for i in range(self.num_uavs):
            for j in range(self.num_users):
                dx = self.uav_positions[i, 0] - self.user_positions[j, 0]
                dy = self.uav_positions[i, 1] - self.user_positions[j, 1]
                dz = self.uav_positions[i, 2] - self.user_positions[j, 2]
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                distances[i, j] = distance
                heights[i, j] = self.uav_positions[i, 2]
                path_loss_db, is_los = self.calculate_path_loss(distance, self.uav_positions[i, 2])
                gains_db[i, j] = -path_loss_db
                is_los_matrix[i, j] = is_los
        
        # Convert gains from dB to linear scale
        gains_linear = 10 ** (gains_db / 10)
        
        # Calculate SNR and user rates for each UAV-user pair
        noise_power_linear = 10 ** (self.noise_power_dbm / 10) / 1000
        tx_power_linear = 10 ** (self.transmit_power_dbm / 10) / 1000
        
        sinr = (tx_power_linear * gains_linear) / noise_power_linear
        rates = self.bandwidth * np.log2(1 + sinr)
        
        # Assign users to the best UAV based on SNR
        best_uav_indices = np.argmax(sinr, axis=0)
        uav_user_rates = [[] for _ in range(self.num_uavs)]
        
        for j in range(self.num_users):
            uav_idx = best_uav_indices[j]
            uav_user_rates[uav_idx].append(rates[uav_idx, j])
        
        # Calculate total throughput and fairness
        total_throughput = 0.0
        user_rates = []
        qos_satisfied_users = 0
        
        for i in range(self.num_uavs):
            if uav_user_rates[i]:
                uav_rates = np.array(uav_user_rates[i])
                total_throughput += np.sum(uav_rates)
                user_rates.extend(uav_rates.tolist())
                qos_satisfied_users += np.sum(uav_rates >= self.min_user_rate * 1e6)  # QoS in bps
        
        user_rates = np.array(user_rates) if user_rates else np.array([0.0])
        mean_rate = np.mean(user_rates)
        fairness = (mean_rate ** 2) / (np.mean(user_rates ** 2) + 1e-9)
        
        # Collision check: if any two UAVs are too close
        collisions = False
        min_distance = 1.0  # Minimum allowed distance between UAVs
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                dx = self.uav_positions[i, 0] - self.uav_positions[j, 0]
                dy = self.uav_positions[i, 1] - self.uav_positions[j, 1]
                dz = self.uav_positions[i, 2] - self.uav_positions[j, 2]
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                if distance < min_distance:
                    collisions = True
                    break
        
        # Reward components
        reward = total_throughput / 1e6  # Normalize to Mbps
        reward += self.qos_bonus * (qos_satisfied_users / self.num_users)
        
        if collisions:
            reward += self.collision_penalty
        
        info = {
            'throughput': float(total_throughput),
            'fairness': float(fairness),
            'collisions': collisions,
            'mean_rate': float(mean_rate),
            'user_rates': user_rates,
            'uav_positions': self.uav_positions.copy(),
            'user_positions': self.user_positions.copy(),
            'is_los': is_los_matrix,
            'distances': distances,
            'heights': heights
        }
        
        return float(reward), info

    def render(self):
        """Render the environment (optional)."""
        if self.enable_plotting:
            self._update_plot()

    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None