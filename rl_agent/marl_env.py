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
        # UE velocity tracking (POINT 2: UE velocity considerations)
        self.user_velocities = None  # Track user velocities
        self.user_velocity_categories = None  # LOW, MEDIUM, HIGH mobility
        # Packet drop parameters (POINT 2: UE velocity considerations)
        self.packet_drop_rates = {
            'LOW_MOBILITY': 0.01,    # 1% packet drop for low mobility users
            'MEDIUM_MOBILITY': 0.05, # 5% packet drop for medium mobility users
            'HIGH_MOBILITY': 0.15    # 15% packet drop for high mobility users
        }
        # Channel parameters
        self.B = 1.0  # Bandwidth (arbitrary units)
        self.noise_power = 1e-3
        self.tx_power = 1.0
        self.path_loss_exp_los = 2.0  # Path loss exponent for LOS (lower due to clear path)
        self.path_loss_exp_nlos = 3.5  # Path loss exponent for NLOS (higher due to obstacles)
        
        # Height-dependent LOS probability parameters
        # At lower altitude, LOS blocked by obstacles; at higher altitude, LOS increases
        # After threshold, distance-based path loss dominates
        self.los_prob_base = 0.1  # Base LOS probability at ground level
        self.los_prob_max = 0.95  # Maximum LOS probability at optimal height
        self.optimal_height = 20.0  # Approximate optimal height (meters) where LOS probability peaks
        self.height_sensitivity = 0.15  # Controls how quickly LOS probability changes with height
        self.height_decay_factor = 0.02  # Controls decay after optimal height
        
        # Shadowing/fading parameters
        self.shadowing_std_los = 3.0  # Shadowing standard deviation for LOS (dB)
        self.shadowing_std_nlos = 8.0  # Shadowing standard deviation for NLOS (dB)
        # Fixed initial UAV and user positions (random, but fixed for the run)
        self.init_uav_positions = np.random.rand(self.num_uavs, 3)
        self.init_uav_positions[:, 0] *= self.grid_size[0]
        self.init_uav_positions[:, 1] *= self.grid_size[1]
        # Initialize heights in range [10, 30] meters to start near optimal region
        self.init_uav_positions[:, 2] = 10 + np.random.rand(self.num_uavs) * 20
        self.init_user_positions = np.random.rand(self.num_users, 3)
        self.init_user_positions[:, 0] *= self.grid_size[0]
        self.init_user_positions[:, 1] *= self.grid_size[1]
        self.init_user_positions[:, 2] = 0
        
        # Initialize user velocities and categories (POINT 2: UE velocity considerations)
        self.init_user_velocities = np.random.uniform(0, 30, self.num_users)  # 0-30 m/s
        self.init_user_velocity_categories = []
        for vel in self.init_user_velocities:
            if vel < 5:
                self.init_user_velocity_categories.append('LOW_MOBILITY')
            elif vel < 15:
                self.init_user_velocity_categories.append('MEDIUM_MOBILITY')
            else:
                self.init_user_velocity_categories.append('HIGH_MOBILITY')

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
        # Target heights in range [10, 30] meters
        self.target_positions[:, 2] = 10 + np.random.rand(self.num_uavs) * 20
        self.user_positions = self.init_user_positions.copy()
        self.user_velocities = self.init_user_velocities.copy()
        self.user_velocity_categories = self.init_user_velocity_categories.copy()
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
        """Check if any UAV is outside the allowed altitude range [5, 50] meters.
        Expanded range to allow learning optimal height."""
        z = self.uav_positions[:, 2]
        return np.any(z < 5) or np.any(z > 50)
    
    def _check_qos_violations(self, user_rates: np.ndarray) -> bool:
        """Check if any user rate is below the minimum required rate (QoS)."""
        return np.any(user_rates < self.min_user_rate)
    
    def _compute_los_probability(self, uav_height: float) -> float:
        """
        Compute LOS probability based on UAV height.
        
        At lower altitude, LOS can be blocked by obstacles (buildings, trees).
        At higher altitude, LOS probability increases.
        After optimal height, distance-based path loss dominates.
        
        Args:
            uav_height: Height of the UAV in meters
            
        Returns:
            LOS probability in [0, 1]
        """
        if uav_height < 5:
            # Very low altitude, high blockage probability
            return self.los_prob_base
        
        # LOS probability increases with height up to optimal height
        if uav_height <= self.optimal_height:
            # Increasing phase: LOS probability grows with height
            height_factor = (uav_height - 5) / (self.optimal_height - 5)
            los_prob = self.los_prob_base + (self.los_prob_max - self.los_prob_base) * (
                1 - np.exp(-self.height_sensitivity * height_factor * (self.optimal_height - 5))
            )
        else:
            # Decreasing phase: After optimal height, distance-based path loss dominates
            # LOS probability still high but slightly decreases due to increased distance
            excess_height = uav_height - self.optimal_height
            los_prob = self.los_prob_max * np.exp(-self.height_decay_factor * excess_height)
        
        return np.clip(los_prob, self.los_prob_base, self.los_prob_max)
    
    def _calculate_packet_drops(self, user_idx: int, uav_idx: int) -> float:
        """
        POINT 2: Calculate packet drop rate based on user velocity and UAV height.
        Higher UAVs provide longer LOS duration for mobile users, reducing packet drops.
        """
        velocity_category = self.user_velocity_categories[user_idx]
        base_drop_rate = self.packet_drop_rates[velocity_category]
        
        # Height factor: higher UAVs reduce packet drops for mobile users
        # Using height range [5, 50] meters from observation space
        uav_height = self.uav_positions[uav_idx, 2]
        min_height, max_height = 5.0, 50.0
        height_factor = 1.0 - min((uav_height - min_height) / (max_height - min_height), 1.0)
        
        # Mobile users benefit more from height
        if velocity_category == 'HIGH_MOBILITY':
            adjusted_drop_rate = base_drop_rate * (0.5 + 0.5 * height_factor)
        elif velocity_category == 'MEDIUM_MOBILITY':
            adjusted_drop_rate = base_drop_rate * (0.7 + 0.3 * height_factor)
        else:
            adjusted_drop_rate = base_drop_rate * (0.9 + 0.1 * height_factor)
        
        return adjusted_drop_rate
    
    def _compute_path_loss(self, distance: float, is_los: bool, uav_height: float) -> float:
        """
        Compute path loss based on LOS/NLOS state and distance.
        
        Args:
            distance: 3D distance between UAV and user
            is_los: Boolean indicating LOS or NLOS state
            uav_height: Height of UAV (for additional height-dependent loss)
            
        Returns:
            Path loss factor (to be divided from signal power)
        """
        if is_los:
            path_loss_exp = self.path_loss_exp_los
        else:
            path_loss_exp = self.path_loss_exp_nlos
        
        # Base path loss: distance^path_loss_exp
        base_path_loss = distance ** path_loss_exp
        
        # Additional height-dependent loss factor (small effect)
        # At very high altitudes, additional loss due to increased distance
        height_loss_factor = 1.0 + 0.01 * max(0, uav_height - self.optimal_height)
        
        return base_path_loss * height_loss_factor
    
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
        """
        Compute total network throughput (sum-rate) and per-user rates, associating each user to its nearest UAV.
        Incorporates height-dependent LOS probability and path loss models.
        """
        total_rate = 0.0
        user_rates = np.zeros(self.num_users)
        new_association = np.zeros(self.num_users, dtype=int)
        
        # Find nearest UAV for each user (based on 3D distance)
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

        # Compute SINR-based rate with height-dependent LOS and path loss
        for user_idx in range(self.num_users):
            uav_idx = self.association[user_idx]
            user_pos = self.user_positions[user_idx]
            uav_pos = self.uav_positions[uav_idx]
            uav_height = uav_pos[2]
            
            # Compute 3D distance
            d_ij = np.linalg.norm(user_pos - uav_pos) + 1e-3
            
            # Determine LOS/NLOS state based on height-dependent probability
            los_prob = self._compute_los_probability(uav_height)
            is_los = np.random.random() < los_prob
            
            # Compute path loss based on LOS/NLOS state
            path_loss = self._compute_path_loss(d_ij, is_los, uav_height)
            
            # Compute shadowing/fading (different for LOS vs NLOS)
            if is_los:
                # LOS: lower shadowing variance
                shadowing_db = np.random.normal(0, self.shadowing_std_los)
            else:
                # NLOS: higher shadowing variance
                shadowing_db = np.random.normal(0, self.shadowing_std_nlos)
            
            # Convert shadowing from dB to linear scale
            shadowing_linear = 10 ** (shadowing_db / 10.0)
            
            # Signal power with path loss and shadowing
            signal = self.tx_power * shadowing_linear / path_loss

            # Compute interference from other UAVs
            interference = 0.0
            for other_uav in range(self.num_uavs):
                if other_uav != uav_idx:
                    other_uav_pos = self.uav_positions[other_uav]
                    other_uav_height = other_uav_pos[2]
                    d_kj = np.linalg.norm(other_uav_pos - user_pos) + 1e-3
                    
                    # Determine LOS/NLOS for interfering link
                    other_los_prob = self._compute_los_probability(other_uav_height)
                    other_is_los = np.random.random() < other_los_prob
                    
                    # Compute path loss for interfering link
                    other_path_loss = self._compute_path_loss(d_kj, other_is_los, other_uav_height)
                    
                    # Shadowing for interfering link
                    if other_is_los:
                        other_shadowing_db = np.random.normal(0, self.shadowing_std_los)
                    else:
                        other_shadowing_db = np.random.normal(0, self.shadowing_std_nlos)
                    other_shadowing_linear = 10 ** (other_shadowing_db / 10.0)
                    
                    interference += self.tx_power * other_shadowing_linear / other_path_loss

            # Compute SINR and rate
            sinr = signal / (self.noise_power + interference)
            rate = self.B * np.log2(1 + sinr)
            
            # Apply packet drop based on velocity and height (POINT 2)
            packet_drop_rate = self._calculate_packet_drops(user_idx, uav_idx)
            effective_rate = rate * (1 - packet_drop_rate)
            
            user_rates[user_idx] = effective_rate
            total_rate += effective_rate

        return total_rate, user_rates
    
    def _move_users(self):
        """Move users with velocity considerations (POINT 2: UE velocity)."""
        for i in range(self.num_users):
            velocity = self.user_velocities[i]
            # Movement distance based on velocity (assuming 1 time step = 1 second)
            max_movement = velocity * 0.1  # Scale down for simulation
            
            movement = np.random.uniform(-max_movement, max_movement, size=2)
            self.user_positions[i, :2] += movement
            
            # Update velocity occasionally
            if np.random.random() < 0.1:  # 10% chance to change velocity
                self.user_velocities[i] = max(0, self.user_velocities[i] + np.random.normal(0, 2))
                
                # Update velocity category
                vel = self.user_velocities[i]
                if vel < 5:
                    self.user_velocity_categories[i] = 'LOW_MOBILITY'
                elif vel < 15:
                    self.user_velocity_categories[i] = 'MEDIUM_MOBILITY'
                else:
                    self.user_velocity_categories[i] = 'HIGH_MOBILITY'
        
        # Clip to grid bounds
        self.user_positions[:, 0] = np.clip(self.user_positions[:, 0], 0, self.grid_size[0])
        self.user_positions[:, 1] = np.clip(self.user_positions[:, 1], 0, self.grid_size[1])
    
    def step(self, actions: List[int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        # Convert actions to movements
        # Height movements are in meters (1m steps for fine control)
        movements = {
            0: np.array([0, 0, 1.0]),   # up (increase height)
            1: np.array([0, 0, -1.0]),  # down (decrease height)
            2: np.array([0, 1, 0]),     # right
            3: np.array([0, -1, 0]),    # left
            4: np.array([1, 0, 0]),     # forward
            5: np.array([-1, 0, 0])     # backward
        }
        
        # Update UAV positions
        for i, action in enumerate(actions):
            self.uav_positions[i] += movements[action]
        
        # Clip heights to valid range [5, 50] meters
        self.uav_positions[:, 2] = np.clip(self.uav_positions[:, 2], 5, 50)
        
        # POINT 2: Prefer higher altitudes for high mobility users
        for uav_idx in range(self.num_uavs):
            served_user_indices = [i for i in range(self.num_users) if self.association[i] == uav_idx]
            if served_user_indices:
                high_mobility_users = sum(1 for i in served_user_indices 
                                        if self.user_velocity_categories[i] == 'HIGH_MOBILITY')
                if high_mobility_users > len(served_user_indices) * 0.3:  # >30% high mobility
                    # Increase height slightly for high mobility users
                    self.uav_positions[uav_idx, 2] = min(self.uav_positions[uav_idx, 2] + 2.0, 50)
        
        # Move users with velocity before reward/throughput calculation
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