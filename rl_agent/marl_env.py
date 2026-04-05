"""
Multi-Agent Reinforcement Learning Environment for UAV coordination.

This environment supports:
- Height-dependent Line-of-Sight (LoS) probability modeling
- UE velocity considerations for mobile users
- Load-aware UAV movement penalties
- Non-Stationary RL: Time-varying traffic demand and channel conditions
- Performative RL: UE distribution adapts to long-term coverage patterns

Non-Stationary RL Features:
    - Traffic demand drifts over episodes (simulating time-of-day patterns)
    - Channel conditions vary (simulating weather/interference changes)
    - Observation space includes context features when enabled

Performative RL Features:
    - Coverage heatmap tracks spatial coverage quality
    - UE distribution weights adapt based on historical coverage
    - Optional policy-induced occupancy (UAV visitation grid) blended into UE placement updates
    - Creates feedback loop: policy affects UE distribution, which affects future policy
    - Call end_episode() after each episode to update performative parameters

Signal-map observations (optional):
    - Per-cell aggregated best-server received power (UE locations not in obs)
    - Each agent's block: pose_i + flat(map) + replicated non-stationary context
    - Use env.agent_obs_dim for slicing; optional reward terms: handovers, movement, spread
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
                 ue_height_range: Tuple[float, float] = (0.0, 5.0),
                 enable_non_stationary: bool = False,
                 enable_performative: bool = False,
                 enable_signal_map_obs: bool = True,
                 handover_penalty: float = 0.0,
                 movement_penalty: float = 0.0,
                 spread_bonus: float = 0.0,
                 use_occupancy_performative: bool = False,
                 enable_occupancy_obs: bool = False,
                 occupancy_performative_weight: float = 0.35):
        super(MARLEnv, self).__init__()
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.grid_size = grid_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_user_rate = min_user_rate
        self.qos_bonus = qos_bonus
        # NEW: UE heights in [0, 5] m (configurable)
        self.ue_height_range = ue_height_range
        
        # Non-Stationary and Performative RL flags
        self.enable_non_stationary = enable_non_stationary
        self.enable_performative = enable_performative
        self.enable_signal_map_obs = enable_signal_map_obs
        self.handover_penalty = handover_penalty
        self.movement_penalty = movement_penalty
        self.spread_bonus = spread_bonus
        self.use_occupancy_performative = bool(
            use_occupancy_performative and enable_performative
        )
        self.enable_occupancy_obs = bool(enable_occupancy_obs)
        self.occupancy_performative_weight = float(
            np.clip(occupancy_performative_weight, 0.0, 1.0)
        )
        self.episode_count = 0  # Track episodes for non-stationary drift
        self.step_count = 0  # Track steps for gradual changes
        
        # Non-Stationary: Time-varying parameters
        self.base_traffic_demand = 1.0  # Base traffic demand multiplier
        self.traffic_drift_rate = 0.001  # How fast traffic patterns change per episode
        self.channel_drift_rate = 0.0005  # Channel condition drift rate
        
        # Performative RL: UE distribution parameters
        self.coverage_heatmap = None  # Grid-based coverage tracking
        self.ue_distribution_weights = None  # Spatial weights for UE spawning
        self.performative_update_rate = 0.1  # How fast UE distribution adapts to coverage
        self.coverage_history_window = 50  # Episodes to average for coverage map
        self.coverage_history = []  # Store recent coverage maps
        self.occupancy_history: List[np.ndarray] = []  # UAV visitation grids (same window as coverage)
        
        # Initialize coverage heatmap (grid cells for spatial tracking)
        self.grid_cells_x = 10  # Number of grid cells in x direction
        self.grid_cells_y = 10  # Number of grid cells in y direction
        self.cell_size_x = grid_size[0] / self.grid_cells_x
        self.cell_size_y = grid_size[1] / self.grid_cells_y
        if self.enable_performative:
            self.coverage_heatmap = np.ones((self.grid_cells_x, self.grid_cells_y)) / (self.grid_cells_x * self.grid_cells_y)
            self.ue_distribution_weights = np.ones((self.grid_cells_x, self.grid_cells_y)) / (self.grid_cells_x * self.grid_cells_y)
        
        # Define action space (6 possible movements: up, down, left, right, forward, backward)
        self.action_space = spaces.MultiDiscrete([6] * num_uavs)
        
        self._signal_map_size = self.grid_cells_x * self.grid_cells_y
        # Per-agent block: [x,y,z] + optional flat signal map + optional occupancy map + optional NS context (7)
        self._ns_context_dim = 7 if enable_non_stationary else 0
        self._signal_flat_dim = self._signal_map_size if enable_signal_map_obs else 0
        self._occupancy_flat_dim = self._signal_map_size if self.enable_occupancy_obs else 0
        self.agent_obs_dim = (
            3 + self._signal_flat_dim + self._occupancy_flat_dim + self._ns_context_dim
        )
        self.total_obs_dim = self.agent_obs_dim * num_uavs
        
        low_per_agent = (
            [0.0, 0.0, 5.0]
            + [0.0] * self._signal_flat_dim
            + [0.0] * self._occupancy_flat_dim
            + [-np.inf] * self._ns_context_dim
        )
        high_per_agent = (
            [float(grid_size[0]), float(grid_size[1]), 50.0]
            + [1.0] * self._signal_flat_dim
            + [1.0] * self._occupancy_flat_dim
            + [np.inf] * self._ns_context_dim
        )
        low = np.array(low_per_agent * num_uavs, dtype=np.float32)
        high = np.array(high_per_agent * num_uavs, dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self._last_signal_map_flat = np.zeros(self._signal_map_size, dtype=np.float32)
        self._last_occupancy_flat = np.zeros(self._signal_map_size, dtype=np.float32)
        self.prev_association = None
        self._step_total_displacement = 0.0
        
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
        
        # UE Movement Non-Stationarity (Type a): Track UE mobility patterns
        self.ue_mobility_history = []  # Track UE position history for mobility analysis
        self.ue_mobility_rates = None  # Average mobility rate per UE
        self.ue_direction_vectors = None  # Current direction of movement per UE
        self.mobility_non_stationarity_window = 10  # Steps to track for mobility patterns
        
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
        # Nominal per-slot energy (J): hover + downlink TX + propulsion ∝ UAV displacement (m).
        # Used for episode energy-efficiency metrics (Mbit/J); slots assumed 1 s for rate→bit accounting.
        self._nominal_slot_duration_s = 1.0
        self._hover_power_per_uav_w = 72.0
        self._propulsion_j_per_m = 6.0
        self._p_tx_watt = float(10 ** (self.transmit_power_dbm / 10.0) / 1000.0)
        
        # Maximum steps per episode
        self.max_steps = 50
        self.current_step = 0
        
        # Optional association function from agent (for novel algorithms)
        self.association_function = None
        
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
        
        # Apply non-stationary drift
        if self.enable_non_stationary:
            self._apply_non_stationary_drift()
        
        # Initialize UAV positions with diversity in height
        self.init_uav_positions = np.random.rand(self.num_uavs, 3)
        self.init_uav_positions[:, 0] *= self.grid_size[0]
        self.init_uav_positions[:, 1] *= self.grid_size[1]
        # Initialize heights in range [10, 30] meters to start near optimal region
        self.init_uav_positions[:, 2] = 10 + np.random.rand(self.num_uavs) * 20
        
        # Initialize user positions (with performative distribution if enabled)
        if self.enable_performative and self.ue_distribution_weights is not None:
            self.init_user_positions = self._sample_users_from_distribution()
        else:
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
        
        # Initialize UE mobility non-stationarity tracking (Type a)
        self.ue_mobility_history = [self.user_positions.copy()]
        self.ue_mobility_rates = np.zeros(self.num_users)
        self.ue_direction_vectors = np.zeros((self.num_users, 2))
        
        # Initialize UAV loads
        self.uav_loads = np.zeros(self.num_uavs)
        
        # Reset step counter
        self.current_step = 0
        self.step_count = 0
        
        # Initialize target positions (for potential trajectory planning)
        self.target_positions = self.uav_positions.copy()
        
        self.prev_association = None
        self._step_total_displacement = 0.0
        self._last_signal_map_flat = np.zeros(self._signal_map_size, dtype=np.float32)
        self._last_occupancy_flat = np.zeros(self._signal_map_size, dtype=np.float32)
        
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

    def _get_non_stationary_context_list(self) -> List[float]:
        """Seven global non-stationary scalars (replicated in each agent's observation block)."""
        if not self.enable_non_stationary:
            return []
        ctx = [
            self.episode_count / 1000.0,
            float(self.base_traffic_demand),
            float(np.sin(self.episode_count * 2 * np.pi / 100.0)),
            float(np.cos(self.episode_count * 2 * np.pi / 100.0)),
        ]
        if len(self.ue_mobility_history) > 1:
            avg_mobility = float(np.mean(self.ue_mobility_rates)) if self.ue_mobility_rates is not None else 0.0
            direction_variance = float(np.var(self.ue_direction_vectors)) if self.ue_direction_vectors is not None else 0.0
            mobile_ues = int(np.sum(self.ue_mobility_rates > 5.0)) if self.ue_mobility_rates is not None else 0
            mobility_concentration = mobile_ues / max(self.num_users, 1.0)
        else:
            avg_mobility = 0.0
            direction_variance = 0.0
            mobility_concentration = 0.0
        ctx.extend([
            avg_mobility / 30.0,
            direction_variance,
            mobility_concentration,
        ])
        return ctx

    def _get_obs(self) -> np.ndarray:
        """
        Per-UAV observation blocks: [pose_i, flat(signal_map), non_stationary_context].
        UE locations are not included; signal map is aggregated received power per cell.
        """
        obs: List[float] = []
        sig = self._last_signal_map_flat.tolist() if self.enable_signal_map_obs else []
        occ = self._last_occupancy_flat.tolist() if self.enable_occupancy_obs else []
        ctx = self._get_non_stationary_context_list()
        for i in range(self.num_uavs):
            obs.extend(self.uav_positions[i].tolist())
            obs.extend(sig)
            obs.extend(occ)
            obs.extend(ctx)
        return np.array(obs, dtype=np.float32)

    def step(self, actions: List[int]):
        """
        Take a step in the environment.
        Actions: list of actions for each UAV (0: up, 1: down, 2: left, 3: right, 4: forward, 5: backward)
        """
        self.current_step += 1
        self.step_count += 1
        
        pos_before = self.uav_positions.copy()
        
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
        
        self._step_total_displacement = float(
            np.sum(np.linalg.norm(self.uav_positions - pos_before, axis=1))
        )
        
        # Move users (for velocity considerations)
        self._move_users()
        
        # Calculate reward, throughput, fairness, collisions
        reward, info = self._calculate_reward_and_metrics()
        
        # Update coverage heatmap for performative RL
        if self.enable_performative and 'user_rates' in info:
            self._update_coverage_heatmap(info['user_rates'])
        
        self._update_uav_occupancy_maps()
        
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
        
        # Simple SNR-based association (can be overridden by agent's association function)
        if hasattr(self, 'association_function') and self.association_function is not None:
            # Use agent-provided association function
            best_uav_indices = self.association_function(sinr, rates, distances, is_los_matrix, 
                                                         self.uav_positions, self.user_positions,
                                                         self.user_velocities, self.coverage_heatmap,
                                                         self.grid_cells_x, self.grid_cells_y,
                                                         self.cell_size_x, self.cell_size_y)
        else:
            # Default: simple SNR-based association
            best_uav_indices = np.argmax(sinr, axis=0)
        
        if self.prev_association is None or self.prev_association.shape != best_uav_indices.shape:
            handovers = 0
        else:
            handovers = int(np.sum(self.prev_association != best_uav_indices))
        self.prev_association = best_uav_indices.copy()
        
        if self.enable_signal_map_obs:
            rx_power = tx_power_linear * gains_linear[best_uav_indices, np.arange(self.num_users)]
            cell_best = np.zeros((self.grid_cells_x, self.grid_cells_y), dtype=np.float32)
            for j in range(self.num_users):
                xg = int(self.user_positions[j, 0] / self.cell_size_x)
                yg = int(self.user_positions[j, 1] / self.cell_size_y)
                if 0 <= xg < self.grid_cells_x and 0 <= yg < self.grid_cells_y:
                    cell_best[xg, yg] = max(float(cell_best[xg, yg]), float(rx_power[j]))
            mx = float(np.max(cell_best))
            if mx > 1e-30:
                cell_best /= mx
            self._last_signal_map_flat = cell_best.flatten()
        else:
            self._last_signal_map_flat = np.zeros(self._signal_map_size, dtype=np.float32)
        
        if self.num_uavs >= 2:
            xy = self.uav_positions[:, :2]
            pair_d = []
            for i in range(self.num_uavs):
                for j in range(i + 1, self.num_uavs):
                    pair_d.append(float(np.linalg.norm(xy[i] - xy[j])))
            mean_uav_ground_dist = float(np.mean(pair_d))
        else:
            mean_uav_ground_dist = 0.0
        
        uav_user_rates = [[] for _ in range(self.num_uavs)]
        
        for j in range(self.num_users):
            uav_idx = best_uav_indices[j]
            uav_user_rates[uav_idx].append(rates[uav_idx, j])
        
        # Apply non-stationary traffic demand multiplier
        traffic_multiplier = self.base_traffic_demand if self.enable_non_stationary else 1.0
        
        # Calculate total throughput and fairness
        total_throughput = 0.0
        user_rates = []
        qos_satisfied_users = 0
        
        for i in range(self.num_uavs):
            if uav_user_rates[i]:
                uav_rates = np.array(uav_user_rates[i]) * traffic_multiplier
                total_throughput += np.sum(uav_rates)
                user_rates.extend(uav_rates.tolist())
                qos_satisfied_users += np.sum(uav_rates >= self.min_user_rate * 1e6)  # QoS in bps
        
        user_rates = np.array(user_rates) if user_rates else np.array([0.0])
        mean_rate = np.mean(user_rates)
        fairness = (mean_rate ** 2) / (np.mean(user_rates ** 2) + 1e-9)
        # ---- EXTRA METRICS FOR YOUR PLOTS ----
        min_rate = float(np.min(user_rates))  # bps

        qos_threshold_bps = self.min_user_rate * 1e6  # since min_user_rate is in Mbps
        qos_ratio = float(np.mean(user_rates >= qos_threshold_bps))  # fraction of users meeting QoS

        # "Goodness" (define it as you like; this is a reasonable default)
        # Mix QoS satisfaction + fairness + mean rate (scaled)
        goodness = 0.5 * qos_ratio + 0.3 * float(fairness) + 0.2 * float(mean_rate / 1e6)
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
        
        reward -= self.handover_penalty * handovers
        reward -= self.movement_penalty * self._step_total_displacement
        reward += self.spread_bonus * mean_uav_ground_dist
        
        dt = self._nominal_slot_duration_s
        e_comm = self.num_uavs * self._p_tx_watt * dt
        e_hover = self.num_uavs * self._hover_power_per_uav_w * dt
        e_prop = self._propulsion_j_per_m * self._step_total_displacement
        step_energy_j = float(e_comm + e_hover + e_prop)
        
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
            'min_rate': float(min_rate),         # bps
            'qos_ratio': float(qos_ratio),        # 0..1
            'goodness': float(goodness),
            'heights': heights,
            'handovers': handovers,
            'total_movement': float(self._step_total_displacement),
            'mean_uav_distance': mean_uav_ground_dist,
            'step_energy_j': step_energy_j,
        }
        
        return float(reward), info

    def render(self):
        """Render the environment (optional)."""
        if self.enable_plotting:
            self._update_plot()

    def _apply_non_stationary_drift(self):
        """Apply non-stationary drift to environment parameters."""
        if not self.enable_non_stationary:
            return
        
        # Drift traffic demand (simulates time-of-day, seasonal changes)
        drift = np.sin(self.episode_count * self.traffic_drift_rate) * 0.2
        self.base_traffic_demand = 1.0 + drift
        
        # Drift channel conditions (simulates weather, interference changes)
        channel_drift = np.cos(self.episode_count * self.channel_drift_rate) * 0.1
        # Adjust shadowing variance slightly
        self.shadowing_std_los = max(2.0, 3.0 + channel_drift)
        self.shadowing_std_nlos = max(7.0, 8.0 + channel_drift * 2)
    
    def _move_users(self):
        """Move users based on their velocities (for velocity considerations).
        Also tracks UE mobility for non-stationarity modeling (Type a)."""
        if self.user_velocities is None:
            return
        
        prev_positions = self.user_positions.copy()
        
        for j in range(self.num_users):
            velocity = self.user_velocities[j]
            # Random direction movement (can be made more realistic with waypoint model)
            angle = np.random.uniform(0, 2 * np.pi)
            dx = velocity * np.cos(angle) * 0.1  # Scale down movement
            dy = velocity * np.sin(angle) * 0.1
            
            self.user_positions[j, 0] += dx
            self.user_positions[j, 1] += dy
            
            # Wrap around boundaries
            self.user_positions[j, 0] = self.user_positions[j, 0] % self.grid_size[0]
            self.user_positions[j, 1] = self.user_positions[j, 1] % self.grid_size[1]
            
            # Track movement for non-stationarity (Type a)
            if self.enable_non_stationary:
                movement_distance = np.sqrt(dx**2 + dy**2)
                self.ue_mobility_rates[j] = movement_distance * 10  # Convert to m/s equivalent
                if movement_distance > 0:
                    self.ue_direction_vectors[j] = np.array([dx, dy]) / movement_distance
                else:
                    self.ue_direction_vectors[j] = np.array([0.0, 0.0])
            
            # Occasionally update velocity category
            if np.random.rand() < 0.01:  # 1% chance per step
                if self.user_velocities[j] < 5:
                    self.user_velocity_categories[j] = 'LOW_MOBILITY'
                elif self.user_velocities[j] < 15:
                    self.user_velocity_categories[j] = 'MEDIUM_MOBILITY'
            else:
                    self.user_velocity_categories[j] = 'HIGH_MOBILITY'
        
        # Update mobility history for non-stationarity tracking (Type a)
        if self.enable_non_stationary:
            self.ue_mobility_history.append(self.user_positions.copy())
            if len(self.ue_mobility_history) > self.mobility_non_stationarity_window:
                self.ue_mobility_history.pop(0)
    
    def set_association_function(self, association_func):
        """Set a custom association function from the agent."""
        self.association_function = association_func
    
    def _sample_users_from_distribution(self) -> np.ndarray:
        """Sample user positions from performative distribution weights."""
        if self.ue_distribution_weights is None:
            # Fallback to uniform distribution
            positions = np.random.rand(self.num_users, 3)
            positions[:, 0] *= self.grid_size[0]
            positions[:, 1] *= self.grid_size[1]
            return positions
        
        # Sample grid cells based on weights
        flat_weights = self.ue_distribution_weights.flatten()
        flat_weights = flat_weights / (flat_weights.sum() + 1e-9)
        
        positions = np.zeros((self.num_users, 3))
        for i in range(self.num_users):
            # Sample a grid cell
            cell_idx = np.random.choice(len(flat_weights), p=flat_weights)
            x_cell = cell_idx % self.grid_cells_x
            y_cell = cell_idx // self.grid_cells_x
            
            # Sample uniformly within the cell
            x = (x_cell + np.random.rand()) * self.cell_size_x
            y = (y_cell + np.random.rand()) * self.cell_size_y
            
            positions[i, 0] = np.clip(x, 0, self.grid_size[0])
            positions[i, 1] = np.clip(y, 0, self.grid_size[1])
        
        return positions
    
    def _uav_cell_occupancy_counts(self) -> np.ndarray:
        """Per-cell counts of UAVs (ground projection); policy-induced spatial occupancy."""
        occ = np.zeros((self.grid_cells_x, self.grid_cells_y), dtype=np.float64)
        for i in range(self.num_uavs):
            xg = int(self.uav_positions[i, 0] / self.cell_size_x)
            yg = int(self.uav_positions[i, 1] / self.cell_size_y)
            if 0 <= xg < self.grid_cells_x and 0 <= yg < self.grid_cells_y:
                occ[xg, yg] += 1.0
        return occ

    def _update_uav_occupancy_maps(self) -> None:
        """Track UAV visitation for optional observations and performative UE placement."""
        occ = self._uav_cell_occupancy_counts()
        if self.enable_occupancy_obs:
            flat = occ.astype(np.float32).flatten()
            mx = float(np.max(flat))
            if mx > 1e-9:
                flat = flat / mx
            self._last_occupancy_flat = flat
        if self.use_occupancy_performative:
            self.occupancy_history.append(occ.copy())
            if len(self.occupancy_history) > self.coverage_history_window:
                self.occupancy_history.pop(0)

    def _update_coverage_heatmap(self, user_rates: np.ndarray):
        """Update the coverage heatmap based on current user rates."""
        if not self.enable_performative:
            return
        
        current_coverage_map = np.zeros((self.grid_cells_x, self.grid_cells_y))
        for user_idx in range(min(len(user_rates), self.num_users)):
            x_grid = int(self.user_positions[user_idx, 0] / self.cell_size_x)
            y_grid = int(self.user_positions[user_idx, 1] / self.cell_size_y)
            
            if 0 <= x_grid < self.grid_cells_x and 0 <= y_grid < self.grid_cells_y:
                current_coverage_map[x_grid, y_grid] += user_rates[user_idx]
        
        # Exponential moving average for heatmap
        if self.coverage_heatmap is None:
            self.coverage_heatmap = current_coverage_map.copy()
        else:
            self.coverage_heatmap = self.coverage_heatmap * 0.9 + current_coverage_map * 0.1
        
        # Store for episodic averaging
        self.coverage_history.append(current_coverage_map.copy())
        if len(self.coverage_history) > self.coverage_history_window:
            self.coverage_history.pop(0)
    
    def _update_ue_distribution_weights(self):
        """Update the UE distribution weights based on long-term average coverage."""
        if not self.enable_performative or not self.coverage_history:
            return
        
        # Average recent coverage maps
        avg_coverage = np.mean(self.coverage_history, axis=0)
        
        # Normalize to get a probability distribution
        normalized_coverage = avg_coverage / (np.sum(avg_coverage) + 1e-9)
        
        target = normalized_coverage
        if self.use_occupancy_performative and self.occupancy_history:
            avg_occ = np.mean(self.occupancy_history, axis=0)
            normalized_occ = avg_occ / (np.sum(avg_occ) + 1e-9)
            w = self.occupancy_performative_weight
            target = (1.0 - w) * normalized_coverage + w * normalized_occ
        
        # Update distribution weights gradually
        if self.ue_distribution_weights is None:
            self.ue_distribution_weights = target.copy()
        else:
            self.ue_distribution_weights = (
                (1 - self.performative_update_rate) * self.ue_distribution_weights +
                self.performative_update_rate * target
            )
        
        # Add some exploration to avoid complete concentration
        exploration_weight = 0.1
        uniform_dist = np.ones_like(self.ue_distribution_weights) / (
            self.grid_cells_x * self.grid_cells_y
        )
        self.ue_distribution_weights = (
            (1 - exploration_weight) * self.ue_distribution_weights +
            exploration_weight * uniform_dist
        )
        self.ue_distribution_weights = self.ue_distribution_weights / (self.ue_distribution_weights.sum() + 1e-9)
    
    def end_episode(self):
        """Called at end of episode to update non-stationary and performative parameters."""
        self.episode_count += 1
        if self.enable_performative:
            self._update_ue_distribution_weights()
            # Clear history for next averaging window periodically
            if self.episode_count % self.coverage_history_window == 0:
                self.coverage_history = []
                self.occupancy_history = []
    
    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None