"""
Novel Adaptive Non-Stationary MARL Algorithm.
Combines both types of non-stationarity:
- Type a: UE movement non-stationarity
- Type b: UAV movement performative RL non-stationarity

This algorithm adapts to both sources of non-stationarity simultaneously.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict


class ContextAwareQNetwork(nn.Module):
    """Q-network that takes state + non-stationary context."""
    def __init__(self, state_dim: int, context_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ContextAwareQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state, context], dim=-1)
        return self.network(combined)


class ContextAwareMixingNetwork(nn.Module):
    """Mixing network that incorporates non-stationary context."""
    def __init__(self, num_agents: int, global_state_dim: int, context_dim: int, hidden_dim: int = 64):
        super(ContextAwareMixingNetwork, self).__init__()
        self.num_agents = num_agents
        input_dim = global_state_dim + context_dim
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hyper_b1 = nn.Linear(input_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, q_values: torch.Tensor, global_state: torch.Tensor, 
                context: torch.Tensor) -> torch.Tensor:
        """Mix Q-values using hypernetworks with context.
        Args:
            q_values: [batch, num_agents] individual Q-values
            global_state: [batch, global_state_dim] global state
            context: [batch, context_dim] non-stationary context
        """
        batch_size = q_values.size(0)
        combined_state = torch.cat([global_state, context], dim=-1)
        
        # First layer
        w1 = torch.abs(self.hyper_w1(combined_state))
        w1 = w1.view(batch_size, self.num_agents, -1)
        b1 = self.hyper_b1(combined_state)
        b1 = b1.view(batch_size, 1, -1)
        
        hidden = torch.bmm(q_values.unsqueeze(1), w1) + b1
        hidden = torch.relu(hidden)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(combined_state))
        w2 = w2.view(batch_size, -1, 1)
        b2 = self.hyper_b2(combined_state)
        
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.squeeze()


class AdaptiveNonStationaryMARL:
    """
    Novel MARL algorithm that adapts to both types of non-stationarity:
    1. Type a: UE movement patterns (mobility non-stationarity)
    2. Type b: UAV movement causing performative RL (policy-induced non-stationarity)
    
    Uses context-aware Q-networks that adapt based on non-stationary context.
    """
    def __init__(self, num_agents: int, state_dim: int, action_dim: int,
                 global_state_dim: int, context_dim: int = 7,  # 4 from non-stationary + 3 from UE mobility
                 learning_rate: float = 0.001, gamma: float = 0.99, epsilon: float = 0.1,
                 device: torch.device = None, buffer_size: int = 10000, batch_size: int = 64,
                 target_update: int = 100):
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.context_dim = context_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_count = 0
        
        # Context-aware Q-networks
        self.q_networks = [
            ContextAwareQNetwork(state_dim, context_dim, action_dim).to(self.device)
            for _ in range(num_agents)
        ]
        
        # Context-aware mixing network
        self.mixing_network = ContextAwareMixingNetwork(
            num_agents, global_state_dim, context_dim).to(self.device)
        
        # Target networks
        self.target_q_networks = [
            ContextAwareQNetwork(state_dim, context_dim, action_dim).to(self.device)
            for _ in range(num_agents)
        ]
        self.target_mixing_network = ContextAwareMixingNetwork(
            num_agents, global_state_dim, context_dim).to(self.device)
        
        # Copy weights to target networks
        for i in range(num_agents):
            self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        # Optimizers
        self.q_optimizers = [optim.Adam(net.parameters(), lr=learning_rate) 
                            for net in self.q_networks]
        self.mixing_optimizer = optim.Adam(self.mixing_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
    
    @staticmethod
    def improved_association_algorithm(sinr: np.ndarray, rates: np.ndarray, 
                                      distances: np.ndarray, is_los_matrix: np.ndarray,
                                      uav_positions: np.ndarray, user_positions: np.ndarray,
                                      user_velocities: np.ndarray = None,
                                      coverage_heatmap: np.ndarray = None,
                                      grid_cells_x: int = 10, grid_cells_y: int = 10,
                                      cell_size_x: float = 1.0, cell_size_y: float = 1.0) -> np.ndarray:
        """
        Improved UE-UAV association algorithm that accounts for:
        1. Performative RL: Prefer associations in well-covered areas (from heatmap)
        2. UE Mobility (Type a non-stationarity): Prefer stable associations for high-mobility UEs
        3. Load balancing: Distribute users across UAVs
        4. LOS probability: Prefer LOS connections
        
        This is the association algorithm specific to the novel AdaptiveNonStationaryMARL algorithm.
        """
        num_uavs, num_users = sinr.shape
        best_uav_indices = np.zeros(num_users, dtype=int)
        
        # Track current load per UAV
        current_loads = np.zeros(num_uavs)
        
        # Score matrix: combines SINR, coverage quality, mobility stability, and load
        scores = sinr.copy()
        
        # 1. Performative RL: Boost scores for well-covered areas
        if coverage_heatmap is not None:
            for j in range(num_users):
                x_grid = int(user_positions[j, 0] / cell_size_x)
                y_grid = int(user_positions[j, 1] / cell_size_y)
                if 0 <= x_grid < grid_cells_x and 0 <= y_grid < grid_cells_y:
                    coverage_boost = coverage_heatmap[x_grid, y_grid]
                    # Normalize and apply boost (well-covered areas get higher scores)
                    scores[:, j] += coverage_boost * 0.1  # 10% boost from coverage
        
        # 2. UE Mobility (Type a): Prefer higher UAVs for high-mobility UEs (longer LOS duration)
        if user_velocities is not None:
            for j in range(num_users):
                velocity = user_velocities[j]
                if velocity > 15:  # High mobility
                    # Boost scores for higher UAVs (better LOS duration)
                    height_bonus = uav_positions[:, 2] / 50.0  # Normalize by max height
                    scores[:, j] += height_bonus * 0.15  # 15% boost for height
                elif velocity > 5:  # Medium mobility
                    height_bonus = uav_positions[:, 2] / 50.0
                    scores[:, j] += height_bonus * 0.08  # 8% boost
        
        # 3. LOS preference: Boost scores for LOS connections
        los_bonus = is_los_matrix.astype(float) * 0.2  # 20% boost for LOS
        scores += los_bonus
        
        # 4. Load balancing: Penalize heavily loaded UAVs
        for j in range(num_users):
            # Calculate load penalty
            load_penalty = current_loads / (np.max(current_loads) + 1e-9)
            scores[:, j] -= load_penalty * 0.1  # 10% penalty for load
        
        # Greedy assignment: assign each user to best UAV considering all factors
        for j in range(num_users):
            best_uav = np.argmax(scores[:, j])
            best_uav_indices[j] = best_uav
            current_loads[best_uav] += 1  # Update load for next assignment
        
        return best_uav_indices
    
    def get_association_function(self):
        """Get the association function to be used by the environment."""
        def association_func(sinr, rates, distances, is_los_matrix, uav_positions, user_positions,
                           user_velocities, coverage_heatmap, grid_cells_x, grid_cells_y,
                           cell_size_x, cell_size_y):
            return self.improved_association_algorithm(
                sinr, rates, distances, is_los_matrix, uav_positions, user_positions,
                user_velocities, coverage_heatmap, grid_cells_x, grid_cells_y,
                cell_size_x, cell_size_y
            )
        return association_func
    
    def extract_context(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Extract non-stationary context from global state.
        Assumes last context_dim elements of global_state are context features.
        """
        if global_state.shape[-1] >= self.global_state_dim + self.context_dim:
            # Extract context (last context_dim elements)
            context = global_state[..., -self.context_dim:]
        else:
            # No context available, use zeros
            if len(global_state.shape) == 1:
                context = torch.zeros(self.context_dim).to(self.device)
            else:
                context = torch.zeros(global_state.shape[0], self.context_dim).to(self.device)
        return context
    
    def get_action(self, state: torch.Tensor, agent_id: int, 
                   global_state: torch.Tensor = None) -> int:
        """Get action with context awareness."""
        # Extract context
        if global_state is not None:
            context = self.extract_context(global_state)
        else:
            context = torch.zeros(self.context_dim).to(self.device)
        
        # Ensure context matches state batch dimension
        if len(state.shape) == 1:
            context = context.unsqueeze(0)
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_networks[agent_id](state, context)
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        return q_values.argmax().item()
    
    def store_transition(self, states: List[torch.Tensor], actions: List[int],
                        rewards: List[float], next_states: List[torch.Tensor],
                        dones: List[bool], global_state: torch.Tensor,
                        next_global_state: torch.Tensor):
        """Store transition with context."""
        self.replay_buffer.append((
            states, actions, rewards, next_states, dones,
            global_state, next_global_state
        ))
    
    def update(self):
        """Update with context-aware learning."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states_batch = [torch.stack([b[0][i] for b in batch]).to(self.device) 
                       for i in range(self.num_agents)]
        actions_batch = torch.tensor([[b[1][i] for b in batch] for i in range(self.num_agents)]).to(self.device)
        rewards_batch = torch.tensor([[b[2][i] for b in batch] for i in range(self.num_agents)], 
                                     dtype=torch.float32).to(self.device)
        next_states_batch = [torch.stack([b[3][i] for b in batch]).to(self.device) 
                            for i in range(self.num_agents)]
        dones_batch = torch.tensor([[b[4][i] for b in batch] for i in range(self.num_agents)], 
                                   dtype=torch.bool).to(self.device)
        global_states_batch = torch.stack([b[5] for b in batch]).to(self.device)
        next_global_states_batch = torch.stack([b[6] for b in batch]).to(self.device)
        
        # Extract context for current and next states
        contexts_batch = self.extract_context(global_states_batch)
        next_contexts_batch = self.extract_context(next_global_states_batch)
        
        # Compute Q-values with context
        q_values = []
        for i in range(self.num_agents):
            q_vals = self.q_networks[i](states_batch[i], contexts_batch)
            q_values.append(q_vals.gather(1, actions_batch[i].unsqueeze(1)))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = []
            for i in range(self.num_agents):
                next_q_vals = self.target_q_networks[i](next_states_batch[i], next_contexts_batch)
                next_q_values.append(next_q_vals.max(1)[0])
            
            # Mix Q-values
            next_q_vals_tensor = torch.stack(next_q_values, dim=1)
            target_q_total = self.target_mixing_network(
                next_q_vals_tensor, next_global_states_batch[:, :self.global_state_dim], next_contexts_batch)
            
            # Compute targets
            rewards_total = rewards_batch.sum(dim=0)
            targets = rewards_total + self.gamma * target_q_total * (~dones_batch.any(dim=0))
        
        # Mix current Q-values
        q_vals_tensor = torch.stack([q.squeeze() for q in q_values], dim=1)
        q_total = self.mixing_network(q_vals_tensor, global_states_batch[:, :self.global_state_dim], contexts_batch)
        
        # Compute loss
        loss = F.mse_loss(q_total, targets)
        
        # Update
        for optimizer in self.q_optimizers:
            optimizer.zero_grad()
        self.mixing_optimizer.zero_grad()
        
        loss.backward()
        
        for i, optimizer in enumerate(self.q_optimizers):
            torch.nn.utils.clip_grad_norm_(self.q_networks[i].parameters(), 10)
            optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), 10)
        self.mixing_optimizer.step()
        
        # Update target networks
        if self.update_count % self.target_update == 0:
            for i in range(self.num_agents):
                self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
            self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
        self.update_count += 1
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'q_networks': [net.state_dict() for net in self.q_networks],
            'mixing_network': self.mixing_network.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        for i, net in enumerate(self.q_networks):
            net.load_state_dict(checkpoint['q_networks'][i])
        self.mixing_network.load_state_dict(checkpoint['mixing_network'])
