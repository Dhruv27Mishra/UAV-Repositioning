import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple

class VDN(nn.Module):
    def __init__(self, num_uavs: int, state_dim: int, action_dim: int, learning_rate: float = 0.001, gamma: float = 0.99, epsilon: float = 0.1, device: torch.device = None):
        super(VDN, self).__init__()
        self.num_uavs = num_uavs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Individual Q-networks for each UAV
        self.q_networks = nn.ModuleList([nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ) for _ in range(num_uavs)])
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # states: [batch, num_uavs, state_dim]
        q_values = []
        for i in range(self.num_uavs):
            q_values.append(self.q_networks[i](states[:, i, :]))  # [batch, action_dim]
        # Stack to shape [batch, num_uavs, action_dim]
        return torch.stack(q_values, dim=1)
    
    def get_action(self, state: torch.Tensor, uav_idx: int, epsilon: float = None) -> int:
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state = state.float().to(self.device).unsqueeze(0)  # [1, state_dim]
            q_values = self.q_networks[uav_idx](state)[0]  # [action_dim]
            return torch.argmax(q_values).item()
    
    def update(self, state, action, reward, next_state, done):
        # state: [1, num_uavs, state_dim]
        # action: [num_uavs]
        # reward: scalar or [num_uavs]
        # next_state: [1, num_uavs, state_dim]
        # done: scalar or [num_uavs]
        state = state.float()
        next_state = next_state.float()
        current_q = self.forward(state)  # [1, num_uavs, action_dim]
        next_q = self.forward(next_state)  # [1, num_uavs, action_dim]
        # Gather Q-values for taken actions for each agent
        action = action.long().unsqueeze(0).unsqueeze(2)  # [1, num_uavs, 1]
        current_q_taken = current_q.gather(2, action).squeeze(2)  # [1, num_uavs]
        max_next_q = next_q.max(dim=2)[0]  # [1, num_uavs]
        # Sum across agents for global Q
        current_q_sum = current_q_taken.sum(dim=1)  # [1]
        max_next_q_sum = max_next_q.sum(dim=1)  # [1]
        # If reward is per-agent, sum; if scalar, keep as is
        if isinstance(reward, torch.Tensor) and reward.ndim > 1:
            reward_sum = reward.sum(dim=1)
        else:
            reward_sum = reward
        if isinstance(done, torch.Tensor) and done.ndim > 1:
            done_sum = done.sum(dim=1)
        else:
            done_sum = done
        target_q = reward_sum + (1 - done_sum) * self.gamma * max_next_q_sum
        loss = nn.MSELoss()(current_q_sum, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item() 