import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple

class Qmix(nn.Module):
    def __init__(self, num_uavs: int, state_dim: int, action_dim: int, learning_rate: float = 0.001, gamma: float = 0.99, epsilon: float = 0.1, device: torch.device = None):
        super(Qmix, self).__init__()
        self.num_uavs = num_uavs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Centralized Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim * num_uavs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * num_uavs)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        # states: [batch, num_uavs, state_dim]
        batch_size = states.size(0)
        flattened_states = states.view(batch_size, -1)  # [batch, num_uavs * state_dim]
        q_values = self.q_network(flattened_states)  # [batch, num_uavs * action_dim]
        return q_values.view(batch_size, self.num_uavs, self.action_dim)
    
    def get_action(self, state: torch.Tensor, uav_idx: int, epsilon: float = None) -> int:
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state = state.float().to(self.device).unsqueeze(0)  # [1, num_uavs, state_dim]
            q_values = self.forward(state)[0, uav_idx]  # [action_dim]
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
        # Gather Q-values for taken actions
        action = action.long().unsqueeze(0).unsqueeze(2)  # [1, num_uavs, 1]
        current_q_taken = current_q.gather(2, action).squeeze(2)  # [1, num_uavs]
        max_next_q = next_q.max(dim=2)[0]  # [1, num_uavs]
        target_q = reward + (1 - done) * self.gamma * max_next_q
        loss = nn.MSELoss()(current_q_taken, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item() 