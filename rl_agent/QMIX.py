"""
QMIX implementation for multi-agent reinforcement learning.
Nonlinear value mixing for complex agent interactions.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple


class QNetwork(nn.Module):
    """Q-network for a single agent."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MixingNetwork(nn.Module):
    """Nonlinear mixing network for QMIX."""
    def __init__(self, num_agents: int, state_dim: int, hidden_dim: int = 64):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, q_values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Mix Q-values using hypernetworks.
        Args:
            q_values: [batch, num_agents] individual Q-values
            states: [batch, state_dim] global state
        """
        batch_size = q_values.size(0)
        
        # First layer
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(batch_size, self.num_agents, -1)
        b1 = self.hyper_b1(states)
        b1 = b1.view(batch_size, 1, -1)
        
        hidden = torch.bmm(q_values.unsqueeze(1), w1) + b1
        hidden = torch.relu(hidden)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(batch_size, -1, 1)
        b2 = self.hyper_b2(states)
        
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.squeeze()


class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *transition):
        self.buffer.append(tuple(transition))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return map(list, zip(*batch))
    
    def __len__(self):
        return len(self.buffer)


class QMIX:
    """QMIX algorithm for multi-agent reinforcement learning."""
    def __init__(self, num_agents: int, state_dim: int, action_dim: int,
                 global_state_dim: int = None, learning_rate: float = 0.001,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 device: torch.device = None, buffer_size: int = 10000,
                 batch_size: int = 64, target_update: int = 100):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim if global_state_dim else state_dim * num_agents
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Create Q-networks and target networks for each agent
        self.q_networks = [QNetwork(state_dim, action_dim).to(self.device)
                          for _ in range(num_agents)]
        self.target_q_networks = [QNetwork(state_dim, action_dim).to(self.device)
                                  for _ in range(num_agents)]
        for i in range(num_agents):
            self.target_q_networks[i].load_state_dict(
                self.q_networks[i].state_dict())
        
        # Mixing networks
        self.mixing_network = MixingNetwork(num_agents, self.global_state_dim).to(
            self.device)
        self.target_mixing_network = MixingNetwork(
            num_agents, self.global_state_dim).to(self.device)
        self.target_mixing_network.load_state_dict(
            self.mixing_network.state_dict())
        
        # Optimizers
        self.q_optimizers = [optim.Adam(net.parameters(), lr=learning_rate)
                            for net in self.q_networks]
        self.mixing_optimizer = optim.Adam(self.mixing_network.parameters(),
                                          lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.learn_step = 0
    
    def get_action(self, state: torch.Tensor, agent_idx: int) -> int:
        """Get action for a specific agent using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state = state.float().unsqueeze(0)
            q_values = self.q_networks[agent_idx](state)[0]
            return q_values.argmax().item()
    
    def store_transition(self, states, actions, rewards, next_states, dones,
                        global_state=None, next_global_state=None):
        """Store transition in replay buffer."""
        if global_state is None:
            global_state = torch.cat(states, dim=0) if isinstance(states[0], torch.Tensor) else np.concatenate(states)
        if next_global_state is None:
            next_global_state = torch.cat(next_states, dim=0) if isinstance(next_states[0], torch.Tensor) else np.concatenate(next_states)
        self.replay_buffer.push(states, actions, rewards, next_states, dones,
                               global_state, next_global_state)
    
    def update(self):
        """Update Q-networks and mixing network."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        transitions = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, global_states, next_global_states = transitions
        
        # Convert to tensors
        states = torch.stack([torch.stack(s).float() for s in states]).to(
            self.device)
        next_states = torch.stack([torch.stack(s).float() for s in next_states]).to(
            self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        
        # Convert global states
        if isinstance(global_states[0], np.ndarray):
            global_states = torch.stack([torch.tensor(gs).float() for gs in global_states]).to(self.device)
        else:
            global_states = torch.stack(global_states).to(self.device)
        
        if isinstance(next_global_states[0], np.ndarray):
            next_global_states = torch.stack([torch.tensor(ngs).float() for ngs in next_global_states]).to(self.device)
        else:
            next_global_states = torch.stack(next_global_states).to(self.device)
        
        # Compute individual Q-values
        q_values = []
        target_q_values = []
        for agent in range(self.num_agents):
            q_vals = self.q_networks[agent](states[:, agent, :])
            q_values.append(q_vals.gather(1, actions[:, agent].unsqueeze(1)).squeeze(1))
            
            with torch.no_grad():
                target_q_vals = self.target_q_networks[agent](next_states[:, agent, :])
                target_q_values.append(target_q_vals.max(1)[0])
        
        q_values = torch.stack(q_values, dim=1)
        target_q_values = torch.stack(target_q_values, dim=1)
        
        # Mix Q-values
        q_total = self.mixing_network(q_values, global_states)
        with torch.no_grad():
            target_q_total = self.target_mixing_network(target_q_values, next_global_states)
        
        # Compute target
        rewards_sum = rewards.sum(dim=1)
        dones_any = dones.any(dim=1).float()
        target_q_total = rewards_sum + (1 - dones_any) * self.gamma * target_q_total.squeeze()
        
        # Loss
        loss = nn.MSELoss()(q_total.squeeze(), target_q_total)
        
        # Update
        for optimizer in self.q_optimizers:
            optimizer.zero_grad()
        self.mixing_optimizer.zero_grad()
        loss.backward()
        for optimizer in self.q_optimizers:
            optimizer.step()
        self.mixing_optimizer.step()
        
        # Update target networks
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            for i in range(self.num_agents):
                self.target_q_networks[i].load_state_dict(
                    self.q_networks[i].state_dict())
            self.target_mixing_network.load_state_dict(
                self.mixing_network.state_dict())
    
    def save(self, path: str) -> None:
        """Save model checkpoints."""
        torch.save({
            'q_networks': [net.state_dict() for net in self.q_networks],
            'target_q_networks': [net.state_dict() for net in self.target_q_networks],
            'mixing_network': self.mixing_network.state_dict(),
            'target_mixing_network': self.target_mixing_network.state_dict(),
            'q_optimizers': [opt.state_dict() for opt in self.q_optimizers],
            'mixing_optimizer': self.mixing_optimizer.state_dict()
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoints."""
        checkpoint = torch.load(path)
        for i, state_dict in enumerate(checkpoint['q_networks']):
            self.q_networks[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['target_q_networks']):
            self.target_q_networks[i].load_state_dict(state_dict)
        self.mixing_network.load_state_dict(checkpoint['mixing_network'])
        self.target_mixing_network.load_state_dict(
            checkpoint['target_mixing_network'])

