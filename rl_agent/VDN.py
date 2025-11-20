"""
VDN (Value Decomposition Networks) implementation for multi-agent reinforcement learning.
Linear decomposition of Q-values per agent.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple
from rl_agent.IQL import QNetwork, ReplayBuffer


class VDN:
    """VDN algorithm for multi-agent reinforcement learning."""
    def __init__(self, num_agents: int, state_dim: int, action_dim: int,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 0.1, device: torch.device = None,
                 buffer_size: int = 10000, batch_size: int = 64,
                 target_update: int = 100):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
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
        
        self.optimizers = [optim.Adam(net.parameters(), lr=learning_rate)
                          for net in self.q_networks]
        
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
    
    def store_transition(self, states, actions, rewards, next_states, dones):
        """Store transition in replay buffer."""
        self.replay_buffer.push(states, actions, rewards, next_states, dones)
    
    def update(self):
        """Update Q-networks using VDN decomposition."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size)
        
        # Convert to tensors
        states = torch.stack([torch.stack(s).float() for s in states]).to(
            self.device)
        next_states = torch.stack([torch.stack(s).float() for s in next_states]).to(
            self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        
        # Compute individual Q-values and sum (VDN: simple sum)
        q_values = []
        target_q_values = []
        for agent in range(self.num_agents):
            q_vals = self.q_networks[agent](states[:, agent, :])
            q_values.append(q_vals.gather(1, actions[:, agent].unsqueeze(1)).squeeze(1))
            
            with torch.no_grad():
                target_q_vals = self.target_q_networks[agent](
                    next_states[:, agent, :])
                target_q_values.append(target_q_vals.max(1)[0])
        
        # VDN: sum of individual Q-values
        q_total = torch.stack(q_values, dim=1).sum(dim=1)
        target_q_total = torch.stack(target_q_values, dim=1).sum(dim=1)
        
        # Compute target
        rewards_sum = rewards.sum(dim=1)
        dones_any = dones.any(dim=1).float()
        target_q_total = rewards_sum + (1 - dones_any) * self.gamma * target_q_total
        
        # Loss
        loss = nn.MSELoss()(q_total, target_q_total)
        
        # Update all networks
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()
        
        # Update target networks
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            for i in range(self.num_agents):
                self.target_q_networks[i].load_state_dict(
                    self.q_networks[i].state_dict())
    
    def save(self, path: str) -> None:
        """Save model checkpoints."""
        torch.save({
            'q_networks': [net.state_dict() for net in self.q_networks],
            'target_q_networks': [net.state_dict() for net in self.target_q_networks],
            'optimizers': [opt.state_dict() for opt in self.optimizers]
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoints."""
        checkpoint = torch.load(path)
        for i, state_dict in enumerate(checkpoint['q_networks']):
            self.q_networks[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['target_q_networks']):
            self.target_q_networks[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['optimizers']):
            self.optimizers[i].load_state_dict(state_dict)

