"""
Deep Nash Q-learning implementation for multi-agent reinforcement learning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def push(self, *transition):
        self.buffer.append(tuple(transition))
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return map(list, zip(*batch))
    def __len__(self):
        return len(self.buffer)

class DeepNashQLearning:
    def __init__(self, num_agents: int, state_dim: int, action_dim: int,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 0.1, device: torch.device = None,
                 buffer_size: int = 10000, batch_size: int = 64, target_update: int = 100):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Calculate joint action space size
        self.joint_action_dim = action_dim ** num_agents
        
        # Create Q-networks and target networks for each agent
        self.q_networks = [QNetwork(state_dim, self.joint_action_dim).to(self.device) for _ in range(num_agents)]
        self.target_networks = [QNetwork(state_dim, self.joint_action_dim).to(self.device) for _ in range(num_agents)]
        for i in range(num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
        self.optimizers = [optim.Adam(net.parameters(), lr=learning_rate) for net in self.q_networks]
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.learn_step = 0
    
    def _find_nash_equilibrium(self, q_values: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # Best response dynamics for Nash equilibrium (batch or single)
        # q_values: [batch, joint_action_dim] or [joint_action_dim]
        # For simplicity, use argmax for each agent (approximate Nash)
        if q_values.dim() == 1:
            # [joint_action_dim]
            best_joint = torch.argmax(q_values).item()
            actions = []
            for _ in range(self.num_agents):
                actions.append(best_joint % self.action_dim)
                best_joint //= self.action_dim
            return torch.tensor(actions, device=self.device), q_values.max().item()
        else:
            # [batch, joint_action_dim]
            best_joints = torch.argmax(q_values, dim=1)
            actions = []
            for idx in range(q_values.size(0)):
                joint = best_joints[idx].item()
                a = []
                for _ in range(self.num_agents):
                    a.append(joint % self.action_dim)
                    joint //= self.action_dim
                actions.append(a)
            return torch.tensor(actions, device=self.device), q_values.max(1)[0]
    
    def get_action(self, state: torch.Tensor, agent_idx: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state = state.float().unsqueeze(0)  # [1, state_dim]
            q_values = self.q_networks[agent_idx](state)[0]  # [joint_action_dim]
            actions, _ = self._find_nash_equilibrium(q_values)
            return actions[agent_idx].item()
    
    def store_transition(self, states, actions, rewards, next_states, dones):
        self.replay_buffer.push(states, actions, rewards, next_states, dones)
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # states: list of batch_size lists of agent states
        # Convert to tensors
        states = torch.stack([torch.stack(s).float() for s in states]).to(self.device)  # [batch, num_agents, state_dim]
        next_states = torch.stack([torch.stack(s).float() for s in next_states]).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        # For each agent, update Q-network
        for agent in range(self.num_agents):
            # Current Q-values
            q_values = self.q_networks[agent](states[:, agent, :])  # [batch, joint_action_dim]
            # Gather Q-values for taken joint actions
            joint_action_indices = []
            for a in actions:
                idx = 0
                for i, act in enumerate(a):
                    idx += act * (self.action_dim ** i)
                joint_action_indices.append(idx)
            joint_action_indices = torch.tensor(joint_action_indices, device=self.device)
            current_q = q_values.gather(1, joint_action_indices.unsqueeze(1)).squeeze(1)
            # Next Q-values (target network)
            with torch.no_grad():
                next_q_values = self.target_networks[agent](next_states[:, agent, :])  # [batch, joint_action_dim]
                _, next_q = self._find_nash_equilibrium(next_q_values)
            target_q = rewards[:, agent] + (1 - dones[:, agent]) * self.gamma * next_q
            # Loss and update
            loss = nn.MSELoss()(current_q, target_q)
            self.optimizers[agent].zero_grad()
            loss.backward()
            self.optimizers[agent].step()
        # Update target networks
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            for i in range(self.num_agents):
                self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
    
    def save(self, path: str) -> None:
        torch.save({
            'q_networks': [net.state_dict() for net in self.q_networks],
            'target_networks': [net.state_dict() for net in self.target_networks],
            'optimizers': [opt.state_dict() for opt in self.optimizers]
        }, path)
    
    def load(self, path: str) -> None:
        checkpoint = torch.load(path)
        for i, state_dict in enumerate(checkpoint['q_networks']):
            self.q_networks[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['target_networks']):
            self.target_networks[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['optimizers']):
            self.optimizers[i].load_state_dict(state_dict) 