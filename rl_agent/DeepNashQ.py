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

class DeepNashQ:
    def __init__(self, num_agents: int, state_dim: int, action_dim: int,
                 learning_rate: float = 0.0003, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, device: torch.device = None,
                 buffer_size: int = 100000, batch_size: int = 128, target_update: int = 200):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Calculate joint action space size
        self.joint_action_dim = action_dim ** num_agents
        
        # Create Q-networks and target networks for each agent with improved architecture
        self.q_networks = [QNetwork(state_dim, self.joint_action_dim).to(self.device) for _ in range(num_agents)]
        self.target_networks = [QNetwork(state_dim, self.joint_action_dim).to(self.device) for _ in range(num_agents)]
        for i in range(num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
        
        # Use Adam optimizer with gradient clipping
        self.optimizers = [optim.Adam(net.parameters(), lr=learning_rate, eps=1e-5) for net in self.q_networks]
        
        # Replay buffer with larger size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.learn_step = 0
        
        # Initialize target networks
        self._update_target_networks()
    
    def _update_target_networks(self):
        for i in range(self.num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
    
    def _update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
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
    
    def update(self, state, action, reward, next_state, done):
        # Store transition in replay buffer
        self.store_transition(state, action, reward, next_state, done)
        
        # Update epsilon
        self._update_epsilon()
        
        # Only update if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert tensors to CPU before stacking
        states = [np.array(s.cpu() if torch.is_tensor(s) else s) for s in states]
        actions = [np.array(a.cpu() if torch.is_tensor(a) else a) for a in actions]
        rewards = [np.array(r.cpu() if torch.is_tensor(r) else r) for r in rewards]
        next_states = [np.array(ns.cpu() if torch.is_tensor(ns) else ns) for ns in next_states]
        dones = [np.array(d.cpu() if torch.is_tensor(d) else d) for d in dones]
        
        # Stack lists into arrays for correct shape
        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        next_states = np.stack(next_states)
        dones = np.stack(dones)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Ensure correct shapes
        if states.dim() == 2:  # [batch, state_dim]
            states = states.unsqueeze(1).expand(-1, self.num_agents, -1)  # [batch, num_agents, state_dim]
        if next_states.dim() == 2:
            next_states = next_states.unsqueeze(1).expand(-1, self.num_agents, -1)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)  # [1, num_agents]
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0).unsqueeze(0)  # [1, 1]
        if dones.dim() == 0:
            dones = dones.unsqueeze(0).unsqueeze(0)  # [1, 1]
        
        for agent in range(self.num_agents):
            # Compute current Q values
            q_values = self.q_networks[agent](states[:, agent, :])
            
            # Compute joint action indices
            joint_indices = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            for i, act in enumerate(actions.T):
                joint_indices += act * (self.action_dim ** i)
            
            current_q = q_values.gather(1, joint_indices.unsqueeze(1))
            
            # Compute next Q values using target network
            with torch.no_grad():
                next_q_values = self.target_networks[agent](next_states[:, agent, :])
                max_next_q = next_q_values.max(1)[0]
                target_q = rewards[:, agent] + (1 - dones[:, agent]) * self.gamma * max_next_q
            
            # Compute loss with Huber loss for stability
            loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
            
            # Optimize
            self.optimizers[agent].zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_networks[agent].parameters(), 1.0)
            self.optimizers[agent].step()
        
        # Update target networks periodically
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self._update_target_networks()
    
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