"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) implementation.
Actor-Critic method for continuous action spaces (adapted for discrete).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Tuple


class Actor(nn.Module):
    """Actor network (policy)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Critic(nn.Module):
    """Critic network (Q-function)."""
    def __init__(self, state_dim: int, action_dim: int, num_agents: int,
                 hidden_dim: int = 64):
        super(Critic, self).__init__()
        # States: state_dim * num_agents
        # Actions: action_dim * num_agents (one-hot or probabilities)
        input_dim = state_dim * num_agents + action_dim * num_agents
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass with all agents' states and actions.
        
        Args:
            states: [batch, state_dim * num_agents] flattened states
            actions: [batch, action_dim * num_agents] flattened action probs/one-hot
        """
        x = torch.cat([states, actions], dim=-1)
        return self.network(x)


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


class MADDPG:
    """MADDPG algorithm for multi-agent reinforcement learning."""
    def __init__(self, num_agents: int, state_dim: int, action_dim: int,
                 learning_rate_actor: float = 0.001, learning_rate_critic: float = 0.001,
                 gamma: float = 0.99, tau: float = 0.01, device: torch.device = None,
                 buffer_size: int = 10000, batch_size: int = 64):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Create actors and critics for each agent
        self.actors = [Actor(state_dim, action_dim).to(self.device)
                      for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim).to(self.device)
                             for _ in range(num_agents)]
        self.critics = [Critic(state_dim, action_dim, num_agents).to(self.device)
                       for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim, action_dim, num_agents).to(self.device)
                              for _ in range(num_agents)]
        
        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
        
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=learning_rate_actor)
                                for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=learning_rate_critic)
                                 for critic in self.critics]
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.learn_step = 0
    
    def get_action(self, state: torch.Tensor, agent_idx: int, explore: bool = True) -> int:
        """Get action for a specific agent."""
        with torch.no_grad():
            state = state.float().unsqueeze(0)
            action_probs = self.actors[agent_idx](state)[0]
            
            if explore:
                # Add noise for exploration
                action_probs = action_probs + torch.randn_like(action_probs) * 0.1
                action_probs = F.softmax(action_probs, dim=0)
            
            return action_probs.multinomial(1).item()
    
    def store_transition(self, states, actions, rewards, next_states, dones):
        """Store transition in replay buffer."""
        self.replay_buffer.push(states, actions, rewards, next_states, dones)
    
    def update(self):
        """Update actors and critics."""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size)
        
        # Convert to tensors
        states = torch.stack([torch.stack(s).float() for s in states]).to(self.device)
        next_states = torch.stack([torch.stack(s).float() for s in next_states]).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        
        # Flatten states for critic
        states_flat = states.view(self.batch_size, -1)
        next_states_flat = next_states.view(self.batch_size, -1)
        
        # Convert discrete actions to one-hot encoding
        actions_one_hot = F.one_hot(actions, self.action_dim).float()
        actions_flat = actions_one_hot.view(self.batch_size, -1)
        
        # Update each agent
        for agent in range(self.num_agents):
            # Get next actions from target actors (use probabilities directly)
            with torch.no_grad():
                next_action_probs = []
                for a in range(self.num_agents):
                    next_action_probs.append(
                        self.target_actors[a](next_states[:, a, :]))
                next_actions_probs = torch.stack(next_action_probs, dim=1)
                next_actions_flat = next_actions_probs.view(self.batch_size, -1)
                
                target_q = self.target_critics[agent](
                    next_states_flat, next_actions_flat)
                target_q = rewards[:, agent].unsqueeze(1) + (
                    1 - dones[:, agent].unsqueeze(1)) * self.gamma * target_q
            
            # Update critic
            current_q = self.critics[agent](states_flat, actions_flat)
            critic_loss = nn.MSELoss()(current_q, target_q)
            
            self.critic_optimizers[agent].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent].step()
            
            # Update actor
            action_probs = self.actors[agent](states[:, agent, :])
            # Create actions with current agent's new action probs
            actions_all_probs = actions_one_hot.clone()
            actions_all_probs[:, agent, :] = action_probs
            actions_all_flat = actions_all_probs.view(self.batch_size, -1)
            
            actor_loss = -self.critics[agent](states_flat, actions_all_flat).mean()
            
            self.actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent].step()
        
        # Soft update target networks
        for i in range(self.num_agents):
            for param, target_param in zip(self.actors[i].parameters(),
                                          self.target_actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data +
                                       (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critics[i].parameters(),
                                          self.target_critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data +
                                       (1 - self.tau) * target_param.data)
    
    def save(self, path: str) -> None:
        """Save model checkpoints."""
        torch.save({
            'actors': [net.state_dict() for net in self.actors],
            'target_actors': [net.state_dict() for net in self.target_actors],
            'critics': [net.state_dict() for net in self.critics],
            'target_critics': [net.state_dict() for net in self.target_critics],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers]
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoints."""
        checkpoint = torch.load(path)
        for i, state_dict in enumerate(checkpoint['actors']):
            self.actors[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['target_actors']):
            self.target_actors[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['critics']):
            self.critics[i].load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['target_critics']):
            self.target_critics[i].load_state_dict(state_dict)

