"""
MAPPO (Multi-Agent Proximal Policy Optimization) implementation.
On-policy actor-critic algorithm for MARL.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import List, Tuple, Dict


class Actor(nn.Module):
    """Actor network (policy) for MAPPO."""
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
    """Critic network (value function) for MAPPO."""
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO).
    On-policy algorithm suitable for non-stationary environments.
    """
    def __init__(self, num_agents: int, state_dim: int, action_dim: int,
                 learning_rate_actor: float = 0.0003, learning_rate_critic: float = 0.001,
                 gamma: float = 0.99, clip_epsilon: float = 0.2, device: torch.device = None):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        
        # Create actor and critic networks for each agent
        self.actors = [Actor(state_dim, action_dim).to(self.device) for _ in range(num_agents)]
        self.critics = [Critic(state_dim).to(self.device) for _ in range(num_agents)]
        
        # Optimizers
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=learning_rate_actor) 
                                for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=learning_rate_critic) 
                                 for critic in self.critics]
        
        # On-policy buffer: stores trajectories
        self.trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'log_probs': []
        }
    
    def get_action(self, state: torch.Tensor, agent_id: int, explore: bool = True) -> int:
        """Get action from actor network."""
        with torch.no_grad():
            probs = self.actors[agent_id](state)
            if explore:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                action = torch.argmax(probs)
                log_prob = torch.log(probs[action] + 1e-9)
        
        return action.item(), log_prob.item()
    
    def store_transition(self, states: List[torch.Tensor], actions: List[int],
                        rewards: List[float], next_states: List[torch.Tensor],
                        dones: List[bool], log_probs: List[float]):
        """Store transition in on-policy buffer."""
        self.trajectories['states'].append(states)
        self.trajectories['actions'].append(actions)
        self.trajectories['rewards'].append(rewards)
        self.trajectories['next_states'].append(next_states)
        self.trajectories['dones'].append(dones)
        self.trajectories['log_probs'].append(log_probs)
    
    def update(self):
        """Update actors and critics using PPO clipped objective."""
        if len(self.trajectories['states']) == 0:
            return
        
        # Convert trajectories to tensors
        states = torch.stack([torch.cat(s) for s in self.trajectories['states']]).to(self.device)
        actions = torch.tensor(self.trajectories['actions']).to(self.device)
        rewards = torch.tensor(self.trajectories['rewards'], dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.trajectories['dones'], dtype=torch.bool).to(self.device)
        old_log_probs = torch.tensor(self.trajectories['log_probs'], dtype=torch.float32).to(self.device)
        
        # Compute returns (discounted rewards)
        returns = self._compute_returns(rewards, dones)
        
        # Update each agent
        for agent_id in range(self.num_agents):
            agent_states = states[:, agent_id * self.state_dim:(agent_id + 1) * self.state_dim]
            agent_actions = actions[:, agent_id]
            agent_old_log_probs = old_log_probs[:, agent_id]
            agent_returns = returns[:, agent_id]
            
            # Compute advantages
            with torch.no_grad():
                values = self.critics[agent_id](agent_states).squeeze()
            advantages = agent_returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
            
            # PPO update
            for _ in range(4):  # Multiple PPO epochs
                # Get current policy
                probs = self.actors[agent_id](agent_states)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(agent_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - agent_old_log_probs)
                
                # Clipped objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                
                # Update actor
                self.actor_optimizers[agent_id].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 0.5)
                self.actor_optimizers[agent_id].step()
                
                # Update critic
                values = self.critics[agent_id](agent_states).squeeze()
                critic_loss = F.mse_loss(values, agent_returns)
                self.critic_optimizers[agent_id].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 0.5)
                self.critic_optimizers[agent_id].step()
        
        # Clear trajectories (on-policy: use data once)
        self._clear_trajectories()
    
    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(self.num_agents).to(self.device)
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (~dones[t]).float()
            returns[t] = running_return
        
        return returns
    
    def _clear_trajectories(self):
        """Clear trajectory buffer."""
        for key in self.trajectories:
            self.trajectories[key] = []
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint['critics'][i])

