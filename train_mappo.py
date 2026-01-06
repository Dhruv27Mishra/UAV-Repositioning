"""
Training script for MAPPO (Multi-Agent Proximal Policy Optimization) algorithm.
On-policy algorithm suitable for non-stationary environments.
"""
import numpy as np
from rl_agent.marl_env import MARLEnv
from rl_agent.MAPPO import MAPPO
import torch
import os
from tqdm import tqdm


def jains_fairness(user_rates):
    """Calculate Jain's fairness index."""
    user_rates = np.array(user_rates)
    numerator = np.sum(user_rates) ** 2
    denominator = len(user_rates) * np.sum(user_rates ** 2)
    return numerator / denominator if denominator != 0 else 0


def train(num_episodes=1000, num_uavs=7, grid_size=(10, 10, 5),
          learning_rate_actor=0.0003, learning_rate_critic=0.001, gamma=0.99,
          clip_epsilon=0.2, save_interval=50, device=None):
    """Train MAPPO agents."""
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = MARLEnv(num_uavs=num_uavs, grid_size=grid_size, device=device,
                 enable_non_stationary=True, enable_performative=True)
    state_dim = env.observation_space.shape[0] // num_uavs if env.enable_non_stationary else 3
    action_dim = env.action_space.nvec[0]
    
    agent = MAPPO(num_agents=num_uavs, state_dim=state_dim, action_dim=action_dim,
                 learning_rate_actor=learning_rate_actor, learning_rate_critic=learning_rate_critic,
                 gamma=gamma, clip_epsilon=clip_epsilon, device=device)
    
    pbar = tqdm(range(num_episodes), desc="Training MAPPO")
    total_collisions = 0
    total_throughput = 0.0
    episode_rewards = []
    episode_throughputs = []
    
    for episode in pbar:
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_collisions = 0
        episode_throughput = 0.0
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
        
        # Collect trajectory
        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_next_states = []
        trajectory_dones = []
        trajectory_log_probs = []
        
        while not done:
            actions = []
            log_probs = []
            for i in range(num_uavs):
                agent_obs = obs_tensor[i*state_dim:(i+1)*state_dim]
                action, log_prob = agent.get_action(agent_obs, i, explore=True)
                actions.append(action)
                log_probs.append(log_prob)
            
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            next_obs_tensor = torch.tensor(next_obs, device=device, dtype=torch.float32)
            
            # Store in trajectory
            states = [obs_tensor[i*state_dim:(i+1)*state_dim] for i in range(num_uavs)]
            next_states = [next_obs_tensor[i*state_dim:(i+1)*state_dim] for i in range(num_uavs)]
            
            trajectory_states.append(states)
            trajectory_actions.append(actions)
            trajectory_rewards.append([reward] * num_uavs)
            trajectory_next_states.append(next_states)
            trajectory_dones.append([done] * num_uavs)
            trajectory_log_probs.append(log_probs)
            
            episode_reward += reward
            if info['collisions']:
                episode_collisions += 1
                total_collisions += 1
            episode_throughput += info['throughput']
            total_throughput += info['throughput']
            
            obs_tensor = next_obs_tensor
        
        # Store full trajectory
        for t in range(len(trajectory_states)):
            agent.store_transition(
                trajectory_states[t], trajectory_actions[t], trajectory_rewards[t],
                trajectory_next_states[t], trajectory_dones[t], trajectory_log_probs[t]
            )
        
        # Update agent (on-policy: update after collecting trajectory)
        agent.update()
        
        # End episode for performative/non-stationary updates
        env.end_episode()
        
        pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'collisions': episode_collisions,
            'throughput': f'{episode_throughput:.2f}'
        })
        
        if (episode + 1) % save_interval == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/mappo_episode_{episode+1}.pt")
        
        episode_rewards.append(episode_reward)
        episode_throughputs.append(episode_throughput)
    
    os.makedirs("models", exist_ok=True)
    agent.save("models/mappo_final.pt")
    
    print("\nTraining completed!")
    print(f"Total episodes: {num_episodes}")
    print(f"Total collisions: {total_collisions}")
    print(f"Total throughput: {total_throughput:.2f}")
    print(f"Average throughput per episode: {total_throughput/num_episodes:.2f}")
    
    return episode_rewards, episode_throughputs


if __name__ == "__main__":
    train()

