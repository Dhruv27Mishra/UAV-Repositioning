"""
Training script for QMIX algorithm.
"""

import repo_paths  # noqa: F401

import numpy as np
from rl_agent.marl_env import MARLEnv
from rl_agent.QMIX import QMIX
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
          learning_rate=0.001, gamma=0.99, epsilon=0.1, save_interval=50,
          buffer_size=10000, batch_size=64, target_update=100):
    """Train QMIX agents."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = MARLEnv(num_uavs=num_uavs, grid_size=grid_size, device=device,
                 enable_non_stationary=True, enable_performative=True)
    state_dim = getattr(env, "agent_obs_dim", env.observation_space.shape[0] // num_uavs)
    action_dim = env.action_space.nvec[0]
    global_state_dim = env.observation_space.shape[0]
    
    agent = QMIX(num_agents=num_uavs, state_dim=state_dim, action_dim=action_dim,
                 global_state_dim=global_state_dim, learning_rate=learning_rate,
                 gamma=gamma, epsilon=epsilon, device=device, buffer_size=buffer_size,
                 batch_size=batch_size, target_update=target_update)
    
    pbar = tqdm(range(num_episodes), desc="Training QMIX")
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
        
        while not done:
            actions = []
            for i in range(num_uavs):
                agent_obs = obs_tensor[i * state_dim:(i + 1) * state_dim]
                action = agent.get_action(agent_obs, i)
                actions.append(action)
            
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            next_obs_tensor = torch.tensor(next_obs, device=device, dtype=torch.float32)
            
            states = [obs_tensor[i * state_dim:(i + 1) * state_dim] for i in range(num_uavs)]
            next_states = [next_obs_tensor[i * state_dim:(i + 1) * state_dim] for i in range(num_uavs)]
            global_state = obs_tensor.flatten()
            next_global_state = next_obs_tensor.flatten()
            
            agent.store_transition(states, actions, [reward] * num_uavs,
                                  next_states, [done] * num_uavs,
                                  global_state, next_global_state)
            agent.update()
            
            obs = next_obs
            obs_tensor = next_obs_tensor
            episode_reward += reward
            
            if info['collisions']:
                episode_collisions += 1
                total_collisions += 1
            episode_throughput += info['throughput']
            total_throughput += info['throughput']
        
        env.end_episode()
        
        pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'collisions': episode_collisions,
            'throughput': f'{episode_throughput:.2f}'
        })
        
        if (episode + 1) % save_interval == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/qmix_episode_{episode+1}.pt")
        
        episode_rewards.append(episode_reward)
        episode_throughputs.append(episode_throughput)
    
    os.makedirs("models", exist_ok=True)
    agent.save("models/qmix_final.pt")
    
    print("\nTraining completed!")
    print(f"Total episodes: {num_episodes}")
    print(f"Total collisions: {total_collisions}")
    print(f"Total throughput: {total_throughput:.2f}")
    print(f"Average throughput per episode: {total_throughput/num_episodes:.2f}")
    
    return episode_rewards, episode_throughputs


if __name__ == "__main__":
    train()

