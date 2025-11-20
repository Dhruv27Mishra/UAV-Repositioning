"""
Training script for MADDPG algorithm.
"""
import numpy as np
from rl_agent.marl_env import MARLEnv
from rl_agent.MADDPG import MADDPG
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
          learning_rate_actor=0.001, learning_rate_critic=0.001, gamma=0.99,
          tau=0.01, save_interval=50, buffer_size=10000, batch_size=64):
    """Train MADDPG agents."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = MARLEnv(num_uavs=num_uavs, grid_size=grid_size, device=device)
    state_dim = 3
    action_dim = env.action_space.nvec[0]
    
    agent = MADDPG(num_agents=num_uavs, state_dim=state_dim, action_dim=action_dim,
                   learning_rate_actor=learning_rate_actor,
                   learning_rate_critic=learning_rate_critic, gamma=gamma, tau=tau,
                   device=device, buffer_size=buffer_size, batch_size=batch_size)
    
    pbar = tqdm(range(num_episodes), desc="Training MADDPG")
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
                agent_obs = obs_tensor[i*3:(i+1)*3]
                action = agent.get_action(agent_obs, i, explore=(episode < num_episodes * 0.8))
                actions.append(action)
            
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            next_obs_tensor = torch.tensor(next_obs, device=device, dtype=torch.float32)
            
            states = [obs_tensor[i*3:(i+1)*3] for i in range(num_uavs)]
            next_states = [next_obs_tensor[i*3:(i+1)*3] for i in range(num_uavs)]
            agent.store_transition(states, actions, [reward] * num_uavs,
                                  next_states, [done] * num_uavs)
            agent.update()
            
            obs = next_obs
            obs_tensor = next_obs_tensor
            episode_reward += reward
            
            if info['collisions']:
                episode_collisions += 1
                total_collisions += 1
            episode_throughput += info['throughput']
            total_throughput += info['throughput']
        
        pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'collisions': episode_collisions,
            'throughput': f'{episode_throughput:.2f}'
        })
        
        if (episode + 1) % save_interval == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/maddpg_episode_{episode+1}.pt")
        
        episode_rewards.append(episode_reward)
        episode_throughputs.append(episode_throughput)
    
    os.makedirs("models", exist_ok=True)
    agent.save("models/maddpg_final.pt")
    
    print("\nTraining completed!")
    print(f"Total episodes: {num_episodes}")
    print(f"Total collisions: {total_collisions}")
    print(f"Total throughput: {total_throughput:.2f}")
    print(f"Average throughput per episode: {total_throughput/num_episodes:.2f}")
    
    return episode_rewards, episode_throughputs


if __name__ == "__main__":
    train()

