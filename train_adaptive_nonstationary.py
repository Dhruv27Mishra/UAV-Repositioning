"""
Training script for AdaptiveNonStationaryMARL algorithm.
This novel algorithm includes the improved association algorithm.
"""
import numpy as np
from rl_agent.marl_env import MARLEnv
from rl_agent.AdaptiveNonStationaryMARL import AdaptiveNonStationaryMARL
import torch
import os
from tqdm import tqdm


def train(num_episodes=1000, num_uavs=3, num_users=20, grid_size=(10, 10, 5),
          learning_rate=0.001, gamma=0.99, epsilon=0.1,
          save_interval=50, device=None):
    """Train AdaptiveNonStationaryMARL with improved association algorithm."""
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment with non-stationary and performative features enabled
    env = MARLEnv(num_uavs=num_uavs, num_users=num_users, grid_size=grid_size, device=device,
                 enable_non_stationary=True, enable_performative=True)
    
    state_dim = getattr(env, "agent_obs_dim", env.observation_space.shape[0] // num_uavs)
    action_dim = env.action_space.nvec[0]
    global_state_dim = env.observation_space.shape[0]
    # NS context is replicated inside each agent observation block; mixing uses full global obs only
    context_dim = 0
    
    # Create novel algorithm
    agent = AdaptiveNonStationaryMARL(
        num_agents=num_uavs, state_dim=state_dim, action_dim=action_dim,
        global_state_dim=global_state_dim, context_dim=context_dim,
        learning_rate=learning_rate, gamma=gamma, epsilon=epsilon, device=device
    )
    
    # Set the improved association function in the environment
    env.set_association_function(agent.get_association_function())
    print("✓ Improved association algorithm enabled for AdaptiveNonStationaryMARL")
    
    pbar = tqdm(range(num_episodes), desc="Training AdaptiveNonStationaryMARL")
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
                agent_obs = obs_tensor[i*state_dim:(i+1)*state_dim]
                action = agent.get_action(agent_obs, i, global_state=obs_tensor)
                actions.append(action)
            
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            next_obs_tensor = torch.tensor(next_obs, device=device, dtype=torch.float32)
            
            # Store transition
            states = [obs_tensor[i*state_dim:(i+1)*state_dim] for i in range(num_uavs)]
            next_states = [next_obs_tensor[i*state_dim:(i+1)*state_dim] for i in range(num_uavs)]
            
            agent.store_transition(states, actions, [reward] * num_uavs, 
                                 next_states, [done] * num_uavs,
                                 obs_tensor, next_obs_tensor)
            
            # Update agent
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
            
            episode_reward += reward
            if info.get('collisions', False):
                episode_collisions += 1
                total_collisions += 1
            episode_throughput += info.get('throughput', 0.0)
            total_throughput += info.get('throughput', 0.0)
            
            obs_tensor = next_obs_tensor
        
        # End episode for performative/non-stationary updates
        env.end_episode()
        
        pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'collisions': episode_collisions,
            'throughput': f'{episode_throughput:.2f}'
        })
        
        if (episode + 1) % save_interval == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/adaptive_nonstationary_episode_{episode+1}.pt")
        
        episode_rewards.append(episode_reward)
        episode_throughputs.append(episode_throughput)
    
    os.makedirs("models", exist_ok=True)
    agent.save("models/adaptive_nonstationary_final.pt")
    
    print("\nTraining completed!")
    print(f"Total episodes: {num_episodes}")
    print(f"Total collisions: {total_collisions}")
    print(f"Total throughput: {total_throughput:.2f}")
    print(f"Average throughput per episode: {total_throughput/num_episodes:.2f}")
    print("\n✓ Novel algorithm with improved association algorithm trained successfully!")
    
    return episode_rewards, episode_throughputs


if __name__ == "__main__":
    train()

