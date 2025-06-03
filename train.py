"""
Main training script for the UAV MARL system using Deep Nash Q-learning.
"""
import numpy as np
from rl_agent.marl_env import MARLEnv
from rl_agent.zoo_marl import DeepNashQLearning
import torch
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess

def train(num_episodes=1000,
          num_uavs=7,
          grid_size=(10, 10, 5),
          learning_rate=0.001,
          gamma=0.99,
          epsilon=0.1,
          eval_interval=10,
          save_interval=50,
          buffer_size=10000,
          batch_size=64,
          target_update=100):
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = MARLEnv(
        num_uavs=num_uavs,
        grid_size=grid_size,
        device=device
    )
    
    # Get state and action dimensions
    state_dim = 3  # Each agent observes its own (x,y,z) position
    action_dim = env.action_space.nvec[0]
    
    # Create Deep Nash Q-learning agent
    marl = DeepNashQLearning(
        num_agents=num_uavs,
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        device=device,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update=target_update
    )
    
    # Training loop with progress bar
    pbar = tqdm(range(num_episodes), desc="Training")
    total_collisions = 0
    total_throughput = 0.0
    episode_rewards = []
    episode_throughputs = []
    last_episode_log = {'uav_positions': [], 'user_positions': []}
    episode_end_positions = {'uav_positions': [], 'user_positions': []}
    
    for episode in pbar:
        episode_start = time.time()
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_collisions = 0
        episode_throughput = 0.0
        
        # Convert observation to tensor
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
        
        while not done:
            # Get actions for all agents
            actions = []
            for i in range(num_uavs):
                agent_obs = obs_tensor[i*3:(i+1)*3]  # Get agent's observation
                action = marl.get_action(agent_obs, i)
                actions.append(action)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # Convert next observation to tensor
            next_obs_tensor = torch.tensor(next_obs, device=device, dtype=torch.float32)
            
            # Store transition in replay buffer
            states = [obs_tensor[i*3:(i+1)*3] for i in range(num_uavs)]
            next_states = [next_obs_tensor[i*3:(i+1)*3] for i in range(num_uavs)]
            marl.store_transition(states, actions, [reward] * num_uavs, next_states, [done] * num_uavs)
            marl.update()
            
            obs = next_obs
            obs_tensor = next_obs_tensor
            episode_reward += reward
            
            # Track collisions and throughput
            if info['collisions']:
                episode_collisions += 1
                total_collisions += 1
            episode_throughput += info['throughput']
            total_throughput += info['throughput']
            
            # Log trajectory for last episode
            if episode == num_episodes - 1:
                env.log_step(last_episode_log)
        
        # Update progress bar with episode metrics
        pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'collisions': episode_collisions,
            'throughput': f'{episode_throughput:.2f}',
            'total_collisions': total_collisions,
            'total_throughput': f'{total_throughput:.2f}'
        })
        
        # Save models periodically
        if (episode + 1) % save_interval == 0:
            os.makedirs("models", exist_ok=True)
            marl.save(f"models/deep_nash_q_episode_{episode+1}.pt")
        
        episode_rewards.append(episode_reward)
        episode_throughputs.append(episode_throughput)
        
        episode_time = time.time() - episode_start
        print(f"\nEpisode {episode} completed in {episode_time:.3f}s")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Collisions: {episode_collisions} (Total: {total_collisions})")
        print(f"Throughput: {episode_throughput:.2f} (Total: {total_throughput:.2f})")
        
        # End of episode: store the new positions for animation
        episode_end_positions['uav_positions'].append(env.uav_positions.copy())
        episode_end_positions['user_positions'].append(env.user_positions.copy())
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    marl.save("models/deep_nash_q_final.pt")
    
    # Print final statistics
    print("\nTraining completed!")
    print(f"Total episodes: {num_episodes}")
    print(f"Total collisions: {total_collisions}")
    print(f"Total throughput: {total_throughput:.2f}")
    print(f"Average collisions per episode: {total_collisions/num_episodes:.2f}")
    print(f"Average throughput per episode: {total_throughput/num_episodes:.2f}")
    
    # Plot throughput and reward curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(episode_throughputs)
    plt.xlabel('Episode')
    plt.ylabel('Throughput')
    plt.title('Throughput per Episode')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Animate episode-end positions over all episodes
    fig, ax = plt.subplots(figsize=(7, 7))
    uav_traj = episode_end_positions['uav_positions']
    user_traj = episode_end_positions['user_positions']
    def update(frame):
        ax.clear()
        uav_xy = uav_traj[frame][:, :2]
        user_xy = user_traj[frame][:, :2]
        ax.scatter(user_xy[:, 0], user_xy[:, 1], c='green', label='Users')
        ax.scatter(uav_xy[:, 0], uav_xy[:, 1], c='blue', s=100, marker='o', label='UAVs')
        for user_idx in range(user_xy.shape[0]):
            user = user_xy[user_idx]
            min_dist = float('inf')
            nearest_uav = 0
            for uav_idx in range(uav_xy.shape[0]):
                uav = uav_traj[frame][uav_idx]
                d = np.linalg.norm(user_traj[frame][user_idx] - uav)
                if d < min_dist:
                    min_dist = d
                    nearest_uav = uav_idx
            uav = uav_xy[nearest_uav]
            ax.plot([user[0], uav[0]], [user[1], uav[1]], 'gray', alpha=0.5)
        ax.set_xlim(0, env.grid_size[0])
        ax.set_ylim(0, env.grid_size[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'UAV and User Positions (Episode {frame+1})')
        ax.legend()
        ax.grid(True)
    print(f"Animating {len(uav_traj)} frames for episode-end positions...")
    ani = animation.FuncAnimation(fig, update, frames=len(uav_traj), interval=400)
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    train() 