"""
Main training script for the UAV MARL system using VDN.
"""
import numpy as np
from rl_agent.marl_env import MARLEnv
from rl_agent.VDN.vdn import VDN
import torch
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import subprocess

def jains_fairness(user_rates):
    import numpy as np
    user_rates = np.array(user_rates)
    numerator = np.sum(user_rates) ** 2
    denominator = len(user_rates) * np.sum(user_rates ** 2)
    return numerator / denominator if denominator != 0 else 0
def get_min_user_rate(user_rates):
    return np.min(user_rates)

def train(num_episodes=1000,
          num_uavs=7,
          grid_size=(10, 10, 5),
          learning_rate_start=0.001,
          learning_rate_end=0.0001,
          learning_rate_decay=0.995,
          gamma=0.99,
          epsilon_start=1.0,
          epsilon_end=0.01,
          epsilon_decay=0.995,
          eval_interval=10,
          save_interval=50,
          buffer_size=10000,
          batch_size=64,
          target_update=100,
          reward_scale=0.1):
    
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
    
    # Initialize learning rate
    learning_rate = learning_rate_start
    
    # Create VDN agent
    marl = VDN(
        num_uavs=num_uavs,
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon_start,
        device=device
    )
    marl.to(device)
    
    # Training loop with progress bar
    pbar = tqdm(range(num_episodes), desc="Training")
    total_collisions = 0
    total_throughput = 0.0
    episode_rewards = []
    episode_throughputs = []
    episode_fairnesses = []

    last_episode_log = {'uav_positions': [], 'user_positions': []}
    episode_end_positions = {'uav_positions': [], 'user_positions': []}
    
    # Initialize epsilon
    epsilon = epsilon_start
    
    # Reward normalization variables
    running_reward_mean = 0.0
    running_reward_var = 1.0
    reward_count = 1
    
    for episode in pbar:
        episode_start = time.time()
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_collisions = 0
        episode_throughput = 0.0
        
        # Convert observation to tensor and reshape for each agent
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).view(1, num_uavs, -1)
        
        while not done:
            # Get actions for all agents with current epsilon
            actions = [marl.get_action(obs_tensor[0, i], i, epsilon) for i in range(num_uavs)]
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # Scale reward
            scaled_reward = reward * reward_scale
            
            # --- Reward normalization ---
            # Update running mean/var (Welford's algorithm)
            delta = scaled_reward - running_reward_mean
            running_reward_mean += delta / reward_count
            delta2 = scaled_reward - running_reward_mean
            running_reward_var += delta * delta2
            reward_count += 1
            reward_std = max(np.sqrt(running_reward_var / reward_count), 1e-6)
            normalized_reward = (scaled_reward - running_reward_mean) / reward_std
            
            # Convert next observation to tensor and reshape for each agent
            next_obs_tensor = torch.tensor(next_obs, device=device, dtype=torch.float32).view(1, num_uavs, -1)
            
            # Update Q-networks
            marl.update(obs_tensor, torch.tensor(actions, device=device), normalized_reward, next_obs_tensor, done)
            
            obs = next_obs
            obs_tensor = next_obs_tensor
            episode_reward += reward  # Use original reward for logging
            
            # Track collisions and throughput
            if info['collisions']:
                episode_collisions += 1
                total_collisions += 1
            episode_throughput += info['throughput']
            if 'user_rates' in info:
                episode_fairness = jains_fairness(info['user_rates'])
            else:
                episode_fairness = 0
            episode_fairnesses.append(episode_fairness)

            total_throughput += info['throughput']
            
            # Log trajectory for last episode
            if episode == num_episodes - 1:
                env.log_step(last_episode_log)
        
        # Decay epsilon and learning rate
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        learning_rate = max(learning_rate_end, learning_rate * learning_rate_decay)
        
        # Update optimizer learning rate
        for param_group in marl.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # Update progress bar with episode metrics
        pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'collisions': episode_collisions,
            'throughput': f'{episode_throughput:.2f}',
            'total_collisions': total_collisions,
            'total_throughput': f'{total_throughput:.2f}',
            'epsilon': f'{epsilon:.3f}',
            'lr': f'{learning_rate:.6f}'
        })
        
        # Save models periodically
        if (episode + 1) % save_interval == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(marl.state_dict(), f"models/vdn_episode_{episode+1}.pt")
        
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
    np.save("results/vdn_fairness.npy", np.array(episode_fairnesses))  # change to vdn_fairness.npy in train_vdn.py

    # Save final model
    os.makedirs("models", exist_ok=True)
    torch.save(marl.state_dict(), "models/vdn_final.pt")
    
    # Save rewards and throughputs
    os.makedirs("results", exist_ok=True)
    np.save("results/vdn_rewards.npy", np.array(episode_rewards))
    np.save("results/vdn_throughputs.npy", np.array(episode_throughputs))
    
    # Print final statistics
    print("\nTraining completed!")
    print(f"Total episodes: {num_episodes}")
    print(f"Total collisions: {total_collisions}")
    print(f"Total throughput: {total_throughput:.2f}")
    print(f"Average collisions per episode: {total_collisions/num_episodes:.2f}")
    print(f"Average throughput per episode: {total_throughput/num_episodes:.2f}")
    
    # Plot throughput and reward curves
    plt.figure(figsize=(12, 5))
    
    # Calculate moving averages
    window_size = 20  # Adjust this value to control smoothing
    rewards_ma = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    throughputs_ma = np.convolve(episode_throughputs, np.ones(window_size)/window_size, mode='valid')
    
    # Plot raw data with low opacity
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Raw')
    plt.plot(range(window_size-1, len(episode_rewards)), rewards_ma, label='Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_throughputs, alpha=0.3, label='Raw')
    plt.plot(range(window_size-1, len(episode_throughputs)), throughputs_ma, label='Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Throughput')
    plt.title('Throughput per Episode')
    plt.legend()
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
    window_size = 10
    fairness_ma = np.convolve(episode_fairnesses, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 4))
    plt.plot(episode_fairnesses, alpha=0.3, label='Raw Fairness')
    plt.plot(range(window_size-1, len(episode_fairnesses)), fairness_ma, label='Moving Average')
    plt.xlabel('Episode')
    plt.ylabel("Jain's Fairness Index")
    plt.title("Fairness per Episode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/vdn_fairness_curve.png")  # or vdn_fairness_curve.png
    plt.show()

    user_counts = [1, 2, 3, 4, 5]
    fairness_scores = []

    num_runs = 1000  # average over 1000 runs

    for num_users in user_counts:
        total_fairness = 0
        for _ in range(num_runs):
            env = MARLEnv(num_uavs=5, num_users=num_users)
            env.reset()
            _, user_rates = env._compute_throughput()
            fairness = jains_fairness(user_rates)
            total_fairness += fairness

        avg_fairness = total_fairness / num_runs
        fairness_scores.append(avg_fairness)
    np.save(f"results/vdn_fairness_vs_density.npy", np.array(fairness_scores))
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(user_counts, fairness_scores, marker='o', linestyle='-', color='darkblue')
    plt.xlabel("Number of Users (UE Density)")
    plt.xticks(user_counts)
    plt.ylabel("Jain's Fairness Index")
    plt.title("Fairness vs UE Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/vdn_fairness_vs_density.png")
    plt.show()
    
    user_counts = [1, 2, 3, 4, 5]
    min_rate_results = []

    num_runs = 1000  # average over multiple resets

    for num_users in user_counts:
        total_min_rate = 0
        for _ in range(num_runs):
            env = MARLEnv(num_uavs=5, num_users=num_users)
            env.reset()
            _, user_rates = env._compute_throughput()
            total_min_rate += get_min_user_rate(user_rates)
        avg_min_rate = total_min_rate / num_runs
        min_rate_results.append(avg_min_rate)

    # Save result
    np.save("results/XYZ_min_rate_vs_density.npy", np.array(min_rate_results))  # change XYZ to deepnash/qmix/vdn

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(user_counts, min_rate_results, marker='o', linestyle='-', color='darkred')
    plt.xlabel("Number of Users (UE Density)")
    plt.ylabel("Minimum Throughput per UE (Mbps)")
    plt.title("Minimum Throughput vs UE Density")
    plt.grid(True)
    plt.xticks(user_counts)
    plt.tight_layout()
    plt.savefig("results/vdn_min_rate_vs_density.png")  # change XYZ accordingly
    plt.show()
    np.save("results/vdn_min_rate_vs_density.npy", np.array(min_rate_results))  
if __name__ == "__main__":
    train() 