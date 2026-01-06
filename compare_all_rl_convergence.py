"""
Compare convergence of novel algorithm (AdaptiveNonStationaryMARL) with enhanced model
vs other RL models (QMIX, IQL, VDN, MADDPG, DeepNashQ) with regular model.
"""
import numpy as np
import matplotlib.pyplot as plt
from rl_agent.marl_env import MARLEnv
from rl_agent.QMIX import QMIX
from rl_agent.IQL import IQL
from rl_agent.VDN import VDN
from rl_agent.MADDPG import MADDPG
from rl_agent.DeepNashQ import DeepNashQ
from rl_agent.MAPPO import MAPPO
from rl_agent.AdaptiveNonStationaryMARL import AdaptiveNonStationaryMARL
import torch
from tqdm import tqdm


def compute_moving_average(data, window_size):
    """Compute moving average for smoothing."""
    if len(data) < window_size:
        return data
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start_idx:end_idx]))
    return np.array(smoothed)


def train_model(agent_class, agent_name, env, num_episodes=500, num_steps_per_episode=50, 
                learning_rate=0.001, gamma=0.99, epsilon=0.1, device=None, context_dim=7):
    """Train a single RL model and return convergence metrics."""
    
    print(f"\n{'='*70}")
    print(f"Training {agent_name}...")
    print(f"{'='*70}")
    
    num_uavs = env.num_uavs
    # Dynamically determine state_dim based on environment's observation space
    state_dim = env.observation_space.shape[0] // num_uavs if env.enable_non_stationary else 3 
    action_dim = env.action_space.nvec[0]
    
    # Initialize agent based on type
    if agent_name == "AdaptiveNonStationaryMARL":
        global_state_dim = env.observation_space.shape[0]
        agent = agent_class(num_agents=num_uavs, state_dim=state_dim, action_dim=action_dim,
                           global_state_dim=global_state_dim, context_dim=context_dim,
                           learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,
                           device=device, buffer_size=10000, batch_size=64, target_update=100)
        # Set the improved association function
        env.set_association_function(agent.get_association_function())
    elif agent_name == "QMIX":
        global_state_dim = env.observation_space.shape[0] # Global state is the full observation
        agent = agent_class(num_agents=num_uavs, state_dim=state_dim, action_dim=action_dim,
                           global_state_dim=global_state_dim, learning_rate=learning_rate,
                           gamma=gamma, epsilon=epsilon, device=device, buffer_size=10000,
                           batch_size=64, target_update=100)
    elif agent_name == "IQL":
        agent = agent_class(num_agents=num_uavs, state_dim=state_dim, action_dim=action_dim,
                           learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,
                           device=device, buffer_size=10000, batch_size=64, target_update=100)
    elif agent_name == "VDN":
        agent = agent_class(num_agents=num_uavs, state_dim=state_dim, action_dim=action_dim,
                           learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,
                           device=device, buffer_size=10000, batch_size=64, target_update=100)
    elif agent_name == "MADDPG":
        # MADDPG doesn't need global_state_dim - critic uses state_dim * num_agents internally
        agent = agent_class(num_agents=num_uavs, state_dim=state_dim, action_dim=action_dim,
                           learning_rate_actor=learning_rate, learning_rate_critic=learning_rate,
                           gamma=gamma, device=device, buffer_size=10000, batch_size=64, tau=0.01)
    elif agent_name == "DeepNashQ":
        agent = agent_class(num_agents=num_uavs, state_dim=state_dim, action_dim=action_dim,
                           learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,
                           device=device, buffer_size=10000, batch_size=64, target_update=100)
    elif agent_name == "MAPPO":
        # MAPPO is on-policy, uses different parameters
        agent = agent_class(num_agents=num_uavs, state_dim=state_dim, action_dim=action_dim,
                           learning_rate_actor=0.0003, learning_rate_critic=0.001,
                           gamma=gamma, clip_epsilon=0.2, device=device)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    episode_rewards = []
    episode_throughputs = []
    
    print(f"  Running {num_episodes} episodes...")
    
    for episode in tqdm(range(num_episodes), desc=f"  {agent_name}", leave=False):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_throughput = 0.0
        step_count = 0
        
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
        
        # For MAPPO (on-policy), collect trajectory
        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_next_states = []
        trajectory_dones = []
        trajectory_log_probs = []
        
        while not done and step_count < num_steps_per_episode:
            # Get actions from agent
            actions = []
            log_probs = []
            for i in range(num_uavs):
                # Adjust agent_obs slicing for potentially expanded observation space
                agent_obs = obs_tensor[i*state_dim:(i+1)*state_dim]
                if agent_name == "AdaptiveNonStationaryMARL":
                    action = agent.get_action(agent_obs, i, global_state=obs_tensor)
                    log_probs.append(0.0)  # Not used for this agent
                elif agent_name == "MAPPO":
                    action, log_prob = agent.get_action(agent_obs, i, explore=True)
                    log_probs.append(log_prob)
                elif agent_name == "MADDPG":
                    action = agent.get_action(agent_obs, i, explore=True)
                    log_probs.append(0.0)  # Not used for this agent
                else:
                    action = agent.get_action(agent_obs, i)
                    log_probs.append(0.0)  # Not used for this agent
                actions.append(action)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            episode_reward += reward
            episode_throughput += info.get('throughput', 0.0)
            step_count += 1
            
            # Convert next observation to tensor
            next_obs_tensor = torch.tensor(next_obs, device=device, dtype=torch.float32)
            
            # Store transition
            states = [obs_tensor[i*state_dim:(i+1)*state_dim] for i in range(num_uavs)]
            next_states = [next_obs_tensor[i*state_dim:(i+1)*state_dim] for i in range(num_uavs)]
            
            if agent_name == "MAPPO":
                # MAPPO is on-policy - store in trajectory for episode-end update
                trajectory_states.append(states)
                trajectory_actions.append(actions)
                trajectory_rewards.append([reward] * num_uavs)
                trajectory_next_states.append(next_states)
                trajectory_dones.append([done] * num_uavs)
                trajectory_log_probs.append(log_probs)
            elif agent_name == "AdaptiveNonStationaryMARL":
                # Novel algorithm needs global state
                global_state = obs_tensor
                next_global_state = next_obs_tensor
                agent.store_transition(states, actions, [reward] * num_uavs, 
                                     next_states, [done] * num_uavs,
                                     global_state, next_global_state)
            elif agent_name == "QMIX":
                # QMIX needs global state
                global_state = obs_tensor
                next_global_state = next_obs_tensor
                agent.store_transition(states, actions, [reward] * num_uavs, 
                                     next_states, [done] * num_uavs,
                                     global_state, next_global_state)
            elif agent_name == "MADDPG":
                # MADDPG store_transition expects states and actions
                agent.store_transition(states, actions, [reward] * num_uavs, 
                                     next_states, [done] * num_uavs)
            else:
                agent.store_transition(states, actions, [reward] * num_uavs, 
                                     next_states, [done] * num_uavs)
            
            # Update agent (off-policy agents update during episode)
            # MAPPO is on-policy and updates after episode
            if agent_name != "MAPPO":
                if hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) > agent.batch_size and step_count % 2 == 0:
                    agent.update()
            
            # Update observation for next step
            obs_tensor = next_obs_tensor
        
        # For MAPPO (on-policy), store full trajectory and update
        if agent_name == "MAPPO":
            for t in range(len(trajectory_states)):
                agent.store_transition(
                    trajectory_states[t], trajectory_actions[t], trajectory_rewards[t],
                    trajectory_next_states[t], trajectory_dones[t], trajectory_log_probs[t]
                )
            # Update after collecting full trajectory
            agent.update()
            
            obs_tensor = next_obs_tensor
        
        # Call end_episode for performative/non-stationary updates
        env.end_episode()
        
        episode_rewards.append(episode_reward)
        episode_throughputs.append(episode_throughput)
    
    return {
        'rewards': episode_rewards,
        'throughputs': episode_throughputs
    }


def plot_rl_convergence_comparison(results_dict_regular, results_novel_enhanced, num_episodes):
    """Create convergence comparison plots: Novel algo (enhanced) vs Others (regular)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # 1x2 layout: throughput and reward
    
    episodes = np.arange(num_episodes)
    window_size = min(100, num_episodes // 10)  # Larger window for better smoothing
    
    colors = {
        'AdaptiveNonStationaryMARL': 'red',
        'QMIX': 'blue',
        'IQL': 'green',
        'VDN': 'orange',
        'MADDPG': 'purple',
        'DeepNashQ': 'brown'
    }
    
    # Plot 1: Throughput Convergence (convert to Gbps)
    ax1 = axes[0]
    # Plot regular models
    for agent_name, results in results_dict_regular.items():
        color = colors.get(agent_name, 'gray')
        # Convert bps to Gbps (divide by 1e9)
        throughputs_gbps = np.array(results['throughputs']) / 1e9
        ax1.plot(episodes, throughputs_gbps, '-', alpha=0.08, linewidth=0.3, color=color, zorder=1)
        smoothed = compute_moving_average(throughputs_gbps, window_size)
        ax1.plot(episodes, smoothed, '-', linewidth=3.5, label=agent_name, color=color, zorder=3)
        
        std_window = window_size
        std_values = []
        for i in range(len(throughputs_gbps)):
            start_idx = max(0, i - std_window // 2)
            end_idx = min(len(throughputs_gbps), i + std_window // 2 + 1)
            std_values.append(np.std(throughputs_gbps[start_idx:end_idx]))
        std_values = np.array(std_values)
        ax1.fill_between(episodes, smoothed - std_values, smoothed + std_values, 
                        alpha=0.15, color=color, zorder=2)
    
    # Plot novel algorithm with enhanced model
    if 'AdaptiveNonStationaryMARL' in results_novel_enhanced:
        results = results_novel_enhanced['AdaptiveNonStationaryMARL']
        color = colors.get('AdaptiveNonStationaryMARL', 'red')
        # Convert bps to Gbps (divide by 1e9)
        throughputs_gbps = np.array(results['throughputs']) / 1e9
        ax1.plot(episodes, throughputs_gbps, '-', alpha=0.08, linewidth=0.3, color=color, zorder=1)
        smoothed = compute_moving_average(throughputs_gbps, window_size)
        ax1.plot(episodes, smoothed, '--', linewidth=3.5, label='AdaptiveNonStationaryMARL', 
                color=color, zorder=3)
        
        std_window = window_size
        std_values = []
        for i in range(len(throughputs_gbps)):
            start_idx = max(0, i - std_window // 2)
            end_idx = min(len(throughputs_gbps), i + std_window // 2 + 1)
            std_values.append(np.std(throughputs_gbps[start_idx:end_idx]))
        std_values = np.array(std_values)
        ax1.fill_between(episodes, smoothed - std_values, smoothed + std_values, 
                        alpha=0.15, color=color, zorder=2)
    
    ax1.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Total Throughput (Gbps)', fontsize=13, fontweight='bold')
    ax1.set_title('Throughput Convergence Comparison', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    
    # Plot 2: Reward Convergence
    ax2 = axes[1]
    # Plot regular models
    for agent_name, results in results_dict_regular.items():
        color = colors.get(agent_name, 'gray')
        ax2.plot(episodes, results['rewards'], '-', alpha=0.08, linewidth=0.3, color=color, zorder=1)
        smoothed = compute_moving_average(results['rewards'], window_size)
        ax2.plot(episodes, smoothed, '-', linewidth=3.5, label=agent_name, color=color, zorder=3)
        
        std_window = window_size
        std_values = []
        for i in range(len(results['rewards'])):
            start_idx = max(0, i - std_window // 2)
            end_idx = min(len(results['rewards']), i + std_window // 2 + 1)
            std_values.append(np.std(results['rewards'][start_idx:end_idx]))
        std_values = np.array(std_values)
        ax2.fill_between(episodes, smoothed - std_values, smoothed + std_values, 
                        alpha=0.15, color=color, zorder=2)
    
    # Plot novel algorithm with enhanced model
    if 'AdaptiveNonStationaryMARL' in results_novel_enhanced:
        results = results_novel_enhanced['AdaptiveNonStationaryMARL']
        color = colors.get('AdaptiveNonStationaryMARL', 'red')
        ax2.plot(episodes, results['rewards'], '-', alpha=0.08, linewidth=0.3, color=color, zorder=1)
        smoothed = compute_moving_average(results['rewards'], window_size)
        ax2.plot(episodes, smoothed, '--', linewidth=3.5, label='AdaptiveNonStationaryMARL', 
                color=color, zorder=3)
        
        std_window = window_size
        std_values = []
        for i in range(len(results['rewards'])):
            start_idx = max(0, i - std_window // 2)
            end_idx = min(len(results['rewards']), i + std_window // 2 + 1)
            std_values.append(np.std(results['rewards'][start_idx:end_idx]))
        std_values = np.array(std_values)
        ax2.fill_between(episodes, smoothed - std_values, smoothed + std_values, 
                        alpha=0.15, color=color, zorder=2)
    
    ax2.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Episode Reward', fontsize=13, fontweight='bold')
    ax2.set_title('Reward Convergence Comparison', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig('rl_models_convergence_comparison.png', dpi=200, bbox_inches='tight')
    print("✓ Saved RL models convergence comparison to 'rl_models_convergence_comparison.png'")
    plt.close('all')


def main():
    """Main function to run RL models convergence comparison."""
    print("="*70)
    print("RL Models Convergence Comparison")
    print("Novel Algorithm (Enhanced) vs Other RL Models (Regular)")
    print(f"{'='*70}")
    
    # Configuration
    num_uavs = 3
    num_users = 20
    num_episodes = 1000  # 1000 episodes
    num_steps_per_episode = 30  # Reduced for faster execution
    grid_size = (10, 10, 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 0.1
    context_dim = 7  # For novel algorithm
    
    print(f"\nConfiguration:")
    print(f"  - Number of UAVs: {num_uavs}")
    print(f"  - Number of Users: {num_users}")
    print(f"  - Episodes: {num_episodes}")
    print(f"  - Steps per episode: {num_steps_per_episode}")
    print(f"  - Grid Size: {grid_size}")
    print(f"  - Device: {device}")
    
    # Define regular models (use regular environment)
    regular_models = {
        'QMIX': QMIX,
        'IQL': IQL,
        'VDN': VDN,
        'MADDPG': MADDPG,
        'DeepNashQ': DeepNashQ,
        'MAPPO': MAPPO
    }
    
    results_dict_regular = {}
    results_novel_enhanced = {}
    
    # Train each regular model with regular environment
    print(f"\n{'='*70}")
    print("Training Regular RL Models (Regular Environment)")
    print(f"{'='*70}")
    for agent_name, agent_class in regular_models.items():
        env = MARLEnv(num_uavs=num_uavs, num_users=num_users, grid_size=grid_size, device=device,
                     enable_non_stationary=False, enable_performative=False)
        results = train_model(agent_class, agent_name, env, num_episodes=num_episodes,
                             num_steps_per_episode=num_steps_per_episode,
                             learning_rate=learning_rate, gamma=gamma, epsilon=epsilon, device=device)
        results_dict_regular[agent_name] = results
    
    # Train novel algorithm with enhanced environment
    print(f"\n{'='*70}")
    print("Training Novel Algorithm (Enhanced Environment: Performative + Non-Stationary)")
    print(f"{'='*70}")
    env = MARLEnv(num_uavs=num_uavs, num_users=num_users, grid_size=grid_size, device=device,
                 enable_non_stationary=True, enable_performative=True)
    results = train_model(AdaptiveNonStationaryMARL, "AdaptiveNonStationaryMARL", env, 
                         num_episodes=num_episodes, num_steps_per_episode=num_steps_per_episode,
                         learning_rate=learning_rate, gamma=gamma, epsilon=epsilon, 
                         device=device, context_dim=context_dim)
    results_novel_enhanced['AdaptiveNonStationaryMARL'] = results
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("Generating convergence comparison plots...")
    print(f"{'='*70}")
    plot_rl_convergence_comparison(results_dict_regular, results_novel_enhanced, num_episodes)
    
    # Print summary
    print(f"\n{'='*70}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*70}")
    last_n = min(100, num_episodes // 4)  # Use last 25% of episodes
    print(f"\nFinal Performance - Regular Models (last {last_n} episodes):")
    for agent_name in regular_models.keys():
        final_tp = np.mean(results_dict_regular[agent_name]['throughputs'][-last_n:])
        final_rwd = np.mean(results_dict_regular[agent_name]['rewards'][-last_n:])
        print(f"  {agent_name:25s}: Throughput = {final_tp:.2e}, Reward = {final_rwd:.2f}")
    
    print(f"\nFinal Performance - Novel Algorithm with Enhanced Model (last {last_n} episodes):")
    if 'AdaptiveNonStationaryMARL' in results_novel_enhanced:
        final_tp = np.mean(results_novel_enhanced['AdaptiveNonStationaryMARL']['throughputs'][-last_n:])
        final_rwd = np.mean(results_novel_enhanced['AdaptiveNonStationaryMARL']['rewards'][-last_n:])
        print(f"  {'AdaptiveNonStationaryMARL':25s}: Throughput = {final_tp:.2e}, Reward = {final_rwd:.2f}")
    
    print(f"\n{'='*70}")
    print("✅ RL models convergence comparison complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

