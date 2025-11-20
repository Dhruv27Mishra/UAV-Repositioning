"""
Comprehensive comparison script for all MARL algorithms.
Compares: IQL, VDN, QMIX, Deep Nash Q, MADDPG
"""
import numpy as np
import matplotlib.pyplot as plt
from rl_agent.marl_env import MARLEnv
from rl_agent.IQL import IQL
from rl_agent.VDN import VDN
from rl_agent.QMIX import QMIX
from rl_agent.DeepNashQ import DeepNashQ
from rl_agent.MADDPG import MADDPG
import torch
from tqdm import tqdm
import time


def jains_fairness(user_rates):
    """Calculate Jain's fairness index."""
    user_rates = np.array(user_rates)
    numerator = np.sum(user_rates) ** 2
    denominator = len(user_rates) * np.sum(user_rates ** 2)
    return numerator / denominator if denominator != 0 else 0


def run_algorithm(algorithm_name, agent_class, env, num_episodes=100, num_uavs=3,
                  state_dim=3, action_dim=6, device=None, **kwargs):
    """Run a MARL algorithm and collect metrics."""
    print(f"\n{'='*70}")
    print(f"Training {algorithm_name}")
    print(f"{'='*70}")
    
    # Initialize agent
    if algorithm_name == "QMIX":
        global_state_dim = state_dim * num_uavs
        agent = agent_class(num_agents=num_uavs, state_dim=state_dim,
                          action_dim=action_dim, global_state_dim=global_state_dim,
                          device=device, **kwargs)
    elif algorithm_name == "MADDPG":
        agent = agent_class(num_agents=num_uavs, state_dim=state_dim,
                          action_dim=action_dim, device=device, **kwargs)
    else:
        agent = agent_class(num_agents=num_uavs, state_dim=state_dim,
                          action_dim=action_dim, device=device, **kwargs)
    
    metrics = {
        'throughputs': [],
        'rewards': [],
        'fairnesses': [],
        'collisions': [],
        'heights': []
    }
    
    for episode in tqdm(range(num_episodes), desc=f"  {algorithm_name}"):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_throughput = 0
        episode_heights = []
        step_count = 0
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
        
        while not done and step_count < 50:
            # Get actions
            actions = []
            for i in range(num_uavs):
                agent_obs = obs_tensor[i*3:(i+1)*3]
                if algorithm_name == "MADDPG":
                    explore = episode < num_episodes * 0.8
                    action = agent.get_action(agent_obs, i, explore=explore)
                else:
                    action = agent.get_action(agent_obs, i)
                actions.append(action)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            next_obs_tensor = torch.tensor(next_obs, device=device, dtype=torch.float32)
            
            # Store transition
            states = [obs_tensor[i*3:(i+1)*3] for i in range(num_uavs)]
            next_states = [next_obs_tensor[i*3:(i+1)*3] for i in range(num_uavs)]
            
            if algorithm_name == "QMIX":
                global_state = obs_tensor.flatten()
                next_global_state = next_obs_tensor.flatten()
                agent.store_transition(states, actions, [reward] * num_uavs,
                                      next_states, [done] * num_uavs,
                                      global_state, next_global_state)
            else:
                agent.store_transition(states, actions, [reward] * num_uavs,
                                      next_states, [done] * num_uavs)
            
            # Update agent
            agent.update()
            
            obs = next_obs
            obs_tensor = next_obs_tensor
            episode_reward += reward
            episode_throughput += info.get('throughput', 0)
            episode_heights.append(env.uav_positions[:, 2].copy())
            
            step_count += 1
        
        # Calculate fairness
        if 'user_rates' in info:
            fairness = jains_fairness(info['user_rates'])
        else:
            fairness = 0
        
        metrics['throughputs'].append(episode_throughput)
        metrics['rewards'].append(episode_reward)
        metrics['fairnesses'].append(fairness)
        metrics['collisions'].append(info.get('collisions', False))
        metrics['heights'].append(np.mean(episode_heights) if episode_heights else 0)
    
    return metrics


def compare_all_algorithms(num_episodes=200, num_uavs=3, num_users=10):
    """Compare all MARL algorithms."""
    print("=" * 70)
    print("COMPREHENSIVE MARL ALGORITHM COMPARISON")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MARLEnv(num_uavs=num_uavs, num_users=num_users, grid_size=(10, 10, 5),
                 device=device)
    state_dim = 3
    action_dim = env.action_space.nvec[0]
    
    # Algorithm configurations
    algorithms = {
        'IQL': (IQL, {'learning_rate': 0.001, 'gamma': 0.99, 'epsilon': 0.1,
                     'buffer_size': 10000, 'batch_size': 64, 'target_update': 100}),
        'VDN': (VDN, {'learning_rate': 0.001, 'gamma': 0.99, 'epsilon': 0.1,
                     'buffer_size': 10000, 'batch_size': 64, 'target_update': 100}),
        'QMIX': (QMIX, {'learning_rate': 0.001, 'gamma': 0.99, 'epsilon': 0.1,
                       'buffer_size': 10000, 'batch_size': 64, 'target_update': 100}),
        'Deep Nash Q': (DeepNashQ, {'learning_rate': 0.001, 'gamma': 0.99,
                                   'epsilon': 0.1, 'buffer_size': 10000,
                                   'batch_size': 64, 'target_update': 100}),
        'MADDPG': (MADDPG, {'learning_rate_actor': 0.001,
                           'learning_rate_critic': 0.001, 'gamma': 0.99,
                           'tau': 0.01, 'buffer_size': 10000, 'batch_size': 64})
    }
    
    results = {}
    
    # Run each algorithm
    for alg_name, (alg_class, alg_kwargs) in algorithms.items():
        metrics = run_algorithm(alg_name, alg_class, env, num_episodes=num_episodes,
                              num_uavs=num_uavs, state_dim=state_dim,
                              action_dim=action_dim, device=device, **alg_kwargs)
        results[alg_name] = metrics
    
    # Print summary
    print_summary(results)
    
    # Create plots
    plot_comparison(results)
    
    return results


def print_summary(results):
    """Print summary statistics for all algorithms."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n📊 THROUGHPUT STATISTICS:")
    throughput_means = {}
    for alg_name, metrics in results.items():
        mean_tp = np.mean(metrics['throughputs'])
        std_tp = np.std(metrics['throughputs'])
        throughput_means[alg_name] = mean_tp
        print(f"  {alg_name:15s}: {mean_tp:7.3f} ± {std_tp:6.3f}")
    
    best_alg = max(throughput_means, key=throughput_means.get)
    print(f"\n🏆 Best Performer: {best_alg} ({throughput_means[best_alg]:.3f})")
    
    print("\n📈 FAIRNESS STATISTICS:")
    for alg_name, metrics in results.items():
        mean_fair = np.mean(metrics['fairnesses'])
        print(f"  {alg_name:15s}: {mean_fair:.4f}")
    
    print("\n💥 COLLISIONS:")
    for alg_name, metrics in results.items():
        total_collisions = np.sum(metrics['collisions'])
        print(f"  {alg_name:15s}: {total_collisions:4d}")
    
    print("\n📏 HEIGHT STATISTICS:")
    for alg_name, metrics in results.items():
        mean_height = np.mean([h for h in metrics['heights'] if h > 0])
        print(f"  {alg_name:15s}: {mean_height:6.2f}m")


def plot_comparison(results):
    """Create comprehensive comparison plots."""
    fig = plt.figure(figsize=(18, 12))
    
    algorithms = list(results.keys())
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot 1: Throughput over episodes
    ax1 = plt.subplot(2, 3, 1)
    for i, alg_name in enumerate(algorithms):
        episodes = range(len(results[alg_name]['throughputs']))
        ax1.plot(episodes, results[alg_name]['throughputs'],
                color=colors[i % len(colors)], alpha=0.6, label=alg_name, linewidth=1.5)
    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput', fontsize=12, fontweight='bold')
    ax1.set_title('Throughput Over Episodes', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average throughput comparison
    ax2 = plt.subplot(2, 3, 2)
    means = [np.mean(results[alg]['throughputs']) for alg in algorithms]
    stds = [np.std(results[alg]['throughputs']) for alg in algorithms]
    bars = ax2.bar(algorithms, means, yerr=stds, capsize=5, color=colors[:len(algorithms)],
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Average Throughput', fontsize=12, fontweight='bold')
    ax2.set_title('Average Throughput Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + stds[i],
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Fairness comparison
    ax3 = plt.subplot(2, 3, 3)
    fairness_means = [np.mean(results[alg]['fairnesses']) for alg in algorithms]
    bars = ax3.bar(algorithms, fairness_means, color=colors[:len(algorithms)],
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel("Jain's Fairness Index", fontsize=12, fontweight='bold')
    ax3.set_title('Fairness Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar, mean in zip(bars, fairness_means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 4: Throughput distribution
    ax4 = plt.subplot(2, 3, 4)
    for i, alg_name in enumerate(algorithms):
        ax4.hist(results[alg_name]['throughputs'], bins=30, alpha=0.5,
                color=colors[i % len(colors)], label=alg_name, density=True)
    ax4.set_xlabel('Throughput', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax4.set_title('Throughput Distribution', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Reward comparison
    ax5 = plt.subplot(2, 3, 5)
    reward_means = [np.mean(results[alg]['rewards']) for alg in algorithms]
    reward_stds = [np.std(results[alg]['rewards']) for alg in algorithms]
    bars = ax5.bar(algorithms, reward_means, yerr=reward_stds, capsize=5,
                  color=colors[:len(algorithms)], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax5.set_title('Reward Comparison', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 6: Summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    table_data = []
    for alg in algorithms:
        metrics = results[alg]
        mean_tp = np.mean(metrics['throughputs'])
        mean_fair = np.mean(metrics['fairnesses'])
        collisions = np.sum(metrics['collisions'])
        table_data.append([
            alg,
            f'{mean_tp:.3f}',
            f'{mean_fair:.3f}',
            f'{collisions}'
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Algorithm', 'Avg Throughput', 'Fairness', 'Collisions'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best performer
    best_idx = np.argmax(means)
    for i in range(4):
        table[(best_idx + 1, i)].set_facecolor('#FFEB3B')
    
    plt.suptitle('MARL Algorithm Comparison: All Methods', fontsize=16,
                fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('marl_comparison_all.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved to 'marl_comparison_all.png'")


if __name__ == "__main__":
    results = compare_all_algorithms(num_episodes=200, num_uavs=3, num_users=10)
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)

