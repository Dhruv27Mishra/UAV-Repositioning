import numpy as np
import matplotlib.pyplot as plt

# Load results
try:
    deepnash_rewards = np.load('results/deepnash_rewards.npy')
    vdn_rewards = np.load('results/vdn_rewards.npy')
    qmix_rewards = np.load('results/qmix_rewards.npy')

    deepnash_throughputs = np.load('results/deepnash_throughputs.npy')
    vdn_throughputs = np.load('results/vdn_throughputs.npy')
    qmix_throughputs = np.load('results/qmix_throughputs.npy')
except FileNotFoundError as e:
    print(f"Missing results file: {e}")
    print("Please ensure all models have been trained and their results saved in the 'results' directory.")
    exit(1)

# Create figure with subplots
plt.figure(figsize=(15, 10))

# 1. Reward Comparison
plt.subplot(2, 2, 1)
window_size = 20
for rewards, label in [(deepnash_rewards, 'Deep Nash Q'), (vdn_rewards, 'VDN'), (qmix_rewards, 'QMIX')]:
    rewards_ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(rewards)), rewards_ma, label=label, linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode (Moving Average)')
plt.legend()
plt.grid(True)

# 2. Throughput Comparison
plt.subplot(2, 2, 2)
for throughputs, label in [(deepnash_throughputs, 'Deep Nash Q'), (vdn_throughputs, 'VDN'), (qmix_throughputs, 'QMIX')]:
    throughputs_ma = np.convolve(throughputs, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(throughputs)), throughputs_ma, label=label, linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Throughput')
plt.title('Throughput per Episode (Moving Average)')
plt.legend()
plt.grid(True)

# 3. Performance Metrics Bar Chart
plt.subplot(2, 2, 3)
metrics = {
    'Deep Nash Q': {
        'Avg Reward': np.mean(deepnash_rewards),
        'Avg Throughput': np.mean(deepnash_throughputs),
        'Total Collisions': 92  # From the logs
    },
    'VDN': {
        'Avg Reward': np.mean(vdn_rewards),
        'Avg Throughput': np.mean(vdn_throughputs),
        'Total Collisions': 0  # From the logs
    },
    'QMIX': {
        'Avg Reward': np.mean(qmix_rewards),
        'Avg Throughput': np.mean(qmix_throughputs),
        'Total Collisions': 0  # From the logs
    }
}

x = np.arange(3)
width = 0.25

plt.bar(x - width, [metrics[algo]['Avg Reward'] for algo in ['Deep Nash Q', 'VDN', 'QMIX']], 
        width, label='Avg Reward')
plt.bar(x, [metrics[algo]['Avg Throughput'] for algo in ['Deep Nash Q', 'VDN', 'QMIX']], 
        width, label='Avg Throughput')
plt.bar(x + width, [metrics[algo]['Total Collisions'] for algo in ['Deep Nash Q', 'VDN', 'QMIX']], 
        width, label='Total Collisions')

plt.xlabel('Algorithm')
plt.ylabel('Value')
plt.title('Performance Metrics Comparison')
plt.xticks(x, ['Deep Nash Q', 'VDN', 'QMIX'])
plt.legend()
plt.grid(True)

# 4. Learning Progress
plt.subplot(2, 2, 4)
for rewards, label in [(deepnash_rewards, 'Deep Nash Q'), (vdn_rewards, 'VDN'), (qmix_rewards, 'QMIX')]:
    cumulative_rewards = np.cumsum(rewards)
    plt.plot(cumulative_rewards, label=label, linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Learning Progress')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
plt.show()
deepnash_fairness = np.load("results/deepnash_fairness.npy")
qmix_fairness = np.load("results/qmix_fairness.npy")
vdn_fairness = np.load("results/vdn_fairness.npy")

def smooth(y, window=20):
    return np.convolve(y, np.ones(window)/window, mode='valid')

# Smooth data
window_size = 20
dn_smooth = smooth(deepnash_fairness, window_size)
qmix_smooth = smooth(qmix_fairness, window_size)
vdn_smooth = smooth(vdn_fairness, window_size)
x_vals = np.arange(len(dn_smooth))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(x_vals, dn_smooth, label="Deep NashQ", linewidth=2)
plt.plot(x_vals, qmix_smooth, label="QMIX", linewidth=2)
plt.plot(x_vals, vdn_smooth, label="VDN", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Jain's Fairness Index")
plt.title("Fairness Comparison Across Algorithms")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/fairness_comparison_smooth.png")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

user_counts = [1, 2, 3, 4, 5,6,7,8,9,10]

# Load saved data
deepnash_fairness = np.load("results/deepnash_fairness_vs_density.npy")
qmix_fairness = np.load("results/qmix_fairness_vs_density.npy")
vdn_fairness = np.load("results/vdn_fairness_vs_density.npy")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(user_counts, deepnash_fairness, marker='o', label='Deep NashQ')
plt.plot(user_counts, qmix_fairness, marker='s', label='QMIX')
plt.plot(user_counts, vdn_fairness, marker='^', label='VDN')

plt.xlabel("Number of Users (UE Density)")
plt.xticks(user_counts)
plt.ylabel("Jain's Fairness Index")
plt.title("Fairness vs UE Density (All Models)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/fairness_vs_density_all_models.png")
plt.show()

# Print detailed statistics
print("\nDetailed Performance Statistics:")
print("-" * 50)
for algo in ['Deep Nash Q', 'VDN', 'QMIX']:
    print(f"\n{algo}:")
    print(f"Average Reward: {metrics[algo]['Avg Reward']:.2f}")
    print(f"Average Throughput: {metrics[algo]['Avg Throughput']:.2f}")
    print(f"Total Collisions: {metrics[algo]['Total Collisions']}")
    print(f"Collision Rate: {metrics[algo]['Total Collisions']/1000:.3f} per episode") 