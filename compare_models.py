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

# Print detailed statistics
print("\nDetailed Performance Statistics:")
print("-" * 50)
for algo in ['Deep Nash Q', 'VDN', 'QMIX']:
    print(f"\n{algo}:")
    print(f"Average Reward: {metrics[algo]['Avg Reward']:.2f}")
    print(f"Average Throughput: {metrics[algo]['Avg Throughput']:.2f}")
    print(f"Total Collisions: {metrics[algo]['Total Collisions']}")
    print(f"Collision Rate: {metrics[algo]['Total Collisions']/1000:.3f} per episode") 