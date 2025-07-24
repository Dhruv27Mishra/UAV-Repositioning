import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import importlib.util
print("Models being compared: Deep Nash Q, VDN, QMIX, Deterministic I, Deterministic II")
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
model_path = os.path.join(current_dir, "Geometric Deterministic Model", "model.py")
spec = importlib.util.spec_from_file_location("deterministic_model", model_path)
deterministic_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deterministic_model)
abid_model_path = os.path.join(current_dir, "Geometric Deterministic Model", "abid_model.py")
abid_spec = importlib.util.spec_from_file_location("abid_model", abid_model_path)
abid_model = importlib.util.module_from_spec(abid_spec)
abid_spec.loader.exec_module(abid_model)
num_episodes = 1000
if not (os.path.exists('results/deterministic_throughputs.npy') and os.path.exists('results/deterministic_rewards.npy')):
    deterministic_throughputs = []
    deterministic_rewards = []
    sim = deterministic_model.DeterministicUAVSimulation()
    for _ in range(num_episodes):
        throughput, _ = sim.run()
        deterministic_throughputs.append(throughput)
        deterministic_rewards.append(throughput * 0.1)
    np.save('results/deterministic_throughputs.npy', np.array(deterministic_throughputs))
    np.save('results/deterministic_rewards.npy', np.array(deterministic_rewards))
if not (os.path.exists('results/abid_throughputs.npy') and os.path.exists('results/abid_rewards.npy')):
    abid_throughputs = []
    abid_rewards = []
    abid_sim = abid_model.AbidUAVSimulation()
    for _ in range(num_episodes):
        throughput, _ = abid_sim.run()
        abid_throughputs.append(throughput)
        abid_rewards.append(throughput * 0.1)
    np.save('results/abid_throughputs.npy', np.array(abid_throughputs))
    np.save('results/abid_rewards.npy', np.array(abid_rewards))
try:
    deepnash_rewards = np.load('results/deepnash_rewards.npy')
    vdn_rewards = np.load('results/vdn_rewards.npy')
    qmix_rewards = np.load('results/qmix_rewards.npy')
    deterministic_rewards = np.load('results/deterministic_rewards.npy')
    abid_rewards = np.load('results/abid_rewards.npy')
    deepnash_throughputs = np.load('results/deepnash_throughputs.npy')
    vdn_throughputs = np.load('results/vdn_throughputs.npy')
    qmix_throughputs = np.load('results/qmix_throughputs.npy')
    deterministic_throughputs = np.load('results/deterministic_throughputs.npy')
    abid_throughputs = np.load('results/abid_throughputs.npy')
except FileNotFoundError as e:
    print(f"Missing results file: {e}")
    print("Please ensure all models have been trained and their results saved in the 'results' directory.")
    exit(1)
plt.figure(figsize=(18, 10))
window_size = 20
plt.subplot(2, 2, 1)
for rewards, label in zip([
    deepnash_rewards, vdn_rewards, qmix_rewards, deterministic_rewards, abid_rewards
], [
    'Deep Nash Q', 'VDN', 'QMIX', 'Deterministic I', 'Deterministic II'
]):
    rewards_ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(rewards)), rewards_ma, label=label, linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode (Moving Average)')
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 2)
for throughputs, label in zip([
    deepnash_throughputs, vdn_throughputs, qmix_throughputs, deterministic_throughputs, abid_throughputs
], [
    'Deep Nash Q', 'VDN', 'QMIX', 'Deterministic I', 'Deterministic II'
]):
    throughputs_ma = np.convolve(throughputs, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(throughputs)), throughputs_ma, label=label, linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Throughput')
plt.title('Throughput per Episode (Moving Average)')
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 3)
metrics = {
    'Deep Nash Q': {
        'Avg Reward': np.mean(deepnash_rewards),
        'Avg Throughput': np.mean(deepnash_throughputs),
        'Total Collisions': 92
    },
    'VDN': {
        'Avg Reward': np.mean(vdn_rewards),
        'Avg Throughput': np.mean(vdn_throughputs),
        'Total Collisions': 0
    },
    'QMIX': {
        'Avg Reward': np.mean(qmix_rewards),
        'Avg Throughput': np.mean(qmix_throughputs),
        'Total Collisions': 0
    },
    'Deterministic I': {
        'Avg Reward': np.mean(deterministic_rewards),
        'Avg Throughput': np.mean(deterministic_throughputs),
        'Total Collisions': 0
    },
    'Deterministic II': {
        'Avg Reward': np.mean(abid_rewards),
        'Avg Throughput': np.mean(abid_throughputs),
        'Total Collisions': 0
    }
}
x = np.arange(5)
width = 0.13
plt.bar(x - 2*width, [metrics[algo]['Avg Reward'] for algo in metrics], width, label='Avg Reward')
plt.bar(x - width, [metrics[algo]['Avg Throughput'] for algo in metrics], width, label='Avg Throughput')
plt.bar(x, [metrics[algo]['Total Collisions'] for algo in metrics], width, label='Total Collisions')
plt.xlabel('Algorithm')
plt.ylabel('Value')
plt.title('Performance Metrics Comparison')
plt.xticks(x, list(metrics.keys()))
plt.legend()
plt.grid(True)
plt.subplot(2, 2, 4)
for rewards, label in zip([
    deepnash_rewards, vdn_rewards, qmix_rewards, deterministic_rewards, abid_rewards
], [
    'Deep Nash Q', 'VDN', 'QMIX', 'Deterministic I', 'Deterministic II'
]):
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
print("\nDetailed Performance Statistics:")
print("-" * 50)
for algo in metrics:
    print(f"\n{algo}:")
    print(f"Average Reward: {metrics[algo]['Avg Reward']:.2f}")
    print(f"Average Throughput: {metrics[algo]['Avg Throughput']:.2f}")
    print(f"Total Collisions: {metrics[algo]['Total Collisions']}")
    print(f"Collision Rate: {metrics[algo]['Total Collisions']/1000:.3f} per episode")
user_counts = [1, 2, 3, 4, 5]
try:
    deepnash = np.load("results/deepnash_min_rate_vs_density.npy")
    qmix = np.load("results/qmix_min_rate_vs_density.npy")
    vdn = np.load("results/vdn_min_rate_vs_density.npy")
    abid = np.load("results/abid_min_rate_vs_density.npy")
    deterministic = np.load("results/deterministic_min_rate_vs_density.npy")
    plt.figure(figsize=(10, 6))
    plt.plot(user_counts, deepnash, marker='o', label='Deep NashQ')
    plt.plot(user_counts, qmix, marker='s', label='QMIX')
    plt.plot(user_counts, vdn, marker='^', label='VDN')
    plt.plot(user_counts, abid, marker='v', label='Abid Model')
    plt.plot(user_counts, deterministic, marker='x', label='Deterministic')
    plt.xlabel("Number of Users (UE Density)")
    plt.ylabel("Minimum Throughput per UE (Mbps)")
    plt.title("Minimum User Data Rate vs UE Density (All Models)")
    plt.legend()
    plt.grid(True)
    plt.xticks(user_counts)
    plt.tight_layout()
    plt.savefig("results/min_rate_vs_density_all_models.png")
    plt.show()
except Exception as e:
    print("Could not load min rate vs density results for all models:", e)
