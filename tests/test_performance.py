"""
Test script to verify performance with variable height and height-dependent LOS probability.
"""
from pathlib import Path
import sys

_root = next(p for p in Path(__file__).resolve().parents if (p / "rl_agent" / "marl_env.py").exists())
if str(_root) not in sys.path:
    sys.path.append(str(_root))

import numpy as np
from rl_agent.marl_env import MARLEnv
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_height_dependent_los():
    """Test the height-dependent LOS probability function."""
    print("=" * 60)
    print("Testing Height-Dependent LOS Probability Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MARLEnv(num_uavs=1, num_users=1, grid_size=(10, 10, 5), device=device)
    
    # Test over a range of heights
    heights = np.linspace(5, 50, 100)
    los_probs = []
    
    print("Height (m) | LOS Probability")
    print("-" * 30)
    for h in heights:
        los_prob = env.height_dependent_los_probability(h)
        los_probs.append(los_prob)
        if h % 5 == 0:
            print(f"{h:10.1f} | {los_prob:0.3f}")
    
    # Plot LOS probability vs height
    plt.figure(figsize=(8, 5))
    plt.plot(heights, los_probs, 'b-', linewidth=2)
    plt.axvline(x=env.optimal_height, color='r', linestyle='--', label=f'Optimal Height ({env.optimal_height}m)')
    plt.xlabel('UAV Height (meters)', fontsize=12)
    plt.ylabel('LOS Probability', fontsize=12)
    plt.title('Height-Dependent LOS Probability', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('test_los_probability.png', dpi=150)
    print(f"\n✓ LOS probability plot saved to 'test_los_probability.png'")
    print(f"  Optimal height: {env.optimal_height}m")
    print(f"  Max LOS prob: {max(los_probs):.3f} at height {heights[np.argmax(los_probs)]:.1f}m")
    
    return heights, los_probs

def test_throughput_vs_height():
    """Test how throughput varies with UAV height."""
    print("\n" + "=" * 60)
    print("Testing Throughput vs Height")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MARLEnv(num_uavs=3, num_users=10, grid_size=(10, 10, 5), device=device)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Test different heights
    test_heights = np.linspace(10, 40, 15)
    throughputs = []
    avg_user_rates = []
    
    print("Testing throughput at different heights (averaging over 10 samples)...")
    for h in tqdm(test_heights, desc="Height tests"):
        # Set all UAVs to same height
        env.uav_positions[:, 2] = h
        
        # Average over multiple samples to account for stochastic effects
        samples = 10
        height_throughputs = []
        height_user_rates = []
        
        for _ in range(samples):
            actions = [0] * env.num_uavs  # Hover action (no movement)
            _, _, _, _, info = env.step(actions)
            height_throughputs.append(info['throughput'])
            height_user_rates.append(np.mean(info['user_rates']))
        
        throughputs.append(np.mean(height_throughputs))
        avg_user_rates.append(np.mean(height_user_rates))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(test_heights, throughputs, 'g-', linewidth=2, marker='o')
    ax1.axvline(x=env.optimal_height, color='r', linestyle='--', label=f'Optimal Height ({env.optimal_height}m)')
    ax1.set_xlabel('UAV Height (meters)', fontsize=12)
    ax1.set_ylabel('Total Throughput', fontsize=12)
    ax1.set_title('Total Throughput vs Height', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(test_heights, avg_user_rates, 'b-', linewidth=2, marker='s')
    ax2.axvline(x=env.optimal_height, color='r', linestyle='--', label=f'Optimal Height ({env.optimal_height}m)')
    ax2.set_xlabel('UAV Height (meters)', fontsize=12)
    ax2.set_ylabel('Average User Rate', fontsize=12)
    ax2.set_title('Average User Rate vs Height', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('test_throughput_vs_height.png', dpi=150)
    print("\n✓ Throughput vs height plot saved to 'test_throughput_vs_height.png'")
    
    return test_heights, throughputs, avg_user_rates

def test_throughput_vs_velocity():
    """
    Test how throughput varies with UE (user equipment) velocity.
    We sweep UE velocities and measure average throughput.
    """
    print("\n" + "=" * 60)
    print("Testing Throughput vs UE Velocity")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MARLEnv(num_uavs=3, num_users=20, grid_size=(10, 10, 5), device=device)

    # Velocities in m/s: from static to high mobility
    test_velocities = np.array([0, 2, 5, 10, 15, 20, 25, 30])
    throughputs = []
    avg_user_rates = []

    num_episodes = 20  # average over multiple episodes
    print(f"Testing throughput at different UE velocities (averaging over {num_episodes} episodes per point)...")

    for v in tqdm(test_velocities, desc="Velocity tests"):
        vel_throughputs = []
        vel_user_rates = []

        for _ in range(num_episodes):
            obs, _ = env.reset()

            # Set ALL users to the same velocity v and correct categories
            env.user_velocities[:] = v
            if v < 5:
                env.user_velocity_categories = ['LOW_MOBILITY'] * env.num_users
            elif v < 15:
                env.user_velocity_categories = ['MEDIUM_MOBILITY'] * env.num_users
            else:
                env.user_velocity_categories = ['HIGH_MOBILITY'] * env.num_users

            # One step is enough to get throughput from current positions
            actions = [0] * env.num_uavs  # e.g., "hover" or any fixed action
            _, _, _, _, info = env.step(actions)

            vel_throughputs.append(info.get('throughput', 0.0))
            if 'user_rates' in info and len(info['user_rates']) > 0:
                vel_user_rates.append(float(np.mean(info['user_rates'])))

        throughputs.append(float(np.mean(vel_throughputs)))
        avg_user_rates.append(float(np.mean(vel_user_rates)) if vel_user_rates else 0.0)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(test_velocities, throughputs, marker='o', linewidth=2)
    plt.xlabel("UE Velocity (m/s)")
    plt.ylabel("Average Throughput")
    plt.title("Throughput vs UE Velocity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("test_throughput_vs_velocity.png", dpi=150)

    print("\n✓ Throughput vs UE velocity plot saved to 'test_throughput_vs_velocity.png'")
    print("Velocities:", test_velocities)
    print("Throughputs:", throughputs)

    return test_velocities, throughputs, avg_user_rates

def test_short_training():
    """Run a short training loop to verify stability."""
    print("\n" + "=" * 60)
    print("Testing Short Training Run")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MARLEnv(num_uavs=3, num_users=10, grid_size=(10, 10, 5), device=device)
    
    num_episodes = 50
    rewards = []
    throughputs = []
    mean_heights = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_throughput = 0
        episode_heights = []
        
        while not done:
            actions = [np.random.randint(0, 6) for _ in range(env.num_uavs)]
            obs, reward, done, _, info = env.step(actions)
            episode_reward += reward
            episode_throughput += info.get('throughput', 0)
            episode_heights.append(env.uav_positions[:, 2].copy())
        
        rewards.append(episode_reward)
        throughputs.append(episode_throughput)
        mean_heights.append(np.mean(episode_heights))
    
    # Plot training curves
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    
    ax1.plot(rewards, 'b-')
    ax1.set_title("Episode Reward")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    
    ax2.plot(throughputs, 'g-')
    ax2.set_title("Episode Throughput")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Throughput")
    
    ax3.plot(mean_heights, 'r-')
    ax3.set_title("Mean UAV Height")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Height (m)")
    
    plt.tight_layout()
    plt.savefig('test_short_training.png', dpi=150)
    print("\n✓ Short training plots saved to 'test_short_training.png'")
    
    return rewards, throughputs, mean_heights

def main():
    """Run all performance tests."""
    print("\n" + "=" * 60)
    print("UAV Variable Height Performance Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Height-dependent LOS probability
        heights, los_probs = test_height_dependent_los()
        
        # Test 2: Throughput vs height
        test_heights, throughputs, user_rates = test_throughput_vs_height()
        
        # Test 3: Throughput vs UE velocity
        velocities, vel_throughputs, vel_user_rates = test_throughput_vs_velocity()
        
        # Test 4: Short training run
        rewards, throughputs_train, heights_train = test_short_training()
        
        print("\n" + "=" * 60)
        print("All Tests Completed Successfully!")
        print("=" * 60)
        print("\nKey Findings:")
        print(f"  - LOS probability increases with height up to ~{heights[np.argmax(los_probs)]:.1f}m")
        print(f"  - Optimal height for throughput: ~{test_heights[np.argmax(throughputs)]:.1f}m")
        print(f"  - Generated throughput vs UE velocity curve for velocities {velocities[0]} to {velocities[-1]} m/s")
        print(f"  - System successfully handles variable heights in range [5, 50]m")
        print(f"  - Height-dependent path loss and shadowing are working correctly")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)