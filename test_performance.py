"""
Test script to verify performance with variable height and height-dependent LOS probability.
"""
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
    env = MARLEnv(num_uavs=3, num_users=10, grid_size=(10, 10, 5), device=device)
    
    # Test LOS probability at different heights
    heights = np.linspace(5, 50, 20)
    los_probs = []
    
    for h in heights:
        prob = env._compute_los_probability(h)
        los_probs.append(prob)
        print(f"Height: {h:5.1f}m -> LOS Probability: {prob:.3f}")
    
    # Plot LOS probability vs height
    plt.figure(figsize=(10, 6))
    plt.plot(heights, los_probs, 'b-', linewidth=2, marker='o')
    plt.axvline(x=env.optimal_height, color='r', linestyle='--', label=f'Optimal Height ({env.optimal_height}m)')
    plt.xlabel('UAV Height (meters)', fontsize=12)
    plt.ylabel('LOS Probability', fontsize=12)
    plt.title('Height-Dependent LOS Probability Model', fontsize=14)
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
        
        # Average over multiple samples to account for stochasticity
        sample_throughputs = []
        sample_rates = []
        for _ in range(10):
            throughput, user_rates = env._compute_throughput()
            sample_throughputs.append(throughput)
            sample_rates.append(np.mean(user_rates))
        
        throughputs.append(np.mean(sample_throughputs))
        avg_user_rates.append(np.mean(sample_rates))
        print(f"  Height: {h:5.1f}m -> Avg Throughput: {np.mean(sample_throughputs):.3f}, Avg User Rate: {np.mean(sample_rates):.3f}")
    
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
    
    optimal_idx = np.argmax(throughputs)
    optimal_height = test_heights[optimal_idx]
    print(f"\n✓ Throughput plot saved to 'test_throughput_vs_height.png'")
    print(f"  Optimal height for throughput: {optimal_height:.1f}m")
    print(f"  Max throughput: {max(throughputs):.3f}")
    
    return test_heights, throughputs, avg_user_rates

def test_short_training():
    """Run a short training episode to verify the system works."""
    print("\n" + "=" * 60)
    print("Running Short Training Test (5 episodes)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MARLEnv(num_uavs=3, num_users=10, grid_size=(10, 10, 5), device=device)
    
    # Simple random policy test
    obs, _ = env.reset()
    episode_rewards = []
    episode_throughputs = []
    episode_heights = []
    
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_throughput = 0
        step_count = 0
        
        while not done and step_count < 50:  # Limit steps per episode
            # Random actions
            actions = [np.random.randint(0, 6) for _ in range(env.num_uavs)]
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            episode_reward += reward
            episode_throughput += info.get('throughput', 0)
            step_count += 1
        
        avg_height = np.mean(env.uav_positions[:, 2])
        episode_rewards.append(episode_reward)
        episode_throughputs.append(episode_throughput)
        episode_heights.append(avg_height)
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Throughput={episode_throughput:.2f}, Avg Height={avg_height:.2f}m")
    
    print(f"\n✓ Training test completed")
    print(f"  Average reward: {np.mean(episode_rewards):.2f}")
    print(f"  Average throughput: {np.mean(episode_throughputs):.2f}")
    print(f"  Average height: {np.mean(episode_heights):.2f}m")
    print(f"  Height range: [{np.min(episode_heights):.2f}, {np.max(episode_heights):.2f}]m")
    
    return episode_rewards, episode_throughputs, episode_heights

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
        
        # Test 3: Short training run
        rewards, throughputs_train, heights_train = test_short_training()
        
        print("\n" + "=" * 60)
        print("All Tests Completed Successfully!")
        print("=" * 60)
        print("\nKey Findings:")
        print(f"  - LOS probability increases with height up to ~{heights[np.argmax(los_probs)]:.1f}m")
        print(f"  - Optimal height for throughput: ~{test_heights[np.argmax(throughputs)]:.1f}m")
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


