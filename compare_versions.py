"""
Comparison script between old version (fixed height) and new version (variable height).
"""
import numpy as np
import matplotlib.pyplot as plt
from rl_agent.marl_env import MARLEnv
import torch
from tqdm import tqdm
import time

class OldMARLEnv(MARLEnv):
    """Old version with fixed height (10-15m) and simple path loss model."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override with old parameters
        self.path_loss_exp = 2.5  # Old single path loss exponent
        # Remove height-dependent parameters
        if hasattr(self, 'path_loss_exp_los'):
            delattr(self, 'path_loss_exp_los')
        if hasattr(self, 'path_loss_exp_nlos'):
            delattr(self, 'path_loss_exp_nlos')
        if hasattr(self, 'los_prob_base'):
            delattr(self, 'los_prob_base')
        if hasattr(self, 'los_prob_max'):
            delattr(self, 'los_prob_max')
        if hasattr(self, 'optimal_height'):
            delattr(self, 'optimal_height')
        if hasattr(self, 'height_sensitivity'):
            delattr(self, 'height_sensitivity')
        if hasattr(self, 'height_decay_factor'):
            delattr(self, 'height_decay_factor')
        if hasattr(self, 'shadowing_std_los'):
            delattr(self, 'shadowing_std_los')
        if hasattr(self, 'shadowing_std_nlos'):
            delattr(self, 'shadowing_std_nlos')
        
        # Reset initial heights to old range [10, 15]
        self.init_uav_positions[:, 2] = 10 + np.random.rand(self.num_uavs) * 5
    
    def _check_altitude_violations(self) -> bool:
        """Old version: fixed altitude range [10, 15] meters."""
        z = self.uav_positions[:, 2]
        return np.any(z < 10) or np.any(z > 15)
    
    def _compute_los_probability(self, uav_height: float) -> float:
        """Old version: constant LOS probability (not height-dependent)."""
        return 0.8  # Fixed LOS probability
    
    def _compute_path_loss(self, distance: float, is_los: bool, uav_height: float) -> float:
        """Old version: simple path loss without LOS/NLOS distinction."""
        return distance ** self.path_loss_exp
    
    def _compute_throughput(self) -> (float, np.ndarray):
        """Old version: simple throughput calculation without height-dependent LOS."""
        total_rate = 0.0
        user_rates = np.zeros(self.num_users)
        fading = np.random.exponential(1.0, size=(self.num_uavs, self.num_users))
        new_association = np.zeros(self.num_users, dtype=int)
        
        # Find nearest UAV for each user
        for user_idx in range(self.num_users):
            user_pos = self.user_positions[user_idx]
            distances = np.linalg.norm(self.uav_positions - user_pos, axis=1)
            new_association[user_idx] = np.argmin(distances)

        # Track handovers
        self.prev_association = self.association.copy()
        self.association = new_association

        # Update handover log
        handovers = (self.association != self.prev_association).astype(int)
        self.handover_log.append(handovers)
        if len(self.handover_log) > self.window_size:
            self.handover_log.pop(0)

        # Compute SINR-based rate (old simple model)
        for user_idx in range(self.num_users):
            uav_idx = self.association[user_idx]
            user_pos = self.user_positions[user_idx]
            uav_pos = self.uav_positions[uav_idx]
            d_ij = np.linalg.norm(user_pos - uav_pos) + 1e-3
            signal = self.tx_power * fading[uav_idx, user_idx] / (d_ij ** self.path_loss_exp)

            interference = 0.0
            for other_uav in range(self.num_uavs):
                if other_uav != uav_idx:
                    d_kj = np.linalg.norm(self.uav_positions[other_uav] - user_pos) + 1e-3
                    interference += self.tx_power * fading[other_uav, user_idx] / (d_kj ** self.path_loss_exp)

            sinr = signal / (self.noise_power + interference)
            rate = self.B * np.log2(1 + sinr)
            user_rates[user_idx] = rate
            total_rate += rate

        return total_rate, user_rates
    
    def reset(self, seed=None):
        """Reset with old height constraints."""
        super().reset(seed=seed)
        # Clip heights to old range [10, 15]
        self.uav_positions[:, 2] = np.clip(self.uav_positions[:, 2], 10, 15)
        self.target_positions[:, 2] = 10 + np.random.rand(self.num_uavs) * 5
        return self._get_observation(), {}
    
    def step(self, actions):
        """Step with old height constraints."""
        obs, reward, done, truncated, info = super().step(actions)
        # Clip heights to old range [10, 15]
        self.uav_positions[:, 2] = np.clip(self.uav_positions[:, 2], 10, 15)
        return obs, reward, done, truncated, info

def run_episodes(env, num_episodes=100, policy='random'):
    """Run episodes and collect metrics."""
    metrics = {
        'throughputs': [],
        'rewards': [],
        'heights': [],
        'los_probs': [],
        'user_rates': [],
        'collisions': []
    }
    
    for episode in tqdm(range(num_episodes), desc=f"Running {policy} policy"):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_throughput = 0
        episode_heights = []
        episode_los_probs = []
        step_count = 0
        
        while not done and step_count < 50:
            if policy == 'random':
                actions = [np.random.randint(0, 6) for _ in range(env.num_uavs)]
            else:
                # Simple policy: move towards center and maintain height
                actions = []
                for i in range(env.num_uavs):
                    center_x, center_y = env.grid_size[0]/2, env.grid_size[1]/2
                    uav_pos = env.uav_positions[i]
                    
                    if uav_pos[0] < center_x:
                        actions.append(4)  # forward
                    elif uav_pos[0] > center_x:
                        actions.append(5)  # backward
                    elif uav_pos[1] < center_y:
                        actions.append(2)  # right
                    elif uav_pos[1] > center_y:
                        actions.append(3)  # left
                    else:
                        actions.append(np.random.randint(0, 6))
            
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            episode_reward += reward
            episode_throughput += info.get('throughput', 0)
            episode_heights.append(env.uav_positions[:, 2].copy())
            
            # Calculate average LOS probability for current heights
            if hasattr(env, '_compute_los_probability'):
                avg_los = np.mean([env._compute_los_probability(h) for h in env.uav_positions[:, 2]])
                episode_los_probs.append(avg_los)
            
            step_count += 1
        
        metrics['throughputs'].append(episode_throughput)
        metrics['rewards'].append(episode_reward)
        metrics['heights'].append(np.mean(episode_heights) if episode_heights else 0)
        metrics['los_probs'].append(np.mean(episode_los_probs) if episode_los_probs else 0)
        metrics['collisions'].append(info.get('collisions', False))
    
    return metrics

def compare_versions(num_episodes=200):
    """Compare old and new versions."""
    print("=" * 70)
    print("COMPARISON: Old Version (Fixed Height) vs New Version (Variable Height)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run old version
    print("\n[1/2] Running OLD VERSION (Fixed Height 10-15m, Simple Path Loss)...")
    old_env = OldMARLEnv(num_uavs=3, num_users=10, grid_size=(10, 10, 5), device=device)
    old_metrics = run_episodes(old_env, num_episodes=num_episodes, policy='random')
    
    # Run new version
    print("\n[2/2] Running NEW VERSION (Variable Height 5-50m, Height-Dependent LOS)...")
    new_env = MARLEnv(num_uavs=3, num_users=10, grid_size=(10, 10, 5), device=device)
    new_metrics = run_episodes(new_env, num_episodes=num_episodes, policy='random')
    
    # Print statistics
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print("\n📊 THROUGHPUT STATISTICS:")
    print(f"  Old Version:")
    print(f"    Mean: {np.mean(old_metrics['throughputs']):.3f}")
    print(f"    Std:  {np.std(old_metrics['throughputs']):.3f}")
    print(f"    Min:  {np.min(old_metrics['throughputs']):.3f}")
    print(f"    Max:  {np.max(old_metrics['throughputs']):.3f}")
    print(f"  New Version:")
    print(f"    Mean: {np.mean(new_metrics['throughputs']):.3f}")
    print(f"    Std:  {np.std(new_metrics['throughputs']):.3f}")
    print(f"    Min:  {np.min(new_metrics['throughputs']):.3f}")
    print(f"    Max:  {np.max(new_metrics['throughputs']):.3f}")
    print(f"  Improvement: {((np.mean(new_metrics['throughputs']) / np.mean(old_metrics['throughputs'])) - 1) * 100:.2f}%")
    
    print("\n📈 HEIGHT STATISTICS:")
    print(f"  Old Version:")
    print(f"    Mean Height: {np.mean(old_metrics['heights']):.2f}m")
    print(f"    Range: [10.0, 15.0]m (fixed)")
    print(f"  New Version:")
    print(f"    Mean Height: {np.mean(new_metrics['heights']):.2f}m")
    print(f"    Min Height: {np.min([h for h in new_metrics['heights'] if h > 0]):.2f}m")
    print(f"    Max Height: {np.max(new_metrics['heights']):.2f}m")
    
    print("\n📡 LOS PROBABILITY STATISTICS:")
    print(f"  Old Version:")
    print(f"    Mean LOS Prob: {np.mean(old_metrics['los_probs']) if old_metrics['los_probs'] else 0.8:.3f} (fixed)")
    print(f"  New Version:")
    print(f"    Mean LOS Prob: {np.mean(new_metrics['los_probs']):.3f}")
    print(f"    Min LOS Prob:  {np.min(new_metrics['los_probs']):.3f}")
    print(f"    Max LOS Prob:  {np.max(new_metrics['los_probs']):.3f}")
    
    print("\n💥 COLLISIONS:")
    print(f"  Old Version: {np.sum(old_metrics['collisions'])} collisions")
    print(f"  New Version: {np.sum(new_metrics['collisions'])} collisions")
    
    # Create comparison plots
    create_comparison_plots(old_metrics, new_metrics)
    
    return old_metrics, new_metrics

def create_comparison_plots(old_metrics, new_metrics):
    """Create comparison visualization plots."""
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Throughput comparison
    ax1 = plt.subplot(2, 3, 1)
    episodes = range(len(old_metrics['throughputs']))
    ax1.plot(episodes, old_metrics['throughputs'], 'r-', alpha=0.6, label='Old (Fixed Height)', linewidth=1)
    ax1.plot(episodes, new_metrics['throughputs'], 'b-', alpha=0.6, label='New (Variable Height)', linewidth=1)
    ax1.axhline(np.mean(old_metrics['throughputs']), color='r', linestyle='--', alpha=0.5, label=f'Old Mean: {np.mean(old_metrics["throughputs"]):.2f}')
    ax1.axhline(np.mean(new_metrics['throughputs']), color='b', linestyle='--', alpha=0.5, label=f'New Mean: {np.mean(new_metrics["throughputs"]):.2f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Throughput')
    ax1.set_title('Throughput Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Throughput distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(old_metrics['throughputs'], bins=30, alpha=0.6, color='r', label='Old (Fixed Height)', density=True)
    ax2.hist(new_metrics['throughputs'], bins=30, alpha=0.6, color='b', label='New (Variable Height)', density=True)
    ax2.axvline(np.mean(old_metrics['throughputs']), color='r', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(new_metrics['throughputs']), color='b', linestyle='--', linewidth=2)
    ax2.set_xlabel('Throughput')
    ax2.set_ylabel('Density')
    ax2.set_title('Throughput Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Height comparison
    ax3 = plt.subplot(2, 3, 3)
    old_heights_plot = [h for h in old_metrics['heights'] if h > 0]
    new_heights_plot = [h for h in new_metrics['heights'] if h > 0]
    ax3.plot(range(len(old_heights_plot)), old_heights_plot, 'r-', alpha=0.6, label='Old (Fixed 10-15m)', linewidth=1)
    ax3.plot(range(len(new_heights_plot)), new_heights_plot, 'b-', alpha=0.6, label='New (Variable 5-50m)', linewidth=1)
    ax3.axhline(12.5, color='r', linestyle='--', alpha=0.5, label='Old Mean: 12.5m')
    ax3.axhline(np.mean(new_heights_plot), color='b', linestyle='--', alpha=0.5, label=f'New Mean: {np.mean(new_heights_plot):.2f}m')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average UAV Height (m)')
    ax3.set_title('Height Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: LOS probability comparison
    ax4 = plt.subplot(2, 3, 4)
    if new_metrics['los_probs']:
        ax4.plot(range(len(new_metrics['los_probs'])), new_metrics['los_probs'], 'b-', alpha=0.6, label='New (Height-Dependent)', linewidth=1)
        ax4.axhline(0.8, color='r', linestyle='--', alpha=0.5, label='Old (Fixed: 0.8)')
        ax4.axhline(np.mean(new_metrics['los_probs']), color='b', linestyle='--', alpha=0.5, label=f'New Mean: {np.mean(new_metrics["los_probs"]):.3f}')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Average LOS Probability')
        ax4.set_title('LOS Probability Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Reward comparison
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(episodes, old_metrics['rewards'], 'r-', alpha=0.6, label='Old (Fixed Height)', linewidth=1)
    ax5.plot(episodes, new_metrics['rewards'], 'b-', alpha=0.6, label='New (Variable Height)', linewidth=1)
    ax5.axhline(np.mean(old_metrics['rewards']), color='r', linestyle='--', alpha=0.5, label=f'Old Mean: {np.mean(old_metrics["rewards"]):.2f}')
    ax5.axhline(np.mean(new_metrics['rewards']), color='b', linestyle='--', alpha=0.5, label=f'New Mean: {np.mean(new_metrics["rewards"]):.2f}')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Reward')
    ax5.set_title('Reward Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
    COMPARISON SUMMARY
    
    Throughput:
      Old:  {np.mean(old_metrics['throughputs']):.3f} ± {np.std(old_metrics['throughputs']):.3f}
      New:  {np.mean(new_metrics['throughputs']):.3f} ± {np.std(new_metrics['throughputs']):.3f}
      Improvement: {((np.mean(new_metrics['throughputs']) / np.mean(old_metrics['throughputs'])) - 1) * 100:.2f}%
    
    Height:
      Old:  Fixed [10, 15]m (mean: 12.5m)
      New:  Variable [5, 50]m (mean: {np.mean([h for h in new_metrics['heights'] if h > 0]):.2f}m)
    
    LOS Probability:
      Old:  Fixed 0.8
      New:  Variable (mean: {np.mean(new_metrics['los_probs']):.3f})
    
    Key Features:
      • New version allows height optimization
      • Height-dependent LOS probability model
      • LOS/NLOS path loss differentiation
      • Expanded height range for learning
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace', 
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('version_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plots saved to 'version_comparison.png'")
    
    # Create height vs throughput analysis
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Height vs Throughput scatter (new version only)
    new_heights_valid = [h for h in new_metrics['heights'] if h > 0]
    new_throughputs_valid = new_metrics['throughputs'][:len(new_heights_valid)]
    ax1.scatter(new_heights_valid, new_throughputs_valid, alpha=0.5, s=20, c='b')
    ax1.axvline(20.0, color='r', linestyle='--', label='Optimal Height (20m)')
    ax1.set_xlabel('Average UAV Height (m)')
    ax1.set_ylabel('Throughput')
    ax1.set_title('Height vs Throughput (New Version)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # LOS Probability vs Throughput
    if new_metrics['los_probs']:
        ax2.scatter(new_metrics['los_probs'], new_metrics['throughputs'], alpha=0.5, s=20, c='g')
        ax2.set_xlabel('Average LOS Probability')
        ax2.set_ylabel('Throughput')
        ax2.set_title('LOS Probability vs Throughput (New Version)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('height_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Height analysis plots saved to 'height_analysis.png'")

if __name__ == "__main__":
    old_metrics, new_metrics = compare_versions(num_episodes=200)
    print("\n" + "=" * 70)
    print("Comparison complete! Check 'version_comparison.png' and 'height_analysis.png'")
    print("=" * 70)

