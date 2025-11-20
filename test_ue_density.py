"""
Test script to compare old and new versions with varying UE (User Equipment) density.
Tests how throughput scales with different numbers of users.
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

def test_ue_density(num_episodes=50, num_uavs=3, ue_densities=[5, 10, 15, 20, 25, 30, 40, 50]):
    """
    Test both versions with varying UE density.
    
    Args:
        num_episodes: Number of episodes to run per density
        num_uavs: Number of UAVs
        ue_densities: List of UE counts to test
    """
    print("=" * 70)
    print("UE DENSITY TEST: Comparing Old vs New Version")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {
        'ue_densities': ue_densities,
        'old_throughputs': [],
        'old_std': [],
        'old_per_user_throughput': [],
        'new_throughputs': [],
        'new_std': [],
        'new_per_user_throughput': [],
        'old_heights': [],
        'new_heights': [],
        'old_los_probs': [],
        'new_los_probs': []
    }
    
    for num_users in ue_densities:
        print(f"\n{'='*70}")
        print(f"Testing with {num_users} UEs")
        print(f"{'='*70}")
        
        # Test old version
        print(f"\n[Old Version] Testing {num_users} UEs...")
        old_env = OldMARLEnv(num_uavs=num_uavs, num_users=num_users, grid_size=(10, 10, 5), device=device)
        old_metrics = run_episodes(old_env, num_episodes=num_episodes, num_users=num_users)
        
        # Test new version
        print(f"\n[New Version] Testing {num_users} UEs...")
        new_env = MARLEnv(num_uavs=num_uavs, num_users=num_users, grid_size=(10, 10, 5), device=device)
        new_metrics = run_episodes(new_env, num_episodes=num_episodes, num_users=num_users)
        
        # Store results
        results['old_throughputs'].append(np.mean(old_metrics['throughputs']))
        results['old_std'].append(np.std(old_metrics['throughputs']))
        results['old_per_user_throughput'].append(np.mean(old_metrics['throughputs']) / num_users)
        results['new_throughputs'].append(np.mean(new_metrics['throughputs']))
        results['new_std'].append(np.std(new_metrics['throughputs']))
        results['new_per_user_throughput'].append(np.mean(new_metrics['throughputs']) / num_users)
        results['old_heights'].append(np.mean(old_metrics['heights']))
        results['new_heights'].append(np.mean(new_metrics['heights']))
        results['old_los_probs'].append(np.mean(old_metrics['los_probs']) if old_metrics['los_probs'] else 0.8)
        results['new_los_probs'].append(np.mean(new_metrics['los_probs']))
        
        # Print summary for this density
        print(f"\n  Results for {num_users} UEs:")
        print(f"    Old - Total Throughput: {np.mean(old_metrics['throughputs']):.3f} ± {np.std(old_metrics['throughputs']):.3f}")
        print(f"    Old - Per-User Throughput: {np.mean(old_metrics['throughputs']) / num_users:.3f}")
        print(f"    New - Total Throughput: {np.mean(new_metrics['throughputs']):.3f} ± {np.std(new_metrics['throughputs']):.3f}")
        print(f"    New - Per-User Throughput: {np.mean(new_metrics['throughputs']) / num_users:.3f}")
        improvement = ((np.mean(new_metrics['throughputs']) / np.mean(old_metrics['throughputs'])) - 1) * 100
        print(f"    Improvement: {improvement:+.2f}%")
    
    return results

def run_episodes(env, num_episodes=50, num_users=None):
    """Run episodes and collect metrics."""
    metrics = {
        'throughputs': [],
        'rewards': [],
        'heights': [],
        'los_probs': [],
        'user_rates': [],
        'collisions': []
    }
    
    for episode in tqdm(range(num_episodes), desc=f"  Episodes", leave=False):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_throughput = 0
        episode_heights = []
        episode_los_probs = []
        step_count = 0
        
        while not done and step_count < 50:
            # Random policy for fair comparison
            actions = [np.random.randint(0, 6) for _ in range(env.num_uavs)]
            
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

def plot_results(results):
    """Create comprehensive plots for UE density comparison."""
    fig = plt.figure(figsize=(16, 12))
    
    ue_densities = results['ue_densities']
    
    # Plot 1: Total Throughput vs UE Density
    ax1 = plt.subplot(2, 3, 1)
    ax1.errorbar(ue_densities, results['old_throughputs'], yerr=results['old_std'], 
                 fmt='ro-', label='Old (Fixed Height)', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.errorbar(ue_densities, results['new_throughputs'], yerr=results['new_std'], 
                 fmt='bs-', label='New (Variable Height)', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.set_xlabel('Number of UEs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Throughput', fontsize=12, fontweight='bold')
    ax1.set_title('Total Throughput vs UE Density', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([min(ue_densities)-2, max(ue_densities)+2])
    
    # Plot 2: Per-User Throughput vs UE Density
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(ue_densities, results['old_per_user_throughput'], 'ro-', 
             label='Old (Fixed Height)', linewidth=2, markersize=8)
    ax2.plot(ue_densities, results['new_per_user_throughput'], 'bs-', 
             label='New (Variable Height)', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of UEs', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Per-User Throughput', fontsize=12, fontweight='bold')
    ax2.set_title('Per-User Throughput vs UE Density', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min(ue_densities)-2, max(ue_densities)+2])
    
    # Plot 3: Improvement Percentage
    ax3 = plt.subplot(2, 3, 3)
    improvements = [(new/old - 1) * 100 for new, old in 
                    zip(results['new_throughputs'], results['old_throughputs'])]
    ax3.bar(ue_densities, improvements, color='green', alpha=0.7, width=2)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Number of UEs', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Throughput Improvement vs UE Density', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim([min(ue_densities)-2, max(ue_densities)+2])
    for i, (ue, imp) in enumerate(zip(ue_densities, improvements)):
        ax3.text(ue, imp + (1 if imp >= 0 else -3), f'{imp:+.1f}%', 
                ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9, fontweight='bold')
    
    # Plot 4: Average Height vs UE Density
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(ue_densities, results['old_heights'], 'ro-', 
             label='Old (Fixed ~12.5m)', linewidth=2, markersize=8)
    ax4.plot(ue_densities, results['new_heights'], 'bs-', 
             label='New (Variable)', linewidth=2, markersize=8)
    ax4.axhline(12.5, color='r', linestyle='--', alpha=0.5, label='Old Fixed Height')
    ax4.set_xlabel('Number of UEs', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average UAV Height (m)', fontsize=12, fontweight='bold')
    ax4.set_title('Average Height vs UE Density', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([min(ue_densities)-2, max(ue_densities)+2])
    
    # Plot 5: LOS Probability vs UE Density
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(ue_densities, results['old_los_probs'], 'ro-', 
             label='Old (Fixed 0.8)', linewidth=2, markersize=8)
    ax5.plot(ue_densities, results['new_los_probs'], 'bs-', 
             label='New (Variable)', linewidth=2, markersize=8)
    ax5.axhline(0.8, color='r', linestyle='--', alpha=0.5, label='Old Fixed LOS')
    ax5.set_xlabel('Number of UEs', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Average LOS Probability', fontsize=12, fontweight='bold')
    ax5.set_title('LOS Probability vs UE Density', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([min(ue_densities)-2, max(ue_densities)+2])
    ax5.set_ylim([0.7, 1.0])
    
    # Plot 6: Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Create summary table
    table_data = []
    for i, ue in enumerate(ue_densities):
        old_tp = results['old_throughputs'][i]
        new_tp = results['new_throughputs'][i]
        imp = improvements[i]
        old_per = results['old_per_user_throughput'][i]
        new_per = results['new_per_user_throughput'][i]
        table_data.append([
            f'{ue}',
            f'{old_tp:.2f}',
            f'{new_tp:.2f}',
            f'{imp:+.1f}%',
            f'{old_per:.3f}',
            f'{new_per:.3f}'
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['UEs', 'Old TP', 'New TP', 'Imp %', 'Old/UE', 'New/UE'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style improvement column
    for i in range(1, len(table_data) + 1):
        imp_val = float(table_data[i-1][3].replace('%', '').replace('+', ''))
        if imp_val > 0:
            table[(i, 3)].set_facecolor('#C8E6C9')
        else:
            table[(i, 3)].set_facecolor('#FFCDD2')
    
    plt.suptitle('UE Density Performance Comparison: Old vs New Version', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('ue_density_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ UE density comparison plot saved to 'ue_density_comparison.png'")
    
    # Create detailed analysis plot
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Throughput scaling
    ax1.plot(ue_densities, results['old_throughputs'], 'ro-', label='Old (Fixed Height)', 
             linewidth=2.5, markersize=10, alpha=0.8)
    ax1.plot(ue_densities, results['new_throughputs'], 'bs-', label='New (Variable Height)', 
             linewidth=2.5, markersize=10, alpha=0.8)
    ax1.fill_between(ue_densities, 
                     [t - s for t, s in zip(results['old_throughputs'], results['old_std'])],
                     [t + s for t, s in zip(results['old_throughputs'], results['old_std'])],
                     alpha=0.2, color='red')
    ax1.fill_between(ue_densities, 
                     [t - s for t, s in zip(results['new_throughputs'], results['new_std'])],
                     [t + s for t, s in zip(results['new_throughputs'], results['new_std'])],
                     alpha=0.2, color='blue')
    ax1.set_xlabel('Number of UEs', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Total Throughput', fontsize=13, fontweight='bold')
    ax1.set_title('Throughput Scaling with UE Density', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Efficiency (per-user throughput)
    ax2.plot(ue_densities, results['old_per_user_throughput'], 'ro-', 
             label='Old (Fixed Height)', linewidth=2.5, markersize=10, alpha=0.8)
    ax2.plot(ue_densities, results['new_per_user_throughput'], 'bs-', 
             label='New (Variable Height)', linewidth=2.5, markersize=10, alpha=0.8)
    ax2.set_xlabel('Number of UEs', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Per-User Throughput', fontsize=13, fontweight='bold')
    ax2.set_title('Network Efficiency vs UE Density', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Improvement trend
    ax3.plot(ue_densities, improvements, 'go-', linewidth=2.5, markersize=10, alpha=0.8)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.fill_between(ue_densities, improvements, 0, where=[i >= 0 for i in improvements], 
                     alpha=0.3, color='green', label='Improvement')
    ax3.fill_between(ue_densities, improvements, 0, where=[i < 0 for i in improvements], 
                     alpha=0.3, color='red', label='Degradation')
    ax3.set_xlabel('Number of UEs', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
    ax3.set_title('Performance Improvement Trend', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Height and LOS comparison
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(ue_densities, results['old_heights'], 'r^-', 
                     label='Old Height (Fixed)', linewidth=2, markersize=8, alpha=0.7)
    line2 = ax4.plot(ue_densities, results['new_heights'], 'b^-', 
                     label='New Height (Variable)', linewidth=2, markersize=8, alpha=0.7)
    line3 = ax4_twin.plot(ue_densities, results['old_los_probs'], 'rs--', 
                          label='Old LOS (Fixed)', linewidth=2, markersize=8, alpha=0.7)
    line4 = ax4_twin.plot(ue_densities, results['new_los_probs'], 'bs--', 
                          label='New LOS (Variable)', linewidth=2, markersize=8, alpha=0.7)
    ax4.set_xlabel('Number of UEs', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Average Height (m)', fontsize=13, fontweight='bold', color='black')
    ax4_twin.set_ylabel('Average LOS Probability', fontsize=13, fontweight='bold', color='gray')
    ax4.set_title('Height and LOS vs UE Density', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ue_density_detailed_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Detailed analysis plot saved to 'ue_density_detailed_analysis.png'")

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    improvements = [(new/old - 1) * 100 for new, old in 
                    zip(results['new_throughputs'], results['old_throughputs'])]
    
    print(f"\n📊 Average Performance Across All UE Densities:")
    print(f"  Old Version - Mean Total Throughput: {np.mean(results['old_throughputs']):.3f}")
    print(f"  New Version - Mean Total Throughput: {np.mean(results['new_throughputs']):.3f}")
    print(f"  Average Improvement: {np.mean(improvements):+.2f}%")
    
    print(f"\n📈 Per-User Efficiency:")
    print(f"  Old Version - Mean Per-User Throughput: {np.mean(results['old_per_user_throughput']):.3f}")
    print(f"  New Version - Mean Per-User Throughput: {np.mean(results['new_per_user_throughput']):.3f}")
    
    print(f"\n📡 Height and LOS:")
    print(f"  Old Version - Mean Height: {np.mean(results['old_heights']):.2f}m (Fixed)")
    print(f"  New Version - Mean Height: {np.mean(results['new_heights']):.2f}m (Variable)")
    print(f"  Old Version - Mean LOS: {np.mean(results['old_los_probs']):.3f} (Fixed)")
    print(f"  New Version - Mean LOS: {np.mean(results['new_los_probs']):.3f} (Variable)")
    
    print(f"\n🎯 Best Performance:")
    best_old_idx = np.argmax(results['old_throughputs'])
    best_new_idx = np.argmax(results['new_throughputs'])
    print(f"  Old - Best at {results['ue_densities'][best_old_idx]} UEs: {results['old_throughputs'][best_old_idx]:.3f}")
    print(f"  New - Best at {results['ue_densities'][best_new_idx]} UEs: {results['new_throughputs'][best_new_idx]:.3f}")
    
    print(f"\n📉 Scaling Analysis:")
    print(f"  Old - Throughput range: [{min(results['old_throughputs']):.3f}, {max(results['old_throughputs']):.3f}]")
    print(f"  New - Throughput range: [{min(results['new_throughputs']):.3f}, {max(results['new_throughputs']):.3f}]")
    old_scaling = (results['old_throughputs'][-1] - results['old_throughputs'][0]) / results['old_throughputs'][0] * 100
    new_scaling = (results['new_throughputs'][-1] - results['new_throughputs'][0]) / results['new_throughputs'][0] * 100
    print(f"  Old - Scaling from {results['ue_densities'][0]} to {results['ue_densities'][-1]} UEs: {old_scaling:+.2f}%")
    print(f"  New - Scaling from {results['ue_densities'][0]} to {results['ue_densities'][-1]} UEs: {new_scaling:+.2f}%")

if __name__ == "__main__":
    # Test with varying UE densities
    ue_densities = [5, 10, 15, 20, 25, 30, 40, 50]
    results = test_ue_density(num_episodes=50, num_uavs=3, ue_densities=ue_densities)
    
    # Create plots
    plot_results(results)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 70)
    print("UE Density Test Complete!")
    print("=" * 70)

