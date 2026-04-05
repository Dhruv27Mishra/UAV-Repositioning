"""
Compare convergence of novel algorithms (PerformativeMFMARL / PerformativeMARL: MAPPO + adaptive association)
vs other RL models (QMIX, IQL, VDN, MADDPG, DeepNashQ) with regular model.
"""
import repo_paths  # noqa: F401

import json
import logging
import os
import re
from typing import Any, Dict, Optional

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

# On-policy PPO variants (optionally with adaptive association and/or extended env)
MAPPO_FAMILY = frozenset({"MAPPO", "PerformativeMFMARL", "PerformativeMARL"})


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


def _safe_checkpoint_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "agent"


def _hp(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return dict(base)
    out = dict(base)
    out.update(overrides)
    return out


def train_model(
    agent_class,
    agent_name,
    env,
    num_episodes=500,
    num_steps_per_episode=50,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1,
    device=None,
    context_dim=0,
    checkpoint_every: Optional[int] = None,
    checkpoint_root: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    hyperparam_overrides: Optional[Dict[str, Any]] = None,
):
    """Train a single RL model and return convergence metrics.

    If ``checkpoint_every`` and ``checkpoint_root`` are set, every ``checkpoint_every``
    episodes saves ``metrics.json`` and ``agent_checkpoint.pt`` (via ``agent.save``)
    under ``checkpoint_root / {agent_name} / episode_{k}``.
    """
    
    print(f"\n{'='*70}")
    print(f"Training {agent_name}...")
    print(f"{'='*70}")
    
    num_uavs = env.num_uavs
    state_dim = getattr(env, "agent_obs_dim", env.observation_space.shape[0] // num_uavs)
    action_dim = env.action_space.nvec[0]
    
    # Initialize agent based on type
    if agent_name == "AdaptiveNonStationaryMARL":
        global_state_dim = env.observation_space.shape[0]
        kw = _hp(
            dict(
                num_agents=num_uavs,
                state_dim=state_dim,
                action_dim=action_dim,
                global_state_dim=global_state_dim,
                context_dim=context_dim,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                device=device,
                buffer_size=10000,
                batch_size=64,
                target_update=100,
            ),
            hyperparam_overrides,
        )
        agent = agent_class(**kw)
        # Set the improved association function
        env.set_association_function(agent.get_association_function())
    elif agent_name in MAPPO_FAMILY:
        kw = _hp(
            dict(
                num_agents=num_uavs,
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate_actor=0.0003,
                learning_rate_critic=0.001,
                gamma=gamma,
                clip_epsilon=0.2,
                device=device,
            ),
            hyperparam_overrides,
        )
        agent = agent_class(**kw)
        if agent_name in ("PerformativeMFMARL", "PerformativeMARL"):
            env.set_association_function(AdaptiveNonStationaryMARL.standalone_association_function())
    elif agent_name.startswith("QMIX"):
        global_state_dim = env.observation_space.shape[0]  # Global state is the full observation
        kw = _hp(
            dict(
                num_agents=num_uavs,
                state_dim=state_dim,
                action_dim=action_dim,
                global_state_dim=global_state_dim,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                device=device,
                buffer_size=10000,
                batch_size=64,
                target_update=100,
            ),
            hyperparam_overrides,
        )
        agent = agent_class(**kw)
    elif agent_name == "IQL":
        kw = _hp(
            dict(
                num_agents=num_uavs,
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                device=device,
                buffer_size=10000,
                batch_size=64,
                target_update=100,
            ),
            hyperparam_overrides,
        )
        agent = agent_class(**kw)
    elif agent_name == "VDN":
        kw = _hp(
            dict(
                num_agents=num_uavs,
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                device=device,
                buffer_size=10000,
                batch_size=64,
                target_update=100,
            ),
            hyperparam_overrides,
        )
        agent = agent_class(**kw)
    elif agent_name == "MADDPG":
        kw = _hp(
            dict(
                num_agents=num_uavs,
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate_actor=learning_rate,
                learning_rate_critic=learning_rate,
                gamma=gamma,
                device=device,
                buffer_size=10000,
                batch_size=64,
                tau=0.01,
            ),
            hyperparam_overrides,
        )
        agent = agent_class(**kw)
    elif agent_name == "DeepNashQ":
        kw = _hp(
            dict(
                num_agents=num_uavs,
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                device=device,
                buffer_size=10000,
                batch_size=64,
                target_update=100,
            ),
            hyperparam_overrides,
        )
        agent = agent_class(**kw)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    episode_rewards = []
    episode_throughputs = []
    episode_min_rates = []
    episode_goodness = []
    episode_energy_efficiencies = []
    episode_energies_j = []
    print(f"  Running {num_episodes} episodes...")
    if logger:
        logger.info("Starting training agent=%s episodes=%d", agent_name, num_episodes)

    def maybe_checkpoint(episode_idx: int) -> None:
        if not checkpoint_every or not checkpoint_root:
            return
        ep = episode_idx + 1
        if ep % checkpoint_every != 0:
            return
        sub = os.path.join(checkpoint_root, _safe_checkpoint_name(agent_name), f"episode_{ep:05d}")
        os.makedirs(sub, exist_ok=True)
        metrics_path = os.path.join(sub, "metrics.json")
        payload = {
            "agent_name": agent_name,
            "episode": ep,
            "rewards": [float(x) for x in episode_rewards],
            "throughputs": [float(x) for x in episode_throughputs],
            "goodness": [float(x) for x in episode_goodness],
            "min_rates": [float(x) for x in episode_min_rates],
            "energy_efficiencies_mbit_per_j": [float(x) for x in episode_energy_efficiencies],
            "episode_energies_j": [float(x) for x in episode_energies_j],
        }
        with open(metrics_path, "w") as f:
            json.dump(payload, f, indent=2)
        ckpt_path = os.path.join(sub, "agent_checkpoint.pt")
        if hasattr(agent, "save"):
            try:
                agent.save(ckpt_path)
            except Exception as ex:
                if logger:
                    logger.warning("Checkpoint save failed for %s: %s", agent_name, ex)
        if logger:
            logger.info("Checkpoint agent=%s episode=%d dir=%s", agent_name, ep, sub)

    for episode in tqdm(range(num_episodes), desc=f"  {agent_name}", leave=False):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_throughput = 0.0
        episode_min_rate = float('inf')
        episode_goodness_sum = 0.0
        episode_energy_j = 0.0
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
                elif agent_name in MAPPO_FAMILY:
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
            episode_energy_j += float(info.get('step_energy_j', 0.0))
            step_count += 1
            episode_min_rate = min(episode_min_rate, info.get('min_rate', episode_min_rate))
            episode_goodness_sum += info.get('goodness', 0.0)
            
            # Convert next observation to tensor
            next_obs_tensor = torch.tensor(next_obs, device=device, dtype=torch.float32)
            
            # Store transition
            states = [obs_tensor[i*state_dim:(i+1)*state_dim] for i in range(num_uavs)]
            next_states = [next_obs_tensor[i*state_dim:(i+1)*state_dim] for i in range(num_uavs)]
            
            if agent_name in MAPPO_FAMILY:
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
            elif agent_name.startswith("QMIX"):
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
            if agent_name not in MAPPO_FAMILY:
                if hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) > agent.batch_size and step_count % 2 == 0:
                    agent.update()
            
            # Update observation for next step
            obs_tensor = next_obs_tensor
        
        # For MAPPO (on-policy), store full trajectory and update
        if agent_name in MAPPO_FAMILY:
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
        if episode_min_rate == float('inf'):
            episode_min_rate = 0.0

        episode_min_rates.append(episode_min_rate)
        episode_goodness.append(episode_goodness_sum / max(1, step_count))
        # Nominal bits per episode ≈ Σ (rate [bit/s] × 1 s) per step; energy efficiency in Mbit/J.
        bits_nominal = max(0.0, episode_throughput)
        ee_mbit_per_j = (bits_nominal / 1e6) / (episode_energy_j + 1e-9)
        episode_energy_efficiencies.append(ee_mbit_per_j)
        episode_energies_j.append(episode_energy_j)

        maybe_checkpoint(episode)

    return {
        'rewards': episode_rewards,
        'throughputs': episode_throughputs,
        'min_rates': episode_min_rates,
        'goodness': episode_goodness,
        'energy_efficiencies': episode_energy_efficiencies,
        'episode_energies_j': episode_energies_j,
    }


def plot_rl_convergence_comparison(results_dict_regular, results_novel_enhanced, num_episodes):
    """Create convergence comparison plots: Novel algo (enhanced) vs Others (regular)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) # 1x2 layout: throughput and reward
    
    episodes = np.arange(num_episodes)
    window_size = min(100, num_episodes // 10)  # Larger window for better smoothing
    
    colors = {
        'PerformativeMFMARL': 'red',
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
    if 'PerformativeMFMARL' in results_novel_enhanced:
        results = results_novel_enhanced['PerformativeMFMARL']
        color = colors.get('PerformativeMFMARL', 'red')
        # Convert bps to Gbps (divide by 1e9)
        throughputs_gbps = np.array(results['throughputs']) / 1e9
        ax1.plot(episodes, throughputs_gbps, '-', alpha=0.08, linewidth=0.3, color=color, zorder=1)
        smoothed = compute_moving_average(throughputs_gbps, window_size)
        ax1.plot(episodes, smoothed, '--', linewidth=3.5, label='PerformativeMFMARL',
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
    if 'PerformativeMFMARL' in results_novel_enhanced:
        results = results_novel_enhanced['PerformativeMFMARL']
        color = colors.get('PerformativeMFMARL', 'red')
        ax2.plot(episodes, results['rewards'], '-', alpha=0.08, linewidth=0.3, color=color, zorder=1)
        smoothed = compute_moving_average(results['rewards'], window_size)
        ax2.plot(episodes, smoothed, '--', linewidth=3.5, label='PerformativeMFMARL',
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

# ===================== SWEEPS + 3-GRAPH PLOTTING =====================

def summarize_last(results, last_n=100):
    last_n = min(last_n, len(results['throughputs']))
    tp_gbps = np.mean(results['throughputs'][-last_n:]) / 1e9
    min_rate_mbps = np.mean(results['min_rates'][-last_n:]) / 1e6
    goodness = np.mean(results['goodness'][-last_n:])
    return tp_gbps, min_rate_mbps, goodness


def sweep_ue_density(agent_class, agent_name, device, grid_size,
                     num_uavs=3, user_list=(5,10,15,20,30,40),
                     num_episodes=400, num_steps_per_episode=30,
                     enable_non_stationary=True, enable_performative=True,
                     learning_rate=0.001, gamma=0.99, epsilon=0.1, context_dim=0,
                     last_n=100):
    xs, y_tp, y_minrate = [], [], []

    for U in user_list:
        env = MARLEnv(num_uavs=num_uavs, num_users=U, grid_size=grid_size, device=device,
                      enable_non_stationary=enable_non_stationary, enable_performative=enable_performative)

        results = train_model(agent_class, agent_name, env,
                              num_episodes=num_episodes, num_steps_per_episode=num_steps_per_episode,
                              learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,
                              device=device, context_dim=context_dim)

        tp_gbps, min_rate_mbps, _ = summarize_last(results, last_n=last_n)
        xs.append(U); y_tp.append(tp_gbps); y_minrate.append(min_rate_mbps)

    return np.array(xs), np.array(y_tp), np.array(y_minrate)


def sweep_threshold(agent_class, agent_name, device, grid_size,
                    num_uavs=3, num_users=20,
                    threshold_list=(0.2,0.4,0.6,0.8,1.0),  # Mbps
                    num_episodes=400, num_steps_per_episode=30,
                    enable_non_stationary=True, enable_performative=True,
                    learning_rate=0.001, gamma=0.99, epsilon=0.1, context_dim=0,
                    last_n=100):
    xs, y_good = [], []

    for thr_mbps in threshold_list:
        env = MARLEnv(num_uavs=num_uavs, num_users=num_users, grid_size=grid_size, device=device,
                      enable_non_stationary=enable_non_stationary, enable_performative=enable_performative)

        # Threshold in your env = QoS min_user_rate (Mbps)
        env.min_user_rate = float(thr_mbps)

        results = train_model(agent_class, agent_name, env,
                              num_episodes=num_episodes, num_steps_per_episode=num_steps_per_episode,
                              learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,
                              device=device, context_dim=context_dim)

        _, _, goodness = summarize_last(results, last_n=last_n)
        xs.append(thr_mbps); y_good.append(goodness)

    return np.array(xs), np.array(y_good)

def plot_3_graphs(x_users, y_tp_gbps, y_minrate_mbps, x_thr_mbps, y_goodness,
                  out_png="three_graphs.png"):
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x_users, y_tp_gbps, marker='o')
    plt.xlabel("UE Density (Number of Users)")
    plt.ylabel("System Throughput (Gbps)")
    plt.title("System Throughput vs UE Density")
    plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

    plt.subplot(1, 3, 2)
    plt.plot(x_users, y_minrate_mbps, marker='o')
    plt.xlabel("UE Density (Number of Users)")
    plt.ylabel("Min User Data Rate (Mbps)")
    plt.title("Min User Data Rate vs UE Density")
    plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

    plt.subplot(1, 3, 3)
    plt.plot(x_thr_mbps, y_goodness, marker='o')
    plt.xlabel("QoS Threshold (min_user_rate, Mbps)")
    plt.ylabel("Goodness")
    plt.title("Goodness vs Threshold")
    plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"✓ Saved {out_png}")
    plt.close()
def summarize_last(results, last_n=100):
    last_n = min(last_n, len(results['throughputs']))
    tp_gbps = np.mean(results['throughputs'][-last_n:]) / 1e9
    min_rate_mbps = np.mean(results['min_rates'][-last_n:]) / 1e6
    goodness = np.mean(results['goodness'][-last_n:])
    return tp_gbps, min_rate_mbps, goodness


def sweep_ue_density_multi(agents_dict, device, grid_size,
                           num_uavs=3, user_list=(5,10,15,20,30,40),
                           num_episodes=300, num_steps_per_episode=30,
                           env_flags=None,
                           learning_rate=0.001, gamma=0.99, epsilon=0.1, context_dim=0,
                           last_n=100):
    """
    env_flags = dict(enable_non_stationary=..., enable_performative=..., min_user_rate=...)
    """
    env_flags = env_flags or {}
    curves = {}

    for agent_name, agent_class in agents_dict.items():
        xs, y_tp, y_minrate = [], [], []

        for U in user_list:
            env = MARLEnv(num_uavs=num_uavs, num_users=U, grid_size=grid_size, device=device,
                          enable_non_stationary=env_flags.get("enable_non_stationary", False),
                          enable_performative=env_flags.get("enable_performative", False))

            # Same QoS threshold for everyone (unless you’re sweeping it separately)
            if "min_user_rate" in env_flags:
                env.min_user_rate = float(env_flags["min_user_rate"])

            results = train_model(agent_class, agent_name, env,
                                  num_episodes=num_episodes, num_steps_per_episode=num_steps_per_episode,
                                  learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,
                                  device=device, context_dim=context_dim)

            tp_gbps, min_rate_mbps, _ = summarize_last(results, last_n=last_n)
            xs.append(U); y_tp.append(tp_gbps); y_minrate.append(min_rate_mbps)

        curves[agent_name] = {"x": np.array(xs), "throughput": np.array(y_tp), "minrate": np.array(y_minrate)}

    return curves


def sweep_threshold_multi(agents_dict, device, grid_size,
                          num_uavs=3, num_users=20,
                          threshold_list=(0.2,0.4,0.6,0.8,1.0),  # Mbps
                          num_episodes=300, num_steps_per_episode=30,
                          env_flags=None,
                          learning_rate=0.001, gamma=0.99, epsilon=0.1, context_dim=0,
                          last_n=100):
    env_flags = env_flags or {}
    curves = {}

    for agent_name, agent_class in agents_dict.items():
        xs, y_good = [], []

        for thr_mbps in threshold_list:
            env = MARLEnv(num_uavs=num_uavs, num_users=num_users, grid_size=grid_size, device=device,
                          enable_non_stationary=env_flags.get("enable_non_stationary", False),
                          enable_performative=env_flags.get("enable_performative", False))

            # Sweep QoS threshold (same meaning for all agents)
            env.min_user_rate = float(thr_mbps)

            results = train_model(agent_class, agent_name, env,
                                  num_episodes=num_episodes, num_steps_per_episode=num_steps_per_episode,
                                  learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,
                                  device=device, context_dim=context_dim)

            _, _, goodness = summarize_last(results, last_n=last_n)
            xs.append(thr_mbps); y_good.append(goodness)

        curves[agent_name] = {"x": np.array(xs), "goodness": np.array(y_good)}

    return curves


def plot_3_graphs_multi(curves_density, curves_threshold, out_png="three_graphs_multi.png"):
    plt.figure(figsize=(18, 5))

    # 1) Throughput vs UE density
    plt.subplot(1, 3, 1)
    for agent_name, data in curves_density.items():
        plt.plot(data["x"], data["throughput"], marker='o', label=agent_name)
    plt.xlabel("UE Density (Number of Users)")
    plt.ylabel("System Throughput (Gbps)")
    plt.title("System Throughput vs UE Density")
    plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    plt.legend(fontsize=9)

    # 2) Min rate vs UE density
    plt.subplot(1, 3, 2)
    for agent_name, data in curves_density.items():
        plt.plot(data["x"], data["minrate"], marker='o', label=agent_name)
    plt.xlabel("UE Density (Number of Users)")
    plt.ylabel("Min User Data Rate (Mbps)")
    plt.title("Min User Data Rate vs UE Density")
    plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    plt.legend(fontsize=9)

    # 3) Goodness vs threshold
    plt.subplot(1, 3, 3)
    for agent_name, data in curves_threshold.items():
        plt.plot(data["x"], data["goodness"], marker='o', label=agent_name)
    plt.xlabel("QoS Threshold (min_user_rate, Mbps)")
    plt.ylabel("Goodness")
    plt.title("Goodness vs Threshold")
    plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    plt.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"✓ Saved {out_png}")
    plt.close()
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
    context_dim = 0  # NS context embedded in each agent_obs block
    
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
    print("Training Novel Algorithm: PerformativeMFMARL (MAPPO + adaptive association, enhanced env)")
    print(f"{'='*70}")
    env = MARLEnv(num_uavs=num_uavs, num_users=num_users, grid_size=grid_size, device=device,
                 enable_non_stationary=True, enable_performative=True)
    results = train_model(MAPPO, "PerformativeMFMARL", env,
                         num_episodes=num_episodes, num_steps_per_episode=num_steps_per_episode,
                         learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,
                         device=device, context_dim=context_dim)
    results_novel_enhanced['PerformativeMFMARL'] = results
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("Generating convergence comparison plots...")
    print(f"{'='*70}")
    plot_rl_convergence_comparison(results_dict_regular, results_novel_enhanced, num_episodes)
    # ===================== 3-GRAPH SWEEP EXPERIMENT =====================

    agent_class = MAPPO
    agent_name = "PerformativeMFMARL"

    user_list = [5, 10, 15, 20, 30, 40]
    thr_list_mbps = [0.2, 0.4, 0.6, 0.8, 1.0]

    xU, yTP, yMIN = sweep_ue_density(
        agent_class, agent_name, device=device, grid_size=grid_size,
        num_uavs=num_uavs, user_list=user_list,
        num_episodes=400, num_steps_per_episode=num_steps_per_episode,
        enable_non_stationary=True, enable_performative=True,
        learning_rate=learning_rate, gamma=gamma, epsilon=epsilon, context_dim=context_dim,
        last_n=100
    )

    xT, yG = sweep_threshold(
        agent_class, agent_name, device=device, grid_size=grid_size,
        num_uavs=num_uavs, num_users=20,
        threshold_list=thr_list_mbps,
        num_episodes=400, num_steps_per_episode=num_steps_per_episode,
        enable_non_stationary=True, enable_performative=True,
        learning_rate=learning_rate, gamma=gamma, epsilon=epsilon, context_dim=context_dim,
        last_n=100
    )

    plot_3_graphs(xU, yTP, yMIN, xT, yG, out_png="three_graphs.png")
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
    if 'PerformativeMFMARL' in results_novel_enhanced:
        final_tp = np.mean(results_novel_enhanced['PerformativeMFMARL']['throughputs'][-last_n:])
        final_rwd = np.mean(results_novel_enhanced['PerformativeMFMARL']['rewards'][-last_n:])
        print(f"  {'PerformativeMFMARL':25s}: Throughput = {final_tp:.2e}, Reward = {final_rwd:.2f}")
    
    print(f"\n{'='*70}")
    print("✅ RL models convergence comparison complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

