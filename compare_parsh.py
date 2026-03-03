"""
Run fair multi-agent comparisons and generate 3 graphs:

1) System Throughput vs UE Density
2) Min User Data Rate vs UE Density
3) Goodness vs Threshold (QoS min_user_rate)

IMPORTANT:
- Your MARLEnv.step() info dict MUST contain:
    info['throughput']  (bps)   [already exists]
    info['min_rate']    (bps)   [you must add]
    info['goodness']    (scalar)[you must add]
- If you didn't add min_rate/goodness in MARLEnv, this script will KeyError.

FAIR SETUP:
All agents are trained/evaluated with the SAME env flags:
(enable_non_stationary, enable_performative, grid_size, etc.)

This script trains per sweep point (rigorous but slower).
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from rl_agent.marl_env import MARLEnv
from rl_agent.QMIX import QMIX
from rl_agent.IQL import IQL
from rl_agent.VDN import VDN
from rl_agent.MADDPG import MADDPG
from rl_agent.DeepNashQ import DeepNashQ
from rl_agent.MAPPO import MAPPO
from rl_agent.AdaptiveNonStationaryMARL import AdaptiveNonStationaryMARL


# ------------------------- Utilities -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def summarize_last(results, last_n=250):
    last_n = min(last_n, len(results["throughputs"]))

    # bps -> Mbps
    tp_mbps = float(np.mean(results["throughputs"][-last_n:]) / 1e6)

    # bps -> Mbps
    min_rate_mbps = float(np.mean(results["min_rates"][-last_n:]) / 1e6)

    goodness = float(np.mean(results["goodness"][-last_n:]))  # keep raw for now
    return tp_mbps, min_rate_mbps, goodness
# ------------------------- Training -------------------------

def train_model(
    agent_class,
    agent_name,
    env,
    num_episodes=1000,
    num_steps_per_episode=30,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1,
    device=None,
    context_dim=7
):
    """
    Train one agent in the given env.
    Logs per-episode:
      - rewards
      - throughputs (sum of per-step throughput in bps)
      - min_rates (min of per-step min_rate in bps)
      - goodness (avg per-step goodness)
    """

    num_uavs = env.num_uavs

    # Determine state/action dims
    # In your original code you used:
    # state_dim = env.observation_space.shape[0] // num_uavs if env.enable_non_stationary else 3
    # Keep same logic to match your agents.
    state_dim = env.observation_space.shape[0] // num_uavs if env.enable_non_stationary else 3
    action_dim = env.action_space.nvec[0]

    # Initialize agent
    if agent_name == "AdaptiveNonStationaryMARL":
        global_state_dim = env.observation_space.shape[0]
        agent = agent_class(
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
            target_update=100
        )
        env.set_association_function(agent.get_association_function())

    elif agent_name == "QMIX":
        global_state_dim = env.observation_space.shape[0]
        agent = agent_class(
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
            target_update=100
        )

    elif agent_name == "IQL":
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            device=device,
            buffer_size=10000,
            batch_size=64,
            target_update=100
        )

    elif agent_name == "VDN":
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            device=device,
            buffer_size=10000,
            batch_size=64,
            target_update=100
        )

    elif agent_name == "MADDPG":
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate_actor=learning_rate,
            learning_rate_critic=learning_rate,
            gamma=gamma,
            device=device,
            buffer_size=10000,
            batch_size=64,
            tau=0.01
        )

    elif agent_name == "DeepNashQ":
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            device=device,
            buffer_size=10000,
            batch_size=64,
            target_update=100
        )

    elif agent_name == "MAPPO":
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate_actor=0.0003,
            learning_rate_critic=0.001,
            gamma=gamma,
            clip_epsilon=0.2,
            device=device
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    episode_rewards = []
    episode_throughputs = []
    episode_min_rates = []
    episode_goodness = []

    for episode in tqdm(range(num_episodes), desc=f"Training {agent_name}", leave=False):
        obs, _ = env.reset()
        done = False
        step_count = 0

        episode_reward = 0.0
        episode_throughput = 0.0
        episode_min_rate = float("inf")
        episode_goodness_sum = 0.0

        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)

        # MAPPO trajectory storage (on-policy)
        trajectory_states = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_next_states = []
        trajectory_dones = []
        trajectory_log_probs = []

        while not done and step_count < num_steps_per_episode:
            actions = []
            log_probs = []

            for i in range(num_uavs):
                agent_obs = obs_tensor[i * state_dim:(i + 1) * state_dim]

                if agent_name == "AdaptiveNonStationaryMARL":
                    action = agent.get_action(agent_obs, i, global_state=obs_tensor)
                    log_probs.append(0.0)

                elif agent_name == "MAPPO":
                    action, log_prob = agent.get_action(agent_obs, i, explore=True)
                    log_probs.append(log_prob)

                elif agent_name == "MADDPG":
                    action = agent.get_action(agent_obs, i, explore=True)
                    log_probs.append(0.0)

                else:
                    action = agent.get_action(agent_obs, i)
                    log_probs.append(0.0)

                actions.append(action)

            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            step_count += 1

            episode_reward += float(reward)
            episode_throughput += float(info.get("throughput", 0.0))

            # These require you to add them in env.info
            if "min_rate" in info:
                episode_min_rate = min(episode_min_rate, float(info["min_rate"]))
            if "goodness" in info:
                episode_goodness_sum += float(info["goodness"])

            next_obs_tensor = torch.tensor(next_obs, device=device, dtype=torch.float32)

            # Build per-agent states
            states = [obs_tensor[i * state_dim:(i + 1) * state_dim] for i in range(num_uavs)]
            next_states = [next_obs_tensor[i * state_dim:(i + 1) * state_dim] for i in range(num_uavs)]

            # Store transition
            if agent_name == "MAPPO":
                trajectory_states.append(states)
                trajectory_actions.append(actions)
                trajectory_rewards.append([reward] * num_uavs)
                trajectory_next_states.append(next_states)
                trajectory_dones.append([done] * num_uavs)
                trajectory_log_probs.append(log_probs)

            elif agent_name in ("AdaptiveNonStationaryMARL", "QMIX"):
                global_state = obs_tensor
                next_global_state = next_obs_tensor
                agent.store_transition(
                    states, actions, [reward] * num_uavs,
                    next_states, [done] * num_uavs,
                    global_state, next_global_state
                )

            elif agent_name == "MADDPG":
                agent.store_transition(states, actions, [reward] * num_uavs, next_states, [done] * num_uavs)

            else:
                agent.store_transition(states, actions, [reward] * num_uavs, next_states, [done] * num_uavs)

            # Update (off-policy during episode, MAPPO at end)
            if agent_name != "MAPPO":
                if hasattr(agent, "replay_buffer") and len(agent.replay_buffer) > agent.batch_size and (step_count % 2 == 0):
                    agent.update()

            obs_tensor = next_obs_tensor

        # MAPPO update after episode
        if agent_name == "MAPPO":
            for t in range(len(trajectory_states)):
                agent.store_transition(
                    trajectory_states[t], trajectory_actions[t], trajectory_rewards[t],
                    trajectory_next_states[t], trajectory_dones[t], trajectory_log_probs[t]
                )
            agent.update()

        env.end_episode()

        if episode_min_rate == float("inf"):
            episode_min_rate = 0.0

        episode_rewards.append(episode_reward)
        episode_throughputs.append(episode_throughput)
        episode_min_rates.append(episode_min_rate)
        episode_goodness.append(episode_goodness_sum / max(1, step_count))

    return {
        "rewards": episode_rewards,
        "throughputs": episode_throughputs,
        "min_rates": episode_min_rates,
        "goodness": episode_goodness
    }


# ------------------------- Sweeps -------------------------

def sweep_ue_density_multi(
    agents_dict,
    device,
    grid_size,
    num_uavs=3,
    user_list=(5, 10, 15, 20, 30, 40),
    num_episodes=1000,
    num_steps_per_episode=30,
    env_flags=None,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1,
    context_dim=7,
    last_n=250
):
    """
    Returns curves:
      curves[agent_name] = {"x": users, "throughput": y, "minrate": y}
    """
    env_flags = env_flags or {}
    curves = {}

    for agent_name, agent_class in agents_dict.items():
        xs, y_tp, y_minrate = [], [], []

        for U in user_list:
            env = MARLEnv(
                num_uavs=num_uavs,
                num_users=U,
                grid_size=grid_size,
                device=device,
                enable_non_stationary=env_flags.get("enable_non_stationary", False),
                enable_performative=env_flags.get("enable_performative", False)
            )

            if "min_user_rate" in env_flags:
                env.min_user_rate = float(env_flags["min_user_rate"])

            results = train_model(
                agent_class, agent_name, env,
                num_episodes=num_episodes,
                num_steps_per_episode=num_steps_per_episode,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                device=device,
                context_dim=context_dim
            )

            tp_gbps, min_rate_mbps, _ = summarize_last(results, last_n=last_n)
            xs.append(U)
            y_tp.append(tp_gbps)
            y_minrate.append(min_rate_mbps)

        curves[agent_name] = {
            "x": np.array(xs),
            "throughput": np.array(y_tp),
            "minrate": np.array(y_minrate),
        }

    return curves


def sweep_threshold_multi(
    agents_dict,
    device,
    grid_size,
    num_uavs=3,
    num_users=20,
    threshold_list=(0.2, 0.4, 0.6, 0.8, 1.0),  # Mbps
    num_episodes=1000,
    num_steps_per_episode=30,
    env_flags=None,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=0.1,
    context_dim=7,
    last_n=250
):
    """
    Returns curves:
      curves[agent_name] = {"x": thresholds_mbps, "goodness": y}
    """
    env_flags = env_flags or {}
    curves = {}

    for agent_name, agent_class in agents_dict.items():
        xs, y_good = [], []

        for thr_mbps in threshold_list:
            env = MARLEnv(
                num_uavs=num_uavs,
                num_users=num_users,
                grid_size=grid_size,
                device=device,
                enable_non_stationary=env_flags.get("enable_non_stationary", False),
                enable_performative=env_flags.get("enable_performative", False)
            )

            # Threshold in your env = QoS minimum user rate (Mbps)
            env.min_user_rate = float(thr_mbps)

            results = train_model(
                agent_class, agent_name, env,
                num_episodes=num_episodes,
                num_steps_per_episode=num_steps_per_episode,
                learning_rate=learning_rate,
                gamma=gamma,
                epsilon=epsilon,
                device=device,
                context_dim=context_dim
            )

            _, _, goodness = summarize_last(results, last_n=last_n)
            xs.append(thr_mbps)
            y_good.append(goodness)

        curves[agent_name] = {"x": np.array(xs), "goodness": np.array(y_good)}

    return curves


# ------------------------- Plotting -------------------------

def plot_3_separate_graphs(curves_density, curves_threshold, user_list, thr_list_mbps,
                           out_prefix="paper_style"):
    """
    Saves:
      {out_prefix}_throughput_vs_density.png
      {out_prefix}_minrate_vs_density.png
      {out_prefix}_goodness_vs_threshold.png
    Paper-style axes:
      - density x-axis shown as 1..K (density level index)
      - throughput in Mbps
      - min-rate label as LBT (Mbps)
      - goodness normalized to 0..1 per agent
    """
    x_density = np.arange(1, len(user_list) + 1)

    # (1) Throughput vs User Density
    plt.figure(figsize=(7.5, 5))
    for agent_name, data in curves_density.items():
        plt.plot(x_density, data["throughput"], marker="o", label=agent_name)
    plt.xlabel("User Density")
    plt.ylabel("System Throughput (Mbps)")
    plt.title("System Throughput vs User Density")
    plt.xticks(x_density)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="upper left")
    out1 = f"{out_prefix}_throughput_vs_density.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out1}")

    # (2) Minimum User Data Rate vs UE Density
    plt.figure(figsize=(7.5, 5))
    for agent_name, data in curves_density.items():
        plt.plot(x_density, data["minrate"], marker="o", label=agent_name)
    plt.xlabel("Number of Users (UE Density)")
    plt.ylabel("LBT (Mbps)")
    plt.title("Minimum User Data Rate vs UE Density")
    plt.xticks(x_density)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="upper right")
    out2 = f"{out_prefix}_minrate_vs_density.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out2}")

    # (3) Goodness vs Threshold
    plt.figure(figsize=(7.5, 5))
    for agent_name, data in curves_threshold.items():
        g = np.array(data["goodness"], dtype=float)
        g_norm = (g - g.min()) / (g.max() - g.min() + 1e-9)  # normalize 0..1
        plt.plot(thr_list_mbps, g_norm, marker="o", label=agent_name)
    plt.xlabel("Threshold (Mbps)")
    plt.ylabel("Goodness")
    plt.title("Goodness vs Threshold Comparison")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc="upper right")
    out3 = f"{out_prefix}_goodness_vs_threshold.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved {out3}")
# ------------------------- Main -------------------------

def main():
    set_seed(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Config
    num_uavs = 3
    grid_size = (10, 10, 5)

    learning_rate = 0.001
    gamma = 0.99
    epsilon = 0.1
    context_dim = 7

    # Agents to compare (fair: same env flags for all)
    agents = {
        "QMIX": QMIX,
        "IQL": IQL,
        "VDN": VDN,
        "MADDPG": MADDPG,
        "DeepNashQ": DeepNashQ,
        "MAPPO": MAPPO,
        "AdaptiveNonStationaryMARL": AdaptiveNonStationaryMARL,
    }

    # SAME env settings for everyone (fair)
    env_flags_density = dict(
        enable_non_stationary=True,
        enable_performative=True,
        min_user_rate=0.5,  # Mbps fixed threshold during UE density sweeps
    )

    env_flags_threshold = dict(
        enable_non_stationary=True,
        enable_performative=True,
    )

    # Sweep points
    user_list = [5, 10, 15, 20, 30]
    thr_list_mbps = [0.2, 0.5, 1.0, 2.0]

    # Training length (increase for better curves; decrease to run faster)
    num_episodes = 1000
    num_steps_per_episode = 30
    last_n = 250

    print("\nRunning UE density sweep (this trains each agent per point)...")
    curves_density = sweep_ue_density_multi(
        agents,
        device=device,
        grid_size=grid_size,
        num_uavs=num_uavs,
        user_list=user_list,
        num_episodes=num_episodes,
        num_steps_per_episode=num_steps_per_episode,
        env_flags=env_flags_density,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        context_dim=context_dim,
        last_n=last_n
    )

    print("\nRunning threshold sweep (this trains each agent per threshold)...")
    curves_threshold = sweep_threshold_multi(
        agents,
        device=device,
        grid_size=grid_size,
        num_uavs=num_uavs,
        num_users=20,
        threshold_list=thr_list_mbps,
        num_episodes=num_episodes,
        num_steps_per_episode=num_steps_per_episode,
        env_flags=env_flags_threshold,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        context_dim=context_dim,
        last_n=last_n
    )

    print("\nPlotting 3 graphs...")
    plot_3_separate_graphs(curves_density, curves_threshold,
                       user_list=user_list, thr_list_mbps=thr_list_mbps,
                       out_prefix="paper_style")
    print("✅ Done.")


if __name__ == "__main__":
    main()