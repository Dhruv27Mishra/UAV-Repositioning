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
    user_rates = np.array(user_rates, dtype=float)
    if len(user_rates) == 0:
        return 0.0
    numerator = np.sum(user_rates) ** 2
    denominator = len(user_rates) * np.sum(user_rates ** 2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def run_algorithm(algorithm_name,
                  agent_class,
                  env,
                  num_episodes=100,
                  num_uavs=3,
                  state_dim=3,
                  action_dim=6,
                  device=None,
                  **kwargs):
    """Run a MARL algorithm and collect metrics."""
    print(f"\n{'=' * 70}")
    print(f"Training {algorithm_name}")
    print(f"{'=' * 70}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Initialize agent with correct constructor for each algorithm -----
    if algorithm_name == "QMIX":
        # QMIX needs global_state_dim
        global_state_dim = state_dim * num_uavs
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=state_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            device=device,
            **kwargs
        )
    else:
        # IQL, VDN, Deep Nash Q, MADDPG
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **kwargs
        )

    metrics = {
        "throughputs": [],
        "rewards": [],
        "fairnesses": [],
        "collisions": [],
        "heights": [],
        "final_heights": None,  # we fill this after all episodes
    }

    for episode in tqdm(range(num_episodes), desc=f"  {algorithm_name}"):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_throughput = 0.0
        episode_heights = []
        step_count = 0

        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)

        while not done and step_count < 50:
            # -------- Get actions for each agent ----------
            actions = []
            for i_agent in range(num_uavs):
                # obs is flattened [x1,y1,z1, x2,y2,z2, ...]
                agent_obs = obs_tensor[i_agent * state_dim:(i_agent + 1) * state_dim]

                if algorithm_name == "MADDPG":
                    # MADDPG supports exploration flag
                    explore = episode < num_episodes * 0.8
                    action = agent.get_action(agent_obs, i_agent, explore=explore)
                else:
                    action = agent.get_action(agent_obs, i_agent)

                actions.append(int(action))

            # -------- Step environment ----------
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            next_obs_tensor = torch.tensor(
                next_obs, device=device, dtype=torch.float32
            )

            # -------- Build per-agent states ----------
            states = [
                obs_tensor[i * state_dim:(i + 1) * state_dim]
                for i in range(num_uavs)
            ]
            next_states = [
                next_obs_tensor[i * state_dim:(i + 1) * state_dim]
                for i in range(num_uavs)
            ]

            # -------- Store transition ----------
            if algorithm_name == "QMIX":
                # QMIX also needs global state and next global state
                global_state = obs_tensor.clone()
                next_global_state = next_obs_tensor.clone()
                agent.store_transition(
                    states,
                    actions,
                    [reward] * num_uavs,
                    next_states,
                    [done] * num_uavs,
                    global_state,
                    next_global_state,
                )
            else:
                agent.store_transition(
                    states,
                    actions,
                    [reward] * num_uavs,
                    next_states,
                    [done] * num_uavs,
                )

            # -------- Update agent ----------
            agent.update()

            # -------- Accumulate metrics ----------
            obs_tensor = next_obs_tensor
            obs = next_obs
            episode_reward += float(reward)
            episode_throughput += float(info.get("throughput", 0.0))
            episode_heights.append(env.uav_positions[:, 2].copy())
            step_count += 1

        # -------- Fairness on last step of episode ----------
        if "user_rates" in info and len(info["user_rates"]) > 0:
            fairness = jains_fairness(info["user_rates"])
        else:
            fairness = 0.0

        metrics["throughputs"].append(episode_throughput)
        metrics["rewards"].append(episode_reward)
        metrics["fairnesses"].append(fairness)
        metrics["collisions"].append(bool(info.get("collisions", False)))
        metrics["heights"].append(
            float(np.mean(episode_heights)) if episode_heights else 0.0
        )

    # After all episodes, record FINAL UAV HEIGHTS
    metrics["final_heights"] = env.uav_positions[:, 2].copy()

    return metrics


def compare_all_algorithms(num_episodes=200, num_uavs=3, num_users=10):
    """Compare all MARL algorithms on the same environment."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Base env just to get action_dim
    env_tmp = MARLEnv(
        num_uavs=num_uavs,
        num_users=num_users,
        grid_size=(10, 10, 5),
        device=device,
    )
    state_dim = 3
    action_dim = int(env_tmp.action_space.nvec[0])

    # Algorithm configurations (matches your rl_agent constructors)
    algorithms = {
        "IQL": (
            IQL,
            {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon": 0.1,
                "buffer_size": 10000,
                "batch_size": 64,
                "target_update": 100,
            },
        ),
        "VDN": (
            VDN,
            {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon": 0.1,
                "buffer_size": 10000,
                "batch_size": 64,
                "target_update": 100,
            },
        ),
        "QMIX": (
            QMIX,
            {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon": 0.1,
                "buffer_size": 10000,
                "batch_size": 64,
                "target_update": 100,
            },
        ),
        "Deep Nash Q": (
            DeepNashQ,
            {
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon": 0.1,
                "buffer_size": 10000,
                "batch_size": 64,
                "target_update": 100,
            },
        ),
        "MADDPG": (
            MADDPG,
            {
                "learning_rate_actor": 0.001,
                "learning_rate_critic": 0.001,
                "gamma": 0.99,
                "tau": 0.01,
                "buffer_size": 10000,
                "batch_size": 64,
            },
        ),
    }

    results = {}

    for alg_name, (agent_class, kwargs) in algorithms.items():
        env = MARLEnv(
            num_uavs=num_uavs,
            num_users=num_users,
            grid_size=(10, 10, 5),
            device=device,
        )
        metrics = run_algorithm(
            alg_name,
            agent_class,
            env,
            num_episodes=num_episodes,
            num_uavs=num_uavs,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **kwargs,
        )
        results[alg_name] = metrics

    return results


def print_summary(results):
    """Print summary statistics for all algorithms."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print("\n📊 THROUGHPUT STATISTICS:")
    throughput_means = {}
    for alg_name, metrics in results.items():
        mean_tp = float(np.mean(metrics["throughputs"]))
        std_tp = float(np.std(metrics["throughputs"]))
        throughput_means[alg_name] = mean_tp
        print(f"  {alg_name:15s}: {mean_tp:10.3f} ± {std_tp:8.3f}")

    best_alg = max(throughput_means, key=throughput_means.get)
    print(f"\n🏆 Best Performer (throughput): {best_alg} ({throughput_means[best_alg]:.3f})")

    print("\n📈 FAIRNESS STATISTICS (Jain's index):")
    for alg_name, metrics in results.items():
        mean_fair = float(np.mean(metrics["fairnesses"]))
        print(f"  {alg_name:15s}: {mean_fair:.4f}")

    print("\n💥 COLLISIONS (episodes with collision):")
    for alg_name, metrics in results.items():
        total_collisions = int(np.sum(metrics["collisions"]))
        print(f"  {alg_name:15s}: {total_collisions:4d}")

    print("\n📏 MEAN UAV HEIGHT PER EPISODE:")
    for alg_name, metrics in results.items():
        valid_heights = [h for h in metrics["heights"] if h > 0]
        mean_height = float(np.mean(valid_heights)) if valid_heights else 0.0
        print(f"  {alg_name:15s}: {mean_height:6.2f} m")

    print("\n🛬 FINAL UAV HEIGHTS AFTER LEARNING:")
    for alg_name, metrics in results.items():
        final_h = metrics.get("final_heights", None)
        if final_h is not None:
            print(f"  {alg_name:15s}: {np.array2string(final_h, precision=2)}")
        else:
            print(f"  {alg_name:15s}: (final heights not recorded)")


def plot_comparison(results):
    """Create simple comparison plots (throughput, reward, fairness, height)."""
    fig = plt.figure(figsize=(18, 10))

    algorithms = list(results.keys())
    colors = ["red", "blue", "green", "orange", "purple"]

    # Throughput over episodes
    ax1 = fig.add_subplot(2, 2, 1)
    for idx, (alg_name, metrics) in enumerate(results.items()):
        ax1.plot(
            metrics["throughputs"],
            label=alg_name,
            color=colors[idx % len(colors)],
            alpha=0.8,
        )
    ax1.set_title("Episode Throughput")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Throughput")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Reward over episodes
    ax2 = fig.add_subplot(2, 2, 2)
    for idx, (alg_name, metrics) in enumerate(results.items()):
        ax2.plot(
            metrics["rewards"],
            label=alg_name,
            color=colors[idx % len(colors)],
            alpha=0.8,
        )
    ax2.set_title("Episode Reward")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Fairness over episodes
    ax3 = fig.add_subplot(2, 2, 3)
    for idx, (alg_name, metrics) in enumerate(results.items()):
        ax3.plot(
            metrics["fairnesses"],
            label=alg_name,
            color=colors[idx % len(colors)],
            alpha=0.8,
        )
    ax3.set_title("Jain's Fairness Index")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Fairness")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Heights over episodes
    ax4 = fig.add_subplot(2, 2, 4)
    for idx, (alg_name, metrics) in enumerate(results.items()):
        ax4.plot(
            metrics["heights"],
            label=alg_name,
            color=colors[idx % len(colors)],
            alpha=0.8,
        )
    ax4.set_title("Mean UAV Height per Episode")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Height (m)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig("marl_comparison_all.png", dpi=150)
    print("\n✓ Comparison plot saved to 'marl_comparison_all.png'")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    results = compare_all_algorithms(
        num_episodes=200,
        num_uavs=3,
        num_users=10,  # match your original default; change to 20 if you want
    )

    print_summary(results)
    plot_comparison(results)

    end = time.time()
    print(f"\nTotal comparison time: {end - start:.1f} s")


if __name__ == "__main__":
    main()