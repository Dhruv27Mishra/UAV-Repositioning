"""
ALL MODELS IN ENHANCED ENV (Performative + Non-Stationary)
- Robust observation slicing (fixes reshape error)
- Disk caching (skip reruns)
- Smooth sweep option (0..40 or custom)
- CLEARER PLOTS for overlapping curves:
  * highlight AdaptiveNonStationaryMARL
  * distinct markers
  * zorder layering
  * optional zoom inset

Outputs:
- throughput_vs_ue_density_all_enhanced.png
- goodness_vs_ue_density_all_enhanced.png
"""

import os
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# optional inset (ships with matplotlib)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from rl_agent.marl_env import MARLEnv
from rl_agent.QMIX import QMIX
from rl_agent.IQL import IQL
from rl_agent.VDN import VDN
from rl_agent.MADDPG import MADDPG
from rl_agent.DeepNashQ import DeepNashQ
from rl_agent.MAPPO import MAPPO
from rl_agent.AdaptiveNonStationaryMARL import AdaptiveNonStationaryMARL


# ------------------------- Config -------------------------

@dataclass(frozen=True)
class TrainConfig:
    num_uavs: int = 3
    grid_size: Tuple[int, int, int] = (10, 10, 5)

    num_episodes: int = 1000
    num_steps_per_episode: int = 30

    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon: float = 0.1
    context_dim: int = 0

    last_n: int = 200  # last-N averaging for plotted point

    enable_non_stationary: bool = True
    enable_performative: bool = True

    # optional QoS threshold applied to all (set <=0 to disable)
    min_user_rate_mbps: float = -1.0

    cache_dir: str = "cache_runs"
    seed: int = 123


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------- Caching helpers -------------------------

def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)

def make_cache_key(agent_name: str, ue_users: int, cfg: TrainConfig) -> str:
    payload = {"agent": agent_name, "ue_users": ue_users, "cfg": asdict(cfg)}
    s = _stable_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def cache_path(cfg: TrainConfig, key: str) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    return os.path.join(cfg.cache_dir, f"{key}.npz")

def try_load_cached(cfg: TrainConfig, agent_name: str, ue_users: int):
    key = make_cache_key(agent_name, ue_users, cfg)
    path = cache_path(cfg, key)
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return {
            "rewards": data["rewards"].tolist(),
            "throughputs": data["throughputs"].tolist(),
            "min_rates": data["min_rates"].tolist(),
            "goodness": data["goodness"].tolist(),
            "_loaded_from_cache": True,
            "_cache_path": path
        }
    return None

def save_cached(cfg: TrainConfig, agent_name: str, ue_users: int, results: Dict[str, List[float]]):
    key = make_cache_key(agent_name, ue_users, cfg)
    path = cache_path(cfg, key)
    np.savez_compressed(
        path,
        rewards=np.asarray(results["rewards"], dtype=np.float32),
        throughputs=np.asarray(results["throughputs"], dtype=np.float32),
        min_rates=np.asarray(results["min_rates"], dtype=np.float32),
        goodness=np.asarray(results["goodness"], dtype=np.float32),
    )
    return path


# ------------------------- Metrics -------------------------

def summarize_last(results: Dict[str, List[float]], last_n: int):
    last_n = min(last_n, len(results["throughputs"]))
    tp_gbps = float(np.mean(results["throughputs"][-last_n:]) / 1e9)
    goodness = float(np.mean(results["goodness"][-last_n:]))
    return tp_gbps, goodness


# ------------------------- Robust obs layout inference -------------------------

def _infer_obs_layout_and_state_dim(obs_len: int, num_uavs: int, enable_non_stationary: bool):
    """
    Handles:
    (A) obs_len divisible by num_uavs -> pure per-agent blocks
    (B) obs_len NOT divisible -> assume local blocks + global tail
        local_size = obs_len // num_uavs
        global_size = obs_len - local_size*num_uavs
        agent_obs = concat(local_i, global_tail)
    """
    if enable_non_stationary and (obs_len % num_uavs == 0):
        per_local = obs_len // num_uavs
        global_size = 0
        eff_state_dim = per_local
        return per_local, global_size, eff_state_dim

    per_local = obs_len // num_uavs
    global_size = obs_len - per_local * num_uavs
    eff_state_dim = per_local + global_size
    return per_local, global_size, eff_state_dim


# ------------------------- Agent init -------------------------

def init_agent(agent_class, agent_name: str, env: MARLEnv, cfg: TrainConfig, device):
    num_uavs = env.num_uavs
    obs_len = env.observation_space.shape[0]
    per_local, global_size, eff_state_dim = _infer_obs_layout_and_state_dim(obs_len, num_uavs, env.enable_non_stationary)

    action_dim = env.action_space.nvec[0]

    if agent_name == "AdaptiveNonStationaryMARL":
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=eff_state_dim,
            action_dim=action_dim,
            global_state_dim=obs_len,
            context_dim=cfg.context_dim,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            epsilon=cfg.epsilon,
            device=device,
            buffer_size=10000,
            batch_size=64,
            target_update=100
        )
        env.set_association_function(agent.get_association_function())
        return agent, per_local, global_size, eff_state_dim

    if agent_name == "QMIX":
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=eff_state_dim,
            action_dim=action_dim,
            global_state_dim=obs_len,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            epsilon=cfg.epsilon,
            device=device,
            buffer_size=10000,
            batch_size=64,
            target_update=100
        )
        return agent, per_local, global_size, eff_state_dim

    if agent_name in ("IQL", "VDN", "DeepNashQ"):
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=eff_state_dim,
            action_dim=action_dim,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            epsilon=cfg.epsilon,
            device=device,
            buffer_size=10000,
            batch_size=64,
            target_update=100
        )
        return agent, per_local, global_size, eff_state_dim

    if agent_name == "MADDPG":
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=eff_state_dim,
            action_dim=action_dim,
            learning_rate_actor=cfg.learning_rate,
            learning_rate_critic=cfg.learning_rate,
            gamma=cfg.gamma,
            device=device,
            buffer_size=10000,
            batch_size=64,
            tau=0.01
        )
        return agent, per_local, global_size, eff_state_dim

    if agent_name == "MAPPO":
        agent = agent_class(
            num_agents=num_uavs,
            state_dim=eff_state_dim,
            action_dim=action_dim,
            learning_rate_actor=3e-4,
            learning_rate_critic=1e-3,
            gamma=cfg.gamma,
            clip_epsilon=0.2,
            device=device
        )
        return agent, per_local, global_size, eff_state_dim

    raise ValueError(f"Unknown agent: {agent_name}")


# ------------------------- Training (NO chunking) -------------------------

def train_model_cached(agent_class, agent_name: str, env: MARLEnv, cfg: TrainConfig, device):
    cached = try_load_cached(cfg, agent_name, env.num_users)
    if cached is not None:
        return cached

    agent, per_local, global_size, eff_state_dim = init_agent(agent_class, agent_name, env, cfg, device)
    num_uavs = env.num_uavs
    obs_len = env.observation_space.shape[0]

    episode_rewards, episode_throughputs, episode_min_rates, episode_goodness = [], [], [], []

    for _ in tqdm(range(cfg.num_episodes), desc=f"{agent_name} U={env.num_users}", leave=False):
        obs, _ = env.reset()
        done = False

        ep_reward = 0.0
        ep_tp = 0.0
        ep_min_rate = float("inf")
        ep_good_sum = 0.0
        step_count = 0

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

        # local blocks + optional global tail
        if global_size == 0:
            obs_mat = obs_t.view(num_uavs, per_local)
            global_tail = None
        else:
            local_flat = obs_t[:per_local * num_uavs]
            obs_mat = local_flat.view(num_uavs, per_local)
            global_tail = obs_t[per_local * num_uavs : per_local * num_uavs + global_size]

        # MAPPO trajectory
        traj_states, traj_actions, traj_rewards = [], [], []
        traj_next_states, traj_dones, traj_log_probs = [], [], []

        while (not done) and (step_count < cfg.num_steps_per_episode):
            actions = []
            log_probs = []

            with torch.no_grad():
                for i in range(num_uavs):
                    local_i = obs_mat[i]
                    agent_obs = torch.cat([local_i, global_tail], dim=0) if global_tail is not None else local_i

                    if agent_name == "AdaptiveNonStationaryMARL":
                        a = agent.get_action(agent_obs, i, global_state=obs_t)
                        log_probs.append(0.0)
                    elif agent_name == "MAPPO":
                        a, lp = agent.get_action(agent_obs, i, explore=True)
                        log_probs.append(lp)
                    elif agent_name == "MADDPG":
                        a = agent.get_action(agent_obs, i, explore=True)
                        log_probs.append(0.0)
                    else:
                        a = agent.get_action(agent_obs, i)
                        log_probs.append(0.0)

                    actions.append(a)

            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            ep_reward += float(reward)
            ep_tp += float(info.get("throughput", 0.0))
            ep_min_rate = min(ep_min_rate, float(info.get("min_rate", ep_min_rate)))
            ep_good_sum += float(info.get("goodness", 0.0))
            step_count += 1

            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

            if global_size == 0:
                next_obs_mat = next_obs_t.view(num_uavs, per_local)
                next_global_tail = None
            else:
                local_flat_n = next_obs_t[:per_local * num_uavs]
                next_obs_mat = local_flat_n.view(num_uavs, per_local)
                next_global_tail = next_obs_t[per_local * num_uavs : per_local * num_uavs + global_size]

            # build states lists
            states, next_states = [], []
            for i in range(num_uavs):
                s = torch.cat([obs_mat[i], global_tail], dim=0) if global_tail is not None else obs_mat[i]
                ns = torch.cat([next_obs_mat[i], next_global_tail], dim=0) if next_global_tail is not None else next_obs_mat[i]
                states.append(s); next_states.append(ns)

            if agent_name == "MAPPO":
                traj_states.append(states)
                traj_actions.append(actions)
                traj_rewards.append([reward] * num_uavs)
                traj_next_states.append(next_states)
                traj_dones.append([done] * num_uavs)
                traj_log_probs.append(log_probs)
            elif agent_name in ("AdaptiveNonStationaryMARL", "QMIX"):
                agent.store_transition(states, actions, [reward] * num_uavs, next_states, [done] * num_uavs, obs_t, next_obs_t)
            elif agent_name == "MADDPG":
                agent.store_transition(states, actions, [reward] * num_uavs, next_states, [done] * num_uavs)
            else:
                agent.store_transition(states, actions, [reward] * num_uavs, next_states, [done] * num_uavs)

            if agent_name != "MAPPO":
                if hasattr(agent, "replay_buffer") and len(agent.replay_buffer) > agent.batch_size and (step_count % 2 == 0):
                    agent.update()

            obs_t = next_obs_t
            obs_mat = next_obs_mat
            global_tail = next_global_tail

        if agent_name == "MAPPO":
            for t in range(len(traj_states)):
                agent.store_transition(traj_states[t], traj_actions[t], traj_rewards[t],
                                       traj_next_states[t], traj_dones[t], traj_log_probs[t])
            agent.update()

        env.end_episode()

        if ep_min_rate == float("inf"):
            ep_min_rate = 0.0

        episode_rewards.append(ep_reward)
        episode_throughputs.append(ep_tp)
        episode_min_rates.append(ep_min_rate)
        episode_goodness.append(ep_good_sum / max(1, step_count))

    results = {
        "rewards": episode_rewards,
        "throughputs": episode_throughputs,
        "min_rates": episode_min_rates,
        "goodness": episode_goodness
    }
    save_cached(cfg, agent_name, env.num_users, results)
    return results


# ------------------------- Sweep (All models, Enhanced) -------------------------

def sweep_ue_density_all_models_enhanced(
    agents: Dict[str, Any],
    cfg: TrainConfig,
    device,
    user_list: List[int],
):
    curves = {}
    for agent_name, agent_class in agents.items():
        xs, y_tp, y_good = [], [], []

        for U in user_list:
            try:
                env = MARLEnv(
                    num_uavs=cfg.num_uavs,
                    num_users=U,
                    grid_size=cfg.grid_size,
                    device=device,
                    enable_non_stationary=True,
                    enable_performative=True
                )
            except Exception as e:
                print(f"[SKIP] env init failed for num_users={U}: {e}")
                continue

            if cfg.min_user_rate_mbps is not None and cfg.min_user_rate_mbps > 0:
                env.min_user_rate = float(cfg.min_user_rate_mbps)

            res = train_model_cached(agent_class, agent_name, env, cfg, device)
            tp_gbps, good = summarize_last(res, last_n=cfg.last_n)

            xs.append(U); y_tp.append(tp_gbps); y_good.append(good)

        curves[agent_name] = {
            "x": np.array(xs, dtype=int),
            "throughput_gbps": np.array(y_tp, dtype=float),
            "goodness": np.array(y_good, dtype=float),
        }

    return curves


# ------------------------- Plotting: clearer curves -------------------------

def plot_throughput_vs_ue_density_all(
    curves: Dict[str, Dict[str, np.ndarray]],
    xticks: List[int],
    marker_xs: List[int],
    out_png: str = "throughput_vs_ue_density_all_enhanced.png",
    add_zoom_inset: bool = True
):
    plt.figure(figsize=(11, 7))
    ax = plt.gca()

    # style configs
    markers = {
        "QMIX": "o",
        "IQL": "s",
        "VDN": "D",
        "MADDPG": "^",
        "DeepNashQ": "v",
        "MAPPO": "P",
        "AdaptiveNonStationaryMARL": "*",
    }

    # draw non-novel first
    for agent_name, d in curves.items():
        if agent_name == "AdaptiveNonStationaryMARL":
            continue
        x = d["x"]; y = d["throughput_gbps"]
        ax.plot(
            x, y,
            linewidth=2.2,
            alpha=0.85,
            marker=markers.get(agent_name, "o"),
            markevery=[i for i, xv in enumerate(x) if int(xv) in set(marker_xs)],
            label=agent_name,
            zorder=2
        )

    # highlight novel on top
    if "AdaptiveNonStationaryMARL" in curves:
        d = curves["AdaptiveNonStationaryMARL"]
        x = d["x"]; y = d["throughput_gbps"]
        ax.plot(
            x, y,
            linewidth=4.0,
            alpha=1.0,
            marker=markers["AdaptiveNonStationaryMARL"],
            markersize=12,
            markevery=[i for i, xv in enumerate(x) if int(xv) in set(marker_xs)],
            label="AdaptiveNonStationaryMARL",
            zorder=10
        )

    ax.set_xlabel("UE Density (Number of Users)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average System Throughput (Gbps)", fontsize=13, fontweight="bold")
    ax.set_title("System Throughput vs UE Density (All Models, Enhanced Env)", fontsize=15, fontweight="bold")
    ax.set_xticks(xticks)
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.9)

    # legend outside
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95)

    # zoom inset where curves overlap
    if add_zoom_inset:
        axins = inset_axes(ax, width="45%", height="45%", loc="lower right", borderpad=2.0)
        for agent_name, d in curves.items():
            x = d["x"]; y = d["throughput_gbps"]
            lw = 3.0 if agent_name == "AdaptiveNonStationaryMARL" else 1.8
            zo = 10 if agent_name == "AdaptiveNonStationaryMARL" else 2
            axins.plot(x, y, linewidth=lw, alpha=0.95, zorder=zo)

        # choose zoom region (you can tweak)
        axins.set_xlim(min(xticks), max(xticks))
        # auto y-limits based on displayed range
        y_all = np.concatenate([curves[k]["throughput_gbps"] for k in curves])
        y_min = np.percentile(y_all, 40)
        y_max = np.percentile(y_all, 98)
        axins.set_ylim(y_min, y_max)
        axins.grid(True, alpha=0.25, linestyle="--")

        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"✓ Saved {out_png}")
    plt.close()


def plot_goodness_vs_ue_density_all(
    curves: Dict[str, Dict[str, np.ndarray]],
    xticks: List[int],
    marker_xs: List[int],
    out_png: str = "goodness_vs_ue_density_all_enhanced.png",
    add_zoom_inset: bool = True
):
    plt.figure(figsize=(11, 7))
    ax = plt.gca()

    markers = {
        "QMIX": "o",
        "IQL": "s",
        "VDN": "D",
        "MADDPG": "^",
        "DeepNashQ": "v",
        "MAPPO": "P",
        "AdaptiveNonStationaryMARL": "*",
    }

    for agent_name, d in curves.items():
        if agent_name == "AdaptiveNonStationaryMARL":
            continue
        x = d["x"]; y = d["goodness"]
        ax.plot(
            x, y,
            linewidth=2.2,
            alpha=0.85,
            marker=markers.get(agent_name, "o"),
            markevery=[i for i, xv in enumerate(x) if int(xv) in set(marker_xs)],
            label=agent_name,
            zorder=2
        )

    if "AdaptiveNonStationaryMARL" in curves:
        d = curves["AdaptiveNonStationaryMARL"]
        x = d["x"]; y = d["goodness"]
        ax.plot(
            x, y,
            linewidth=4.0,
            alpha=1.0,
            marker=markers["AdaptiveNonStationaryMARL"],
            markersize=12,
            markevery=[i for i, xv in enumerate(x) if int(xv) in set(marker_xs)],
            label="AdaptiveNonStationaryMARL",
            zorder=10
        )

    ax.set_xlabel("UE Density (Number of Users)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Goodness", fontsize=13, fontweight="bold")
    ax.set_title("Goodness vs UE Density (All Models, Enhanced Env)", fontsize=15, fontweight="bold")
    ax.set_xticks(xticks)
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.9)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95)

    if add_zoom_inset:
        axins = inset_axes(ax, width="45%", height="45%", loc="lower right", borderpad=2.0)
        for agent_name, d in curves.items():
            x = d["x"]; y = d["goodness"]
            lw = 3.0 if agent_name == "AdaptiveNonStationaryMARL" else 1.8
            zo = 10 if agent_name == "AdaptiveNonStationaryMARL" else 2
            axins.plot(x, y, linewidth=lw, alpha=0.95, zorder=zo)

        axins.set_xlim(min(xticks), max(xticks))
        y_all = np.concatenate([curves[k]["goodness"] for k in curves])
        y_min = np.percentile(y_all, 40)
        y_max = np.percentile(y_all, 98)
        axins.set_ylim(y_min, y_max)
        axins.grid(True, alpha=0.25, linestyle="--")
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"✓ Saved {out_png}")
    plt.close()


# ------------------------- Main -------------------------

def main():
    cfg = TrainConfig(
        num_uavs=3,
        grid_size=(10, 10, 5),
        num_episodes=1000,
        num_steps_per_episode=30,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=0.1,
        context_dim=0,
        last_n=200,
        enable_non_stationary=True,
        enable_performative=True,
        min_user_rate_mbps=-1.0,  # set e.g. 0.6 if you want fixed QoS
        cache_dir="cache_runs",
        seed=123
    )

    set_seed(cfg.seed)

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("ALL MODELS in ENHANCED ENV | Cached | Clear plots")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    agents = {
        "QMIX": QMIX,
        "IQL": IQL,
        "VDN": VDN,
        "MADDPG": MADDPG,
        "DeepNashQ": DeepNashQ,
        "MAPPO": MAPPO,
        "AdaptiveNonStationaryMARL": AdaptiveNonStationaryMARL,
    }

    # If you want smooth curves: use all users 0..40
    # user_list = list(range(0, 41))
    # If you want fewer (faster) like your current plot:
    user_list = list(range(1, 21))

    # Show ticks/markers at these points
    tick_points = list(range(0, 21, 2))  # every 2 users]
    marker_points = [0, 5, 10, 15, 20]  # where to put markers

    t0 = time.time()
    curves = sweep_ue_density_all_models_enhanced(agents, cfg, device, user_list=user_list)
    t1 = time.time()

    plot_throughput_vs_ue_density_all(
        curves,
        xticks=tick_points,
        marker_xs=marker_points,
        out_png="throughput_vs_ue_density_all_enhanced.png",
        add_zoom_inset=True
    )

    plot_goodness_vs_ue_density_all(
        curves,
        xticks=tick_points,
        marker_xs=marker_points,
        out_png="goodness_vs_ue_density_all_enhanced.png",
        add_zoom_inset=True
    )

    print(f"\nDone in {t1 - t0:.1f} seconds.")
    print("Generated:")
    print("  - throughput_vs_ue_density_all_enhanced.png")
    print("  - goodness_vs_ue_density_all_enhanced.png")
    print(f"Cache directory: {cfg.cache_dir}")


if __name__ == "__main__":
    main()