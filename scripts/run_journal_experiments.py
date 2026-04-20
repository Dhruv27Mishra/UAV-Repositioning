#!/usr/bin/env python3
"""
STEP 1 — TRAIN:
    Train each algorithm for 2000 episodes on the base environment.
    Save the final neural-network weights (.pt files).
    No more weight updates after this step.

STEP 2 — EVALUATE:
    Load saved weights (frozen — no training).
    For each x-axis value (gamma / call-rate / velocity / traffic-load):
        Run 500 evaluation episodes with that specific parameter.
        Average the 500 results → one smooth data point.
    This averaging over 500 runs removes noise automatically.

STEP 3 — G6/G7 TRAINING CURVES:
    Record per-episode throughput during training (2000 episodes × 5 seeds).
    Smooth with moving average + spline in the plot script.
"""
from __future__ import annotations

import repo_paths  # noqa: F401

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
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
from rl_agent.ab_qmix_algorithm import ABQMIX
from rl_agent.dmtd_algorithm import DMTD

# ─── constants ────────────────────────────────────────────────────────────────

ALGO_NAMES = [
    "QMIX", "ABQMIX", "IQL", "VDN", "MADDPG",
    "DMTD", "DeepNashQ", "PerformativeMFMARL", "PerformativeMARL",
]

MAPPO_FAMILY = frozenset({"PerformativeMFMARL", "PerformativeMARL"})
QMIX_FAMILY  = frozenset({"QMIX", "ABQMIX"})

# x-axis parameter values
GAMMAS        = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
CALL_RATES    = [5, 10, 15, 25, 35, 50]
VELOCITIES    = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
TRAFFIC_LOADS = [10, 25, 50, 75, 100, 125, 150]

NUM_UAVS     = 3
NUM_USERS    = 20
GRID_SIZE    = (10, 10, 5)
STEPS_PER_EP = 50
LOW_VEL_MAX  = 1.0
HIGH_VEL_MIN = 5.0

TRAIN_EPISODES    = 2000   # train every algorithm for 2000 episodes
EVAL_RUNS         = 500    # evaluation runs per x-axis value (no training)
N_SEEDS           = 5

# quick-mode overrides (smoke-test only)
TRAIN_EPISODES_QUICK = 50
EVAL_RUNS_QUICK      = 20
N_SEEDS_QUICK        = 2

MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models", "journal")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "journal_data.json")


# ─── device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── environment factory ──────────────────────────────────────────────────────

def make_env(
    device: torch.device,
    *,
    traffic_model: str   = "pareto",
    pareto_shape:  float = 1.5,
    pareto_scale:  float = 1.0,
    traffic_load:  float = 1.0,
    low_velocity_max:  float = LOW_VEL_MAX,
    high_velocity_min: float = HIGH_VEL_MIN,
) -> MARLEnv:
    return MARLEnv(
        num_uavs=NUM_UAVS, num_users=NUM_USERS, grid_size=GRID_SIZE,
        device=device, min_user_rate=0.5, qos_bonus=10.0,
        enable_non_stationary=True, enable_performative=True,
        enable_signal_map_obs=True,
        traffic_model=traffic_model, pareto_shape=pareto_shape,
        pareto_scale=pareto_scale, traffic_load=traffic_load,
        low_velocity_max=low_velocity_max,
        high_velocity_min=high_velocity_min,
    )


def _set_velocity_split(env: MARLEnv, seed: int) -> None:
    rng  = np.random.default_rng(seed)
    n    = env.num_users
    n_lo = n // 2
    env.user_velocities[:n_lo] = rng.uniform(0.1, LOW_VEL_MAX * 0.95, n_lo)
    env.user_velocities[n_lo:] = rng.uniform(HIGH_VEL_MIN * 1.05,
                                              HIGH_VEL_MIN * 2.0, n - n_lo)
    cats = ["LOW"] * n_lo + ["HIGH"] * (n - n_lo)
    env.user_velocity_categories = cats


def _set_all_velocity(env: MARLEnv, v: float, seed: int) -> None:
    rng  = np.random.default_rng(seed)
    vels = rng.uniform(max(0.05, v * 0.9), v * 1.1, env.num_users)
    env.user_velocities = vels
    cats = []
    for vi in vels:
        if vi < env.low_velocity_max:     cats.append("LOW")
        elif vi >= env.high_velocity_min: cats.append("HIGH")
        else:                             cats.append("MEDIUM")
    env.user_velocity_categories = cats


# ─── agent factory ────────────────────────────────────────────────────────────

def create_agent(
    name:   str,
    env:    MARLEnv,
    device: torch.device,
    gamma:  float = 0.99,
    seed:   int   = 0,
) -> Any:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n    = env.num_uavs
    sdim = getattr(env, "agent_obs_dim",
                   env.observation_space.shape[0] // n)
    adim = int(env.action_space.nvec[0])
    gdim = env.observation_space.shape[0]

    if name == "QMIX":
        return QMIX(num_agents=n, state_dim=sdim, action_dim=adim,
                    global_state_dim=gdim, learning_rate=2.2e-4, gamma=gamma,
                    epsilon=0.32, device=device, buffer_size=10000,
                    batch_size=40, target_update=320)
    if name == "ABQMIX":
        return ABQMIX(num_agents=n, obs_dim=sdim, global_state_dim=gdim,
                      traj_action_dim=adim, bf_action_dim=1, gamma=gamma,
                      learning_rate=1e-4, epsilon=1.0, epsilon_min=0.1,
                      c_min=1000, batch_size=4, device=device)
    if name == "IQL":
        return IQL(num_agents=n, state_dim=sdim, action_dim=adim,
                   learning_rate=1e-3, gamma=gamma, epsilon=0.1,
                   device=device, buffer_size=10000, batch_size=64,
                   target_update=100)
    if name == "VDN":
        return VDN(num_agents=n, state_dim=sdim, action_dim=adim,
                   learning_rate=1e-3, gamma=gamma, epsilon=0.1,
                   device=device, buffer_size=10000, batch_size=64,
                   target_update=100)
    if name == "MADDPG":
        return MADDPG(num_agents=n, state_dim=sdim, action_dim=adim,
                      learning_rate_actor=1.8e-4, learning_rate_critic=1.8e-4,
                      gamma=gamma, device=device, buffer_size=10000,
                      batch_size=40, tau=0.004)
    if name == "DMTD":
        return DMTD(num_uavs=n, state_dim=sdim, action_dim=adim,
                    gamma=gamma, learning_rate=1e-3, epsilon=1.0,
                    epsilon_min=0.05, buffer_capacity=10000, batch_size=64,
                    device=device)
    if name == "DeepNashQ":
        return DeepNashQ(num_agents=n, state_dim=sdim, action_dim=adim,
                         learning_rate=1e-3, gamma=gamma, epsilon=0.1,
                         device=device, buffer_size=10000, batch_size=64,
                         target_update=100)
    if name in MAPPO_FAMILY:
        agent = MAPPO(num_agents=n, state_dim=sdim, action_dim=adim,
                      learning_rate_actor=5e-4, learning_rate_critic=1.2e-3,
                      gamma=gamma, clip_epsilon=0.22, device=device)
        if hasattr(env, "set_association_function"):
            env.set_association_function(
                AdaptiveNonStationaryMARL.standalone_association_function())
        return agent
    raise ValueError(f"Unknown algorithm: {name}")


def _set_greedy(name: str, agent: Any) -> None:
    """Freeze exploration — pure greedy policy for evaluation."""
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0


def _model_path(exp: str, algo: str, seed: int, tag: str = "trained") -> str:
    d = os.path.join(MODELS_DIR, exp, algo, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{tag}.pt")


# ─── single episode (train OR eval) ──────────────────────────────────────────

def _run_episode(
    name:    str,
    agent:   Any,
    env:     MARLEnv,
    device:  torch.device,
    sdim:    int,
    gstep:   int  = 0,
    explore: bool = True,
    track_mobility_tp: bool = False,
) -> Dict[str, float]:
    obs, _ = env.reset()
    obs_t  = torch.tensor(obs, device=device, dtype=torch.float32)
    num_uavs = env.num_uavs
    done = False

    ep_reward = ep_tp = ep_ho = ep_ej = ep_pdr = 0.0
    ep_tp_lo  = ep_tp_hi = 0.0
    steps = 0

    traj_s = traj_a = traj_r = traj_ns = traj_d = traj_lp = None
    if name in MAPPO_FAMILY:
        traj_s, traj_a, traj_r, traj_ns, traj_d, traj_lp = [], [], [], [], [], []

    if name == "ABQMIX":
        agent.start_episode()

    while not done and steps < STEPS_PER_EP:
        obs_np = [obs_t[i * sdim:(i + 1) * sdim].cpu().numpy()
                  for i in range(num_uavs)]

        if name == "ABQMIX":
            ab_j    = agent.get_actions(obs_np)
            actions = [p[0] for p in ab_j]; lps = [0.0] * num_uavs
        elif name == "DMTD":
            actions = agent.get_actions(obs_np)
            lps = [0.0] * num_uavs; ab_j = None
        elif name in MAPPO_FAMILY:
            actions, lps = [], []
            for i in range(num_uavs):
                a, lp = agent.get_action(obs_t[i * sdim:(i + 1) * sdim],
                                         i, explore=explore)
                actions.append(a); lps.append(lp)
            ab_j = None
        elif name == "MADDPG":
            actions = [agent.get_action(obs_t[i * sdim:(i + 1) * sdim],
                                        i, explore=explore)
                       for i in range(num_uavs)]
            lps = [0.0] * num_uavs; ab_j = None
        else:
            actions = [agent.get_action(obs_t[i * sdim:(i + 1) * sdim], i)
                       for i in range(num_uavs)]
            lps = [0.0] * num_uavs; ab_j = None

        next_obs, reward, term, trunc, info = env.step(actions)
        done = term or trunc

        ep_reward += reward
        ep_tp     += float(info.get("throughput",    0.0))
        ep_ho     += int  (info.get("handovers",     0))
        ep_ej     += float(info.get("step_energy_j", 0.0))
        ep_pdr    += float(info.get("packet_drop_rate", 0.0))

        if track_mobility_tp:
            ur = info.get("user_rates", None)
            if ur is not None and env.user_velocities is not None:
                lm = env.user_velocities <  env.low_velocity_max
                hm = env.user_velocities >= env.high_velocity_min
                ep_tp_lo += float(np.sum(ur[lm])) if lm.any() else 0.0
                ep_tp_hi += float(np.sum(ur[hm])) if hm.any() else 0.0

        next_obs_t = torch.tensor(next_obs, device=device, dtype=torch.float32)
        st  = [obs_t[i * sdim:(i + 1) * sdim]          for i in range(num_uavs)]
        nst = [next_obs_t[i * sdim:(i + 1) * sdim]      for i in range(num_uavs)]
        nob = [next_obs_t[i * sdim:(i + 1) * sdim].cpu().numpy()
               for i in range(num_uavs)]

        if explore:
            if name in MAPPO_FAMILY:
                traj_s.append(st);  traj_a.append(actions)
                traj_r.append([reward] * num_uavs)
                traj_ns.append(nst); traj_d.append([done] * num_uavs)
                traj_lp.append(lps)
            elif name == "ABQMIX":
                agent.store_transition(
                    obs_t.flatten().cpu().numpy(), obs_np, ab_j,
                    reward, next_obs_t.flatten().cpu().numpy(), nob)
            elif name in QMIX_FAMILY:
                agent.store_transition(st, actions, [reward] * num_uavs,
                                       nst, [done] * num_uavs, obs_t, next_obs_t)
            elif name == "DMTD":
                agent.store_transitions(obs_np, actions,
                                        [reward] * num_uavs, nob, gstep + steps)
            else:
                agent.store_transition(st, actions, [reward] * num_uavs,
                                       nst, [done] * num_uavs)

            if name == "ABQMIX":
                agent.update()
            elif name == "DMTD":
                agent.update(gstep + steps)
            elif name not in MAPPO_FAMILY:
                buf = getattr(agent, "replay_buffer", None)
                if buf is not None and len(buf) > getattr(agent, "batch_size", 64):
                    if steps % 2 == 0:
                        agent.update()

        obs_t  = next_obs_t
        steps += 1

    if explore and name in MAPPO_FAMILY and traj_s:
        for t in range(len(traj_s)):
            agent.store_transition(traj_s[t], traj_a[t], traj_r[t],
                                   traj_ns[t], traj_d[t], traj_lp[t])
        agent.update()

    if name == "ABQMIX":
        agent.end_episode()
    env.end_episode()

    n  = max(steps, 1)
    ee = (ep_tp / 1e6) / (ep_ej + 1e-9)
    return {
        "reward":       ep_reward,
        "throughput":   ep_tp  / n,
        "handovers":    ep_ho,
        "energy_eff":   ee,
        "pdr":          ep_pdr    / n * 100.0,
        "tp_low_gbps":  ep_tp_lo  / n / 1e9,
        "tp_high_gbps": ep_tp_hi  / n / 1e9,
    }


# ─── STEP 1: train one algorithm, save weights ───────────────────────────────

def train_and_save(
    algo:    str,
    device:  torch.device,
    seed:    int,
    gamma:   float,
    model_path: str,
    n_train: int,
    env_factory,
) -> Any:
    """
    Train algo for n_train episodes on env_factory().
    Save weights to model_path.
    Return the trained agent.
    """
    env   = env_factory()
    env.reset(seed=seed)
    agent = create_agent(algo, env, device, gamma=gamma, seed=seed)
    sdim  = getattr(env, "agent_obs_dim",
                    env.observation_space.shape[0] // env.num_uavs)
    gstep = 0

    for ep in range(n_train):
        _run_episode(algo, agent, env, device, sdim,
                     gstep=gstep, explore=True)
        gstep += STEPS_PER_EP

    if hasattr(agent, "save"):
        try:
            agent.save(model_path)
        except Exception as e:
            print(f"  [warn] save failed {algo}: {e}")

    return agent


# ─── STEP 2: load weights, evaluate 500 times, return averaged metrics ────────

def load_and_evaluate(
    algo:       str,
    device:     torch.device,
    seed:       int,
    gamma:      float,
    model_path: str,
    n_eval:     int,
    env_factory,
    metric_keys: List[str],
    post_reset_fn=None,
) -> Dict[str, float]:
    """
    Load saved weights (no more training).
    Run n_eval greedy episodes.
    Return averaged metrics — averaging removes noise automatically.
    """
    env   = env_factory()
    env.reset(seed=seed + 9999)
    agent = create_agent(algo, env, device, gamma=gamma, seed=seed)
    sdim  = getattr(env, "agent_obs_dim",
                    env.observation_space.shape[0] // env.num_uavs)

    if os.path.isfile(model_path) and hasattr(agent, "load"):
        try:
            agent.load(model_path)
            print(f"    Loaded {os.path.basename(model_path)}")
        except Exception as e:
            print(f"    [warn] load failed: {e} — using untrained agent")

    _set_greedy(algo, agent)   # freeze weights — no more updates

    accum: Dict[str, List[float]] = {k: [] for k in metric_keys}
    gstep = 0
    for run in range(n_eval):
        if post_reset_fn is not None:
            post_reset_fn(env, run)
        m = _run_episode(algo, agent, env, device, sdim,
                         gstep=gstep, explore=False)
        for k in metric_keys:
            accum[k].append(m[k])
        gstep += STEPS_PER_EP

    # Average over all 500 runs → one smooth data point per x-axis value
    return {k: float(np.mean(accum[k])) for k in metric_keys}


# ─── G1: Energy Efficiency vs Discount Factor ─────────────────────────────────
# For gamma sweep: train a SEPARATE model per gamma value (gamma affects learning)

def run_g1(device: torch.device, quick: bool) -> Dict:
    print("\n=== G1: Energy Efficiency vs Discount Factor ===")
    n_train = TRAIN_EPISODES_QUICK if quick else TRAIN_EPISODES
    n_eval  = EVAL_RUNS_QUICK      if quick else EVAL_RUNS
    n_seed  = N_SEEDS_QUICK        if quick else N_SEEDS

    def _ef():
        return make_env(device, traffic_model="pareto",
                        pareto_shape=1.5, pareto_scale=1.0, traffic_load=1.0)

    data: Dict = {"gammas": GAMMAS}
    for algo in tqdm(ALGO_NAMES, desc="G1 algos"):
        means, stds = [], []
        for g in tqdm(GAMMAS, desc=f"  {algo} train+eval", leave=False):
            seed_vals = []
            for seed in range(n_seed):
                mp = _model_path("g1", algo, seed, f"gamma{g}")

                # STEP 1: train with this specific gamma
                print(f"  [TRAIN] {algo} gamma={g} seed={seed} ({n_train} eps)")
                train_and_save(algo, device, seed, g, mp, n_train, _ef)

                # STEP 2: evaluate 500 times — average removes noise
                print(f"  [EVAL]  {algo} gamma={g} seed={seed} ({n_eval} runs)")
                res = load_and_evaluate(
                    algo, device, seed, g, mp, n_eval, _ef,
                    metric_keys=["energy_eff"])
                seed_vals.append(res["energy_eff"])

            means.append(float(np.mean(seed_vals)))
            stds.append(float(np.std(seed_vals)))
        data[algo] = {"mean": means, "std": stds}
    return data


# ─── G2 & G3: vs Call Arrival Rate ────────────────────────────────────────────
# Train ONCE on default env → evaluate at different call rates (500 runs each)

def run_g2g3(device: torch.device, quick: bool) -> Dict:
    print("\n=== G2/G3: Handover Rate & Energy Efficiency vs Call Arrival Rate ===")
    n_train = TRAIN_EPISODES_QUICK if quick else TRAIN_EPISODES
    n_eval  = EVAL_RUNS_QUICK      if quick else EVAL_RUNS
    n_seed  = N_SEEDS_QUICK        if quick else N_SEEDS
    max_r   = float(max(CALL_RATES))

    def _base_ef():
        return make_env(device, traffic_model="pareto",
                        pareto_shape=1.5, pareto_scale=1.0, traffic_load=1.0)

    data: Dict = {"call_rates": CALL_RATES}
    for algo in tqdm(ALGO_NAMES, desc="G2G3 algos"):
        # STEP 1: train ONCE on default env per seed
        for seed in range(n_seed):
            mp = _model_path("g2g3", algo, seed, "base")
            if not os.path.isfile(mp):
                print(f"  [TRAIN] {algo} seed={seed} ({n_train} eps)")
                train_and_save(algo, device, seed, 0.99, mp, n_train, _base_ef)

        # STEP 2: evaluate at each call rate (500 runs) using saved weights
        ho_m, ho_s, ee_m, ee_s = [], [], [], []
        for rate in tqdm(CALL_RATES, desc=f"  {algo} eval", leave=False):
            tload = rate / max_r

            def _eval_ef(tload=tload):
                return make_env(device, traffic_model="pareto",
                                pareto_shape=1.5, pareto_scale=1.0,
                                traffic_load=tload)

            ho_sv, ee_sv = [], []
            for seed in range(n_seed):
                mp  = _model_path("g2g3", algo, seed, "base")
                print(f"  [EVAL]  {algo} rate={rate} seed={seed} ({n_eval} runs)")
                res = load_and_evaluate(
                    algo, device, seed, 0.99, mp, n_eval, _eval_ef,
                    metric_keys=["handovers", "energy_eff"])
                ho_sv.append(res["handovers"])
                ee_sv.append(res["energy_eff"])

            ho_m.append(float(np.mean(ho_sv))); ho_s.append(float(np.std(ho_sv)))
            ee_m.append(float(np.mean(ee_sv))); ee_s.append(float(np.std(ee_sv)))

        data[algo] = {"ho_mean": ho_m, "ho_std": ho_s,
                      "ee_mean": ee_m, "ee_std": ee_s}
    return data


# ─── G4: Packet Drop Rate vs User Velocity ────────────────────────────────────

def run_g4(device: torch.device, quick: bool) -> Dict:
    print("\n=== G4: Packet Drop Rate vs User Velocity ===")
    n_train = TRAIN_EPISODES_QUICK if quick else TRAIN_EPISODES
    n_eval  = EVAL_RUNS_QUICK      if quick else EVAL_RUNS
    n_seed  = N_SEEDS_QUICK        if quick else N_SEEDS

    def _base_ef():
        return make_env(device, traffic_model="pareto",
                        pareto_shape=1.5, pareto_scale=1.0, traffic_load=0.8)

    data: Dict = {"velocities": VELOCITIES}
    for algo in tqdm(ALGO_NAMES, desc="G4 algos"):
        # STEP 1: train once
        for seed in range(n_seed):
            mp = _model_path("g4", algo, seed, "base")
            if not os.path.isfile(mp):
                print(f"  [TRAIN] {algo} seed={seed} ({n_train} eps)")
                train_and_save(algo, device, seed, 0.99, mp, n_train, _base_ef)

        # STEP 2: evaluate at each velocity (500 runs, force velocity after reset)
        pdr_m, pdr_s = [], []
        for v in tqdm(VELOCITIES, desc=f"  {algo} eval", leave=False):
            def _vel_ef(v=v):
                return make_env(device, traffic_model="pareto",
                                pareto_shape=1.5, pareto_scale=1.0,
                                traffic_load=0.8,
                                low_velocity_max=v, high_velocity_min=v)

            def _post(env, run, v=v):
                _set_all_velocity(env, v, run)

            pdr_sv = []
            for seed in range(n_seed):
                mp  = _model_path("g4", algo, seed, "base")
                print(f"  [EVAL]  {algo} vel={v} seed={seed} ({n_eval} runs)")
                res = load_and_evaluate(
                    algo, device, seed, 0.99, mp, n_eval, _vel_ef,
                    metric_keys=["pdr"], post_reset_fn=_post)
                pdr_sv.append(res["pdr"])

            pdr_m.append(float(np.mean(pdr_sv)))
            pdr_s.append(float(np.std(pdr_sv)))
        data[algo] = {"pdr_mean": pdr_m, "pdr_std": pdr_s}
    return data


# ─── G5: Packet Drop Rate vs Traffic Load ─────────────────────────────────────

def run_g5(device: torch.device, quick: bool) -> Dict:
    print("\n=== G5: Packet Drop Rate vs Traffic Load ===")
    n_train = TRAIN_EPISODES_QUICK if quick else TRAIN_EPISODES
    n_eval  = EVAL_RUNS_QUICK      if quick else EVAL_RUNS
    n_seed  = N_SEEDS_QUICK        if quick else N_SEEDS
    max_l   = float(max(TRAFFIC_LOADS))

    def _base_ef():
        return make_env(device, traffic_model="pareto",
                        pareto_shape=1.5, pareto_scale=1.0, traffic_load=1.0)

    data: Dict = {"traffic_loads_mbps": TRAFFIC_LOADS}
    for algo in tqdm(ALGO_NAMES, desc="G5 algos"):
        # STEP 1: train once
        for seed in range(n_seed):
            mp = _model_path("g5", algo, seed, "base")
            if not os.path.isfile(mp):
                print(f"  [TRAIN] {algo} seed={seed} ({n_train} eps)")
                train_and_save(algo, device, seed, 0.99, mp, n_train, _base_ef)

        # STEP 2: evaluate at each traffic load (500 runs)
        pdr_m, pdr_s = [], []
        for tmbps in tqdm(TRAFFIC_LOADS, desc=f"  {algo} eval", leave=False):
            tload = tmbps / max_l

            def _eval_ef(tload=tload):
                return make_env(device, traffic_model="pareto",
                                pareto_shape=1.5, pareto_scale=1.0,
                                traffic_load=tload)

            pdr_sv = []
            for seed in range(n_seed):
                mp  = _model_path("g5", algo, seed, "base")
                print(f"  [EVAL]  {algo} load={tmbps} seed={seed} ({n_eval} runs)")
                res = load_and_evaluate(
                    algo, device, seed, 0.99, mp, n_eval, _eval_ef,
                    metric_keys=["pdr"])
                pdr_sv.append(res["pdr"])

            pdr_m.append(float(np.mean(pdr_sv)))
            pdr_s.append(float(np.std(pdr_sv)))
        data[algo] = {"pdr_mean": pdr_m, "pdr_std": pdr_s}
    return data


# ─── G6 & G7: Training Convergence Curves ────────────────────────────────────
# Record per-episode throughput during training — plot script smooths it

def run_g6g7(device: torch.device, quick: bool) -> Dict:
    print("\n=== G6/G7: Throughput Convergence by Mobility Group ===")
    n_ep   = TRAIN_EPISODES_QUICK if quick else TRAIN_EPISODES
    n_seed = N_SEEDS_QUICK        if quick else N_SEEDS

    data: Dict = {"num_episodes": n_ep}
    for algo in tqdm(ALGO_NAMES, desc="G6G7 algos"):
        seeds_low:  List[List[float]] = []
        seeds_high: List[List[float]] = []

        for seed in tqdm(range(n_seed), desc=f"  {algo}", leave=False):
            env = make_env(device, traffic_model="pareto",
                           pareto_shape=1.5, pareto_scale=1.0, traffic_load=1.0)
            env.reset(seed=seed)
            _set_velocity_split(env, seed)

            agent = create_agent(algo, env, device, gamma=0.99, seed=seed)
            sdim  = getattr(env, "agent_obs_dim",
                            env.observation_space.shape[0] // env.num_uavs)
            gstep = 0
            ep_lo: List[float] = []
            ep_hi: List[float] = []

            for ep_idx in range(n_ep):
                obs, _ = env.reset()
                _set_velocity_split(env, seed * 100000 + ep_idx)
                obs_t = torch.tensor(obs, device=device, dtype=torch.float32)
                done  = False
                accum_lo = accum_hi = 0.0
                steps = 0

                traj_s, traj_a, traj_r, traj_ns, traj_d, traj_lp = \
                    ([] for _ in range(6)) if algo in MAPPO_FAMILY else ([None]*6)
                if algo == "ABQMIX":
                    agent.start_episode()

                while not done and steps < STEPS_PER_EP:
                    obs_np = [obs_t[i * sdim:(i + 1) * sdim].cpu().numpy()
                              for i in range(env.num_uavs)]

                    if algo == "ABQMIX":
                        ab_j    = agent.get_actions(obs_np)
                        actions = [p[0] for p in ab_j]; lps = [0.0]*env.num_uavs
                    elif algo == "DMTD":
                        actions = agent.get_actions(obs_np)
                        lps = [0.0]*env.num_uavs; ab_j = None
                    elif algo in MAPPO_FAMILY:
                        actions, lps = [], []
                        for i in range(env.num_uavs):
                            a, lp = agent.get_action(
                                obs_t[i*sdim:(i+1)*sdim], i, explore=True)
                            actions.append(a); lps.append(lp)
                        ab_j = None
                    elif algo == "MADDPG":
                        actions = [agent.get_action(
                            obs_t[i*sdim:(i+1)*sdim], i, explore=True)
                            for i in range(env.num_uavs)]
                        lps = [0.0]*env.num_uavs; ab_j = None
                    else:
                        actions = [agent.get_action(obs_t[i*sdim:(i+1)*sdim], i)
                                   for i in range(env.num_uavs)]
                        lps = [0.0]*env.num_uavs; ab_j = None

                    next_obs, reward, term, trunc, info = env.step(actions)
                    done = term or trunc

                    ur = info.get("user_rates", None)
                    if ur is not None and env.user_velocities is not None:
                        lm = env.user_velocities < env.low_velocity_max
                        hm = env.user_velocities >= env.high_velocity_min
                        accum_lo += float(np.sum(ur[lm])) if lm.any() else 0.0
                        accum_hi += float(np.sum(ur[hm])) if hm.any() else 0.0

                    next_obs_t = torch.tensor(next_obs, device=device,
                                              dtype=torch.float32)
                    st  = [obs_t[i*sdim:(i+1)*sdim]     for i in range(env.num_uavs)]
                    nst = [next_obs_t[i*sdim:(i+1)*sdim] for i in range(env.num_uavs)]
                    nob = [next_obs_t[i*sdim:(i+1)*sdim].cpu().numpy()
                           for i in range(env.num_uavs)]

                    if algo in MAPPO_FAMILY:
                        traj_s.append(st);  traj_a.append(actions)
                        traj_r.append([reward]*env.num_uavs)
                        traj_ns.append(nst); traj_d.append([done]*env.num_uavs)
                        traj_lp.append(lps)
                    elif algo == "ABQMIX":
                        agent.store_transition(
                            obs_t.flatten().cpu().numpy(), obs_np, ab_j,
                            reward, next_obs_t.flatten().cpu().numpy(), nob)
                    elif algo in QMIX_FAMILY:
                        agent.store_transition(st, actions, [reward]*env.num_uavs,
                                               nst, [done]*env.num_uavs,
                                               obs_t, next_obs_t)
                    elif algo == "DMTD":
                        agent.store_transitions(obs_np, actions,
                                                [reward]*env.num_uavs, nob,
                                                gstep + steps)
                    else:
                        agent.store_transition(st, actions, [reward]*env.num_uavs,
                                               nst, [done]*env.num_uavs)

                    if algo == "ABQMIX":
                        agent.update()
                    elif algo == "DMTD":
                        agent.update(gstep + steps)
                    elif algo not in MAPPO_FAMILY:
                        buf = getattr(agent, "replay_buffer", None)
                        if buf is not None and len(buf) > getattr(agent, "batch_size", 64):
                            if steps % 2 == 0:
                                agent.update()

                    obs_t  = next_obs_t
                    steps += 1

                if algo in MAPPO_FAMILY and traj_s:
                    for t in range(len(traj_s)):
                        agent.store_transition(traj_s[t], traj_a[t], traj_r[t],
                                               traj_ns[t], traj_d[t], traj_lp[t])
                    agent.update()

                if algo == "ABQMIX":
                    agent.end_episode()
                env.end_episode()

                n = max(steps, 1)
                ep_lo.append(accum_lo / n / 1e9)
                ep_hi.append(accum_hi / n / 1e9)
                gstep += STEPS_PER_EP

            # save final weights
            mp = _model_path("g6g7", algo, seed, "final")
            if hasattr(agent, "save"):
                try: agent.save(mp)
                except Exception as e: print(f"  [warn] {e}")

            seeds_low.append(ep_lo)
            seeds_high.append(ep_hi)

        data[algo] = {"tp_low": seeds_low, "tp_high": seeds_high}
    return data


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="Smoke-test with tiny episode/eval counts.")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated IDs: g1,g2g3,g4,g5,g6g7")
    p.add_argument("--step", type=str, default="",
                   choices=["", "train", "eval"],
                   help="Run only 'train' or 'eval' step (default: both)")
    p.add_argument("--out", type=str, default=RESULTS_PATH)
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Train episodes : {TRAIN_EPISODES_QUICK if args.quick else TRAIN_EPISODES}")
    print(f"Eval runs/param: {EVAL_RUNS_QUICK      if args.quick else EVAL_RUNS}")
    print(f"Seeds          : {N_SEEDS_QUICK        if args.quick else N_SEEDS}")

    only = set(args.only.lower().split(",")) if args.only else set()

    results: Dict = {}
    if os.path.isfile(args.out):
        with open(args.out) as fh:
            results = json.load(fh)
        print(f"Loaded existing results from {args.out}")

    def _run(key: str, fn):
        if only and key not in only:
            return
        if key in results:
            print(f"Skipping {key} — already done (delete results to re-run)")
            return
        t0 = time.time()
        results[key] = fn(device, args.quick)
        print(f"  {key} done in {time.time()-t0:.1f}s")
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  Saved → {args.out}")

    _run("g1",   run_g1)
    _run("g2g3", run_g2g3)
    _run("g4",   run_g4)
    _run("g5",   run_g5)
    _run("g6g7", run_g6g7)

    print(f"\nAll done. Results → {args.out}")
    print(f"Weights  → {os.path.abspath(MODELS_DIR)}/")


if __name__ == "__main__":
    main()
