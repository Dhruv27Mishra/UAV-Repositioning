#!/usr/bin/env python3
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
from rl_agent.AdaptiveNonStationaryMARL import AdaptiveNonStationaryMARL
from rl_agent.ab_qmix_algorithm import ABQMIX
from rl_agent.dmtd_algorithm import DMTD

# Regular-env algorithms
REGULAR_ALGOS = ["IQL", "VDN", "QMIX", "DeepNashQ", "MADDPG", "ABQMIX", "DMTD"]

# Enhanced-env algorithms (non-stationary + performative)
ENHANCED_ALGOS = ["PerformativeMFMARL", "PerformativeMARL"]

ALGO_NAMES      = REGULAR_ALGOS + ENHANCED_ALGOS
ADAPTIVE_FAMILY = frozenset({"PerformativeMFMARL", "PerformativeMARL"})
QMIX_FAMILY     = frozenset({"QMIX", "ABQMIX"})

# ── evaluation x-axis values ──────────────────────────────────────────────────
GAMMAS             = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
CALL_RATES         = [5, 10, 15, 25, 35, 50]
VELOCITIES         = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
TRAFFIC_LOADS      = [10, 25, 50, 75, 100, 125, 150]
EVAL_NUM_UES       = [5, 10, 15, 20, 25, 30, 40, 50]   # UE-count sweep (new plots)
# QoS SINR threshold sweep (dB) — converted to min_user_rate via Shannon capacity
# R_min = B·log2(1 + 10^(γ_dB/10)),  B = 10 MHz (matches marl_env.py bandwidth)
SINR_THRESHOLDS_DB = [-20, -15, -10, -5, 0, 5, 10]
_BANDWIDTH_HZ      = 10e6   # must match MARLEnv.bandwidth

# ── training randomisation pools ─────────────────────────────────────────────
TRAIN_UE_COUNTS     = [10, 15, 20, 25, 30, 40]
TRAIN_VELOCITIES    = [0.5, 1.0, 3.0, 5.0, 8.0, 10.0]
TRAIN_TRAFFIC_LOADS = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5]

# ── environment defaults ──────────────────────────────────────────────────────
NUM_UAVS     = 3
NUM_USERS    = 20
GRID_SIZE    = (10, 10, 5)
STEPS_PER_EP = 50
LOW_VEL_MAX  = 1.0
HIGH_VEL_MIN = 5.0

TRAIN_EPISODES    = 2000
EVAL_RUNS         = 500
N_SEEDS           = 5

TRAIN_EPISODES_QUICK = 50
EVAL_RUNS_QUICK      = 10
N_SEEDS_QUICK        = 2

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(_SCRIPT_DIR, "..", "models")
RESULTS_PATH = os.path.join(_SCRIPT_DIR, "..", "results", "journal_data.json")


# ─── device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── model paths (flat structure) ────────────────────────────────────────────

def _base_model_path(algo: str, seed: int) -> str:
    """models/{algo_lower}_seed{s}_final.pt"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, f"{algo.lower()}_seed{seed}_final.pt")




# ─── environment factories ────────────────────────────────────────────────────

def make_env(
    device: torch.device,
    *,
    num_users:         int   = NUM_USERS,
    traffic_model:     str   = "pareto",
    pareto_shape:      float = 1.5,
    pareto_scale:      float = 1.0,
    traffic_load:      float = 1.0,
    low_velocity_max:  float = LOW_VEL_MAX,
    high_velocity_min: float = HIGH_VEL_MIN,
    enhanced:          bool  = False,
    min_user_rate:     float = 0.5,   # Mbps — QoS threshold swept in goodness-vs-SINR
) -> MARLEnv:
    return MARLEnv(
        num_uavs=NUM_UAVS, num_users=num_users, grid_size=GRID_SIZE,
        device=device, min_user_rate=min_user_rate, qos_bonus=10.0,
        enable_non_stationary=enhanced, enable_performative=enhanced,
        enable_signal_map_obs=True,
        traffic_model=traffic_model, pareto_shape=pareto_shape,
        pareto_scale=pareto_scale, traffic_load=traffic_load,
        low_velocity_max=low_velocity_max,
        high_velocity_min=high_velocity_min,
    )


def _is_enhanced(algo: str) -> bool:
    return algo in ENHANCED_ALGOS


def _make_env_for(algo: str, device: torch.device, **kwargs) -> MARLEnv:
    return make_env(device, enhanced=_is_enhanced(algo), **kwargs)


# ─── velocity helpers ─────────────────────────────────────────────────────────

def _set_velocity_split(env: MARLEnv, seed: int) -> None:
    rng  = np.random.default_rng(seed)
    n    = env.num_users
    n_lo = n // 2
    env.user_velocities[:n_lo] = rng.uniform(0.1, LOW_VEL_MAX * 0.95, n_lo)
    env.user_velocities[n_lo:] = rng.uniform(HIGH_VEL_MIN * 1.05,
                                              HIGH_VEL_MIN * 2.0, n - n_lo)
    env.user_velocity_categories = ["LOW"] * n_lo + ["HIGH"] * (n - n_lo)


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
    if name == "QMIX":
        return QMIX(num_agents=n, state_dim=sdim, action_dim=adim,
                    global_state_dim=gdim, learning_rate=2.2e-4, gamma=gamma,
                    epsilon=0.32, device=device, buffer_size=10000,
                    batch_size=40, target_update=320)
    if name == "DeepNashQ":
        return DeepNashQ(num_agents=n, state_dim=sdim, action_dim=adim,
                         learning_rate=1e-3, gamma=gamma, epsilon=0.1,
                         device=device, buffer_size=10000, batch_size=64,
                         target_update=100)
    if name == "MADDPG":
        return MADDPG(num_agents=n, state_dim=sdim, action_dim=adim,
                      learning_rate_actor=1.8e-4, learning_rate_critic=1.8e-4,
                      gamma=gamma, device=device, buffer_size=10000,
                      batch_size=40, tau=0.004)
    if name == "ABQMIX":
        return ABQMIX(num_agents=n, obs_dim=sdim, global_state_dim=gdim,
                      traj_action_dim=adim, bf_action_dim=1, gamma=gamma,
                      learning_rate=1e-4, epsilon=1.0, epsilon_min=0.1,
                      c_min=1000, batch_size=4, device=device)
    if name == "DMTD":
        return DMTD(num_uavs=n, state_dim=sdim, action_dim=adim,
                    gamma=gamma, learning_rate=1e-3, epsilon=1.0,
                    epsilon_min=0.05, buffer_capacity=10000, batch_size=64,
                    device=device)
    if name in ADAPTIVE_FAMILY:
        agent = AdaptiveNonStationaryMARL(
            num_agents=n, state_dim=sdim, action_dim=adim,
            global_state_dim=gdim, context_dim=0,
            learning_rate=1e-3, gamma=gamma, epsilon=0.1, device=device,
            buffer_size=10000, batch_size=64, target_update=100)
        # PerformativeMFMARL uses the improved association algorithm
        if name == "PerformativeMFMARL" and hasattr(env, "set_association_function"):
            env.set_association_function(agent.get_association_function())
        return agent
    raise ValueError(f"Unknown algorithm: {name}")


def _set_greedy(agent: Any) -> None:
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0


# ─── single episode ───────────────────────────────────────────────────────────

def _run_episode(
    name:    str,
    agent:   Any,
    env:     MARLEnv,
    device:  torch.device,
    sdim:    int,
    gstep:   int  = 0,
    explore: bool = True,
    track_mobility_tp: bool = False,
    post_reset_fn=None,   # called with (env) after env.reset()
) -> Dict[str, float]:
    obs, _ = env.reset()
    if post_reset_fn is not None:
        post_reset_fn(env)

    obs_t    = torch.tensor(obs, device=device, dtype=torch.float32)
    num_uavs = env.num_uavs
    done     = False

    ep_reward = ep_tp = ep_ho = ep_ej = ep_pdr = ep_goodness = 0.0
    ep_tp_lo  = ep_tp_hi = 0.0
    steps     = 0


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
        elif name in ADAPTIVE_FAMILY:
            actions = [agent.get_action(obs_t[i * sdim:(i + 1) * sdim],
                                        i, global_state=obs_t)
                       for i in range(num_uavs)]
            lps = [0.0] * num_uavs; ab_j = None
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
        ep_tp     += float(info.get("throughput",       0.0))
        ep_ho     += int  (info.get("handovers",        0))
        ep_ej     += float(info.get("step_energy_j",    0.0))
        ep_pdr     += float(info.get("packet_drop_rate", 0.0))
        ep_goodness += float(info.get("goodness",        0.0))

        if track_mobility_tp:
            ur = info.get("user_rates", None)
            if ur is not None and env.user_velocities is not None:
                lm = env.user_velocities <  env.low_velocity_max
                hm = env.user_velocities >= env.high_velocity_min
                ep_tp_lo += float(np.sum(ur[lm])) if lm.any() else 0.0
                ep_tp_hi += float(np.sum(ur[hm])) if hm.any() else 0.0

        next_obs_t = torch.tensor(next_obs, device=device, dtype=torch.float32)
        st  = [obs_t[i * sdim:(i + 1) * sdim]           for i in range(num_uavs)]
        nst = [next_obs_t[i * sdim:(i + 1) * sdim]       for i in range(num_uavs)]
        nob = [next_obs_t[i * sdim:(i + 1) * sdim].cpu().numpy()
               for i in range(num_uavs)]

        if explore:
            if name in ADAPTIVE_FAMILY:
                agent.store_transition(st, actions, [reward] * num_uavs,
                                       nst, [done] * num_uavs, obs_t, next_obs_t)
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
            else:
                buf = getattr(agent, "replay_buffer", None)
                if buf is not None and len(buf) > getattr(agent, "batch_size", 64):
                    if steps % 2 == 0:
                        agent.update()

        obs_t  = next_obs_t
        steps += 1

    if name == "ABQMIX":
        agent.end_episode()
    env.end_episode()

    n  = max(steps, 1)
    ee = (ep_tp / 1e6) / (ep_ej + 1e-9)
    return {
        "reward":       ep_reward,
        "throughput":   ep_tp      / n,
        "handovers":    ep_ho,
        "energy_eff":   ee,
        "pdr":          ep_pdr     / n * 100.0,
        "goodness":     ep_goodness / n,
        "tp_low_gbps":  ep_tp_lo   / n / 1e9,
        "tp_high_gbps": ep_tp_hi   / n / 1e9,
    }


# ─── STEP 1: train one algorithm, vary conditions each episode ────────────────

def train_and_save(
    algo:       str,
    device:     torch.device,
    seed:       int,
    gamma:      float,
    model_path: str,
    n_train:    int,
) -> None:
    """
    Train algo for n_train episodes with randomised UEs/velocity/traffic.
    Save weights to model_path.
    """
    env   = _make_env_for(algo, device)
    env.reset(seed=seed)
    agent = create_agent(algo, env, device, gamma=gamma, seed=seed)
    sdim  = getattr(env, "agent_obs_dim",
                    env.observation_space.shape[0] // env.num_uavs)
    rng   = np.random.default_rng(seed + 12345)
    gstep = 0

    for ep in range(n_train):
        env.num_users    = int(rng.choice(TRAIN_UE_COUNTS))
        env.traffic_load = float(rng.choice(TRAIN_TRAFFIC_LOADS))
        vel              = float(rng.choice(TRAIN_VELOCITIES))

        def _post(e, _vel=vel, _s=seed * 100000 + ep):
            _set_all_velocity(e, _vel, _s)

        _run_episode(algo, agent, env, device, sdim,
                     gstep=gstep, explore=True, post_reset_fn=_post)
        gstep += STEPS_PER_EP

    if hasattr(agent, "save"):
        try:
            agent.save(model_path)
            print(f"    Saved -> {model_path}")
        except Exception as e:
            print(f"  [warn] save failed for {algo}: {e}")


# ─── STEP 1 (full): train all algorithms, all seeds ──────────────────────────

def train_all(device: torch.device, n_train: int, n_seed: int) -> None:
    """
    Train every algorithm n_seed times.
    Skip any seed whose model file already exists.
    """
    print("\n" + "="*60)
    print("STEP 1 — TRAINING ALL ALGORITHMS")
    print("="*60)
    for algo in tqdm(ALGO_NAMES, desc="Algorithms"):
        env_type = "enhanced" if _is_enhanced(algo) else "regular"
        print(f"\n[{algo}]  env={env_type}  episodes={n_train}  seeds={n_seed}")
        for seed in range(n_seed):
            mp = _base_model_path(algo, seed)
            if os.path.isfile(mp):
                print(f"  seed={seed} — already trained, skipping")
                continue
            print(f"  [TRAIN] seed={seed} ...")
            train_and_save(algo, device, seed, gamma=0.99, model_path=mp,
                           n_train=n_train)


# ─── STEP 2 helper: load frozen agent ────────────────────────────────────────

def _load_frozen_agent(
    algo:       str,
    device:     torch.device,
    seed:       int,
    gamma:      float,
    model_path: str,
) -> tuple:
    """Return (agent, sdim) with weights loaded and epsilon=0."""
    env   = _make_env_for(algo, device)
    agent = create_agent(algo, env, device, gamma=gamma, seed=seed)
    sdim  = getattr(env, "agent_obs_dim",
                    env.observation_space.shape[0] // env.num_uavs)
    if os.path.isfile(model_path) and hasattr(agent, "load"):
        try:
            agent.load(model_path)
        except Exception as e:
            print(f"  [warn] load failed ({algo} seed={seed}): {e}")
    else:
        print(f"  [warn] no weights found at {model_path} — using untrained agent")
    _set_greedy(agent)
    return agent, sdim, env


def _evaluate(
    algo:        str,
    device:      torch.device,
    seed:        int,
    gamma:       float,
    model_path:  str,
    n_eval:      int,
    env_factory,
    metric_keys: List[str],
    post_reset_fn=None,
) -> Dict[str, float]:
    """Load frozen weights, run n_eval greedy episodes, return averaged metrics."""
    env   = env_factory()
    agent = create_agent(algo, env, device, gamma=gamma, seed=seed)
    sdim  = getattr(env, "agent_obs_dim",
                    env.observation_space.shape[0] // env.num_uavs)

    if os.path.isfile(model_path) and hasattr(agent, "load"):
        try:
            agent.load(model_path)
        except Exception as e:
            print(f"  [warn] load failed ({algo} seed={seed}): {e}")
    else:
        print(f"  [warn] no weights at {model_path} — using untrained agent")
    _set_greedy(agent)

    accum: Dict[str, List[float]] = {k: [] for k in metric_keys}
    gstep = 0
    for run in range(n_eval):
        def _prf(e, _run=run, _fn=post_reset_fn):
            if _fn is not None:
                _fn(e, _run)
        m = _run_episode(algo, agent, env, device, sdim,
                         gstep=gstep, explore=False, post_reset_fn=_prf)
        for k in metric_keys:
            accum[k].append(m[k])
        gstep += STEPS_PER_EP

    return {k: float(np.mean(accum[k])) for k in metric_keys}


# ─── G1: Energy Efficiency vs Discount Factor ─────────────────────────────────
# Uses base trained models — evaluates energy efficiency at each gamma value.

def run_g1(device: torch.device, quick: bool) -> Dict:
    print("\n=== G1: Energy Efficiency vs Discount Factor ===")
    n_eval  = EVAL_RUNS_QUICK      if quick else EVAL_RUNS
    n_seed  = N_SEEDS_QUICK        if quick else N_SEEDS

    data: Dict = {"gammas": GAMMAS}
    for algo in tqdm(ALGO_NAMES, desc="G1 algos"):
        means, stds = [], []
        for g in tqdm(GAMMAS, desc=f"  {algo} gamma sweep", leave=False):
            seed_vals = []
            for seed in range(n_seed):
                mp = _base_model_path(algo, seed)

                def _ef():
                    return _make_env_for(algo, device, traffic_load=1.0)

                print(f"  [EVAL]  {algo} gamma={g} seed={seed} ({n_eval} runs)")
                res = _evaluate(algo, device, seed, g, mp, n_eval, _ef,
                                metric_keys=["energy_eff"])
                seed_vals.append(res["energy_eff"])

            means.append(float(np.mean(seed_vals)))
            stds.append(float(np.std(seed_vals)))
        data[algo] = {"mean": means, "std": stds}
    return data


# ─── G2 & G3: Handover Rate & Energy Efficiency vs Call Arrival Rate ──────────

def run_g2g3(device: torch.device, quick: bool) -> Dict:
    print("\n=== G2/G3: Handover Rate & Energy Efficiency vs Call Arrival Rate ===")
    n_eval  = EVAL_RUNS_QUICK if quick else EVAL_RUNS
    n_seed  = N_SEEDS_QUICK   if quick else N_SEEDS
    max_r   = float(max(CALL_RATES))

    data: Dict = {"call_rates": CALL_RATES}
    for algo in tqdm(ALGO_NAMES, desc="G2G3 algos"):
        ho_m, ho_s, ee_m, ee_s = [], [], [], []
        for rate in tqdm(CALL_RATES, desc=f"  {algo} eval", leave=False):
            tload = rate / max_r

            def _ef(tload=tload):
                return _make_env_for(algo, device, traffic_load=tload)

            ho_sv, ee_sv = [], []
            for seed in range(n_seed):
                mp = _base_model_path(algo, seed)
                print(f"  [EVAL]  {algo} rate={rate} seed={seed} ({n_eval} runs)")
                res = _evaluate(algo, device, seed, 0.99, mp, n_eval, _ef,
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
    n_eval = EVAL_RUNS_QUICK if quick else EVAL_RUNS
    n_seed = N_SEEDS_QUICK   if quick else N_SEEDS

    data: Dict = {"velocities": VELOCITIES}
    for algo in tqdm(ALGO_NAMES, desc="G4 algos"):
        pdr_m, pdr_s = [], []
        for v in tqdm(VELOCITIES, desc=f"  {algo} eval", leave=False):

            def _ef(v=v):
                return _make_env_for(algo, device, traffic_load=0.8,
                                     low_velocity_max=v, high_velocity_min=v)

            def _post(env, run, v=v):
                _set_all_velocity(env, v, run)

            pdr_sv = []
            for seed in range(n_seed):
                mp = _base_model_path(algo, seed)
                print(f"  [EVAL]  {algo} vel={v} seed={seed} ({n_eval} runs)")
                res = _evaluate(algo, device, seed, 0.99, mp, n_eval, _ef,
                                metric_keys=["pdr"], post_reset_fn=_post)
                pdr_sv.append(res["pdr"])

            pdr_m.append(float(np.mean(pdr_sv)))
            pdr_s.append(float(np.std(pdr_sv)))
        data[algo] = {"pdr_mean": pdr_m, "pdr_std": pdr_s}
    return data


# ─── G5: Packet Drop Rate vs Traffic Load ─────────────────────────────────────

def run_g5(device: torch.device, quick: bool) -> Dict:
    print("\n=== G5: Packet Drop Rate vs Traffic Load ===")
    n_eval = EVAL_RUNS_QUICK if quick else EVAL_RUNS
    n_seed = N_SEEDS_QUICK   if quick else N_SEEDS
    max_l  = float(max(TRAFFIC_LOADS))

    data: Dict = {"traffic_loads_mbps": TRAFFIC_LOADS}
    for algo in tqdm(ALGO_NAMES, desc="G5 algos"):
        pdr_m, pdr_s = [], []
        for tmbps in tqdm(TRAFFIC_LOADS, desc=f"  {algo} eval", leave=False):
            tload = tmbps / max_l

            def _ef(tload=tload):
                return _make_env_for(algo, device, traffic_load=tload)

            pdr_sv = []
            for seed in range(n_seed):
                mp = _base_model_path(algo, seed)
                print(f"  [EVAL]  {algo} load={tmbps} seed={seed} ({n_eval} runs)")
                res = _evaluate(algo, device, seed, 0.99, mp, n_eval, _ef,
                                metric_keys=["pdr"])
                pdr_sv.append(res["pdr"])

            pdr_m.append(float(np.mean(pdr_sv)))
            pdr_s.append(float(np.std(pdr_sv)))
        data[algo] = {"pdr_mean": pdr_m, "pdr_std": pdr_s}
    return data


# ─── G6 & G7: Convergence Curves (LOW vs HIGH mobility) ──────────────────────
# Records per-episode throughput during a dedicated training run with a fixed
# LOW/HIGH velocity split so both groups can be tracked separately.
# Also randomises num_users and traffic_load for robustness.

def run_g6g7(device: torch.device, quick: bool) -> Dict:
    print("\n=== G6/G7: Throughput Convergence by Mobility Group ===")
    n_ep   = TRAIN_EPISODES_QUICK if quick else TRAIN_EPISODES
    n_seed = N_SEEDS_QUICK        if quick else N_SEEDS

    data: Dict = {"num_episodes": n_ep}
    for algo in tqdm(ALGO_NAMES, desc="G6G7 algos"):
        seeds_low:  List[List[float]] = []
        seeds_high: List[List[float]] = []

        for seed in tqdm(range(n_seed), desc=f"  {algo}", leave=False):
            rng = np.random.default_rng(seed + 77777)

            env   = _make_env_for(algo, device, traffic_load=1.0)
            env.reset(seed=seed)
            _set_velocity_split(env, seed)

            agent = create_agent(algo, env, device, gamma=0.99, seed=seed)
            sdim  = getattr(env, "agent_obs_dim",
                            env.observation_space.shape[0] // env.num_uavs)
            gstep = 0
            ep_lo: List[float] = []
            ep_hi: List[float] = []

            for ep_idx in range(n_ep):
                env.num_users    = int(rng.choice(TRAIN_UE_COUNTS))
                env.traffic_load = float(rng.choice(TRAIN_TRAFFIC_LOADS))

                obs, _ = env.reset()
                _set_velocity_split(env, seed * 100000 + ep_idx)
                obs_t = torch.tensor(obs, device=device, dtype=torch.float32)
                done  = False
                accum_lo = accum_hi = 0.0
                steps = 0

                if algo == "ABQMIX":
                    agent.start_episode()

                while not done and steps < STEPS_PER_EP:
                    obs_np = [obs_t[i * sdim:(i + 1) * sdim].cpu().numpy()
                              for i in range(env.num_uavs)]

                    if algo == "ABQMIX":
                        ab_j    = agent.get_actions(obs_np)
                        actions = [p[0] for p in ab_j]
                    elif algo == "DMTD":
                        actions = agent.get_actions(obs_np); ab_j = None
                    elif algo in ADAPTIVE_FAMILY:
                        actions = [agent.get_action(obs_t[i*sdim:(i+1)*sdim],
                                                    i, global_state=obs_t)
                                   for i in range(env.num_uavs)]
                        ab_j = None
                    elif algo == "MADDPG":
                        actions = [agent.get_action(
                            obs_t[i*sdim:(i+1)*sdim], i, explore=True)
                            for i in range(env.num_uavs)]
                        ab_j = None
                    else:
                        actions = [agent.get_action(obs_t[i*sdim:(i+1)*sdim], i)
                                   for i in range(env.num_uavs)]
                        ab_j = None

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
                    st  = [obs_t[i*sdim:(i+1)*sdim]      for i in range(env.num_uavs)]
                    nst = [next_obs_t[i*sdim:(i+1)*sdim]  for i in range(env.num_uavs)]
                    nob = [next_obs_t[i*sdim:(i+1)*sdim].cpu().numpy()
                           for i in range(env.num_uavs)]

                    if algo in ADAPTIVE_FAMILY:
                        agent.store_transition(st, actions, [reward]*env.num_uavs,
                                               nst, [done]*env.num_uavs,
                                               obs_t, next_obs_t)
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
                    else:
                        buf = getattr(agent, "replay_buffer", None)
                        if buf is not None and len(buf) > getattr(agent, "batch_size", 64):
                            if steps % 2 == 0:
                                agent.update()

                    obs_t  = next_obs_t
                    steps += 1

                if algo == "ABQMIX":
                    agent.end_episode()
                env.end_episode()

                n = max(steps, 1)
                ep_lo.append(accum_lo / n / 1e9)
                ep_hi.append(accum_hi / n / 1e9)
                gstep += STEPS_PER_EP

            seeds_low.append(ep_lo)
            seeds_high.append(ep_hi)

        data[algo] = {"tp_low": seeds_low, "tp_high": seeds_high}
    return data


# ─── G8: Energy Efficiency, PDR & Goodness vs. Number of UEs ─────────────────
# Sweeps EVAL_NUM_UES; loads saved checkpoints (base model, gamma=0.99).
# PDR = Σ_k max(0, d_k − r_k) / Σ_k d_k  (fraction of unserved demand, in %)
# Goodness = 0.5·QoS_ratio + 0.3·Jain_fairness + 0.2·mean_rate_Mbps  (env def.)

def run_g_ue_sweep(device: torch.device, quick: bool) -> Dict:
    print("\n=== G_UE: Energy Efficiency / PDR / Goodness vs. Number of UEs ===")
    n_eval = EVAL_RUNS_QUICK if quick else EVAL_RUNS
    n_seed = N_SEEDS_QUICK   if quick else N_SEEDS

    data: Dict = {"num_ues": EVAL_NUM_UES}
    for algo in tqdm(ALGO_NAMES, desc="G_UE algos"):
        ee_m,  ee_s,  ee_seeds  = [], [], []
        pdr_m, pdr_s, pdr_seeds = [], [], []
        gd_m,  gd_s,  gd_seeds  = [], [], []

        for n_ue in tqdm(EVAL_NUM_UES, desc=f"  {algo} UE sweep", leave=False):

            def _ef(n=n_ue):
                return _make_env_for(algo, device, num_users=n, traffic_load=1.0)

            ee_sv, pdr_sv, gd_sv = [], [], []
            for seed in range(n_seed):
                mp = _base_model_path(algo, seed)
                print(f"  [EVAL]  {algo} num_ue={n_ue} seed={seed} ({n_eval} runs)")
                res = _evaluate(algo, device, seed, 0.99, mp, n_eval, _ef,
                                metric_keys=["energy_eff", "pdr", "goodness"])
                ee_sv.append(res["energy_eff"])
                pdr_sv.append(res["pdr"])
                gd_sv.append(res["goodness"])

            ee_m.append(float(np.mean(ee_sv)));   ee_s.append(float(np.std(ee_sv)))
            pdr_m.append(float(np.mean(pdr_sv))); pdr_s.append(float(np.std(pdr_sv)))
            gd_m.append(float(np.mean(gd_sv)));   gd_s.append(float(np.std(gd_sv)))
            ee_seeds.append([float(v) for v in ee_sv])
            pdr_seeds.append([float(v) for v in pdr_sv])
            gd_seeds.append([float(v) for v in gd_sv])

        data[algo] = {
            "ee_mean":    ee_m,     "ee_std":    ee_s,   "ee_seeds":   ee_seeds,
            "pdr_mean":   pdr_m,    "pdr_std":   pdr_s,  "pdr_seeds":  pdr_seeds,
            "good_mean":  gd_m,     "good_std":  gd_s,   "good_seeds": gd_seeds,
        }
    return data


# ─── G9: Packet Drop Rate vs. Packet Arrival Rate ────────────────────────────
# Reuses CALL_RATES x-axis (same as G2/G3) but collects PDR instead of HO/EE.
# traffic_load = call_rate / max(CALL_RATES) normalises to [0,1] for the env.

def run_g_pdr_arrival(device: torch.device, quick: bool) -> Dict:
    print("\n=== G_PDR_ARR: Packet Drop Rate vs. Packet Arrival Rate ===")
    n_eval = EVAL_RUNS_QUICK if quick else EVAL_RUNS
    n_seed = N_SEEDS_QUICK   if quick else N_SEEDS
    max_r  = float(max(CALL_RATES))

    data: Dict = {"call_rates": CALL_RATES}
    for algo in tqdm(ALGO_NAMES, desc="G_PDR_ARR algos"):
        pdr_m, pdr_s, pdr_seeds = [], [], []

        for rate in tqdm(CALL_RATES, desc=f"  {algo} eval", leave=False):
            tload = rate / max_r

            def _ef(tload=tload):
                return _make_env_for(algo, device, traffic_load=tload)

            pdr_sv = []
            for seed in range(n_seed):
                mp = _base_model_path(algo, seed)
                print(f"  [EVAL]  {algo} rate={rate} seed={seed} ({n_eval} runs)")
                res = _evaluate(algo, device, seed, 0.99, mp, n_eval, _ef,
                                metric_keys=["pdr"])
                pdr_sv.append(res["pdr"])

            pdr_m.append(float(np.mean(pdr_sv)))
            pdr_s.append(float(np.std(pdr_sv)))
            pdr_seeds.append([float(v) for v in pdr_sv])

        data[algo] = {"pdr_mean": pdr_m, "pdr_std": pdr_s, "pdr_seeds": pdr_seeds}
    return data


# ─── G10: Goodness vs. QoS SINR Threshold ────────────────────────────────────
# Sweeps SINR_THRESHOLDS_DB.  Each threshold is converted to a min_user_rate
# (Mbps) via Shannon capacity: R_min = B·log2(1 + 10^(γ_dB/10)), B = 10 MHz.
# Higher SINR threshold -> harder to satisfy -> lower goodness expected.

def run_g_goodness_sinr(device: torch.device, quick: bool) -> Dict:
    print("\n=== G_GOOD_SINR: Goodness vs. QoS SINR Threshold ===")
    n_eval = EVAL_RUNS_QUICK if quick else EVAL_RUNS
    n_seed = N_SEEDS_QUICK   if quick else N_SEEDS

    # Shannon: R_min (Mbps) at each SINR threshold, with B = _BANDWIDTH_HZ
    min_rates_mbps: List[float] = [
        float(_BANDWIDTH_HZ * np.log2(1.0 + 10.0 ** (s / 10.0)) / 1e6)
        for s in SINR_THRESHOLDS_DB
    ]

    data: Dict = {
        "sinr_thresholds_db":  SINR_THRESHOLDS_DB,
        "min_user_rates_mbps": min_rates_mbps,
    }

    for algo in tqdm(ALGO_NAMES, desc="G_GOOD_SINR algos"):
        gd_m, gd_s, gd_seeds = [], [], []

        for sinr_db, mr in zip(SINR_THRESHOLDS_DB, min_rates_mbps):

            def _ef(mr=mr):
                return _make_env_for(algo, device, min_user_rate=mr, traffic_load=1.0)

            gd_sv = []
            for seed in range(n_seed):
                mp = _base_model_path(algo, seed)
                print(f"  [EVAL]  {algo} sinr={sinr_db:+d}dB seed={seed} ({n_eval} runs)")
                res = _evaluate(algo, device, seed, 0.99, mp, n_eval, _ef,
                                metric_keys=["goodness"])
                gd_sv.append(res["goodness"])

            gd_m.append(float(np.mean(gd_sv)))
            gd_s.append(float(np.std(gd_sv)))
            gd_seeds.append([float(v) for v in gd_sv])

        data[algo] = {"good_mean": gd_m, "good_std": gd_s, "good_seeds": gd_seeds}
    return data


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="Smoke-test with tiny episode/eval counts.")
    p.add_argument("--only", type=str, default="",
                   help="train | g1 | g2g3 | g4 | g5 | g6g7 | "
                        "g_ue_sweep | g_pdr_arrival | g_goodness_sinr  (comma-separated)")
    p.add_argument("--out", type=str, default=RESULTS_PATH)
    args = p.parse_args()

    device  = get_device()
    n_train = TRAIN_EPISODES_QUICK if args.quick else TRAIN_EPISODES
    n_eval  = EVAL_RUNS_QUICK      if args.quick else EVAL_RUNS
    n_seed  = N_SEEDS_QUICK        if args.quick else N_SEEDS

    print(f"Device         : {device}")
    print(f"Train episodes : {n_train}")
    print(f"Eval runs/pt   : {n_eval}")
    print(f"Seeds          : {n_seed}")
    print(f"Models dir     : {os.path.abspath(MODELS_DIR)}/")

    only = set(args.only.lower().split(",")) if args.only else set()

    # ── STEP 1: train all base models ─────────────────────────────────────────
    if not only or "train" in only:
        train_all(device, n_train, n_seed)

    # ── STEP 2: graph evaluations ──────────────────────────────────────────────
    results: Dict = {}
    if os.path.isfile(args.out):
        with open(args.out) as fh:
            results = json.load(fh)
        print(f"\nLoaded existing results from {args.out}")

    def _run(key: str, fn):
        if only and key not in only:
            return
        if key in results:
            print(f"Skipping {key} — already in results (delete key to re-run)")
            return
        t0 = time.time()
        results[key] = fn(device, args.quick)
        print(f"  {key} done in {time.time()-t0:.1f}s")
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  Saved -> {args.out}")

    _run("g1",             run_g1)
    _run("g2g3",           run_g2g3)
    _run("g4",             run_g4)
    _run("g5",             run_g5)
    _run("g6g7",           run_g6g7)
    _run("g_ue_sweep",     run_g_ue_sweep)
    _run("g_pdr_arrival",  run_g_pdr_arrival)
    _run("g_goodness_sinr", run_g_goodness_sinr)

    print(f"\nAll done.")
    print(f"Results -> {args.out}")
    print(f"Weights -> {os.path.abspath(MODELS_DIR)}/")

if __name__ == "__main__":
    main()
