"""
Experiment Analysis — UAV Repositioning with Transaction-Paper Baselines.

Compares proposed algorithms (PerformativeMFMARL, PerformativeMARL) against
baselines from the transaction paper: QMIX, AB-QMIX, MADDPG, DeepNashQ, DMTD, IQL.

Two output figures (publication style, smooth lines only, 300 DPI PNG):

  Figure 1  –  experiment_gamma_arrival.png  (2×2)
    (a) Energy efficiency vs discount factor γ  [QMIX, parametric]
    (b) Handover switching — all algorithms under Pareto VBR traffic
    (c) Energy efficiency  — all algorithms under Pareto VBR traffic
    (d) Packet drop rate   — all algorithms, LOW (solid) vs HIGH (dashed) mobility

  Figure 2  –  experiment_pdr_velocity.png  (1×2)
    (a) PDR — Low mobility users (v < 1 m/s) — all algorithms
    (b) PDR — High mobility users (v ≥ 5 m/s) — all algorithms

─────────────────────────────────────────────────────────────────────────────
Pareto VBR Traffic Model
─────────────────────────────────────────────────────────────────────────────
  Per-user call demand at each slot (Pareto Type-I, minimum 1):

      d_i = scale × (X_i + 1) × load   [bps]

  where  X_i ~ Pareto(α)  via numpy (np.random.pareto(α) returns Y
  such that Y+1 is Pareto Type-I with shape α and x_min = 1).

  Parameters: α = 1.5 (shape / tail index), scale = 1 Mbps, load ∈ [0.5,6].

  Mean demand per user:
      E[d_i] = scale × load × α/(α−1) = scale × load × 3.0  [bps]

─────────────────────────────────────────────────────────────────────────────
Call (Packet) Drop Rate Formula
─────────────────────────────────────────────────────────────────────────────
  At each step, for each user i:
      dropped_i = max(0,  d_i − r_i)

  where r_i is the channel-capacity-limited served rate (bps).

  Step-level CDR:
      CDR = Σ_i dropped_i  /  Σ_i d_i           ∈ [0, 1]

  Per-mobility-category CDR (reported separately for LOW and HIGH):
      CDR_LOW  = Σ_{i: v_i < 1}  dropped_i  /  Σ_{i: v_i < 1}  d_i
      CDR_HIGH = Σ_{i: v_i ≥ 5}  dropped_i  /  Σ_{i: v_i ≥ 5}  d_i

Usage (from repo root):
    python scripts/experiment_analysis.py
"""

import repo_paths  # noqa: F401

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from rl_agent.marl_env       import MARLEnv
from rl_agent.QMIX           import QMIX
from rl_agent.IQL            import IQL
from rl_agent.VDN            import VDN
from rl_agent.MADDPG         import MADDPG
from rl_agent.DeepNashQ      import DeepNashQ
from rl_agent.ab_qmix_algorithm import ABQMIX
from rl_agent.dmtd_algorithm    import DMTD
from rl_agent.MAPPO             import MAPPO
from publication_marl_plots  import apply_publication_style

# ─────────────────────────── configuration ──────────────────────────────────
NUM_UAVS   = 7
NUM_USERS  = 15
GRID_SIZE  = (10, 10, 5)
EPISODES   = 2000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'figures')

# 15 % of episodes → smooth without over-flattening
SMOOTH_W = max(50, int(EPISODES * 0.15))   # = 300 for 2000 episodes

# Okabe-Ito colorblind-safe palette (same as publication_marl_plots.py)
_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Algorithm display order — all 8 transaction-paper algorithms
ALGO_ORDER = [
    'QMIX',
    'AB-QMIX',
    'MADDPG',
    'DeepNashQ',
    'PerformativeMFMARL',
    'PerformativeMARL',
    'DMTD',
    'IQL',
    'VDN',
]

# Proposed algorithms get thicker lines
PROPOSED_ALGOS = {'PerformativeMFMARL', 'PerformativeMARL'}
ALGO_COLORS    = {name: _COLORS[i % len(_COLORS)] for i, name in enumerate(ALGO_ORDER)}
THICK_LW       = 1.8
PROPOSED_LW    = 2.6

# Gamma sweep values (Experiment 1, QMIX only)
GAMMAS = [0.7, 0.8, 0.9, 0.95, 0.99]


# ─────────────────────────── smoothing helper ────────────────────────────────

def smooth(y, w=SMOOTH_W):
    """Moving-average with edge-padding — no raw trace, no band."""
    y = np.asarray(y, dtype=np.float64)
    if w <= 1 or len(y) < w:
        return y
    pad = w // 2
    yp  = np.pad(y, (pad, pad), mode='edge')
    return np.convolve(yp, np.ones(w) / w, mode='valid')[:len(y)]


def _xticks(n):
    ticks = {1, n}
    for s in (500, 1000, 1500):
        if s < n:
            ticks.add(s)
    return sorted(ticks)


def set_xaxis(ax, n=EPISODES):
    ax.set_xlim(1, n)
    ax.set_xlabel("Training episodes")
    ax.set_xticks(_xticks(n))


# ─────────────────────────── env / agent factory ─────────────────────────────

def make_env(pareto_scale=1.0, traffic_load=1.0,
             traffic_model='pareto', pareto_shape=1.5):
    """Shared env constructor for all experiments."""
    return MARLEnv(
        num_uavs=NUM_UAVS,
        num_users=NUM_USERS,
        grid_size=GRID_SIZE,
        device=DEVICE,
        enable_non_stationary=False,
        enable_performative=False,
        enable_signal_map_obs=False,
        traffic_model=traffic_model,
        pareto_shape=pareto_shape,
        pareto_scale=pareto_scale,
        traffic_load=traffic_load,
        low_velocity_max=1.0,
        high_velocity_min=5.0,
    )


def make_agent(algo_name, env, gamma=0.99):
    """Construct the correct agent for each algorithm."""
    sd = env.agent_obs_dim
    ad = int(env.action_space.nvec[0])
    gs = int(env.observation_space.shape[0])

    if algo_name == 'QMIX':
        return QMIX(
            num_agents=NUM_UAVS, state_dim=sd, action_dim=ad,
            global_state_dim=gs, gamma=gamma,
            learning_rate=0.001, epsilon=0.1,
            buffer_size=10000, batch_size=64, target_update=100,
            device=DEVICE,
        )
    elif algo_name == 'IQL':
        return IQL(
            num_agents=NUM_UAVS, state_dim=sd, action_dim=ad,
            gamma=gamma, learning_rate=0.001, epsilon=0.1,
            buffer_size=10000, batch_size=64, target_update=100,
            device=DEVICE,
        )
    elif algo_name == 'MADDPG':
        return MADDPG(
            num_agents=NUM_UAVS, state_dim=sd, action_dim=ad,
            gamma=gamma, learning_rate_actor=0.001,
            learning_rate_critic=0.001, tau=0.01,
            buffer_size=10000, batch_size=64,
            device=DEVICE,
        )
    elif algo_name == 'DeepNashQ':
        return DeepNashQ(
            num_agents=NUM_UAVS, state_dim=sd, action_dim=ad,
            gamma=gamma, learning_rate=0.001, epsilon=0.1,
            buffer_size=10000, batch_size=64, target_update=100,
            device=DEVICE,
        )
    elif algo_name == 'VDN':
        return VDN(
            num_agents=NUM_UAVS, state_dim=sd, action_dim=ad,
            gamma=gamma, learning_rate=0.001, epsilon=0.1,
            buffer_size=10000, batch_size=64, target_update=100,
            device=DEVICE,
        )
    elif algo_name == 'AB-QMIX':
        # traj_action_dim = env action space; bf_action_dim = 1 (no beamforming)
        return ABQMIX(
            num_agents=NUM_UAVS,
            obs_dim=sd,
            global_state_dim=gs,
            traj_action_dim=ad,
            bf_action_dim=1,
            gamma=gamma,
            learning_rate=1e-4,
            epsilon=1.0,
            epsilon_min=0.1,
            c_min=1000,           # lower threshold so learning starts earlier
            batch_size=4,
            device=DEVICE,
        )
    elif algo_name == 'DMTD':
        return DMTD(
            num_uavs=NUM_UAVS,
            state_dim=sd,
            action_dim=ad,
            gamma=gamma,
            learning_rate=1e-3,
            epsilon=1.0,
            epsilon_min=0.05,
            buffer_capacity=10000,
            batch_size=64,
            device=DEVICE,
        )
    elif algo_name in ('PerformativeMFMARL', 'PerformativeMARL'):
        return MAPPO(
            num_agents=NUM_UAVS,
            state_dim=sd,
            action_dim=ad,
            gamma=gamma,
            learning_rate_actor=0.0003,
            learning_rate_critic=0.001,
            device=DEVICE,
        )
    raise ValueError(f"Unknown algorithm: {algo_name}")


# ─────────────────────────── training loop ───────────────────────────────────

def run_training(env, agent, algo_name, num_episodes=EPISODES, desc=''):
    """
    Train agent on env for num_episodes.
    Returns dict of per-episode numpy arrays: ee, ho, pdr, pdr_low, pdr_high.

    Handles each algorithm's unique interface:
      QMIX          — store_transition needs global_state args
      AB-QMIX       — get_actions returns [[traj,bf],...]; episode-level replay;
                       store_transition(global_state, obs, actions, reward, ng, no)
      DMTD          — get_actions(np_states_list); store_transitions(…, t); update(t)
      MAPPO         — get_action returns (action, log_prob); store_transition needs log_probs
      MADDPG        — get_action needs explore kwarg
      IQL/DeepNashQ — standard interface
    """
    sd  = env.agent_obs_dim
    is_abqmix  = (algo_name == 'AB-QMIX')
    is_dmtd    = (algo_name == 'DMTD')
    is_mappo   = (algo_name in ('PerformativeMFMARL', 'PerformativeMARL'))
    is_maddpg  = (algo_name == 'MADDPG')
    is_qmix    = (algo_name == 'QMIX')

    ee_ep, ho_ep, pdr_ep, pdrl_ep, pdrh_ep = [], [], [], [], []
    global_step = 0  # cumulative step counter for DMTD

    for ep in tqdm(range(num_episodes), desc=desc or algo_name, leave=False):
        obs, _ = env.reset()
        done   = False
        ep_ee, ep_ho = [], 0
        ep_pdr, ep_pdrl, ep_pdrh = [], [], []

        obs_t = torch.tensor(obs, device=DEVICE, dtype=torch.float32)
        obs_np = [obs[i * sd:(i + 1) * sd] for i in range(NUM_UAVS)]

        if is_abqmix:
            agent.start_episode()

        step = 0
        while not done:
            # ── actions ──────────────────────────────────────────────────────
            if is_abqmix:
                # get_actions returns [[traj_a, bf_a], ...] — use traj_a for env
                joint = agent.get_actions(obs_np)
                actions     = [pair[0] for pair in joint]  # traj actions → env
                raw_actions = joint                         # full pair stored
            elif is_dmtd:
                actions = agent.get_actions(obs_np)
                raw_actions = actions
            elif is_mappo:
                explore  = (ep < num_episodes * 0.8)
                actions, log_probs = [], []
                for i in range(NUM_UAVS):
                    a, lp = agent.get_action(obs_t[i * sd:(i + 1) * sd], i,
                                             explore=explore)
                    actions.append(int(a))
                    log_probs.append(lp)
                raw_actions = actions
            elif is_maddpg:
                explore = (ep < num_episodes * 0.8)
                actions = [int(agent.get_action(obs_t[i * sd:(i + 1) * sd],
                                                i, explore=explore))
                           for i in range(NUM_UAVS)]
                raw_actions = actions
            else:
                # QMIX, IQL, DeepNashQ
                actions = [int(agent.get_action(obs_t[i * sd:(i + 1) * sd], i))
                           for i in range(NUM_UAVS)]
                raw_actions = actions

            nobs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            nt   = torch.tensor(nobs, device=DEVICE, dtype=torch.float32)
            nobs_np = [nobs[i * sd:(i + 1) * sd] for i in range(NUM_UAVS)]

            # ── store transition ──────────────────────────────────────────────
            states      = [obs_t[i * sd:(i + 1) * sd] for i in range(NUM_UAVS)]
            next_states = [nt[i * sd:(i + 1) * sd]    for i in range(NUM_UAVS)]

            if is_qmix:
                agent.store_transition(
                    states, actions, [reward] * NUM_UAVS,
                    next_states, [done] * NUM_UAVS,
                    obs_t.flatten(), nt.flatten(),
                )
            elif is_abqmix:
                agent.store_transition(
                    obs_t.flatten().cpu().numpy(),  # global_state
                    obs_np,                          # obs (list of np arrays)
                    raw_actions,                     # [[traj,bf],...]
                    reward,
                    nt.flatten().cpu().numpy(),      # next_global_state
                    nobs_np,                         # next_obs
                )
            elif is_dmtd:
                agent.store_transitions(
                    obs_np, actions, [reward] * NUM_UAVS,
                    nobs_np, global_step,
                )
            elif is_mappo:
                agent.store_transition(
                    states, actions, [reward] * NUM_UAVS,
                    next_states, [done] * NUM_UAVS,
                    log_probs,
                )
            else:
                # IQL, MADDPG, DeepNashQ — standard 5-arg store_transition
                agent.store_transition(
                    states, actions, [reward] * NUM_UAVS,
                    next_states, [done] * NUM_UAVS,
                )

            # ── update ───────────────────────────────────────────────────────
            if is_dmtd:
                agent.update(global_step)
            else:
                agent.update()

            # ── metrics ───────────────────────────────────────────────────────
            ep_ee.append(info.get('energy_efficiency_mbitpj', 0.0))
            ep_ho   += info.get('handovers', 0)
            ep_pdr.append(  np.clip(info.get('packet_drop_rate',      0.0), 0, 1))
            ep_pdrl.append( np.clip(info.get('packet_drop_rate_low',  0.0), 0, 1))
            ep_pdrh.append( np.clip(info.get('packet_drop_rate_high', 0.0), 0, 1))

            obs_t  = nt
            obs_np = nobs_np
            step        += 1
            global_step += 1

        if is_abqmix:
            agent.end_episode()
        env.end_episode()

        ee_ep.append(  float(np.mean(ep_ee)))
        ho_ep.append(  float(ep_ho))
        pdr_ep.append( float(np.mean(ep_pdr))  * 100)
        pdrl_ep.append(float(np.mean(ep_pdrl)) * 100)
        pdrh_ep.append(float(np.mean(ep_pdrh)) * 100)

    return {
        'ee':       np.array(ee_ep),
        'ho':       np.array(ho_ep),
        'pdr':      np.array(pdr_ep),
        'pdr_low':  np.array(pdrl_ep),
        'pdr_high': np.array(pdrh_ep),
    }


# ─────────────────────────── experiments ─────────────────────────────────────

def experiment_gamma():
    """Vary gamma for QMIX (parametric study of discount factor → EE)."""
    print("\n" + "=" * 60)
    print("Experiment 1: Energy Efficiency vs Discount Factor (QMIX)")
    print("=" * 60)
    out = {}
    for g in GAMMAS:
        env   = make_env(pareto_scale=1.0, traffic_load=1.0)
        agent = make_agent('QMIX', env, gamma=g)
        out[g] = run_training(env, agent, 'QMIX', desc=f'QMIX γ={g}')
        env.close()
    return out


def experiment_all_algos_pareto(pareto_scale=2.0, traffic_load=1.0):
    """Train all algorithms under Pareto VBR traffic — algorithm comparison."""
    print("\n" + "=" * 60)
    print(f"Experiment 2: All Algorithms — Pareto VBR (scale={pareto_scale} Mbps)")
    print("=" * 60)
    out = {}
    for name in ALGO_ORDER:
        env   = make_env(pareto_scale=pareto_scale, traffic_load=traffic_load)
        agent = make_agent(name, env, gamma=0.99)
        out[name] = run_training(env, agent, name,
                                 desc=f'{name} Pareto={pareto_scale}Mbps')
        env.close()
    return out


def experiment_all_algos_highload(traffic_load=4.0):
    """Train all algorithms under high traffic load — PDR comparison."""
    print("\n" + "=" * 60)
    print(f"Experiment 3: All Algorithms — High Traffic Load ({traffic_load}×)")
    print("=" * 60)
    out = {}
    for name in ALGO_ORDER:
        env   = make_env(pareto_scale=1.0, traffic_load=traffic_load)
        agent = make_agent(name, env, gamma=0.99)
        out[name] = run_training(env, agent, name,
                                 desc=f'{name} load={traffic_load}×')
        env.close()
    return out


# ─────────────────────────── plotting ────────────────────────────────────────

def _draw(ax, eps, y, color, label, ls='-', lw=THICK_LW):
    """Single smooth curve — no raw trace, no band."""
    ax.plot(eps, smooth(y), color=color, linestyle=ls,
            linewidth=lw, label=label, zorder=2)


def _legend(ax, ncol=1):
    ax.legend(frameon=True, fancybox=False, edgecolor='0.4',
              loc='best', ncol=ncol, fontsize=8)


def plot_all(gamma_res, pareto_res, highload_res):
    apply_publication_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eps = np.arange(1, EPISODES + 1, dtype=np.float64)

    # ════════════════════════════════════════════════════════════════
    # Figure 1 — 2×2
    # ════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # ── (a) EE vs gamma — QMIX parametric ────────────────────────────
    gamma_colors = _COLORS[:len(GAMMAS)]
    for i, g in enumerate(GAMMAS):
        lw = PROPOSED_LW if g == 0.99 else THICK_LW
        _draw(ax_a, eps, gamma_res[g]['ee'],
              gamma_colors[i], f'γ = {g}', lw=lw)
    set_xaxis(ax_a)
    ax_a.set_ylabel('Energy efficiency (Mbit/J)')
    ax_a.set_title('(a) Energy Efficiency vs Discount Factor (γ)\n[QMIX]')
    _legend(ax_a)

    # ── (b) Handovers — all algorithms, Pareto VBR ───────────────────
    for name in ALGO_ORDER:
        if name not in pareto_res:
            continue
        lw = PROPOSED_LW if name in PROPOSED_ALGOS else THICK_LW
        _draw(ax_b, eps, pareto_res[name]['ho'],
              ALGO_COLORS[name], name, lw=lw)
    set_xaxis(ax_b)
    ax_b.set_ylabel('Handovers per episode')
    ax_b.set_title('(b) Handover Switching — All Algorithms\n(Pareto VBR Traffic)')
    _legend(ax_b, ncol=2)

    # ── (c) EE — all algorithms, Pareto VBR ──────────────────────────
    for name in ALGO_ORDER:
        if name not in pareto_res:
            continue
        lw = PROPOSED_LW if name in PROPOSED_ALGOS else THICK_LW
        _draw(ax_c, eps, pareto_res[name]['ee'],
              ALGO_COLORS[name], name, lw=lw)
    set_xaxis(ax_c)
    ax_c.set_ylabel('Energy efficiency (Mbit/J)')
    ax_c.set_title('(c) Energy Efficiency — All Algorithms\n(Pareto VBR Traffic)')
    _legend(ax_c, ncol=2)

    # ── (d) PDR LOW (solid) vs HIGH (dashed) — all algorithms ─────────
    for name in ALGO_ORDER:
        if name not in highload_res:
            continue
        color = ALGO_COLORS[name]
        lw    = PROPOSED_LW if name in PROPOSED_ALGOS else THICK_LW
        ax_d.plot(eps, smooth(highload_res[name]['pdr_low']),
                  color=color, linestyle='-',  linewidth=lw,
                  label=f'{name} LOW', zorder=2)
        ax_d.plot(eps, smooth(highload_res[name]['pdr_high']),
                  color=color, linestyle='--', linewidth=lw,
                  label=f'{name} HIGH', zorder=2)
    set_xaxis(ax_d)
    ax_d.set_ylabel('Packet drop rate (%)')
    ax_d.set_ylim(bottom=0)
    ax_d.set_title('(d) Packet Drop Rate — All Algorithms\n'
                   'solid = Low (v<1 m/s)   dashed = High (v≥5 m/s)')
    _legend(ax_d, ncol=2)

    p1 = os.path.join(OUTPUT_DIR, 'experiment_gamma_arrival.png')
    fig.savefig(p1, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved → {p1}")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════
    # Figure 2 — 1×2  PDR LOW vs HIGH per algorithm
    # ════════════════════════════════════════════════════════════════
    fig2, (ax_l, ax_h) = plt.subplots(1, 2, figsize=(10.5, 3.8),
                                       constrained_layout=True)
    for name in ALGO_ORDER:
        if name not in highload_res:
            continue
        color = ALGO_COLORS[name]
        lw    = PROPOSED_LW if name in PROPOSED_ALGOS else THICK_LW
        _draw(ax_l, eps, highload_res[name]['pdr_low'],  color, name, lw=lw)
        _draw(ax_h, eps, highload_res[name]['pdr_high'], color, name, lw=lw)

    for ax, title in [
        (ax_l, '(a) PDR — Low Mobility Users (v < 1 m/s)'),
        (ax_h, '(b) PDR — High Mobility Users (v ≥ 5 m/s)'),
    ]:
        set_xaxis(ax)
        ax.set_ylabel('Packet drop rate (%)')
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        _legend(ax, ncol=2)

    p2 = os.path.join(OUTPUT_DIR, 'experiment_pdr_velocity.png')
    fig2.savefig(p2, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved → {p2}")
    plt.close(fig2)


# ─────────────────────────── entry point ─────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("UAV Repositioning — Experiment Analysis")
    print(f"Device      : {DEVICE}")
    print(f"Episodes    : {EPISODES} per configuration")
    print(f"Smooth W    : {SMOOTH_W}  ({100 * SMOOTH_W // EPISODES}% of episodes)")
    print(f"Algorithms  : {ALGO_ORDER}")
    print(f"Proposed    : {sorted(PROPOSED_ALGOS)}")
    print("=" * 60)
    print()
    print("Pareto VBR formula : d_i = scale × (Pareto(1.5) + 1) × load  [bps]")
    print("                     Mean demand = scale × load × 3.0  Mbps")
    print("Call drop rate     : CDR = Σ max(0, d_i − r_i) / Σ d_i")
    print()

    gamma_res    = experiment_gamma()
    pareto_res   = experiment_all_algos_pareto(pareto_scale=2.0)
    highload_res = experiment_all_algos_highload(traffic_load=4.0)

    print("\nGenerating figures …")
    plot_all(gamma_res, pareto_res, highload_res)
    print("Done.")
