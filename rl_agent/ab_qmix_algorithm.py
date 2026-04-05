"""
AB-QMIX: Action-Branching QMIX for Multi-UAV mmWave Communications
Based on: "Multi-UAV aided energy-aware transmissions in mmWave communication
           network: Action-branching QMIX network"
Do et al., Journal of Network and Computer Applications, 230 (2024) 103948
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Agent Network  (per UAV)
# FC(256) → LSTM(128) → two action-branching dueling heads
# Q_m = Q_traj + Q_bf   (Eq. 38, Section 3.2.2)
# ─────────────────────────────────────────────────────────────────────────────
class AgentNetwork(nn.Module):
    def __init__(
        self,
        obs_dim:         int,
        traj_action_dim: int = 9,
        bf_action_dim:   int = 16,
        hidden1:         int = 256,
        hidden2:         int = 128,
    ):
        super(AgentNetwork, self).__init__()
        self.traj_action_dim = traj_action_dim
        self.bf_action_dim   = bf_action_dim
        self.hidden2         = hidden2

        self.fc1  = nn.Sequential(nn.Linear(obs_dim, hidden1), nn.ReLU())
        self.lstm = nn.LSTM(hidden1, hidden2, batch_first=True)

        # Trajectory branch (dueling)
        self.traj_value = nn.Linear(hidden2, 1)
        self.traj_adv   = nn.Linear(hidden2, traj_action_dim)

        # Beamforming branch (dueling)
        self.bf_value = nn.Linear(hidden2, 1)
        self.bf_adv   = nn.Linear(hidden2, bf_action_dim)

    def forward(
        self, obs: torch.Tensor, hidden=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """
        obs    : (batch, obs_dim)  or  (batch, seq_len, obs_dim)
        Returns: q_traj, q_bf, q_agent (summed), hidden
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)                              # (B, 1, obs_dim)

        x = self.fc1(obs)                                       # (B, seq, 256)
        lstm_out, hidden = self.lstm(x, hidden)
        feat = lstm_out[:, -1, :]                               # (B, 128)

        # Trajectory Q  (Eq. 38)
        v_t    = self.traj_value(feat)
        a_t    = self.traj_adv(feat)
        q_traj = v_t + (a_t - a_t.mean(dim=1, keepdim=True))   # (B, traj_actions)

        # Beamforming Q  (Eq. 38)
        v_b  = self.bf_value(feat)
        a_b  = self.bf_adv(feat)
        q_bf = v_b + (a_b - a_b.mean(dim=1, keepdim=True))     # (B, bf_actions)

        # Combined agent Q for mixing network  (Section 3.2.2)
        q_agent = q_traj.max(dim=1).values + q_bf.max(dim=1).values  # (B,)

        return q_traj, q_bf, q_agent, hidden


# ─────────────────────────────────────────────────────────────────────────────
# Hypernetwork  —  global state → non-negative mixing-net weights  (Eq. 29)
# ─────────────────────────────────────────────────────────────────────────────
class Hypernetwork(nn.Module):
    def __init__(self, global_state_dim: int, n_agents: int, embed_dim: int = 32):
        super(Hypernetwork, self).__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # ReLU guarantees non-negative weights → monotonicity constraint
        self.hyper_w1 = nn.Sequential(
            nn.Linear(global_state_dim, n_agents * embed_dim), nn.ReLU())
        self.hyper_w2 = nn.Sequential(
            nn.Linear(global_state_dim, embed_dim), nn.ReLU())
        self.hyper_b1 = nn.Linear(global_state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(global_state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1))

    def forward(self, global_state: torch.Tensor):
        B  = global_state.shape[0]
        w1 = self.hyper_w1(global_state).view(B, self.n_agents, self.embed_dim)
        w2 = self.hyper_w2(global_state).view(B, self.embed_dim, 1)
        b1 = self.hyper_b1(global_state).view(B, 1, self.embed_dim)
        b2 = self.hyper_b2(global_state).view(B, 1, 1)
        return w1, w2, b1, b2


# ─────────────────────────────────────────────────────────────────────────────
# Mixing Network  —  Q_tot = f(Q_1, ..., Q_M ; s_t)  (Eq. 28)
# ─────────────────────────────────────────────────────────────────────────────
class MixingNetwork(nn.Module):
    def __init__(self, n_agents: int, global_state_dim: int, embed_dim: int = 32):
        super(MixingNetwork, self).__init__()
        self.hyper = Hypernetwork(global_state_dim, n_agents, embed_dim)

    def forward(self, agent_qs: torch.Tensor,
                global_state: torch.Tensor) -> torch.Tensor:
        """
        agent_qs     : (batch, n_agents)
        global_state : (batch, global_state_dim)
        Returns q_tot: (batch, 1)
        """
        w1, w2, b1, b2 = self.hyper(global_state)
        x      = agent_qs.unsqueeze(1)                         # (B, 1, n_agents)
        hidden = F.elu(torch.bmm(x, w1) + b1)                  # (B, 1, embed)
        q_tot  = torch.bmm(hidden, w2) + b2                    # (B, 1, 1)
        return q_tot.squeeze(2)                                 # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Episode Replay Buffer  (stores full episodes for LSTM training)
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeReplayBuffer:
    def __init__(self, capacity: int = 500, seq_len: int = 500):
        self.capacity = capacity
        self.seq_len  = seq_len
        self.buffer   = deque(maxlen=capacity)
        self._episode: List[dict] = []

    def start_episode(self):
        self._episode = []

    def push(self, global_state, obs, actions, reward, next_global_state, next_obs):
        self._episode.append(dict(
            global_state      = np.array(global_state,      dtype=np.float32),
            obs               = [np.array(o, dtype=np.float32) for o in obs],
            actions           = list(actions),
            reward            = float(reward),
            next_global_state = np.array(next_global_state, dtype=np.float32),
            next_obs          = [np.array(o, dtype=np.float32) for o in next_obs],
        ))

    def end_episode(self):
        if self._episode:
            self.buffer.append(list(self._episode))
            self._episode = []

    def sample(self, B: int) -> List[List[dict]]:
        episodes = random.choices(list(self.buffer), k=B)
        seqs = []
        for ep in episodes:
            if len(ep) <= self.seq_len:
                seqs.append(ep)
            else:
                start = random.randint(0, len(ep) - self.seq_len)
                seqs.append(ep[start: start + self.seq_len])
        return seqs

    @property
    def total_transitions(self) -> int:
        return sum(len(ep) for ep in self.buffer)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# AB-QMIX  —  full controller  (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────
class ABQMIX:
    def __init__(
        self,
        num_agents:          int,
        obs_dim:             int,
        global_state_dim:    int,
        traj_action_dim:     int   = 9,
        bf_action_dim:       int   = 16,
        hidden1:             int   = 256,
        hidden2:             int   = 128,
        mixing_embed_dim:    int   = 32,
        learning_rate:       float = 1e-4,
        gamma:               float = 0.99,
        epsilon:             float = 1.0,
        epsilon_min:         float = 0.1,
        delta_epsilon:       float = 5e-6,
        buffer_capacity:     int   = 500,
        batch_size:          int   = 4,
        seq_len:             int   = 500,
        c_min:               int   = 10_000,
        target_update_freq:  int   = 10_000,
        device:              torch.device = None,
    ):
        self.num_agents       = num_agents
        self.traj_action_dim  = traj_action_dim
        self.bf_action_dim    = bf_action_dim
        self.gamma            = gamma
        self.epsilon          = epsilon
        self.epsilon_min      = epsilon_min
        self.delta_epsilon    = delta_epsilon
        self.batch_size       = batch_size
        self.c_min            = c_min
        self.target_update_freq = target_update_freq
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Primary networks
        self.agent_networks = nn.ModuleList([
            AgentNetwork(obs_dim, traj_action_dim, bf_action_dim, hidden1, hidden2)
            for _ in range(num_agents)
        ])
        self.mixing_network = MixingNetwork(num_agents, global_state_dim, mixing_embed_dim)

        # Target networks
        self.target_agent_networks = nn.ModuleList([
            AgentNetwork(obs_dim, traj_action_dim, bf_action_dim, hidden1, hidden2)
            for _ in range(num_agents)
        ])
        self.target_mixing_network = MixingNetwork(num_agents, global_state_dim, mixing_embed_dim)
        self._copy_to_targets()
        for net in self.target_agent_networks:
            net.eval()
        self.target_mixing_network.eval()

        all_params = list(self.agent_networks.parameters()) + \
                     list(self.mixing_network.parameters())
        self.optimizer = optim.Adam(all_params, lr=learning_rate)

        self.replay     = EpisodeReplayBuffer(buffer_capacity, seq_len)
        self._hidden    = [None] * num_agents
        self._grad_steps = 0

    # ── ε-greedy joint action selection  (Algorithm 1 Lines 8-15) ────────────
    def get_actions(
        self,
        obs:                 List[np.ndarray],
        unavailable_actions: Optional[List[List[int]]] = None,
    ) -> List[List[int]]:
        """
        Returns [[traj_action, bf_action], ...] for each agent.
        unavailable_actions: per-agent list of action indices to mask.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon - self.delta_epsilon)
        actions = []
        for m, net in enumerate(self.agent_networks):
            if random.random() < self.epsilon:
                traj_a = random.randint(0, self.traj_action_dim - 1)
                bf_a   = random.randint(0, self.bf_action_dim   - 1)
            else:
                x = torch.FloatTensor(obs[m]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_traj, q_bf, _, self._hidden[m] = net(x, self._hidden[m])
                q_traj = q_traj.squeeze(0).cpu().numpy()
                q_bf   = q_bf.squeeze(0).cpu().numpy()

                # Action filter  (Section 3.2.2)
                if unavailable_actions is not None and m < len(unavailable_actions):
                    for a_idx in unavailable_actions[m]:
                        if a_idx < self.traj_action_dim:
                            q_traj[a_idx] = -1e9
                        else:
                            q_bf[a_idx - self.traj_action_dim] = -1e9

                traj_a = int(np.argmax(q_traj))
                bf_a   = int(np.argmax(q_bf))

            actions.append([traj_a, bf_a])
        return actions

    # ── store one step into the active episode  (Algorithm 1 Line 17) ─────────
    def store_transition(self, global_state, obs, actions, reward,
                         next_global_state, next_obs):
        self.replay.push(global_state, obs, actions, reward,
                         next_global_state, next_obs)

    # ── start / end episode helpers ───────────────────────────────────────────
    def start_episode(self):
        self.replay.start_episode()
        self._hidden = [None] * self.num_agents

    def end_episode(self):
        self.replay.end_episode()

    # ── gradient update  (Algorithm 1 Lines 18-28) ────────────────────────────
    def update(self) -> Optional[float]:
        if self.replay.total_transitions < self.c_min:
            return None

        episodes = self.replay.sample(self.batch_size)
        total_loss = torch.tensor(0.0)

        for seq in episodes:
            T = len(seq)
            if T == 0:
                continue

            gs  = torch.FloatTensor(np.stack([s["global_state"]      for s in seq])).to(self.device)
            ngs = torch.FloatTensor(np.stack([s["next_global_state"] for s in seq])).to(self.device)
            rw  = torch.FloatTensor(np.array([s["reward"]            for s in seq])).to(self.device)

            obs_seqs      = [torch.FloatTensor(np.stack([s["obs"][m]      for s in seq])).to(self.device)
                             for m in range(self.num_agents)]
            next_obs_seqs = [torch.FloatTensor(np.stack([s["next_obs"][m] for s in seq])).to(self.device)
                             for m in range(self.num_agents)]
            traj_acts     = [torch.LongTensor(np.array([s["actions"][m][0] for s in seq])).to(self.device)
                             for m in range(self.num_agents)]
            bf_acts       = [torch.LongTensor(np.array([s["actions"][m][1] for s in seq])).to(self.device)
                             for m in range(self.num_agents)]

            # Current Q_tot  (Algorithm 1 Line 20)
            agent_qs, h = [], [None] * self.num_agents
            for m in range(self.num_agents):
                qt, qb, _, h[m] = self.agent_networks[m](obs_seqs[m].unsqueeze(0), h[m])
                chosen_t = qt.squeeze(0).gather(1, traj_acts[m].unsqueeze(1)).squeeze(1)
                chosen_b = qb.squeeze(0).gather(1, bf_acts[m].unsqueeze(1)).squeeze(1)
                agent_qs.append(chosen_t + chosen_b)
            q_tot = self.mixing_network(torch.stack(agent_qs, dim=1), gs)   # (T, 1)

            # Target Q_tot  (Algorithm 1 Line 21)
            with torch.no_grad():
                t_qs, th = [], [None] * self.num_agents
                for m in range(self.num_agents):
                    tqt, tqb, _, th[m] = self.target_agent_networks[m](
                        next_obs_seqs[m].unsqueeze(0), th[m])
                    t_qs.append(tqt.squeeze(0).max(dim=1).values +
                                tqb.squeeze(0).max(dim=1).values)
                q_tot_target = self.target_mixing_network(
                    torch.stack(t_qs, dim=1), ngs)                         # (T, 1)

            # y_tot = r + γ * Q_tot_target  (Line 22)
            y_tot = (rw + self.gamma * q_tot_target.squeeze(1)).detach()

            total_loss = total_loss + F.mse_loss(q_tot.squeeze(1), y_tot)  # Eq. 39

        total_loss = total_loss / max(len(episodes), 1)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._grad_steps += 1
        if self._grad_steps % self.target_update_freq == 0:                # Lines 26-28
            self._copy_to_targets()

        return total_loss.item()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _copy_to_targets(self):
        for src, tgt in zip(self.agent_networks, self.target_agent_networks):
            tgt.load_state_dict(src.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

    def save(self, path: str):
        torch.save({
            "agent_networks":        [net.state_dict() for net in self.agent_networks],
            "target_agent_networks": [net.state_dict() for net in self.target_agent_networks],
            "mixing_network":        self.mixing_network.state_dict(),
            "target_mixing_network": self.target_mixing_network.state_dict(),
            "optimizer":             self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for i, sd in enumerate(ckpt["agent_networks"]):
            self.agent_networks[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt["target_agent_networks"]):
            self.target_agent_networks[i].load_state_dict(sd)
        self.mixing_network.load_state_dict(ckpt["mixing_network"])
        self.target_mixing_network.load_state_dict(ckpt["target_mixing_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────
def compute_reward(
    sum_rate:           float,
    propulsion_energy:  float,
    reached_destination:bool,
    failed:             bool,
    service_duration:   int,
    w1:                 float = 1.2,
    w2:                 float = 1.0,
    w3_fail:            float = 2_500.0,
    w3_success:         float = 1.0,
) -> float:
    """
    Eq. 37 — r_m = w1*r1 - w2*r2 + r3
    r1 = sum_rate  (Eq. 34)
    r2 = propulsion energy  (Eq. 35)
    r3 = terminal reward    (Eq. 36)
    """
    r3 = w3_success * service_duration if reached_destination else \
         (-w3_fail if failed else 0.0)
    return w1 * sum_rate - w2 * propulsion_energy + r3


def propulsion_power(
    V:     float,
    P_i:   float = 88.6279,
    P_b:   float = 79.8563,
    v0:    float = 7.2,
    U_tip: float = 200.0,
    d0:    float = 0.3,
    rho:   float = 1.225,
    s:     float = 0.05,
    A:     float = 0.79,
) -> float:
    """Eq. 13 — P_m(V), default values from Table 3."""
    induced  = P_i * np.sqrt(np.sqrt(1.0 + V**4 / (4*v0**2)) - V**2 / (2*v0**2))
    blade    = P_b * (1.0 + 3.0 * V**2 / U_tip**2)
    parasite = 0.5 * d0 * rho * s * A * V**3
    return induced + blade + parasite


def build_bf_codebook(N: int, n_beams: int) -> np.ndarray:
    """
    Eq. 31-32 — steering-vector codebook, angles uniformly quantised over (0, 2π).
    Returns (n_beams, N) complex array.
    """
    angles   = np.linspace(0, 2 * np.pi, n_beams, endpoint=False)
    n_idx    = np.arange(N)
    codebook = (1 / np.sqrt(N)) * np.exp(
        1j * 2 * np.pi * 0.5 * n_idx[None, :] * np.cos(angles[:, None]))
    return codebook
