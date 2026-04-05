"""
DMTD: Distributed 3D Multi-UAV Trajectory Design
Based on: "Deep Reinforcement Learning-Based Distributed 3D UAV Trajectory Design"
He et al., IEEE Transactions on Communications, Vol. 72, No. 6, June 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Q-Network  (Double DQN backbone, 3 hidden layers as per Table II)
# ─────────────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Priority Experience Replay  (Algorithm 2)
# p1(t) = 1 / rank(|delta|)
# p2(t) = exp(-(t_sample - t_generate))
# p(t)  = gamma1 * p1 + gamma2 * p2
# ─────────────────────────────────────────────────────────────────────────────
class HybridPriorityReplayBuffer:
    def __init__(self, capacity: int, gamma1: float = 0.5, gamma2: float = 0.5):
        assert abs(gamma1 + gamma2 - 1.0) < 1e-6, "gamma1 + gamma2 must equal 1"
        self.capacity = capacity
        self.gamma1   = gamma1
        self.gamma2   = gamma2
        self.buffer: List[dict] = []

    def push(self, state, action: int, reward: float,
             next_state, td_error: float, t: int):
        entry = dict(state=state, action=action, reward=reward,
                     next_state=next_state, td_error=abs(td_error),
                     t_generate=t, p1=0.0, p2=1.0, hybrid=0.0)
        self.buffer.append(entry)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        self._recompute_weights(t)

    def _recompute_weights(self, current_t: int):
        sorted_buf = sorted(self.buffer, key=lambda x: x["td_error"], reverse=True)
        for rank, e in enumerate(sorted_buf, start=1):
            e["p1"]     = 1.0 / rank
            e["p2"]     = math.exp(-(current_t - e["t_generate"]))
            e["hybrid"] = self.gamma1 * e["p1"] + self.gamma2 * e["p2"]

    def sample(self, k: int, current_t: int) -> Tuple[List[dict], List[int]]:
        self._recompute_weights(current_t)
        total = sum(e["hybrid"] for e in self.buffer)
        probs = [e["hybrid"] / total for e in self.buffer]
        idxs  = random.choices(range(len(self.buffer)), weights=probs, k=k)
        return [self.buffer[i] for i in idxs], idxs

    def update_td_errors(self, indices: List[int],
                         new_errors: List[float], current_t: int):
        for i, err in zip(indices, new_errors):
            self.buffer[i]["td_error"] = abs(err)
        self._recompute_weights(current_t)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# DMTD  —  one instance manages ALL UAV agents  (Algorithm 3)
#
# State  sn(t) = { en(t), c1..cN(t), a(t) }          (Section IV-B.1)
# Action (8 horizontal moves + hover) × 3 altitudes = 27 discrete actions
# Reward rn(t) = J(t)*sum_cm(t)/en(t) + Kn(t)/M - p*lambda   (Eq. 19)
# ─────────────────────────────────────────────────────────────────────────────
class DMTD:
    def __init__(
        self,
        num_uavs:             int,
        state_dim:            int,
        action_dim:           int   = 27,
        learning_rate:        float = 1e-3,
        gamma:                float = 0.99,
        epsilon:              float = 1.0,
        epsilon_min:          float = 0.05,
        epsilon_decay:        float = 0.95,
        buffer_capacity:      int   = 10_000,
        batch_size:           int   = 64,
        target_update_freq:   int   = 10,
        gamma1:               float = 0.5,
        gamma2:               float = 0.5,
        delta_max:            float = 1.0,
        epsilon_online_boost: float = 0.3,
        device:               torch.device = None,
    ):
        self.num_uavs             = num_uavs
        self.action_dim           = action_dim
        self.gamma                = gamma
        self.epsilon              = epsilon
        self.epsilon_min          = epsilon_min
        self.epsilon_decay        = epsilon_decay
        self.batch_size           = batch_size
        self.target_update_freq   = target_update_freq
        self.delta_max            = delta_max
        self.epsilon_online_boost = epsilon_online_boost
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # One Q-network + target + replay buffer per UAV  (Algorithm 3, Line 3)
        self.q_networks      = [QNetwork(state_dim, action_dim).to(self.device)
                                 for _ in range(num_uavs)]
        self.target_networks = [QNetwork(state_dim, action_dim).to(self.device)
                                 for _ in range(num_uavs)]
        for i in range(num_uavs):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
            self.target_networks[i].eval()

        self.optimizers = [optim.Adam(net.parameters(), lr=learning_rate)
                           for net in self.q_networks]
        self.replay_buffers = [
            HybridPriorityReplayBuffer(buffer_capacity, gamma1, gamma2)
            for _ in range(num_uavs)
        ]
        self._learn_steps = [0] * num_uavs

    # ── ε-greedy action selection ─────────────────────────────────────────────
    def get_actions(self, states: List[np.ndarray],
                    greedy: bool = False) -> List[int]:
        actions = []
        for i, net in enumerate(self.q_networks):
            if not greedy and random.random() < self.epsilon:
                actions.append(random.randint(0, self.action_dim - 1))
            else:
                s = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    actions.append(int(net(s).argmax(dim=1).item()))
        return actions

    # ── store transitions + HPER update  (Algorithm 3 Lines 15-16) ───────────
    def store_transitions(
        self,
        states:      List[np.ndarray],
        actions:     List[int],
        rewards:     List[float],
        next_states: List[np.ndarray],
        t:           int,
    ) -> List[float]:
        """Store one transition per UAV. Returns per-UAV |delta| values."""
        td_errors = []
        for i in range(self.num_uavs):
            td = self._compute_td_error(i, states[i], actions[i],
                                        rewards[i], next_states[i])
            self.replay_buffers[i].push(
                states[i], actions[i], rewards[i], next_states[i], td, t)
            td_errors.append(td)
        return td_errors

    # ── one gradient step per UAV  (Algorithm 2) ─────────────────────────────
    def update(self, t: int):
        for i in range(self.num_uavs):
            if len(self.replay_buffers[i]) < self.batch_size:
                continue

            batch, idxs = self.replay_buffers[i].sample(self.batch_size, t)

            states      = torch.FloatTensor(np.array([b["state"]      for b in batch])).to(self.device)
            actions     = torch.LongTensor( np.array([b["action"]     for b in batch])).to(self.device)
            rewards     = torch.FloatTensor(np.array([b["reward"]     for b in batch])).to(self.device)
            next_states = torch.FloatTensor(np.array([b["next_state"] for b in batch])).to(self.device)

            # Double DQN target  (Algorithm 2 Line 9)
            with torch.no_grad():
                best_acts  = self.q_networks[i](next_states).argmax(dim=1)
                td_targets = rewards + self.gamma * \
                             self.target_networks[i](next_states)\
                                 .gather(1, best_acts.unsqueeze(1)).squeeze(1)

            current_q = self.q_networks[i](states)\
                            .gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = nn.MSELoss()(current_q, td_targets)

            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

            # Update HPER weights  (Algorithm 2 Line 10)
            new_errors = (td_targets - current_q).detach().cpu().numpy().tolist()
            self.replay_buffers[i].update_td_errors(idxs, new_errors, t)

            # Periodically sync target network  (Algorithm 2 Line 13)
            self._learn_steps[i] += 1
            if self._learn_steps[i] % self.target_update_freq == 0:
                self.target_networks[i].load_state_dict(
                    self.q_networks[i].state_dict())

    # ── offline pre-training step  (Algorithm 3 Lines 8-17) ──────────────────
    def step_offline(
        self,
        states:      List[np.ndarray],
        actions:     List[int],
        rewards:     List[float],
        next_states: List[np.ndarray],
        t:           int,
    ):
        self.store_transitions(states, actions, rewards, next_states, t)
        self.update(t)
        self._decay_epsilon()

    # ── online training step  (Algorithm 3 Lines 23-33) ──────────────────────
    def step_online(
        self,
        states:      List[np.ndarray],
        actions:     List[int],
        rewards:     List[float],
        next_states: List[np.ndarray],
        t:           int,
    ):
        td_errors = self.store_transitions(states, actions, rewards, next_states, t)
        for td in td_errors:
            # Environmental change detected → boost exploration  (Line 29-31)
            if abs(td) > self.delta_max:
                self.epsilon = self.epsilon_online_boost
                self.update(t)
                return
        self._decay_epsilon()

    # ── switch to online mode  (Algorithm 3 Line 22: ε = 0) ─────────────────
    def set_online_mode(self):
        self.epsilon = 0.0

    # ── helpers ───────────────────────────────────────────────────────────────
    def _compute_td_error(self, agent_idx: int, state, action: int,
                          reward: float, next_state) -> float:
        s  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            best_a    = self.q_networks[agent_idx](ns).argmax(dim=1)
            td_target = reward + self.gamma * \
                        self.target_networks[agent_idx](ns)[0, best_a].item()
            td_curr   = self.q_networks[agent_idx](s)[0, action].item()
        return td_target - td_curr

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save({
            "q_networks":      [net.state_dict() for net in self.q_networks],
            "target_networks": [net.state_dict() for net in self.target_networks],
            "optimizers":      [opt.state_dict() for opt in self.optimizers],
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for i, sd in enumerate(ckpt["q_networks"]):
            self.q_networks[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt["target_networks"]):
            self.target_networks[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt["optimizers"]):
            self.optimizers[i].load_state_dict(sd)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────
def jain_fairness_index(access_ratios: List[float]) -> float:
    """Eq. 15 — J(T) = (sum Am)^2 / (M * sum Am^2)"""
    M  = len(access_ratios)
    s1 = sum(access_ratios)
    s2 = sum(a ** 2 for a in access_ratios)
    return (s1 ** 2) / (M * s2) if s2 > 0 else 0.0


def compute_reward(
    J_t:               float,
    total_covered_ues: int,
    energy_n:          float,
    connected_ues_n:   int,
    total_ues:         int,
    is_illegal:        bool,
    penalty_lambda:    float = 1.0,
) -> float:
    """Eq. 19 — rn(t) = J(t)*sum_cm(t)/en(t)  +  Kn(t)/M  -  p*lambda"""
    energy_n    = max(energy_n, 1e-9)
    global_term = J_t * total_covered_ues / energy_n
    local_term  = connected_ues_n / total_ues
    penalty     = penalty_lambda if is_illegal else 0.0
    return global_term + local_term - penalty


def dynamic_handover(
    uavs_covered_ues: dict,
    channel_gains:    dict,
    uav_loads:        dict,
    f1: float = 0.6,
    f2: float = 0.4,
) -> dict:
    """
    Algorithm 1 — F(Bn',um)(t) = f1*g(Bn',um)(t) - f2*Mn'(t)
    Returns {ue_id: best_uav_id}
    """
    ue_to_uavs: dict = {}
    for uav_id, ue_list in uavs_covered_ues.items():
        for ue in ue_list:
            ue_to_uavs.setdefault(ue, []).append(uav_id)

    assignment = {}
    for ue, accessible_uavs in ue_to_uavs.items():
        best_uav, best_util = None, -float("inf")
        for uav in accessible_uavs:
            util = f1 * channel_gains.get((uav, ue), 0.0) \
                 - f2 * uav_loads.get(uav, 0)
            if util > best_util:
                best_util = util
                best_uav  = uav
        assignment[ue] = best_uav
    return assignment
