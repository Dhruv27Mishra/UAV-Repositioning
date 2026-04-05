# Mean-field style performative MARL with signal-map observations

This document describes the **enhanced UAV coordination MDP** used for fair comparison across algorithms, and the proposed **PerformativeMFMARL** / **PerformativeMARL** MAPPO variants (see also `PERFORMATIVE_MARL.md` for a dedicated overview of those two methods).

## High-level flow

```text
reset / spawn UEs (performative placement if enabled)
  -> each step: all UAVs choose movement actions
  -> channel / SINR / association / rates
  -> optional: handover count, signal map, movement cost, spread metric
  -> team reward + info metrics
  -> optional: update coverage heatmap; end_episode updates UE distribution
```

The environment couples **physical layer** (path loss, LoS, rates), **multi-agent control** (UAV motion), and **performative feedback** (long-run UE placement reacts to observed coverage). Observations expose a **spatial field of radio quality** without revealing per-UE coordinates to the policy.

---

## State space (what changed)

Observations are stacked **per-UAV blocks** so every agent receives the same global radio summary (mean-field style global knowledge) plus its own pose.

For UAV index `i`, one block is:

| Component | Content |
|-----------|---------|
| Local pose | `(x_i, y_i, z_i)` in the operational grid / height limits |
| Signal map | Flattened `grid_cells_x × grid_cells_y` vector: per cell, **max** (over UEs in that cell) of **best-server received power** at the last step, **normalized** to about `[0, 1]` by dividing by the global max. **UE positions are not in the observation.** |
| Non-stationary context (if enabled) | Seven scalars replicated in every block: episode-normalized index, traffic demand drift, sin/cos phase features, and UE mobility statistics (aggregated). |

Total vector length = `num_uavs * agent_obs_dim`, with  
`agent_obs_dim = 3 + (grid_x * grid_y if signal map on) + (same size again if occupancy map on) + (7 if non-stationary on)`.

Enable flags on `MARLEnv`:

- `enable_signal_map_obs` — include the map (default `True`).
- `enable_non_stationary` — include drift + mobility context (comparison runs use `True`).
- `enable_performative` — UE spawn distribution tracks historical coverage (`True` in comparison runs).
- `use_occupancy_performative` — blend **UAV visitation** (grid occupancy) with coverage when updating UE placement (PerformativeMFMARL comparison env).
- `enable_occupancy_obs` — append a normalized per-cell UAV-count map to each agent block (**PerformativeMFMARL** only in `run_rl_comparison_500ep.py`; **PerformativeMARL** turns this off).

---

## Action space (unchanged)

- **MultiDiscrete** with six discrete moves per UAV per step: up / down / left / right / forward / backward on the (clipped) position grid.
- **Multiple UAVs may move every step** (no single-mover restriction in the current implementation).

---

## Reward (what changed)

The base reward remains **throughput- and QoS-oriented** (normalized throughput plus QoS bonus, minus collision penalty when UAVs are too close).

Optional shaping terms (weights default to `0`; **QMIX_Shaped** in the 500-episode comparison uses non-zero values):

| Term | Meaning |
|------|---------|
| `handover_penalty` | Subtracts `handover_penalty * (number of association changes vs previous step)` to discourage unnecessary handovers. |
| `movement_penalty` | Subtracts `movement_penalty * sum_i ||Δp_i||` over UAVs (L2 displacement after clipping) to penalize excessive motion. |
| `spread_bonus` | Adds `spread_bonus * mean pairwise ground distance` between UAVs to encourage coverage-friendly separation. |

`info` exposes `handovers`, `total_movement`, `mean_uav_distance`, `goodness`, `throughput`, etc., for logging and analysis.

---

## Performative (distribution shift) loop

When `enable_performative=True`:

1. During episodes, a **coverage heatmap** is updated from user-rate aggregates over the grid.
2. Optionally, a **UAV occupancy** history tracks which grid cells the fleet visits (policy-induced state occupancy).
3. At **`env.end_episode()`**, **UE placement weights** are nudged toward a blend of historical coverage and occupancy (when enabled), with mild uniform mixing for exploration.

Thus the **data distribution** depends on **long-run policy-induced coverage** and optionally **where the policy spends time** — a performative / strategic interaction setting. Training scripts call `end_episode()` after each episode where this mode is on.

---

## Proposed variant (in `run_rl_comparison_500ep.py`)

Baselines share the **same** performative + non-stationary + **signal-map** MDP (no extra occupancy block, no reward shaping). **PerformativeMFMARL** uses **MAPPO** with:

- **Custom UE–UAV association** via `AdaptiveNonStationaryMARL.standalone_association_function()` (SINR, performative heatmap bias, mobility-aware height preference, LOS bias, load balancing — same rule set as `improved_association_algorithm` in `AdaptiveNonStationaryMARL.py`).
- **Reward shaping** (`handover_penalty`, `movement_penalty`, `spread_bonus`) as in the former MAPPO_Shaped recipe.
- **Occupancy-aware performative feedback**: UE spawn weights track a blend of historical **coverage** and **UAV grid occupancy**; observations add a normalized **occupancy map** per agent block.

**PerformativeMARL** is the same stack with **signal-map and occupancy observations disabled** (pose + non-stationary context only), for an ablation without mean-field spatial fields in the policy input.

**Baselines**: **QMIX**, **IQL**, **VDN**, **MADDPG**, **DeepNashQ**, **MAPPO** (default shaping weights = 0, no occupancy obs / occupancy performative).

**Note:** `train_adaptive_nonstationary.py` still trains the original **AdaptiveNonStationaryMARL** (QMIX + adaptive association) if you want that baseline for ablations.

---

## Running the 500-episode comparison

From the repository root:

```bash
python3 run_rl_comparison_500ep.py --episodes 500 --steps-per-episode 30
```

Long runs can be started in the background and tailed, for example:

```bash
nohup python3 run_rl_comparison_500ep.py --episodes 500 --steps-per-episode 30 \
  > figures/comparison_500ep_run.log 2>&1 &
tail -f figures/comparison_500ep_run.log
```

Outputs:

- `figures/throughput_vs_episodes_500.png` — episode throughput (raw + smoothed).
- `figures/reward_vs_episodes_500.png` — episode return (raw + smoothed).
- `figures/rl_comparison_500ep_results.json` — numeric traces for replotting.

Optional: `--seed`, `--num-uavs`, `--num-users`, `--out-dir`.

---

## Implementation pointers

| Topic | Location |
|-------|----------|
| Signal map, rewards, handovers, obs layout | `rl_agent/marl_env.py` |
| Adaptive association rule + optional QMIX agent | `rl_agent/AdaptiveNonStationaryMARL.py` (`standalone_association_function` for MAPPO) |
| MAPPO | `rl_agent/MAPPO.py` |
| Shared training loop for comparisons | `compare_all_rl_convergence.py` (`train_model`) |
| 500-episode runner | `run_rl_comparison_500ep.py` |

---

## Notes

- **Mean-field** here is realized by **sharing the same low-dimensional spatial field** (signal map + NS scalars) across agents instead of listing every UE state.
- **“Better than everyone else”** is an **empirical** claim: use the saved JSON and plots, multiple seeds, and identical episode budgets per method.
- Pretrained weights are not required; the comparison script trains from scratch each run.
