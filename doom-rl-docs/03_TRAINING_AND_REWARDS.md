# 03 — Training Pipeline, Reward Shaping & Curriculum

## 1. Overview: Two Stages + a Curriculum Running Throughout
To reach high zero-shot scores fast without wasting hours on random RL exploration:
1. **Stage A (Optional, 15 min human + 5 min GPU): Behavior Cloning (BC) Warm-Start**
   - Record 10–15 minutes of a teammate playing across Tier 1–3 maps (`training/record_demos.py`). `doom_env` has no visible window or keyboard input, so the recorder needs a training-only subclass that runs ViZDoom in `Mode.SPECTATOR` with the window visible, reads the human's buttons via `game.get_last_action()`, and still computes the `doom_env` observation each step. Map the recorded buttons onto our 6 factored action heads and drop contradictory presses.
   - Train the actor + encoders with per-head cross-entropy (`training/train_bc.py`) for 5 minutes. 10–15 min of demos is only ~10–20k steps: expect a rough prior, not a competent agent. Keep a small BC/KL term early in PPO so RL doesn't immediately erase it.
   - *Result:* Agent immediately knows how to move forward, open doors, pick up keycards, aim at enemies, and hit exit switches before RL even starts.
2. **Stage B: Custom Recurrent PPO + RND Curiosity + Auxiliary Losses (`training/train_ppo.py`)**
   - Clean, custom PyTorch implementation (CleanRL-style truncated BPTT over sequences of length `T=64` or `128`).
   - Separate extrinsic (`gamma=0.995`) and intrinsic (`gamma=0.99`) GAE advantages.
3. **Throughout Stage B: Prioritized Level Replay (PLR) Adaptive Curriculum (`training/plr.py`)**
   - Not a separate stage: PLR picks which map each env loads on every reset during PPO. It continuously samples from our procedural map pool, prioritizing maps with the highest learning potential (positive value loss / partial progress) so the agent progresses cleanly from Tier 1 → Tier 4 without forgetting earlier skills.

---

## 2. Survival-Aware Progression Reward Shaping (`training/rewards.py`)
> **Rule Note:** During *training*, we can read internal engine state to compute dense rewards. The neural network NEVER receives global coordinates—only the official `doom_env` observation dict.

`doom_env`'s built-in reward is only `+1 exit, −1 death, +0.1/kill`; ignore it and compute our own from `env.privileged_state()`, whose keys are (verified): `x, y, z, angle, health, kills, items, damage_taken, keys, dead, timeout, exit_reached, tic, sector_floor, sector_ceiling, door_lock`. The "Source" column says where each signal comes from.

| Event | Training Reward | Source | Purpose |
| :--- | :---: | :--- | :--- |
| **Reach Level Exit** | `+15.0` (+ `2.0 * hp_frac`) | `exit_reached` | #1 priority matching official scorer |
| **Unlock Locked Door** | `+4.0` | sector with `door_lock > 0` whose `sector_ceiling` rises | Directly reinforces Tier 3 backtracking |
| **Pick Up Keycard** | `+3.0` | `keys` flips 0→1 | High-value subgoal |
| **Open Regular Door** | `+0.6` (first time per door) | door sector's `sector_ceiling` rises above its floor | Teaches USE/door interaction |
| **New Sector Entered** | `+0.3 × sqrt(sector area / mean area)` (first visit) | point-in-polygon on `x, y` vs `env.sector_polygons()` | Matches how the scorer measures exploration |
| **New Grid Cell Visited** | `+0.02` (64×64-unit grid, capped per episode) | `x, y` | Dense filler between sectors |
| **Enemy Killed** | `+1.0` | `kills` delta | Clears threats blocking progression |
| **Enemy Damaged** | `+0.15` per hit | ⚠️ *not in `privileged_state`* — needs a training-only wrapper subclass reading ViZDoom's `DAMAGECOUNT` game variable; skip if short on time | Dense combat aiming signal |
| **Health / Armor / Ammo Pickup** | `+0.2` (when below 80%) | `vitals` deltas | Survival resource management |
| **Damage Taken** | `-0.03 * hp_lost` | `damage_taken` delta | Teaches strafing & dodging fireballs |
| **Death** | `-4.0` | `dead` | Strongly penalizes reckless rushing |
| **Time / Stuck Penalty** | `-0.002` / step (`-0.01` if displacement < 4 units for 15 steps) | `x, y` | Prevents wall-hugging & camping |
| **USE spam** | `-0.01` per USE press while a door sector is moving | `sector_ceiling` delta | Spamming USE reverses doors (contract §3) |

*(Door detection verified on `two_rooms.wad`: `door_lock` is per sector, `-1` = not a door, `0` = unlocked door. The door sector's `sector_ceiling` starts at its floor (0) and rises once USE is pressed. Locked doors presumably have `door_lock > 0` (key id); confirm on a map that has one. ViZDoom does have `GameVariable.DAMAGECOUNT` for the enemy-damaged reward.)*

---

## 3. Intrinsic Curiosity (Lightweight RND)
- **Why:** Neither RND nor the grid/sector rewards exist at test time; all of them only shape *training*. What RND adds is a novelty signal in observation space (new rooms, first sight of a door/key) that doesn't depend on engine coordinates. At test time, the "explore new places" behaviour has to come from the learned policy reading `EgoLocalMap`'s visited/frontier channels. If time is short, cut RND first: sector + grid rewards already cover exploration.
- **Implementation:**
  - Fixed random target network `f_target(ray_obs, vitals) -> 64-dim` + trainable predictor network `f_pred(ray_obs, vitals) -> 64-dim`.
  - Intrinsic reward = running-normalized MSE `||f_pred - f_target||^2`, weighted by `beta_int = 0.05` (annealed down slightly in late training).
  - RND is **training-only**—never loaded in `submission/agent.py` (0 inference cost).

---

## 4. Custom Recurrent PPO Hyperparameters (Starting Baseline)
```yaml
num_envs: 16              # 16 on Laptop 1 (i7-13th + 4050); 32-64 on Cloud VM
rollout_steps: 128        # 16 * 128 = 2,048 transitions per update (4,096-8,192 on cloud)
seq_len: 64               # Truncated BPTT chunk length for GRU
num_minibatches: 4        # Split along environment dimension (NEVER shuffle time steps!)
update_epochs: 4
lr: 3e-4                  # Linear/cosine decay to 5e-5
gamma_ext: 0.995          # Long horizon for Tier 3-4 keycard backtracking
gamma_int: 0.99
gae_lambda: 0.95
clip_coef: 0.2
ent_coef: 0.015           # Decay to 0.005 in late training
vf_coef: 0.5
aux_coef: 0.1
max_grad_norm: 0.5
```

> **CRITICAL RL BUG TO AVOID:** When minibatching for Recurrent PPO, **always split across environments/chunks and preserve sequential time order**, resetting the GRU hidden state `h` at exact `done` boundaries inside the sequence!

---

## 5. Prioritized Level Replay (PLR) (`training/plr.py`)
1. Maintain a pool of generated maps tagged by tier (`T1`, `T2`, `T3`, `T4`).
2. Start curriculum with **50% T1 + 50% T2** until T1 exit rate > 70% and T2 exit rate > 50%.
3. Introduce **T3 (Keycard → Locked Door → Exit)** gradually (`20% T1, 25% T2, 45% T3, 10% T4`), then shift toward **`15% T1, 20% T2, 35% T3, 30% T4`** for final generalization.
4. Within each tier, sample maps from the standard PLR mixture (Jiang et al., 2021). Don't add score and staleness together; they're two separate distributions:
   - `score(map)` = mean over the last episode of `max(GAE advantage, 0)` ("positive value loss").
   - `P_score` = rank-based: `h(rank)^(1/β)`, `β = 0.1`, normalized. `P_stale` ∝ steps since the map was last played.
   - `P = (1 − ρ) · P_score + ρ · P_stale`, with `ρ = 0.1`–`0.3`.
   - Periodically inject freshly generated unseen maps (every 500k steps) so the agent never memorizes the training pool!

---

## 6. Checkpointing & Monitoring
Log every update to TensorBoard / CSV:
- `fps` (target: ≥ 1,500 env steps/sec locally, ≥ 4,000+ on cloud). Measured baseline: one `doom_env` instance does **~1,000–1,300 steps/s on one CPU core** with random actions (two_rooms + 2 fixture maps). So 16 envs in subprocesses can feed far more than 1,500/s; the real bottleneck will be our numpy grid update and the PPO update. Larger maps with many monsters will be slower; re-measure on Obsidian maps.
- `odom_error` (mean `|pose_hat − true pose|` per episode, from `privileged_state`)
- `ep_return`, `exit_rate_by_tier`, `doors_opened`, `keycards_picked`, `kills`, `deaths`, `unique_cells`
- Every **1M steps**, automatically run `training/eval_zeroshot.py` on the **12 held-out unseen maps** and save `best_zeroshot.pt` whenever the tier-weighted zero-shot score improves.
