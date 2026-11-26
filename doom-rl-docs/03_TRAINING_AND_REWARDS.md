# 03 — Training Pipeline, Reward Shaping & Curriculum

## 1. Overview: Three-Stage Training Strategy
To reach high zero-shot scores fast without wasting hours on random RL exploration:
1. **Stage A (Optional, 15 min human + 5 min GPU): Behavior Cloning (BC) Warm-Start**
   - Record 10–15 minutes of a teammate playing inside `doom_env` (`training/record_demos.py`) across Tier 1–3 maps.
   - Train the actor + encoders with supervised cross-entropy (`training/train_bc.py`) for 5 minutes.
   - *Result:* Agent immediately knows how to move forward, open doors, pick up keycards, aim at enemies, and hit exit switches before RL even starts.
2. **Stage B: Custom Recurrent PPO + RND Curiosity + Auxiliary Losses (`training/train_ppo.py`)**
   - Clean, custom PyTorch implementation (CleanRL-style truncated BPTT over sequences of length `T=64` or `128`).
   - Separate extrinsic (`gamma=0.995`) and intrinsic (`gamma=0.99`) GAE advantages.
3. **Stage C: Prioritized Level Replay (PLR) Adaptive Curriculum (`training/plr.py`)**
   - Continuously samples from our procedural map pool, prioritizing maps with the highest learning potential (positive value loss / partial progress) so the agent progresses cleanly from Tier 1 → Tier 4 without forgetting earlier skills.

---

## 2. Survival-Aware Progression Reward Shaping (`training/rewards.py`)
> **Rule Note:** During *training*, we can read internal engine state (player `(x, y)` coords, sector/door state, enemy HP, keycards) to compute dense rewards. The neural network NEVER receives global coordinates—only the official `doom_env` observation dict.

| Event | Training Reward | Purpose |
| :--- | :---: | :--- |
| **Reach Level Exit** | `+15.0` (+ `2.0 * hp_frac`) | #1 priority matching official scorer |
| **Unlock Locked Door** | `+4.0` | Directly reinforces Tier 3 backtracking |
| **Pick Up Keycard** | `+3.0` | High-value subgoal |
| **Open Regular Door** | `+0.6` (first time per door) | Teaches USE/door interaction |
| **New Grid Cell Visited** | `+0.06` (64×64 Doom-unit grid, capped per episode) | Dense branch exploration reward |
| **Enemy Killed** | `+1.0` | Clears threats blocking progression |
| **Enemy Damaged** | `+0.15` per hit | Dense combat aiming signal |
| **Health / Armor / Ammo Pickup** | `+0.2` (when below 80%) | Survival resource management |
| **Damage Taken** | `-0.03 * hp_lost` | Teaches strafing & dodging fireballs |
| **Death** | `-4.0` | Strongly penalizes reckless rushing |
| **Time / Stuck Penalty** | `-0.002` / step (`-0.01` if displacement < 4 units for 15 steps) | Prevents wall-hugging & camping |

*(Tune reward magnitudes once we inspect the exact reward/info keys already exposed by `doom_env`!)*

---

## 3. Intrinsic Curiosity (Lightweight RND)
- **Why:** On unseen Tier 2–4 maps, training-only grid-cell rewards aren't available at test time, and even during training RND rewards novel sensory states (new rooms, new keycard/door sights).
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
4. Within each tier, sample maps with probability proportional to:
   - `score(map) = 0.7 * mean_positive_advantage + 0.3 * staleness` (PLR).
   - Periodically inject freshly generated unseen maps (every 500k steps) so the agent never memorizes the training pool!

---

## 6. Checkpointing & Monitoring
Log every update to TensorBoard / CSV:
- `fps` (target: ≥ 1,500 env steps/sec locally, ≥ 4,000+ on cloud)
- `ep_return`, `exit_rate_by_tier`, `doors_opened`, `keycards_picked`, `kills`, `deaths`, `unique_cells`
- Every **1M steps**, automatically run `training/eval_zeroshot.py` on the **12 held-out unseen maps** and save `best_zeroshot.pt` whenever the tier-weighted zero-shot score improves.
