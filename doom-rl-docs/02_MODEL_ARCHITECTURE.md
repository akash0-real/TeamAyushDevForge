# 02 — Custom Neural Architecture Specification

## 1. Design Goals
- **Custom & Judge-Worthy:** Bespoke spatial-intelligence architecture designed specifically for partial observability, variable enemy counts, and long-horizon keycard backtracking.
- **Fast & Lightweight:** ~1.5M–2.5M parameters, **< 5 ms/decision** on 1 CPU thread / RTX 3090, **< 400 MB VRAM**.
- **Zero-Shot Generalizable:** Operates purely on ego-centric spatial features—no global coordinate memorization.

---

## 2. End-to-End Architecture Diagram
```text
                   ┌──────────────────────────┐
                   │   doom_env Observation   │
                   └─┬──────────┬───────────┬─┘
                     │          │           │
       64 Rays       ▼          ▼           ▼      Vitals + Prev Action
   (dist + hit_type) │    Visible Entities  │    (HP, armor, ammo, keys,
                     │   (dx,dy,dist,type)  │       vel, prev_action)
                     ▼                      ▼               ▼
         ┌──────────────────┐   ┌────────────────────┐   ┌──────┐
         │ Circular 1D-CNN  │   │ Entity Encoder MLP │   │ MLP  │
         │ (3 layers, k=5,3)│   │ + Masked Multi-Head│   │(64d) │
         │   -> 128-dim     │   │  Attention Pooling │   └──┬───┘
         └────────┬─────────┘   │      -> 128-dim    │      │
                  │             └─────────┬──────────┘      │
                  │                       │                 │
                  └───────────┬───────────┴─────────────────┘
                              ▼
                  Concat [320-dim] + Ego-Centric Local Grid Summary [64-dim]
                              │
                              ▼
                 ┌────────────────────────┐
                 │  GRU Recurrent Core    │◄── hidden h_t (reset per map)
                 │     (hidden = 384)     │
                 └──────┬──────────┬──────┘
                        │          │
          ┌─────────────┘          └─────────────┐
          ▼                                      ▼
   ┌─────────────┐                       ┌──────────────┐
   │ Policy Head │                       │  Value Head  │
   │ (Action     │                       │  V(s_t)      │
   │  Logits)    │                       │  (extrinsic  │
   └─────────────┘                       │  + intrinsic)│
                                         └──────────────┘
   [Training Only] Auxiliary Heads:
     - Next-step HP delta / enemy proximity / door-in-front prediction
```

---

## 3. Module-by-Module Specification (`submission/model.py`)

### Module 1: Ray Encoder (`CircularRayEncoder`)
- **Input:** Tensor `[B, C_ray, 64]` where channels = normalized ray distance (`log1p(d) / scale` or `1 / (1 + d)`) + one-hot hit type (`wall`, `door`, `locked_door`, `enemy`, `item`, `none`).
- **Topology:**
  - `Conv1d(C_ray, 32, kernel_size=5, stride=2, padding=2)` + LayerNorm/SiLU (use `padding_mode='circular'` if rays span 360°, else `'replicate'` for forward FOV — verify FOV in `docs/CONTRACT.md`!).
  - `Conv1d(32, 64, kernel_size=3, stride=2, padding=1)` + SiLU.
  - `Conv1d(64, 64, kernel_size=3, stride=2, padding=1)` + SiLU.
  - Flatten (`64 * 8 = 512`) → `Linear(512, 128)` + LayerNorm + SiLU.
- **Why:** Preserves angular neighborhood structure so the agent immediately detects doorways, gaps, corridors, and cover edges.

### Module 2: Variable-Length Entity Attention (`EntityAttentionEncoder`)
- **Input:** Padded entity tensor `[B, N_max, D_ent]` (e.g., `N_max = 16` closest visible entities) + boolean validity mask `[B, N_max]`.
  - Per-entity features: relative `(dx, dy)` normalized, `sin(angle), cos(angle)`, `log1p(dist)`, one-hot entity category (`enemy`, `projectile`, `keycard`, `health/ammo`, `door/switch`, `other`).
- **Topology:**
  - Per-entity MLP: `Linear(D_ent, 64) → SiLU → Linear(64, 64)`.
  - 1-layer **Multi-Head Self-Attention** (`embed_dim=64, num_heads=4`, key_padding_mask for padded slots) so entities interact (e.g., comparing nearest projectile vs nearest enemy vs keycard).
  - **Learned Query Cross-Attention Pooling** (2 learned queries: e.g., threat focus + objective focus) → outputs `[B, 128]`.
  - Safe zero-fallback when 0 entities are visible (`mask.sum() == 0` → zeros, no NaN softmax!).

### Module 3: Vitals & Action History Encoder (`VitalsEncoder`)
- **Input:** `[health/100, armor/100, ammo/50, has_keycards (one-hot/bits), vx, vy, yaw_rate, prev_action (one-hot)]`.
- **Topology:** `Linear(D_vit, 64) → LayerNorm → SiLU → Linear(64, 64) → SiLU`.

### Module 4: Lightweight Ego-Centric Spatial Memory Grid (`EgoLocalMap`)
- **Purpose:** Solves Tier 2–4 exploration and **Tier 3 keycard → locked door backtracking** without global map coordinates!
- **How it works (inside `DoomAgent`, reset on `reset()`):**
  - Maintain a compact local 2D grid `[C=4, H=21, W=21]` (e.g., each cell = ~64 Doom units, covering ~1344×1344 units around the agent's local odometry frame):
    - Channel 0: **Visited count / recency** (decays slightly or saturates at 1.0 — tells the agent which corridors are already explored!).
    - Channel 1: **Wall/obstacle hits** projected from the 64 rays.
    - Channel 2: **Door / Locked Door sightings** projected from rays/entities (remembers where the locked door was!).
    - Channel 3: **Keycard / Objective sightings**.
  - On each step, update local estimated pose `(x_hat, y_hat, theta_hat)` by integrating ego-centric velocity `(vx, vy)` and turn action/yaw delta (dead-reckoning relative to spawn—no global coords used!).
  - Crop/rotate an ego-centric `11×11` patch around `(x_hat, y_hat)` and encode with a tiny 2-layer 2D CNN → **64-dim** spatial map embedding.
- **Fallback Safety:** Even if odometry drifts over 1,000 steps, the GRU hidden state + recent local patch still provide accurate room-level memory.

### Module 5: Recurrent Core & Heads
- **GRU:** `nn.GRU(input_size=384, hidden_size=384, num_layers=1)` (`128 rays + 128 entities + 64 vitals + 64 local_map = 384`).
- **Actor Head:** Matches `action_space` exactly (Discrete or MultiDiscrete/MultiBinary — verify in `docs/CONTRACT.md`).
- **Dual Value Heads:**
  - `V_ext(s)`: predicts discounted extrinsic reward (for PPO).
  - `V_int(s)`: predicts discounted intrinsic curiosity reward (RND) — keeping two separate value heads prevents non-stationary curiosity from destabilizing extrinsic value estimates!
- **Auxiliary Heads (Training-Only, dropped/unused in `act()`):**
  - Predicts `has_enemy_nearby`, `door_ahead`, `damage_taken_next_step` from GRU output (MSE/BCE aux loss weight `0.1`).

---

## 4. Implementation Checkpoints for `submission/agent.py`
1. `reset()` must zero BOTH the GRU hidden state `self.h` AND `self.ego_map`.
2. `act(obs)` must run inside `with torch.inference_mode():` and handle single-step batching `[1, ...]` with zero memory leaks.
3. Verify action sampling: use `argmax` (or low-temperature sampling `T=0.2` if stuck-detection needs tiny stochasticity—test both on the 12-map holdout suite!).
4. Run a latency benchmark: 1,000 consecutive `act()` calls on 1 CPU thread must average **< 10 ms/step**.
