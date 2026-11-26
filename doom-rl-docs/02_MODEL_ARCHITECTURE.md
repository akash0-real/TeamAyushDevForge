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
   64 Rays (360°)    ▼          ▼           ▼      Vitals + Prev Action
   (dist + hit_type) │  Visible Entities ≤32│    (HP, armor, ammo, keys,
                     │ (dist,sin,cos,relvel,│   weapon, vel, prev_action)
                     │       type)          │
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
   │ (6 factored │                       │  V(s_t)      │
   │ categorical)│                       │  (extrinsic  │
   └─────────────┘                       │  + intrinsic)│
                                         └──────────────┘
   [Training Only] Auxiliary Heads:
     - Next-step HP delta / enemy proximity / door-in-front prediction
```

---

## 3. Module-by-Module Specification (`submission/model.py`)

### Module 1: Ray Encoder (`CircularRayEncoder`)
- **Input:** Tensor `[B, 7, 64]`: `rays.distance` (already in `[0,1]`; optionally add `sqrt(d)` for more near-range resolution) + one-hot `rays.hit_type` (6: `none`, `wall`, `door`, `locked_door`, `enemy`, `item`).
- **Topology:**
  - `Conv1d(7, 32, kernel_size=5, stride=2, padding=2, padding_mode='circular')` + LayerNorm/SiLU. Rays span 360° (index 0 = ahead, CCW), so **circular padding is correct**.
  - `Conv1d(32, 64, kernel_size=3, stride=2, padding=1)` + SiLU.
  - `Conv1d(64, 64, kernel_size=3, stride=2, padding=1)` + SiLU.
  - Flatten (`64 * 8 = 512`) → `Linear(512, 128)` + LayerNorm + SiLU.
- **Why:** Preserves angular neighborhood structure so the agent immediately detects doorways, gaps, corridors, and cover edges.

### Module 2: Variable-Length Entity Attention (`EntityAttentionEncoder`)
- **Input:** `visible_entities` as given: `features [B, 32, 5]` = `[dist/1024, sin(rel_angle), cos(rel_angle), rel_vel_fwd, rel_vel_right]`, `entity_type [B, 32]` (12 classes, 0 = pad), `mask [B, 32]`. Entities only come from the **90° forward FOV**; there are no `(dx, dy)` coords (derive `dx = d·cos, dy = d·sin` as extra features if useful — that's encoding, not control). Doors/switches are **not** entities — they only appear in rays.
  - Per-entity input: the 5 features (clip `rel_vel` to ±30, divide by 30) + `nn.Embedding(12, 16)` of `entity_type` → `D_ent = 21`.
- **Topology:**
  - Per-entity MLP: `Linear(21, 64) → SiLU → Linear(64, 64)`.
  - 1-layer **Multi-Head Self-Attention** (`embed_dim=64, num_heads=4`, key_padding_mask for padded slots) so entities interact (e.g., comparing nearest projectile vs nearest enemy vs keycard).
  - **Learned Query Cross-Attention Pooling** (2 learned queries: e.g., threat focus + objective focus) → outputs `[B, 128]`.
  - Safe zero-fallback when 0 entities are visible (`mask.sum() == 0` → zeros, no NaN softmax!).

### Module 3: Vitals & Action History Encoder (`VitalsEncoder`)
- **Input (from the 17-dim `vitals`):** `health/100, armor/100` (both range 0–200), `selected_weapon` one-hot(8), `selected_weapon_ammo/50`, `log1p` of the 4 ammo counts, 6 key flags, `vel_forward/16, vel_right/16, vel_up/16`, plus `prev_action` (9 bits — the policy needs this to learn the USE release rule) and `turn_hold_count` (steps the current turn button has been held, capped at 3). **No `yaw_rate` exists in the observation** — heading change must come from our own previous turn actions.
- **Topology:** `Linear(D_vit, 64) → LayerNorm → SiLU → Linear(64, 64) → SiLU`.

### Module 4: Lightweight Ego-Centric Spatial Memory Grid (`EgoLocalMap`)
- **Purpose:** Solves Tier 2–4 exploration and **Tier 3 keycard → locked door backtracking** without global map coordinates!
- **How it works (inside `DoomAgent`, reset on `reset()`):**
  - Maintain a **spawn-anchored** 2D grid `[C=5, H=128, W=128]`, `float16`, each cell = 64 Doom units (covers 8192×8192 units, ~160 KB). It must be anchored to the spawn frame, **not** re-centered on the agent: a 21×21 agent-centred window (old plan) forgets the locked door as soon as you walk ~670 units away, which breaks Tier 3 backtracking.
    - Channel 0: **Visited** (saturating count / recency).
    - Channel 1: **Seen free space** (cells along each ray before its hit point). `seen-free AND not-visited` = frontier, which is what drives Tier 2 exploration.
    - Channel 2: **Wall/obstacle hits** projected from the 64 rays.
    - Channel 3: **Door / Locked Door hits** (`hit_type` 2/3; remembers where the locked door was).
    - Channel 4: **Key / item sightings** (entity type `key`, projected from distance + angle).
  - Projection: ray `i` points at angle `theta_hat + i·(2π/64)` (CCW), hit at `distance·1024` units; entity at `theta_hat + atan2(sin, cos)` and `dist·1024`. (Trig here is *encoding* the observation into memory, allowed by the rules; it never chooses an action.)
  - **Pose estimate (dead-reckoning from spawn, no global coords):**
    - `theta_hat`: integrate our own TURN buttons with a turn table indexed by how long the button has been held. Measured on `two_rooms.wad`: a single-step tap ≈ **7.03°**, a held turn ≈ **14.06°/step** after the first step (Doom slow-turn ramp). Calibrate the full table against `privileged_state()["angle"]` in training.
    - `(x_hat, y_hat)`: integrate `vel_forward, vel_right` × 4 tics, rotated by `theta_hat`. Measured error is large (e.g. 0.1 true vs 14.2 estimated units when blocked by a wall), because velocity is only sampled at the last tic. Better: average previous and current velocity, and consider correcting yaw/translation by circular cross-correlation of consecutive 360° ray scans.
    - Log `|pose_hat − true pose|` per episode during training (from `privileged_state`) so we know how bad drift actually is on 1,000-step maps.
  - Network input: crop an ego-centric, heading-rotated `21×21` patch around `(x_hat, y_hat)` + a `16×16` max-pooled view of the whole grid (keeps the far-away locked door visible) → tiny 2-layer 2D CNN → **64-dim** embedding.
- **Fallback Safety:** If odometry drifts, the GRU hidden state + the local patch still give room-level memory; also train with injected pose noise so the policy doesn't over-trust the map.

### Module 5: Recurrent Core & Heads
- **GRU:** `nn.GRU(input_size=384, hidden_size=384, num_layers=1)` (`128 rays + 128 entities + 64 vitals + 64 local_map = 384`).
- **Actor Head:** `action_space` is `MultiBinary(9)`. Use **6 factored categorical heads** instead of 9 independent Bernoullis, so contradictory presses (forward+back, turn left+right) can't happen: `move {none, fwd, back}`, `strafe {none, left, right}`, `turn {none, left, right}`, `attack {0,1}`, `use {0,1}`, `next_weapon {0,1}` → 3+3+3+2+2+2 = 15 logits. Log-prob = sum of the 6 heads; map back to the 9-bit vector in `act()`.
- **Dual Value Heads:**
  - `V_ext(s)`: predicts discounted extrinsic reward (for PPO).
  - `V_int(s)`: predicts discounted intrinsic curiosity reward (RND) — keeping two separate value heads prevents non-stationary curiosity from destabilizing extrinsic value estimates!
- **Auxiliary Heads (Training-Only, dropped/unused in `act()`):**
  - Predicts `has_enemy_nearby`, `door_ahead`, `damage_taken_next_step` from GRU output (MSE/BCE aux loss weight `0.1`).

---

## 4. Implementation Checkpoints for `submission/agent.py`
1. `reset()` must zero BOTH the GRU hidden state `self.h` AND `self.ego_map`.
2. `act(obs)` must run inside `with torch.inference_mode():` and handle single-step batching `[1, ...]` with zero memory leaks.
3. Verify action sampling: per-head `argmax` vs. sampling at low temperature (`T≈0.2–0.5`). A deterministic recurrent policy often loops in place, so test both on the 12-map holdout suite. Return `np.ndarray` of shape `(9,)`, dtype `int8`, values in `{0,1}`.
4. Run a latency benchmark: 1,000 consecutive `act()` calls on 1 CPU thread must average **< 5 ms/step** (same target as doc 01; hard limit is 40 ms avg per map). Benchmark both `device="cuda"` and CPU; for a ~2M-param model with batch 1, CPU can beat CUDA launch overhead, and you're allowed to ignore the `device` hint.
5. The grid update (projecting 64 rays + entities) runs in numpy every step, so include it in the latency benchmark.
