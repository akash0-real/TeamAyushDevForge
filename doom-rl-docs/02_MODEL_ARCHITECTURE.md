# 02 — Neural Architecture (as built: `submission/model.py`, `submission/preprocess.py`)

## 1. Design Goals
- **Custom and worth a judge's time:** a bespoke spatial-memory architecture for partial observability, variable
  enemy counts and long-horizon keycard backtracking.
- **Fast and lightweight:** **596,624 parameters** for the plain model and **759,056 with EgoMap** (the shipped
  agent). Measured step latency with `validate.py` is p50 **4.3 ms** and p99 8.1 ms.
- **Generalizes zero-shot:** the policy works only on ego-centric features plus a memory map built from its own
  observations and actions. It never sees engine coordinates.

---

## 2. End-to-End Architecture Diagram
```text
                   ┌──────────────────────────┐
                   │   doom_env Observation   │
                   └─┬──────────┬───────────┬─┘
                     │          │           │
   64 Rays (360°)    ▼          ▼           ▼      Vitals 17 + Prev Action 9      EgoMap 128×128×6
   (dist + hit_type) │  Visible Entities ≤32│      + 4 front-cone summary        (built from obs +
                     │ (dist,sin,cos,relvel,│               = 30                  our own actions)
                     │       type)          │                │                          │
                     ▼                      ▼                ▼                          ▼
         ┌──────────────────┐   ┌────────────────────┐   ┌──────┐          two 15×15 heading-up crops
         │ Circular 1D-CNN  │   │ Type emb 16 + MLP  │   │ MLP  │          (fine + 8× coarse) → CNN → 64
         │ 7→32→64→64 ch,   │   │ + 4-head self-attn │   │(64d) │                          │
         │ stride 2 → 128   │   │ + 2-query pooling  │   └──┬───┘                          │
         └────────┬─────────┘   │      → 128         │      │                              │
                  │             └─────────┬──────────┘      │                              │
                  └───────────┬───────────┴─────────────────┴──────────────────────────────┘
                              ▼
                  Concat 320 (plain) / 384 (with EgoMap, shipped)
                              ▼
                 ┌────────────────────────┐
                 │  GRU (hidden = 256)    │◄── hidden h_t (reset per map)
                 │  → LayerNorm           │
                 └──────┬──────────┬──────┘
                        ▼          ▼
               ┌──────────────┐  ┌──────────────┐
               │ Actor: 6     │  │ Critic: V(s) │
               │ factored     │  │ (single head)│
               │ categorical  │  └──────────────┘
               └──────────────┘
```

---

## 3. Module-by-Module Specification

### Module 1: Ray Encoder (`CircularRayEncoder`)
- **Input:** tensor `[B, 7, 64]`: `rays.distance` (already in `[0,1]`) plus one-hot `rays.hit_type`. The 6 hit
  types are `none`, `wall`, `door`, `locked_door`, `enemy` and `item`.
- **Topology:**
  - `Conv1d(7, 32, kernel_size=5, stride=2, padding=2, padding_mode='circular')` + SiLU. The rays span 360°, with
    index 0 straight ahead and indices counter-clockwise, so **circular padding is correct**.
  - `Conv1d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular')` + SiLU.
  - `Conv1d(64, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular')` + SiLU.
  - Flatten (`64 * 8 = 512`) → `Linear(512, 128)` + LayerNorm + SiLU.
- **Why:** it keeps the angular neighbourhood structure, so the agent picks out doorways, gaps, corridors and cover
  edges immediately.

### Module 2: Variable-Length Entity Attention (`EntityAttentionEncoder`)
- **Input:** `visible_entities` as given:
  - `features [B, 32, 5]` = `[dist, sin(rel_angle), cos(rel_angle), rel_vel_fwd, rel_vel_right]`
  - `entity_type [B, 32]`: 12 classes, 0 = pad
  - `mask [B, 32]`
- **What the entities cover:** only the **90° forward FOV**, and there are no `(dx, dy)` coordinates. Doors and
  switches are not entities; they appear only in the rays.
- **Per-entity input:** the 5 features (relative velocity squashed with `tanh(v / 15)`) plus a learned 16-d
  embedding of `entity_type`.
- **Topology:**
  - Per-entity MLP.
  - 1-layer **multi-head self-attention** with 4 heads and a key padding mask, so entities interact (for example,
    comparing the nearest projectile, the nearest enemy and a keycard).
  - **Learned-query cross-attention pooling** with 2 learned queries → `[B, 128]`.
  - When no entity is visible, it falls back safely to zeros instead of a NaN softmax.

### Module 3: Vitals & Action History Encoder
- **Input (30):**
  - The 17 normalized vitals: health, armor, selected weapon, ammo, 6 key flags, velocity.
  - The previous action (9 bits). The policy needs it to learn the USE release rule.
  - 4 front-cone summary features from the rays and entities just ahead: nearest distance, door ahead, locked door
    ahead, enemy visible.
- **No `yaw_rate` exists in the observation.** Heading change comes from our own previous turn actions (Module 4).
- **Topology:** `Linear(30, 64) → LayerNorm → SiLU → Linear(64, 64) → SiLU`.

### Module 4: EgoMap Spatial Memory (`EgoMap` in `preprocess.py`, `EgoMapEncoder` in `model.py`)
- **Purpose:** Tier 2–4 exploration and **Tier 3 keycard → locked door backtracking** without global map coordinates.
- **How it works (inside `DoomAgent`, cleared on `reset()`):**
  - A **spawn-anchored** grid `[C=6, H=128, W=128]`. Each cell is 64 map units, so the grid covers 8192×8192 units
    around spawn.
  - **Why spawn-anchored and not re-centred on the agent:** an agent-centred window forgets the locked door as soon
    as you walk away from it, which breaks Tier 3 backtracking.
  - **Channels:**
    - 0: **Visited**.
    - 1: **Free space seen**, the cells along each ray before its hit point. "Seen free and not visited" is the
      frontier, which drives exploration.
    - 2: **Wall** hits.
    - 3: **Door** hits.
    - 4: **Locked door** hits. This is what remembers where the locked door was.
    - 5: **Key** sightings, projected from the entity's distance and angle.
  - **Projection:** each ray and entity is placed using our heading estimate plus its observed angle and distance.
    The trigonometry here *encodes* the observation into memory, which the rules allow. It never chooses an action.
  - **Pose estimate (dead reckoning from spawn, no global coordinates):**
    - **Heading** comes from our own turn presses, using Doom's turning rules measured per tic: a slow-turn ramp, and
      turn input ignored during the first 7 tics of an episode. **Heading error vs. the engine is 0.0° over 600
      steps** on three maps (`tests/test_egomap.py`).
    - **Position** comes from the observed player-frame velocity (`vel_forward`, `vel_right`), rotated by the heading
      and averaged over the previous and current step. Drift is **47–160 map units after 4,500–7,300 units
      travelled**, which is 1–2 grid cells.
  - **Network input:** two ego-centric, **heading-up 15×15 crops**, a fine one (1 cell/px) and a coarse one
    (8 cells/px, which keeps a far-away locked door in view). They go through a small 2D CNN (`EgoMapEncoder`)
    → **64-dim**.
- **Warm start:** the EgoMap model was warm-started from the plain model with the new map inputs initialised to zero,
  so behaviour is identical at the switch (`tests/test_egomap.py`).
- **Fallback safety:** if position drifts, the GRU hidden state and the fine crop still give room-level memory.

### Module 5: Recurrent Core & Heads
- **GRU:** `nn.GRU(input_size=320 or 384, hidden_size=256, num_layers=1)` → LayerNorm. The input is 128 rays + 128
  entities + 64 vitals, plus 64 from EgoMap in the shipped model.
- **Actor head:** `action_space` is `MultiBinary(9)`. We use **6 factored categorical heads** instead of 9
  independent Bernoullis, so contradictory presses (forward + back, turn left + right) can't happen. The heads are
  `move {none, fwd, back}`, `strafe {none, left, right}`, `turn {none, left, right}`, `attack {0,1}`, `use {0,1}`
  and `next_weapon {0,1}`: 3+3+3+2+2+2 = 15 logits. The log-prob is the sum over the 6 heads; `act()` maps the
  result back to the 9-bit vector.
- **Value head:** a single `V(s)` for the extrinsic (shaped) return.

---

## 4. Implementation Checkpoints for `submission/agent.py`
1. `reset()` zeroes both the GRU hidden state and the EgoMap.
2. `act(obs)` runs under `torch.inference_mode()` with single-step batching `[1, ...]`.
3. The model is warmed up in `__init__`, so the first step takes 23 ms instead of seconds.
4. The latency benchmark uses the official `validate.py`: p50 4.3 ms, p99 8.1 ms. That includes the numpy EgoMap
   update, which runs every step.

---

## Not implemented (future work)
- **Dual value heads (`V_ext` + `V_int` for RND).** Planned to keep non-stationary curiosity rewards from
  destabilizing the extrinsic value. We cut it because RND was not built (see doc 03). There is a single value head.
- **Auxiliary heads** (enemy nearby, door ahead, next-step damage; training-only, loss weight 0.1). Cut for time.
- **Original EgoMap read-out:** an agent-centred 21×21 crop plus a 16×16 max-pooled view of the whole grid. We built
  two 15×15 heading-up crops (fine + 8× coarse) instead.
- **A larger model (~1.5M–2.5M parameters, GRU hidden 384).** The built model is 759K parameters with GRU hidden 256.
- **Correcting position drift** by circular cross-correlation of consecutive 360° ray scans.
- **Training with injected pose noise**, so the policy doesn't over-trust the map.
- **Logging pose error** (estimated vs. true pose) per training episode. Instead, the heading and position error
  were measured offline: 0.0° heading error, and 47–160 units of drift.
