# Team Ayush — Project Brief

**Yugaantar 2026 · DevForge Track 1 "Can It Play DOOM?"**

> A recurrent PPO agent with a self-built spatial memory map, trained on an infinite stream of procedurally
> generated DOOM levels plus real Freedoom maps.

## 1. What makes it different

- **EgoMap, a spatial memory built only from the agent's own senses and actions.** No engine coordinates: heading
  is tracked by dead reckoning from our own turn presses, with **0.0° error vs. the engine over 600 steps** on three
  maps. The map is an *input* to the network; it never picks actions.
- **An infinite procedural map generator** (`maps/gen_maps.py`) for tiers 1–4, every map load a brand-new level, with
  a **solvability proof** for key/lock levels in its self-test.
- **A "tutor" training loop:** we read the training logs, diagnose the failing skill, adjust rewards or curriculum,
  and resume from the same weights (three rounds so far, §4).

## 2. Architecture (`submission/model.py`, 759,056 parameters as shipped)

```text
 rays: 64 x 360°        entities: <=32 visible     vitals 17 + previous        EgoMap 128x128x6 (spawn-anchored)
 (distance + 6 types)   (5 features + type)        action 9 + 4 front-cone     -> two 15x15 heading-up crops
        |                        |                          |                     (fine 1 cell/px, coarse 8 cells/px)
 circular 1D CNN         16-d type embedding         MLP -> 64                            |
 7->32->64->64 ch,       -> MLP -> 4-head                   |                       small CNN -> 64
 stride 2 -> 128         self-attention -> 2 learned        |                             |
        |                query pooling -> 128               |                             |
        +------------------------+--------------------------+-----------------------------+
                                 v
                      concat 384 (320 without EgoMap)
                                 v
                     GRU (hidden 256) -> LayerNorm
                    +------------+--------------+
                    v                           v
     actor: 6 factored categorical heads   critic: V(s)
     move 3 | strafe 3 | turn 3 | attack 2 | use 2 | next-weapon 2  = 15 logits  ->  MultiBinary(9)
```

- **Factored action heads** make contradictory buttons (forward + back, left + right) impossible.
- **Entity attention** is safe when no entity is visible.
- **EgoMap channels:** visited, free space seen, wall, door, locked door, key. One cell is 64 map units, so the
  128×128 grid covers 8192×8192 units around spawn.

## 3. How Tier 3 (key → locked door → exit) works without global coordinates

1. **Pose by dead reckoning.** Heading comes from our own turn presses, using Doom's turning rules measured per tic
   (slow-turn ramp; turn input ignored for the first 7 tics of an episode). Position comes from the observed
   player-frame velocity and drifts only **47–160 map units after 4,500–7,300 units travelled** (1–2 grid cells).
2. **Remember.** Every step, the 64 rays and visible entities are projected into the spawn-anchored grid. A locked
   door the agent walked past stays in the *locked door* channel, and a key it saw stays in the *key* channel.
3. **Backtrack.** The coarse crop (8 cells/px) keeps far-away remembered doors in view. The vitals tell the network
   which keys it holds, and the GRU carries the episode context, so after the pickup the policy heads back to the
   locked door and then the exit. All decisions come from trained weights, as `docs/RULES.md` allows ("learned
   memory modules", "preprocessing and encoding observations").

## 4. Training (`training/train_ppo.py`, `training/rewards.py`)

**Custom recurrent PPO** (Schulman et al., 2017) in PyTorch, not Stable-Baselines3:

- 12 parallel envs, 128-step rollouts.
- BPTT over each full rollout, with the hidden state reset at episode boundaries.
- 4 epochs × 4 per-env minibatches.
- γ 0.995, GAE λ 0.95, clip 0.2, entropy 0.015, value coefficient 0.5.
- Adam, lr 3e-4 → 5e-5 linear; gradient-norm clip 0.5.
- Episode length random 2,100–4,200 tics, like evaluation.

**Rewards** come from privileged engine state and are used only in training; the agent never sees them.

| Event | Reward | Event | Reward |
|---|---:|---|---:|
| Exit | +25 (+2 × health fraction) | Kill | +1 |
| Key picked up | +8 | Damage dealt | +0.01 / hp |
| First sighting of a key | +0.5 | Item picked up | +0.15 |
| Locked door opened | +4 | Health gained while hurt | +0.01 / hp |
| Door opened | +0.8 | Damage taken | −0.015 / hp |
| New 64-unit cell | +0.04 (cap 600 / episode) | Death | −3 |

There is no per-step penalty, so the agent can't learn that dying early pays.

**Map mix per map load:**

- **25 real slots:** Freedoom 2 MAP01–24 + `two_rooms`.
- **48 generated slots:** tier 1: 6, tier 2: 12, tier 3: 18, tier 4: 12.

**Tutor rounds:**

1. At 460k steps: tier 1 exits 83–92%, tier 2 30–55%, tier 3 stuck at about 0.1 keys/episode. We raised the key
   reward from 3 to 8, added +0.5 for the first key sighting, and gave tier 3 a bigger share.
2. Tier 3 was still at 0% exits. We made 35% of tier-3 maps small (3 rooms), and fixed a generator bug where chain
   layouts produced tier-3 maps with no lock.
3. We trained the plain model first (184k steps), then warm-started the EgoMap model from it with the new map inputs
   initialised to zero. Behaviour is identical at the switch, and a test proves it.

## 5. Procedural map factory (`maps/gen_maps.py`)

The generator writes node-less UDMF maps: rooms on a grid joined by 2-cell-wide corridors, plus doors, enemies and
pickups.

| Tier | Layout |
|---|---|
| 1 | Combat, 1–2 rooms |
| 2 | Exploration, 4–7 rooms with loops |
| 3 | Exit room behind a locked keycard door, key in another branch |
| 4 | Two locks (key 1 is itself behind lock 2), 7–11 rooms |

- **Solvability:** the self-test proves every key is reachable, and that the exit is reachable only after unlocking.
- **Loads cleanly:** generated maps load in the official `doom_env` with exactly the doors and locks the generator
  placed.
- **Holdout, never trained on:** Freedoom MAP25–32 and 8 generated maps from reserved seeds (≥ 1e9). The eval also
  plays `two_rooms`, but that's a training map, so we report MAP25–32 separately as the truly unseen real maps.

## 6. Results (holdout; proxy score from `training/eval_zeroshot.py`, higher is better)

<!-- RESULTS:START -->
| Agent | Mean proxy score | Exit % | Notes |
|---|---:|---:|---|
| Random weights | −13.2 | 0% | Freedoom + `two_rooms` holdout |
| Plain model (184k steps) | 9.9 | 18% | holdout v1 |
| **EgoMap model, shipped (184k plain + 460k EgoMap)** | **34.1** | **29%** | holdout v1 |
| Shipped, current eval (generated maps at 2,100 tics, 2 seeds) | 27.7 | — | T0 = MAP25–32 + `two_rooms` 7.9 (11% exits, all on `two_rooms`), T1 101 (100%), T2 59 (50%), T3 30, T4 9.8 |
<!-- RESULTS:END -->

**Tier 3 works.** The shipped agent solves key → locked door → exit on generated tier-3 holdout maps: 100% and 50%
exits in a frozen-set eval at 4,200 tics, with keys picked up. It also reaches the exit of the official test map (a
training map) on its own in 137 steps.

**Footprint** (official `validate.py`, PASSED):

| Measure | Value |
|---|---|
| Step latency | p50 4.3 ms, p99 8.1 ms |
| First step | 23 ms (warm-up in `__init__`) |
| Compute used on the test map | 0.6 s of the 21 s per-map budget |
| RAM, Linux pinned to 1 CPU | 355 MB of 8,192 MB (random weights) |
| Repo size | 3.4 MB (weights 3.05 MB via Git LFS) |

## 7. Honest limitations and next steps

**Limitations:**

- Tier 4 (two locks) is still at 0% exits.
- On real Freedoom levels it has never seen (MAP25–32), it hasn't reached an exit yet: 0 of 80 episodes for both
  shipped checkpoints. It explores and fights there for partial points.
- At the shortest time cap (2,100 tics), tier-3 maps often time out. Backtracking works, but it is slow.
- Position dead reckoning drifts 1–2 cells over long episodes. Heading stays exact.
- Only about 0.7M training steps so far, at roughly 250–300 env steps/s on one laptop (RTX 4050, i7-13620H).

**With more time:**

- Many more training steps.
- Prioritized Level Replay instead of our hand-set map mix.
- RND curiosity.
- Correcting position drift.

PLR and RND are planned but **not implemented**.
