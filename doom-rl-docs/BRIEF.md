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
  and resume from the same weights (seven rounds, §4). Highlights:
  - an adaptive **reverse curriculum** that starts episodes near the exit or a key on real maps and moves the start back
    after each success;
  - **geodesic progress shaping** toward the next key, then the exit, which pays exactly zero for loops (training only).

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

- 12–56 parallel envs (our laptops plus one rented 192-core machine), 128–256-step rollouts.
- BPTT over each full rollout, with the hidden state reset at episode boundaries.
- 4 epochs × 4 per-env minibatches.
- γ 0.995, GAE λ 0.95, clip 0.2, entropy 0.015, value coefficient 0.5.
- Adam, lr 3e-4 → 5e-5 linear; gradient-norm clip 0.5.
- Episode length random 2,100–4,200 tics, like evaluation.

**Rewards** come from privileged engine state and are used only in training; the agent never sees them.

| Event | Reward | Event | Reward |
|---|---:|---|---:|
| Exit | +25 + 6 × health + 6 × time left | Kill | +1 |
| Key picked up | +8 | Damage dealt | +0.01 / hp |
| First sighting of a key | +0.5 | Item picked up | +0.15 |
| Locked door opened | +4 | Health gained while hurt | +0.025 / hp |
| Door opened | +0.8 | Damage taken | −0.03 / hp |
| New 64-unit cell | +0.04 (cap 600 / episode) | Death | −5 − 10 × time left |
| Geodesic progress (round 6) | Φ(s′) − Φ(s), Φ = −walking distance / 1000 | | |

There is no per-step penalty, so the agent can't learn that dying early pays.

**Map mix per map load:**

- **41 real slots:** Freedoom 2 MAP01–24 ×2 (without the 4 giant maps) + `two_rooms`. Half of the real-map episodes start
  near an exit or a key (reverse curriculum).
- **45 generated slots:** tier 1: 3, tier 2: 12, tier 3: 18, tier 4: 12.

**Tutor rounds** (each one diagnosed from the logs or from teammate review):

1. Tier 3 stuck at 0.1 keys per episode: key reward 3 → 8, plus a first-sighting bonus.
2. Tier 3 at 0% exits: small 3-room tier-3 maps, and a generator fix for lockless layouts.
3. Plain model → EgoMap model warm start; behaviour is identical at the switch, and a test proves it.
4. Replays showed slow, damaged exits: exit bonus for health and time left, plus a stronger damage penalty.
5. Real maps had 0 exits: a reverse curriculum (start near an exit or a key) and an early-death penalty.
6. Geodesic progress shaping toward the next key, then the exit; loops (teleporters included) sum to 0.
7. Tier 4 at 0% exits: start near the exit/key on generated tier-3/4 maps too, with a tier-4-heavy mix.

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
**Final submission: v11** (`shipped-v11`, weights `t2r7b_0447`, round 7). Seeded, 20 seeds per map; exits (v8 in brackets).

| Test set | T1 | T2 | T3 | T4 | Unseen big real maps | Mean |
|---|---|---|---|---|---|---:|
| Final exam 2: 16 untouched maps | 33/40 (29) | 26/40 (22) | 33/40 (37) | **1/40** (0) | 0/160 (0) | 42.2 (38.4) |
| Exam 1: 16 maps | 40/40 (40) | 34/40 (31) | 36/40 (36) | 0/40 (0) | 0/160 (0) | 53.8 (48.9) |
| Holdout: 17 maps | 40/40 (40) | 35/40 (34) | 34/40 (34) | 0/40 (0) | 0/160 (0) | 48.4 (45.2) |

- **v11 vs v8, paired by seed:** exam 2 +3.8 ± 2.1, exam 1 +4.9 ± 1.5, holdout +3.2 ± 1.8; unseen real maps +2.7 ± 1.0
  and +5.9 ± 2.3 on the two exams.
- **Holdout mean over time:** random −13.2 → first EgoMap model 31.2 → v3 40.5 → v8 45.2 → v11 48.4.
<!-- RESULTS:END -->

**Footprint** (official `validate.py`, PASSED):

| Measure | Value |
|---|---|
| Step latency | p50 6.2 ms (v11, official Docker image); MAP32 agent-only p50 5.1 ms on 1 core |
| Official Docker image (`validate.py --docker`) | PASSED, 0 warnings (v11, Linux), 788 MB RAM |
| Compute used on the test map | 0.6 s of the 21 s per-map budget |
| Repo size | 3.4 MB (weights 3.05 MB via Git LFS) |

## 7. Honest limitations and next steps

**Limitations:**

- Tier 4 (two locks): 1 exit in 40 on the final exam; the agent usually dies halfway, after finding keys.
- On big real levels it has never seen (Freedoom 2 MAP25–32, Freedoom 1), it hasn't reached an exit: 0 of 480 episodes.
  It explores and fights there for partial points.
- Selection bias: the holdout over-rated v9, so final picks use untouched exam sets.
- Position dead reckoning drifts 1–2 cells over long episodes. Heading stays exact.
- One night of training: about 12M steps behind the shipped weights (RTX 3050 laptop), plus experiment runs.

**With more time:** many more training steps; a tiny zero-initialized camera branch (`rgb`) to learn to spot exit
switches; Prioritized Level Replay instead of the hand-set map mix; RND curiosity; correcting position drift.
PLR, RND and the camera branch are planned but **not implemented**.
