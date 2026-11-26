# 03 — Training Pipeline, Reward Shaping & Curriculum (as built)

## 1. Overview
1. **Custom recurrent PPO** (`training/train_ppo.py`): our own PyTorch implementation (PPO, Schulman et al., 2017),
   not Stable-Baselines3.
2. **Training-only reward shaping** (`training/rewards.py`), computed from privileged engine state.
3. **Curriculum = a hand-set weighted map mix plus "tutor rounds".** We read the training logs, diagnose the failing
   skill, adjust rewards or curriculum, and resume from the same weights.
4. **Model switch:** we trained the plain model first (184k steps), then warm-started the EgoMap model from it
   (doc 02, Module 4).

---

## 2. Survival-Aware Progression Reward Shaping (`training/rewards.py`)
> **Rule note:** during *training* we read internal engine state (`env.privileged_state()` and ViZDoom's
> `DAMAGECOUNT`) to compute dense rewards. The network NEVER receives these, only the official `doom_env`
> observation dict.

`doom_env`'s built-in reward is only `+1 exit, −1 death, +0.1/kill`. We ignore it and use the following:

| Event | Training Reward | Purpose |
| :--- | :---: | :--- |
| **Reach level exit** | `+25` (+ `2 × health fraction`) | The exit is the largest scoring category; kept above the maximum exploration reward (+24) |
| **Key picked up** | `+8` (tutor round 1: was `+3`) | High-value subgoal |
| **First sighting of a key** | `+0.5` (added in tutor round 1) | Gets the agent to notice keys |
| **Locked door opened** | `+4` | Directly reinforces Tier 3 backtracking |
| **Door opened** | `+0.8` | Teaches USE / door interaction |
| **New 64-unit grid cell** | `+0.04` (cap 600 cells per episode) | Dense exploration signal |
| **Enemy killed** | `+1` | Clears threats blocking progression |
| **Damage dealt** | `+0.01` per hp (`DAMAGECOUNT`) | Dense combat aiming signal |
| **Item picked up** | `+0.15` | Resource management |
| **Health gained while hurt** | `+0.01` per hp | Survival |
| **Damage taken** | `−0.015` per hp | Teaches dodging |
| **Death** | `−3` | Penalizes reckless rushing |

**No per-step time penalty.** A per-step penalty can teach the agent that dying early pays.

Door detection: `door_lock` is per sector (`-1` = not a door, `0` = unlocked door, `> 0` = locked). A door counts as
opened when its sector's ceiling rises.

---

## 3. Custom Recurrent PPO Hyperparameters (as used)
```yaml
num_envs: 12              # parallel doom_env workers (AsyncVectorEnv)
rollout_steps: 128        # 12 * 128 = 1,536 transitions per update
bptt: full rollout        # BPTT over each full 128-step rollout; GRU hidden reset at episode boundaries
num_minibatches: 4        # split along the environment dimension (never shuffle time steps)
update_epochs: 4
lr: 3e-4                  # Adam, linear decay to 5e-5
gamma: 0.995              # long horizon for Tier 3-4 keycard backtracking
gae_lambda: 0.95
clip_coef: 0.2
ent_coef: 0.015
vf_coef: 0.5
max_grad_norm: 0.5
episode_ticks: random 2,100-4,200   # like evaluation
```
`train_ppo.py` still accepts a `--seq-len` flag, but it has no effect: BPTT always spans the full rollout.

> **Critical RL bug to avoid:** when minibatching for recurrent PPO, always split across environments and keep
> sequential time order, resetting the GRU hidden state `h` at exact `done` boundaries inside the sequence.

---

## 4. Curriculum: Weighted Map Mix + Tutor Rounds
**Map mix per map load.** Each env picks one of 73 slots:

| Slots | Maps |
|---:|---|
| 25 | Real maps: Freedoom 2 MAP01–24 + `two_rooms` |
| 6 | Generated tier 1 |
| 12 | Generated tier 2 |
| 18 | Generated tier 3 |
| 12 | Generated tier 4 |

Each generated slot creates a **brand-new map** on every load, with seeds below 1e9. Seeds of 1e9 and above are
reserved for holdout maps.

**Tutor rounds** (from the `[core] tutor` commits):
1. **Memory-grid model at 460k steps:** tier 1 exits 83–92%, tier 2 30–55%, tier 3 stuck at about 0.1
   keys/episode. We raised the key reward from 3 to 8, added +0.5 for the first key sighting, and raised the tier-3
   share.
2. **Tier 3 still at 0% exits:** 35% of tier-3 maps are now small (3 rooms). We also fixed a generator bug where
   chain layouts produced tier-3 maps with no lock.
3. **Model switch:** we trained the plain model first (184k steps), then warm-started the EgoMap model from it with
   the new map inputs initialised to zero. Behaviour is identical at the switch, and a test proves it.

---

## 5. Checkpointing & Monitoring
- `train_ppo.py` writes `<save-path>` (the latest, every 10 updates) and `<stem>.best.pt` (best training return).
- **The final checkpoint is chosen with `training/eval_zeroshot.py` on the holdout maps, not by training return**
  (see doc 04 and `doom-rl-docs/RESULTS.md`).
- **Speed:**
  - On Windows, several ViZDoom games stall each other on the default 15.6 ms system timer. `rewards.py` raises it to
    1 ms in every worker. That gave **7–10× more env steps/s**: 4 games went from 51 to 380 steps/s, and 12 envs
    reach about 1,400 steps/s on small maps.
  - On the real training mix we measured about **250–300 env steps/s** on one laptop. That's about 0.7M training
    steps by the time of writing.

---

## Not implemented (future work)
- **Behavior cloning warm-start from human demos** (`record_demos.py`, `train_bc.py`). It was cut because 10–15 min
  of demos (~10–20k steps) is too little data, and there wasn't time. The original reasoning: a BC prior teaches
  moving, doors and aiming before RL starts. It needs a KL/BC term early in PPO, or RL erases it.
- **RND curiosity** (a fixed random target network plus a trained predictor; intrinsic reward = normalized
  prediction error, `β ≈ 0.05`; a separate `γ_int = 0.99` and its own value head). The original reasoning: a novelty
  signal in observation space that doesn't depend on engine coordinates. It was cut because the grid-cell and
  first-key-sighting rewards already cover exploration during training.
- **Prioritized Level Replay** (Jiang et al., 2021): rank-based sampling by positive value loss, mixed with a
  staleness distribution. We use the fixed weighted map mix plus tutor rounds instead.
- **Auxiliary losses** (`aux_coef 0.1`; see doc 02).
- **Rewards from the plan that were not built:**
  - A sector-area exploration reward (`env.sector_polygons()`). We use 64-unit grid cells instead.
  - A stuck penalty.
  - A USE-spam penalty.
  - A per-step time penalty. This one was dropped on purpose (see §2).
- **Truncated BPTT in 64-step chunks.** We use BPTT over the full 128-step rollout.
- **Automatic holdout eval every 1M steps with `best_zeroshot.pt`.** Holdout evals are run by hand with
  `eval_zeroshot.py`.
