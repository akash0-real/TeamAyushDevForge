# AGENTS.md — Yugaantar DevForge Track 1 ("Can It Play DOOM?")

> **Purpose:** Master context file for any AI coding agent or teammate starting from scratch. Read this file first, then read the linked specs for the module you are working on.

---

## 1. Mission Summary
We are building an autonomous **Deep Reinforcement Learning** agent for **Yugaantar DevForge — Track 1 ("Can It Play DOOM?")** using the **ViZDoom** engine (`vizdoom==1.3.3`) and the official `devforge-doom-template` (`doom_env` Gymnasium wrapper).

The agent must play **unseen Freedoom 2 maps zero-shot** across 4 difficulty tiers:
- **Tier 1:** Combat & Survival
- **Tier 2:** Multi-Room Exploration
- **Tier 3:** Long-Horizon Progression (`Keycard → Locked Door → Exit`)
- **Tier 4:** Full Gauntlet

**Core Strategy:** Optimize for **survival-aware progression** (clear threats blocking the path, explore branches systematically, remember locked doors/keycards, backtrack to unlock, and reach the exit alive with high HP) using a **custom spatial-memory neural architecture** (EgoMap + GRU) trained with **custom recurrent PPO**. Training uses an **infinite stream of procedurally generated maps plus real Freedoom 2 maps**, with a hand-set weighted map mix and manual "tutor" curriculum rounds. The 2-page summary is [`doom-rl-docs/BRIEF.md`](./doom-rl-docs/BRIEF.md).

---

## 2. Hard Rules (NEVER VIOLATE)
1. **No Hardcoded Control Logic in `submission/agent.py`:** No rule trees (`if enemy: shoot`), no classical pathfinders or planners (`A*`, `Dijkstra`, BFS over our memory grid), no trigonometry aimbots (`atan2`). Decisions must come from trained neural parameters. Allowed: observation preprocessing/encoding, action masking from observation fields, frame stacking, learned memory (so `EgoMap` is fine as long as it is only an *input* to the network, never used to pick actions).
2. **No Pretrained External Models / Vanilla SB3:** Do not download pretrained ViZDoom weights from Hugging Face/GitHub, and do not submit an off-the-shelf Stable-Baselines3 PPO script. Build our custom PyTorch architecture and custom Recurrent PPO loop cleanly.
3. **Engine Rules:** You MAY modify rewards/curriculum/maps during *training* (e.g., dense exploration reward using engine coords). At *evaluation*, the agent runs strictly on the unmodified DOOM engine and ONLY sees the `observation` dictionary passed to `act(obs)`.
4. **Submission Limits:**
   - Public GitHub repo **created during the hackathon from [`Reinforce-SST/devforge-doom-template`](https://github.com/Reinforce-SST/devforge-doom-template)** ("Use this template" on GitHub), total size **≤ 1 GB** (weights via Git LFS). The template files were copied into this repo (commit `35a32ee`) rather than generated with GitHub's template button; contents match the official template exactly.
   - Scorer runs the organizers' Python 3.13 image; don't rely on any Dockerfile of ours. Keep the template's own `Dockerfile` — `validate.py --docker` uses it.
   - Eval sandbox: RTX 3090, **≤ 2 GB VRAM**, **≤ 8 GB RAM** (incl. `/tmp`), **1 CPU thread**, offline, read-only repo.
   - Timeouts: 60s `__init__`, 10s `reset()`, 10s first `act()`, 5s any other `act()`, plus a **per-map compute budget of 40 ms × decision steps** (e.g. 21 s on a 2,100-tick map). Source of truth: `env/doom_env/limits.py`.
   - Must pass `python validate.py` (ideally `--docker`) before any checkpoint is considered valid.

---

## 3. Repository Layout (as built)
```text
TeamAyushDevForge/
├── docs/                           # CONTRACT.md (obs/action), EVALUATION.md, RULES.md, SUBMISSION.md
├── env/doom_env/                   # Official wrapper, spaces.py, limits.py (scorer ignores our env/ folder)
├── tests/                          # Official tests + fixtures; test_egomap.py, test_submission.py (ours)
├── Dockerfile, docker/             # Official image recipe for `validate.py --docker` (keep as-is)
├── submission/
│   ├── agent.py                    # DoomAgent entrypoint (__init__ + warm-up, reset, act, load_weights)
│   ├── model.py                    # CircularRayEncoder + EntityAttentionEncoder + EgoMapEncoder + GRU (DoomSpatialPolicy)
│   ├── preprocess.py               # Observation dict -> normalized arrays; EgoMap (dead-reckoned spatial memory)
│   ├── weights/best_agent.pt       # Shipped checkpoint (Git LFS, ~3 MB)
│   └── requirements.txt            # Empty: no extra packages
├── training/
│   ├── train_ppo.py                # Custom recurrent PPO (BPTT over full rollouts, per-env minibatches)
│   ├── rewards.py                  # Training-only reward shaping + weighted map mix (ShapedDoomEnv)
│   ├── eval_zeroshot.py            # Scores a checkpoint through submission/agent.py on holdout maps
│   ├── inspect_maps.py             # Map inspection helper
│   └── results/                    # Eval CSVs
├── maps/
│   ├── gen_maps.py                 # Infinite procedural generator, tiers 1-4, solvability self-test
│   ├── holdout/                    # 8 generated holdout maps (reserved seeds >= 1e9)
│   └── two_rooms.wad               # Official test map
├── doom-rl-docs/                   # Design docs 01-05, RESULTS.md, BRIEF.md / BRIEF.pdf
└── validate.py                     # Official local validator
```

---

## 4. Documentation Map
Read the specific file for your task:
1. [01_PROBLEM_AND_CONTRACT.md](./doom-rl-docs/01_PROBLEM_AND_CONTRACT.md) — Official scoring, observation/action contract, sandbox limits, and compliance rules.
2. [02_MODEL_ARCHITECTURE.md](./doom-rl-docs/02_MODEL_ARCHITECTURE.md) — The built PyTorch architecture: Ray 1D-CNN, Entity Attention, EgoMap spatial memory, GRU, factored action heads.
3. [03_TRAINING_AND_REWARDS.md](./doom-rl-docs/03_TRAINING_AND_REWARDS.md) — Custom recurrent PPO, reward shaping, weighted map mix and tutor rounds.
4. [04_PROCEDURAL_MAPS.md](./doom-rl-docs/04_PROCEDURAL_MAPS.md) — The procedural generator (`maps/gen_maps.py`), its solvability self-test, and the holdout suite.
5. [05_EXECUTION_AND_HARDWARE.md](./doom-rl-docs/05_EXECUTION_AND_HARDWARE.md) — Two-laptop workflow, what was built in order, and the final verification checklist.
6. [RESULTS.md](./doom-rl-docs/RESULTS.md) and [BRIEF.md](./doom-rl-docs/BRIEF.md) — Holdout results, and the 2-page project brief.

---

## 5. First Step for Any Agent Starting Now
1. Check if the team repo (created from `devforge-doom-template`) is cloned. If yes, read `docs/CONTRACT.md`, `env/doom_env/spaces.py`, `env/doom_env/limits.py`, and `validate.py` **before writing any code** so tensor shapes and action indices match 100%. [01_PROBLEM_AND_CONTRACT.md](./doom-rl-docs/01_PROBLEM_AND_CONTRACT.md) already mirrors contract `1.0.0-draft`; re-check if `doom_env.CONTRACT_VERSION` changes.
2. Run `python validate.py` on the baseline template to confirm the local environment works.
3. Follow the final verification checklist in [05_EXECUTION_AND_HARDWARE.md](./doom-rl-docs/05_EXECUTION_AND_HARDWARE.md).

---

## Not implemented (future work)
These were in the original plan and were cut or replaced. Each doc's own "Not implemented" list has the details.

- **Prioritized Level Replay (PLR).** Replaced by a fixed weighted map mix plus manual "tutor" curriculum rounds
  (doc 03).
- **Behavior cloning warm-start from human demos.** Cut, because there was too little demo data for the time
  available (doc 03).
- **RND curiosity and dual value heads.** Not built; there is a single value head (docs 02, 03).
- **Auxiliary heads.** Not built (doc 02).
- **Obsidian-generated tier-4 maps.** Not built; our own generator makes tiers 1–4 (doc 04).
- **Truncated BPTT in 64-step chunks.** Replaced by BPTT over each full 128-step rollout (doc 03).
- **Cloud VM burst training.** Not done; all training so far ran on one laptop (doc 05).
