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

**Core Strategy:** Optimize for **survival-aware progression** (clear threats blocking the path, explore branches systematically, remember locked doors/keycards, backtrack to unlock, and reach the exit alive with high HP) using a **custom spatial-memory neural architecture** trained on **continuously generated procedural maps with Prioritized Level Replay (PLR)**.

---

## 2. Hard Rules (NEVER VIOLATE)
1. **No Hardcoded Control Logic in `submission/agent.py`:** No rule trees (`if enemy: shoot`), no classical pathfinders (`A*`, `Dijkstra`), no trigonometry aimbots (`atan2`). Decisions must come from trained neural parameters.
2. **No Pretrained External Models / Vanilla SB3:** Do not download pretrained ViZDoom weights from Hugging Face/GitHub, and do not submit an off-the-shelf Stable-Baselines3 PPO script. Build our custom PyTorch architecture and custom Recurrent PPO loop cleanly.
3. **Engine Rules:** You MAY modify rewards/curriculum/maps during *training* (e.g., dense exploration reward using engine coords). At *evaluation*, the agent runs strictly on the unmodified DOOM engine and ONLY sees the `observation` dictionary passed to `act(obs)`.
4. **Submission Limits:**
   - Public GitHub repo created from `devforge-doom-template`, total size **≤ 1 GB** (weights via Git LFS).
   - **No Dockerfile** in submission (runs in organizers' Python 3.13 base image).
   - Eval sandbox: RTX 3090, **≤ 2 GB VRAM**, **≤ 8 GB RAM**, **1 CPU thread**, offline, read-only repo.
   - Timeouts: 60s `__init__`, 10s `reset()`, 10s first `act()`, 5s any other `act()`, **~40 ms/step average**.
   - Must pass `python validate.py` before any checkpoint is considered valid.

---

## 3. Target Repository Layout (Inside `devforge-doom-template`)
```text
devforge-doom-template/
├── docs/
│   └── CONTRACT.md                 # Official obs/action contract (READ FIRST once cloned)
├── env/doom_env/                   # Official wrapper & limits.py (DO NOT MODIFY FOR EVAL)
├── submission/
│   ├── agent.py                    # DoomAgent entrypoint (__init__, reset, act)
│   ├── model.py                    # Custom Ray1DCNN + EntityAttention + GRU + SpatialMemory
│   ├── preprocess.py               # Observation dict -> normalized tensors + odometry update
│   ├── weights/                    # Final trained checkpoint (Git LFS, < 50 MB)
│   └── requirements.txt            # Only extra pinned packages not in base image
├── training/
│   ├── train_ppo.py                # Custom Recurrent PPO (truncated BPTT) + RND + PLR
│   ├── train_bc.py                 # Optional 5-min Behavior Cloning warm-start from human demos
│   ├── record_demos.py             # Keyboard recorder inside doom_env for 10-15 min human demos
│   ├── rewards.py                  # Training-only dense reward shaping wrapper
│   ├── plr.py                      # Prioritized Level Replay buffer & difficulty scheduler
│   └── eval_zeroshot.py            # Evaluates checkpoint on 12 held-out unseen maps
├── maps/
│   ├── gen_udmf.py                 # Procedural Python UDMF generator (Tiers 1-3)
│   ├── gen_obsidian.py             # Batch wrapper for Obsidian CLI (Tier 4 Freedoom gauntlets)
│   ├── train_pool/                 # Generated training WADs (gitignored or small subset)
│   └── eval_holdout/               # 12 strictly held-out unseen maps (3 per tier)
└── validate.py                     # Official local validator
```

---

## 4. Documentation Map
Read the specific file for your task:
1. [01_PROBLEM_AND_CONTRACT.md](./doom-rl-docs/01_PROBLEM_AND_CONTRACT.md) — Official scoring, observation/action contract, sandbox limits, and compliance rules.
2. [02_MODEL_ARCHITECTURE.md](./doom-rl-docs/02_MODEL_ARCHITECTURE.md) — Exact PyTorch architecture (Ray 1D-CNN, Entity Attention, GRU, Ego-Centric Spatial Grid, Aux Heads).
3. [03_TRAINING_AND_REWARDS.md](./doom-rl-docs/03_TRAINING_AND_REWARDS.md) — Custom Recurrent PPO, Behavior Cloning warm-start, Reward Shaping, RND Curiosity, and PLR Curriculum.
4. [04_PROCEDURAL_MAPS.md](./doom-rl-docs/04_PROCEDURAL_MAPS.md) — Procedural UDMF generator, Obsidian CLI pipeline, and 12-map Zero-Shot Holdout Suite.
5. [05_EXECUTION_AND_HARDWARE.md](./doom-rl-docs/05_EXECUTION_AND_HARDWARE.md) — Two-laptop + Cloud Credits workflow, build order, validation checklist, and 2-page PDF brief structure.

---

## 5. First Step for Any Agent Starting Now
1. Check if `devforge-doom-template` is cloned. If yes, read `docs/CONTRACT.md`, `env/doom_env/limits.py`, and `validate.py` **before writing any code** so tensor shapes and action indices match 100%.
2. Run `python validate.py` on the baseline template to confirm the local environment works.
3. Follow the build order in [05_EXECUTION_AND_HARDWARE.md](./05_EXECUTION_AND_HARDWARE.md).
