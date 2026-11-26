# 05 — Execution Plan, Hardware Split & Submission Checklist

## 1. Hardware Allocation (Do NOT Interconnect Laptops via DDP)
Interconnecting a 3050 laptop and 4050 laptop over LAN/Wi-Fi adds huge sync overhead for a ~2M-param model. Use parallel roles instead:

| Machine | Role |
| :--- | :--- |
| **Laptop 1** (i7-13th Gen, RTX 4050 6GB, 16GB RAM) | Primary local dev, 12–16 parallel `doom_env` PPO trainer, BC warm-start |
| **Laptop 2** (RTX 3050 6GB) | Procedural map factory (UDMF + Obsidian), `eval_zeroshot.py` video renders, `validate.py`, hyperparameter/reward sweeps |
| **Cloud VM** (using cloud credits) | Final high-throughput run: **16–32 vCPUs + 1× GPU (L4/T4/3090/A100)** running 32–64 parallel envs for 20M–50M steps |

---

## 2. Step-by-Step Build Order

### Phase 1: Contract & Skeleton (First 60–90 mins)
- [x] Template files from [`Reinforce-SST/devforge-doom-template`](https://github.com/Reinforce-SST/devforge-doom-template) copied into `TeamAyushDevForge` (commit `35a32ee`). Keep the repo public.
- [ ] Read `docs/CONTRACT.md`, `docs/EVALUATION.md`, `docs/RULES.md`, `env/doom_env/spaces.py`, `env/doom_env/limits.py`.
- [ ] Run `python validate.py` on the untouched template to verify ViZDoom + Python 3.13 environment.
- [ ] Confirm `observation_space`/`action_space` still match doc 01 (contract `1.0.0-draft`).
- [ ] Implement `submission/preprocess.py`, `submission/model.py`, and `submission/agent.py` (untrained random weights).
- [ ] Re-run `python validate.py` — verify **0 errors**, **< 5 ms/step**, **< 400 MB VRAM**.
- [ ] Calibrate odometry: turn table + velocity integration vs `privileged_state()` angle/x/y (see doc 02, Module 4).

### Phase 2: Map Factory & Zero-Shot Eval (Parallel with Phase 1)
- [ ] Build `maps/gen_udmf.py` (Tiers 1–3) + `maps/verify_wads.py` (confirms every WAD loads cleanly in `doom_env`).
- [ ] Set up Obsidian CLI (`maps/gen_obsidian.py`) for Tier 4 maps.
- [ ] Freeze the **12-map holdout suite** in `maps/eval_holdout/`.
- [ ] Build `training/eval_zeroshot.py` to score any checkpoint across all 12 unseen maps + save gameplay MP4s.

### Phase 3: BC Warm-Start + Recurrent PPO Sanity Check
- [ ] Build `training/record_demos.py` — record **10–15 minutes** of human gameplay across Tier 1–3 maps (clean movement, door opening, keycard pickup, aiming, exiting).
- [ ] Run `training/train_bc.py` (5 mins) and verify on 1 map that the BC agent already moves, opens doors, and shoots.
- [ ] Build `training/rewards.py`, `training/train_ppo.py` (Custom Recurrent PPO + RND + Aux Heads), and `training/plr.py`.
- [ ] Sanity-check PPO on Tier 1–2 locally on Laptop 1 (verify FPS ≥ 1,000 steps/sec, no NaNs, rising exit rate).

### Phase 4: Full Curriculum Training (Local → Cloud Burst)
- [ ] Launch full Tier 1 → Tier 4 PLR curriculum run on Cloud VM (or overnight on Laptop 1).
- [ ] Monitor `eval_zeroshot.py` every 1M steps; watch gameplay videos on holdout maps to diagnose failure modes (e.g., stuck on doors → boost door reward; dying to fireballs → increase damage penalty; failing Tier 3 backtracking → increase Tier 3 PLR weight & verify `EgoLocalMap` channels).
- [ ] Save best zero-shot checkpoint into `submission/weights/best_agent.pt` (tracked via Git LFS).

### Phase 5: Final Verification & 2-Page Project Brief PDF
- [ ] Run `python validate.py --docker` on the final **committed and pushed** `submission/` with `torch.set_num_threads(1)` and verify all latency/memory limits. The validator fails if anything under `submission/` is untracked, ignored or uncommitted.
- [ ] `git lfs ls-files` lists the weights; repo size (including Git LFS) is **≤ 1 GB**; template `Dockerfile` untouched; all training code included in repo.
- [ ] Evaluated code = last commit on the default branch **at the deadline**; push the final checkpoint well before.
- [ ] Write the **2-Page Project Brief PDF**:
  - **Page 1:** USP (*Ego-Centric Spatial Memory + Entity Attention + PLR Procedural Curriculum*), full architecture diagram, observation encoding, and how `EgoLocalMap + GRU` solves Tier 3 backtracking without global coordinates.
  - **Page 2:** Training pipeline (BC warm-start + Custom Recurrent PPO + RND + PLR), procedural map factory (UDMF + Obsidian), zero-shot benchmark table across Tiers 1–4, latency/VRAM footprint, and failure mode analysis.
