# 05 — Execution, Hardware Split & Submission Checklist (as built)

## 1. Hardware Allocation (laptops are NOT interconnected via DDP)
Linking two laptops over LAN/Wi-Fi adds heavy sync overhead for a model under 1M parameters, so each machine has its
own role instead:

| Machine | Role |
| :--- | :--- |
| **Laptop 1** (i7-13620H, RTX 4050 6 GB) | Main PPO training run (12 parallel `doom_env` workers) |
| **Laptop 2** (RTX 3050 6 GB, Linux) | Holdout evaluation (`eval_zeroshot.py`), `validate.py` with 1-CPU pinning (`doom-rl-docs/RESULTS.md`) |

**Windows speed fix:** several ViZDoom games stall each other on Windows' default 15.6 ms timer. Raising it to 1 ms
in every worker gave **7–10× more env steps/s** (see doc 03 §5).

---

## 2. What Was Built, in Order (from `git log`)
1. Template files from [`Reinforce-SST/devforge-doom-template`](https://github.com/Reinforce-SST/devforge-doom-template)
   copied into this repo (commit `35a32ee`), plus the first model, preprocessing and training files.
2. CI sanity check and submission unit tests.
3. Trainer fixes: a crash on episode end, the Windows timer speedup, rewards, and the Freedoom train/holdout split
   (MAP01–24 / MAP25–32).
4. Infinite procedural maps (`maps/gen_maps.py`) mixed into training.
5. EgoMap spatial memory grid, as an optional model input that can be warm-started.
6. 8 generated holdout maps from reserved seeds.
7. Agent warm-up in `__init__`, plus `load_weights()`.
8. Teammate 2: baseline `validate.py` and random-weights holdout results.
9. Tutor round 1: key reward 3 → 8, +0.5 for the first key sighting, map mix shifted towards tier 3.
10. Teammate 2: generated holdout maps added to the eval, with per-tier summaries.
11. Shipped the memory-grid checkpoint (holdout proxy 34.1 vs. 9.9 for the plain model and −13.2 for random weights),
    and fixed lockless tier-3 layouts.
12. Tutor round 2: tier-3 curriculum (35% of tier-3 maps have 3 rooms). The holdout set is frozen from this point.

---

## 3. Final Verification Checklist
- [ ] Run `python validate.py --docker` on the final **committed and pushed** `submission/`. The validator fails if
  anything under `submission/` is untracked, ignored or uncommitted.
- [ ] `git lfs ls-files` lists the weights; repo size (including Git LFS) is **≤ 1 GB**; template `Dockerfile`
  untouched; all training code is in the repo.
- [ ] The evaluated code is the last commit on the default branch **at the deadline**, so push the final checkpoint
  well before it.
- [ ] Refresh the `RESULTS` block in `doom-rl-docs/BRIEF.md` and re-export `BRIEF.pdf` (2 pages or fewer).

The last `validate.py` run passed: p50 4.3 ms, p99 8.1 ms, first step 23 ms, repo 3.4 MB, weights 3.05 MB via Git
LFS. On Linux pinned to 1 CPU (random weights), RAM was 355 MB of 8,192 MB.

**Project brief:** [`BRIEF.md`](BRIEF.md) / [`BRIEF.pdf`](BRIEF.pdf).

---

## Not implemented (future work)
- **A cloud VM burst run** (16–32 vCPUs + 1 GPU, 32–64 envs, 20M–50M steps). All training so far ran on one laptop:
  about 0.7M steps at roughly 250–300 env steps/s.
- **Build-order items that were cut:**
  - BC warm-start (`record_demos.py`, `train_bc.py`)
  - RND and auxiliary heads
  - PLR (`plr.py`)
  - Obsidian maps (`gen_obsidian.py`)
  - Saving gameplay MP4s from `eval_zeroshot.py`

  See docs 02–04 for each one.
- **Laptop 2 as a map factory or sweep machine.** Maps are generated on the fly inside each training worker, so it
  wasn't needed.
