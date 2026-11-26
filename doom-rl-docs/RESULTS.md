# Teammate 2 (RTX 3050) — Validation, Holdout Baselines & Leaderboard

Environment: Linux (12 threads, 15 GB RAM, NVIDIA GeForce RTX 3050 6GB Laptop GPU), Python 3.13.18, `torch==2.15.0+cu132`, `doom_env` contract `1.0.0-draft`.
Docker validation: skipped locally (running `validate.py` directly on Linux with 1-CPU pinning + RTX 3050 CUDA).

---

## Baseline validate.py (random weights)

Command: `.venv/bin/python validate.py --device cuda`

```text
Validating /home/akash/TeamAyushDevForge  (contract 1.0.0-draft)

 ✓ Repository size (tracked files): 472.0 KB of 1.0 GB
 ✓ requirements.txt: no extra packages
 ✓ DoomAgent interface: __init__, reset, act
 ✓ CPU limit: agent pinned to 1 CPU (core 11), like the evaluation sandbox
 ✓ Initialization: 3.8s (limit 60s) on device 'cuda'
 ✓ Episode on 'two_rooms.wad:MAP01': 525 steps, tick cap, agent time 9.8s of 21.0s budget, 10.9s wall-clock
 ✓ Step latency 'two_rooms.wad:MAP01': first 1492 ms, p50 15.1 ms, p99 22.7 ms, max 1492 ms
 ✓ Peak RAM: 1836 MB resident (limit 8192 MB, including files written to /tmp)
 ✓ Peak GPU memory: 130 MB (limit 2048 MB)

PASSED
```

- **Status:** PASSED (0 warnings, 0 failures)
- **Latency:** first `1492 ms`, p50 `15.1 ms/step`, p99 `22.7 ms/step` (budget `40 ms/step`)
- **Memory:** Peak RAM `1836 MB` / `8192 MB`, Peak GPU memory `130 MB` / `2048 MB`

---

## Random-weights holdout baseline

Command: `.venv/bin/python training/eval_zeroshot.py --name random --seeds 0 --device cuda`
CSV: `training/results/random.csv` (17 maps: `two_rooms` + Freedoom `MAP25–32` + 8 generated holdout `gen_t1..t4`)

```text
[T0] two_rooms.wad:MAP01        | score  106.85 | exit  100% | keys 0.0 | doors  1.0 | kills  0.0 | cells  22.0 | hp  49.0 | 12.67 ms/step
[T0] freedoom2.wad:MAP25        | score  -15.00 | exit    0% | keys 0.0 | doors  1.0 | kills  1.0 | cells  23.0 | hp   0.0 | 8.36 ms/step
[T0] freedoom2.wad:MAP26        | score  -21.65 | exit    0% | keys 0.0 | doors  0.0 | kills  2.0 | cells   8.0 | hp   0.0 | 8.64 ms/step
[T0] freedoom2.wad:MAP27        | score  -22.60 | exit    0% | keys 0.0 | doors  0.0 | kills  0.0 | cells  14.0 | hp   0.0 | 10.34 ms/step
[T0] freedoom2.wad:MAP28        | score  -13.40 | exit    0% | keys 0.0 | doors  1.0 | kills  3.0 | cells  20.0 | hp   0.0 | 7.70 ms/step
[T0] freedoom2.wad:MAP29        | score  -14.65 | exit    0% | keys 0.0 | doors  1.0 | kills  0.0 | cells  29.0 | hp   0.0 | 7.94 ms/step
[T0] freedoom2.wad:MAP30        | score  -24.00 | exit    0% | keys 0.0 | doors  0.0 | kills  0.0 | cells  12.0 | hp   0.0 | 6.05 ms/step
[T0] freedoom2.wad:MAP31        | score  -27.80 | exit    0% | keys 0.0 | doors  0.0 | kills  0.0 | cells   1.0 | hp   0.0 | 6.24 ms/step
[T0] freedoom2.wad:MAP32        | score   -6.45 | exit    0% | keys 0.0 | doors  2.0 | kills  4.0 | cells  24.0 | hp   0.0 | 6.22 ms/step
[T1] gen_t1_1.wad:MAP01         | score  102.80 | exit  100% | keys 0.0 | doors  0.0 | kills  0.0 | cells  14.0 | hp 100.0 | 5.86 ms/step
[T1] gen_t1_2.wad:MAP01         | score  -23.85 | exit    0% | keys 0.0 | doors  0.0 | kills  0.0 | cells   7.0 | hp   0.0 | 5.85 ms/step
[T2] gen_t2_1.wad:MAP01         | score  -23.85 | exit    0% | keys 0.0 | doors  0.0 | kills  0.0 | cells   7.0 | hp   0.0 | 5.95 ms/step
[T2] gen_t2_2.wad:MAP01         | score  -16.10 | exit    0% | keys 0.0 | doors  1.0 | kills  1.0 | cells  18.0 | hp   0.0 | 5.90 ms/step
[T3] gen_t3_1.wad:MAP01         | score    5.20 | exit    0% | keys 0.0 | doors  0.0 | kills  0.0 | cells  26.0 | hp 100.0 | 5.89 ms/step
[T3] gen_t3_2.wad:MAP01         | score    3.15 | exit    0% | keys 0.0 | doors  0.0 | kills  3.0 | cells  27.0 | hp   0.0 | 5.99 ms/step
[T4] gen_t4_1.wad:MAP01         | score  -22.60 | exit    0% | keys 0.0 | doors  0.0 | kills  1.0 | cells   9.0 | hp   0.0 | 6.13 ms/step
[T4] gen_t4_2.wad:MAP01         | score  -12.85 | exit    0% | keys 0.0 | doors  1.0 | kills  2.0 | cells  28.0 | hp   0.0 | 5.96 ms/step
-------------------------------------------------------------------------------------------------------------------------
MEAN                            | score   -1.58 | exit   12% | keys 0.0 | doors  0.5 | kills  1.0 | cells  17.0 | 7.16 ms/step
MEAN score: -1.58
T0 mean score:   -4.30 (n=9, exit=  11%)
T1 mean score:   39.47 (n=2, exit=  50%)
T2 mean score:  -19.98 (n=2, exit=   0%)
T3 mean score:    4.18 (n=2, exit=   0%)
T4 mean score:  -17.73 (n=2, exit=   0%)
```

---

## Seed-7 & EgoMap Seed-77 training runs (RTX 3050 CUDA)

- **Smoke test (`two_rooms`, 4 envs, 20k steps):** PASSED (`19,968` steps, `142–146 fps` on CUDA, zero crashes).
- **Plain Seed-7 run (`training/checkpoints/seed7.pt`):**
  - **Command:** `.venv/bin/python -u training/train_ppo.py --num-envs 8 --seed 7 --total-steps 20000000 --save-path training/checkpoints/seed7.pt`
  - **Start time:** `2026-11-26 18:16 IST` (`207–216 fps`, reached `911,360` steps, peak train return `11.78`, `T1: 86%`, `T2: 36%`).
- **EgoMap Seed-77 run (`training/checkpoints/egomap_s77.pt`, `--use-map`):**
  - **Warm-started from:** `900k`-step `seed7.pt` snapshot (`--use-map --init-from seed7_900k_snap.pt`)
  - **Config:** `24` envs, `--rollout-steps 256 --num-minibatches 2 --swap-every 8`, `OMP/OPENBLAS/MKL_NUM_THREADS=1`, performance power profile, 20 compact Freedoom 2 train maps (`MAP01–10, 13–14, 16–22, 24` + `two_rooms`, never touching `MAP25–32`) + `gen1–4 × 12`.
  - **Steady-state throughput:** **`974–982 fps`** on RTX 3050 CUDA (`~3.5M steps/hr`, `62°C`, `1.9 GB` VRAM).
  - **Saved record backup:** `training/checkpoints/egomap_s77_1948_score31.pt` (3.0 MB).
- **Task H Primary Run (`training/checkpoints/t2h.pt`, `--use-map`, scrolling EgoMap + torch-free workers):**
  - **Phase 1 (`20:19–21:50 IST`, round-3 rewards):** `5.25M` new steps (`~8.6M` total) at **`966–969 fps`** (`rollout 5.2–5.8s`, `update 0.9–1.0s`, `6.6 GiB` RAM free, `63.6°C`). Peaked at **`46.62` MEAN score** (`T0: 9.45`, `T1: 101.78`, `T2: 98.51`, **`T3: 124.42 / 83% exits!`**, **`T4: 29.05`**) — pushed to `origin/candidates` as `t2h_2150_score46_62.pt`!
  - **Phase 2 (`21:57 IST` → running, round-4 rewards from `376340f`):** Resumed from `t2h_2150_score46_62.pt` (`--lr 1.5e-4`, `24` envs, `256` steps, `2` minibatches, `--seed 78`, 20 compact Freedoom + `two_rooms` + `6× gen3` + `4× gen4`).

---

## Checkpoint leaderboard

| time | run | steps (from the log) | MEAN score | exit % | keys | doors | kills | ms/step | validate.py |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | random (17 maps, seed 0, cuda) | 0 | -1.58 | 12% | 0.0 | 0.5 | 1.0 | 7.16 | PASS (15.1 ms p50, 130 MB VRAM) |
| 19:00 | seed7_1900 (plain; T0:-6.77 T1:100.52 T2:15.51 T3:6.62 T4:-9.75) | 563,200 | 9.70 | 12% | 0.0 | 0.4 | 1.1 | 8.53 | PASS (16.2 ms p50, 130 MB VRAM) |
| 19:48 | egomap_s77_1948 (EgoMap latest; T0:3.15 T1:101.20 T2:56.03 T3:52.89 T4:18.36) | ~1,610,000 | 28.55 | 24% | 0.1 | 1.0 | 1.1 | 9.44 | — |
| 19:48 | egomap_s77_best_1948 (EgoMap best; T0:7.31 T1:101.47 T2:92.83 T3:20.32 T4:15.99) | ~1,580,000 | 31.00 | 25% | 0.1 | 1.1 | 1.1 | 8.13 | PASS (exit in 384 steps, 12.4 ms p50, 130 MB VRAM) |
| 21:50 | t2h_2150 (Task H latest; T0:9.03 T1:100.90 T2:112.97(100%) T3:111.51(67%) T4:12.85) | ~8,650,000 | 44.57 | 37% | 0.1 | 1.2 | **1.7** | 5.46 | — |
| 21:50 | **t2h_best_2150** *(pushed to `candidates`!)* (T0:**9.45** T1:**101.78** T2:98.51 T3:**124.42(83%)** T4:**29.05**) | ~8,300,000 | **46.62** | **37%** | **0.2** | **1.3** | 1.4 | **3.57** | **PASS (exit in 68 steps, 9.0 ms p50, 130 MB VRAM)** |

---

## Notes for Core (Havish) & Docs (Ayush) — Video Review & Optimization Proposals

*(Written by Teammate 2 after watching side-by-side DOOM + EgoMap video replays of `egomap_s77_1948_score31.pt` and shipped `best_agent.pt`. We did not edit code/README outside our owned files—Core/Docs can apply any of these.)*

### 1. Faster Exits + Higher Remaining HP (Tutor Round 3 Reward Suggestions for `training/rewards.py`)
From watching replays (`02_tier2_unseen_exit.mp4` 123 steps, `03_tier3_keycard_backtrack.mp4` 516 steps, `01_two_rooms` & `MAP27` deaths):
- **Add a positive time-remaining bonus ONLY on exit (zero suicide exploit):** Currently `exit_reached` pays `+25 + 2*hp_frac` regardless of speed, so meandering for 950 steps pays the same as exiting in 150 steps—and causes Tier-3 timeouts at 2,100 tics. Suggest changing to:
  `reward += 25.0 + 6.0 * hp_frac + 6.0 * max(0.0, 1.0 - ps["tic"] / ps["episode_ticks"])`
  No negative per-step penalty is added, so early suicide never becomes profitable, while fast high-HP exits earn up to `+37` (matching `docs/EVALUATION.md`'s health + time efficiency bonus).
- **Stop face-tanking enemies (higher HP & fewer early deaths):** Currently `damage_taken` is `-0.015/hp` (100 HP lost = `-1.5`), while killing one enemy pays `+1.0 + 0.6` (`DAMAGECOUNT`) = `+1.6`, making face-tanking trades net-positive. Suggest:
  - `damage_taken`: `-0.015` → **`-0.03 / hp`**
  - `hp_gain` while hurt: `+0.01` → **`+0.025 / hp`** (incentivizes grabbing medkits on real Freedoom maps)
  - `dead`: `-3.0` → **`-5.0`** (cuts early deaths on `MAP26/31/32`)

### 2. Free Inference-Time Score Boost to Test (`submission/agent.py`)
- Currently `agent.py` samples stochastically at temperature `T=1.0` (`deterministic=False`), which caused noisy turns and an avoidable death on `two_rooms` even on our `31.00` checkpoint.
- **Experiment for Core:** Benchmark **`logits / T` with `T ∈ {0.5, 0.7, 1.0}`** (and `deterministic=True`) on the frozen 17-map holdout suite. Low-temperature sampling (`T ≈ 0.5–0.6`) eliminates twitchy wasted turns (faster exits, cleaner aiming, higher HP) without pure-argmax corner loops—100% neural, zero retraining cost.

### 3. Training Throughput Recipe (`974 fps` measured on RTX 3050!)
- Setting `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, `--num-envs 24 --rollout-steps 256 --num-minibatches 2 --swap-every 8`, performance power profile, and excluding the 4 giant 539–863 sector straggler maps (`MAP11, MAP12, MAP15, MAP23`) pushed RTX 3050 EgoMap training from `214 → 974 fps` (~3.5M steps/hr) and jumped holdout score to **`31.00`** (`T0: +7.31`, `T1: 101.47`, `T2: 92.83`, `T4: 15.99`).
- Core can try this on the RTX 4050 too! Also minor bug note: `training/rewards.py:103` uses `GEN_CACHE / f"{os.getpid()}.wad"`, which collides if two PID-namespaced processes share `/tmp`.

### 4. Doc / README Polish Notes for Docs (Ayush)
- **`README.md` (lines 25–28) & `training/README.md`:** `uv pip install torch==2.15.0` followed by `uv run python ...` can cause `uv run` to re-sync `uv.lock` and uninstall `torch`. Suggest changing `uv run python` → `uv run --no-sync python` (or `.venv/bin/python`).
- **`BRIEF.md` §6–§7:** Before final lock (`03:30 IST`), replace the mixed Windows/Linux footprint table with one consistent Linux 1-CPU CUDA `validate.py` run (`p50 12.4 ms`, `first 164 ms`, `1671 MB RAM`, `130 MB VRAM`), and update the step count / speed & `31.00+` holdout numbers.
