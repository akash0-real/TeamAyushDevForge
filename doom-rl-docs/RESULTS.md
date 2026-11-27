# Teammate 2 (RTX 3050) — Validation, Holdout Baselines & Leaderboard

## T2 UPDATES (teammate 2 → core + teammate 3; newest first, per 22:32 team-channel rule)
- **UPDATE 07:24 (`ACK 07:17 / 07:22` — Round-9 `t2r9.pt` LIVE on RTX 3050 CUDA + ready for `07:55` push):**
  - Started Havish's exact Round-9 command (`COMBAT_W=2`, default balanced mix, `lr 5e-5`, `--num-envs 24 --seed 105 --save-path training/checkpoints/t2r9.pt`) resumed from `shipped-v11` (`8ab22c0c...`).
  - Will snapshot + run deterministic 5-seed CUDA holdout check vs `v11` at `07:52` and push `t2r9.pt` to `origin/candidates` **at `07:55` sharp**, then run final Docker validation at `~08:20` before the `08:30` deadline.
- **UPDATE 07:17 (FINAL `[core] FINAL` (`9045f05` / `f4e4196`) VERIFIED — FRESH CLONE DOCKER `PASSED` + `sha256 8ab22c0c` + `v11` T1–T4 Demo Pushed — WE ARE DONE!):**
  - **Fresh anonymous HTTPS clone `validate.py --docker --image devforge-doom --no-gpu` on `main` (`f4e4196`):** **`PASSED` (0 warnings)** — `6.6 MB / 1.0 GB` repo, `init 6.0s` (`cpu`, 1 CPU pin), `two_rooms` exit in **`58` steps** (`0.4s / 21.0s` budget), `first 16 ms`, **`p50 6.0 ms`**, `p99 6.3 ms`, **`788 MB` RAM**.
  - **Linux 1-CPU CUDA `validate.py --device cuda` on `f4e4196`:** **`PASSED` (0 warnings)** — `172` steps (`1.7s / 21.0s`), `first 110 ms`, `p50 9.1 ms`, `p99 10.1 ms`, `1663 MB` RAM, **`130 MB` VRAM**.
  - **Weights `sha256sum submission/weights/best_agent.pt` confirmed:** **`8ab22c0c1dd939c634095db2e25647a890d8e1622d21b952e238aa6bda38ba76`** (`3.0 MB`, matches `t2r7b_0447` on both local `main` and fresh anonymous HTTPS clone; `git lfs ls-files` verified; repo PUBLIC `HTTP 200`).
  - **Showcase video pushed to `origin/demos` (`a23fa62`, per `AUDIT.md` 06:12 #1):** `SHOWCASE_V11_FINAL.mp4` (`8.6 MB`, side-by-side DOOM + live EgoMap: unseen T1 exit `16`st, unseen T2 `70`st, unseen T3 `171`st, **unseen T4 2-key 2-lock EXIT `505`st** on `gen_t4_exam2` seed 13). Ack to Core + T3: done!
- **UPDATE 07:01 (Round-8 `COMBAT_W` + `WASTE_W` cutoff wrap-up — all 3 snapshots `SHIP RULE: FAIL` vs `v11`; none pushed to `candidates`, `v11` stands!):**
  - **Paired deterministic 5-seed CUDA checks (`seeds 0..4`, masks OFF, vs `v11` = `t2r7b_0447_s04.csv` `51.57`, `T2+T3+T4 19/30`, `MAP25–32 +3.42`):**
    - **`t2r8a_0649_best` (`COMBAT_W=3`, `~1.8M` R8 steps from `v11`):** `43.27` vs `51.57`, `T2+T3+T4 14/30` (`T1 10/10`, `T2 8/10`, `T3 6/10`, `T4 0/10 / 13.78`), `MAP25–32 +2.25` (`Δ -1.17 ± 1.67`) — **`SHIP RULE: FAIL`**.
    - **`t2r8b_0656` (`COMBAT_W=2 WASTE_W=1`, `522k` R8 steps from `v11`):** `42.24` vs `51.57`, `T2+T3+T4 15/30` (`T1 10/10`, `T2 8/10`, `T3 7/10`, `T4 0/10 / 18.94`), `MAP25–32 -4.83` (`Δ -8.24 ± 1.28`) — **`SHIP RULE: FAIL`**.
    - **`t2r8b_0656_best` (`COMBAT_W=2 WASTE_W=1`, best snap):** `41.71` vs `51.57`, `T2+T3+T4 13/30` (`T1 10/10`, `T2 6/10`, `T3 7/10`, **`T4 0/10 / 29.07`** — `gen_t4_2 35.00` w/ `4.4` kills, `0.4` keys), `MAP25–32 -2.59` (`Δ -6.00 ± 1.69`) — **`SHIP RULE: FAIL`**.
  - **Takeaway:** `WASTE_W=1` + `COMBAT_W=2` lifted T4 combat score (`29.07` vs `23.20`), but both Round-8 runs eroded T2/T3 exits and real maps vs `v11`. Stopped local trainers for the `07:05` cutoff; standing by for `maps/exam3` (`~07:20–07:35`) and final Docker/Linux validation before `08:00 IST`.
- **UPDATE 05:55 (Repo PUBLIC verified + latest `7353c8a` Docker re-PASS + `--mask use` 5-seed CUDA check):**
  - **Repo & LFS ready:** `https://github.com/akash0-real/TeamAyushDevForge` is now **PUBLIC (`HTTP 200`)**; `git lfs ls-files` → `submission/weights/best_agent.pt` (`3,051,854 B`, `8ab22c0c...`).
  - **Re-validated `7353c8a` (`shipped-v11`):** Official Docker `PASSED (0 warnings)` (`55` steps, `first 16 ms`, `p50 6.2 ms`, `p99 6.9 ms`, `788 MB` RAM); Linux 1-CPU CUDA `PASSED (0 warnings)` (`82` steps, `p50 9.1 ms`, `130 MB` VRAM); `pytest tests/test_submission.py tests/test_validate.py`: `12/12` pass.
  - **`--mask use` on `v11` (5 deterministic CUDA seeds, `cuda_v11_usemask_s04.csv` vs `t2r7b_0447_s04.csv`):** **`52.10` vs `51.57`** (`+0.53 ± 2.73`, `worse 3/5`, `T1 10/10`, **`T2 10/10 (113.42)` vs `9/10`**, **`T3 9/10 (131.49)` vs `10/10`**, `T4 0/10 (22.88 vs 23.20)`, `T2–T4 19 vs 19`, `exit_points 57 vs 58`, **`MAP25–32 +4.74 vs +3.42`**, `SHIP RULE: PASS`) — neutral overall on 5 CUDA seeds; leaving decision to Core's 20-seed holdout / `exam2`.
- **UPDATE 05:50 (`shipped-v11` = `t2r7b_0447`, sha `8ab22c0c` PASSED BOTH official Docker + Linux 1-CPU CUDA `validate.py` — 0 warnings!):**
  - Pulled `74a1d49` (`shipped-v11`, `3.0 MB` LFS verified, `sha256 8ab22c0c...`) — huge win seeing `t2r7b_0447` beat `v8` on both untouched `maps/exam2` (`+3.8 ± 2.1`, `188 vs 184` exit_pts, first T4 exit!) and 20-seed holdout (`+3.2 ± 1.8`).
  - **Official Docker `validate.py --docker --image devforge-doom --no-gpu` on `shipped-v11`:** **PASSED (0 warnings)** — `5.7 MB` repo, `init 8.3s`, `two_rooms` exit in **`51` steps** (`0.3s / 21.0s`), **`first 7 ms`, `p50 6.2 ms`, `p99 6.9 ms`, `790 MB` RAM**.
  - **Linux 1-CPU CUDA `validate.py --device cuda` on `shipped-v11`:** **PASSED (0 warnings)** — `init 2.2s`, `two_rooms` exit in `109` steps (`1.2s / 21.0s`), `first 109 ms`, `p50 10.0 ms`, `p99 11.2 ms`, `1597 MB` RAM, `130 MB` VRAM.
  - Standing by on 10-min sync for `[core]`'s `"use"`-cooldown check and **`[core] FINAL`** (`~06:15 IST`).
- **UPDATE 05:34 (`05:30` cutoff wrap-up — `t2r7b_0447_score51_57.pt` stays #1 `t2` finalist + training stopped for `exam2`!):**
  - **`t2r7b_0526` (`t2r7b_0526_s04.csv`, `lr 8e-5` +215k extra steps):** `48.51` vs `v8 42.65` (`+5.86 ± 3.84`, `T1 10/10`, `T2 9/10`, `T3 9/10`, `T4 0/10 (19.92)`, `exit_points 55 vs 45`, `MAP25–32 +0.62`, `SHIP RULE: PASS`) — good consistency, but trails **`t2r7b_0447_score51_57.pt`** (`51.57`, `58` exit_pts, `MAP25–32 +3.42`, pooled `+4.9 ± 2.1 sig` per `49ee73a`). Keeping `t2r7b_0447_score51_57.pt` as our single `exam2` finalist!
  - **`t2r7c_0526` (`t2r7c_0526_s04.csv`, `core_r7best_0350` + 338k steps @ `lr 1e-4`):** `45.63` (`+2.98 ± 4.11`, `exit_points 53 vs 45`), **`SHIP RULE: FAIL`** on `MAP25–32` (`-4.40`, `-4.51 ± 1.39`), again confirming further R7 training drifts real maps while `core_r7best_0350` (+`t2r7b_0447`) stay strongest.
  - **All local trainers stopped** (`9.5 GiB` RAM free); standing by on 10-min sync for `maps/exam2` results (`~05:45–05:55`) and **`[core] FINAL` (`~06:15 IST`)** to run final **Task F** (Linux 1-CPU CUDA + official Docker `validate.py` + `FINAL` eval). Also ready to 5-seed CUDA test `"use"` cooldown mask if Core builds it before `05:45`.
- **UPDATE 05:19 (`core_r7best_0350` CONFIRMED on RTX 3050 CUDA — `SHIP RULE: PASS`, `+5.91 ± 3.39`, `0/5` worse + `t2r7c` training from it!):**
  - **2nd-machine check (`core_r7best_0350.pt`, sha `868dd63d...`, 5 deterministic CUDA seeds `0..4`, `v8` defaults; `cuda_core_r7best_s04.csv` vs `cuda_v8_seeded_s04.csv`):**
    - **(1) Overall MEAN:** **`48.56` vs `42.65`** (**`Δ = +5.91 ± 3.39`**, `worse_seeds = 0/5`!) — **PASS**
    - **(2) Exits per tier:** **`T1 10/10 (101.41)`**, **`T2 8/10 (94.02 vs 7/10 76.70)`**, **`T3 8/10 (119.85 vs 7/10 109.38)`**, `T4 0/10 (16.18 vs 20.12)` → **`T2+T3+T4` exits `16/30 vs 14/30`**, **`exit_points = 50 vs 45 (+5)`** — **PASS**
    - **(3) Unseen real `MAP25–32`:** **`+6.70` (MASSIVE POSITIVE!) vs `+0.10`** (`Δ = +6.59 ± 3.94`; `MAP28 +25.80`, `MAP30 +27.33`, `MAP32 +42.29` with `0.2` keys, `16.6` kills) — **PASS**
    - **`ship_rule.py`: `SHIP RULE: PASS`!**
  - **Stopped `t2r7a` line** per Core `05:12` order; running **`t2r7c` (`24` envs from `core_r7best_0350` at `lr 1e-4`, balanced mix)** + `t2r7b` (`12` envs from `t2r7b_0447` at `lr 8e-5`) until `~05:25`, then screening and pushing any improvement **before `05:30 IST`**.
- **UPDATE 05:09 (HUGE Round-7 CANDIDATE `t2r7b_0447_score51_57.pt` PUSHED TO `origin/candidates` — SIGNIFICANT `+8.92 ± 3.53` (`0/5` worse!), `10/10` T3, `9/10` T2, `exit_points 58 vs 45`!):**
  - **Paired deterministic 5-seed CUDA pre-screen (`seeds 0..4`, `v8` default head temps; `t2r7b_0447_s04.csv` vs `cuda_v8_seeded_s04.csv`, `~1.67M` R7 steps from `t2h_0037` at `lr 1.0e-4`):**
    - **(1) Overall MEAN:** **`51.57` vs `42.65`** (**`Δ = +8.92 ± 3.53`**, `t = 2.53`, **`worse_seeds = 0/5`**!) — **PASS**
    - **(2) Exits per tier:** **`T1 10/10 (101.58)`**, **`T2 9/10 (101.35 vs 7/10 76.70)`**, **`T3 10/10 (144.33 vs 7/10 109.38)`**, `T4 0/10 (23.20 vs 20.12; gen_t4_2 +34.60)` → **`T2+T3+T4` exits `19/30 vs 14/30`**, **`exit_points = 58 vs 45 (+13)`** — **PASS**
    - **(3) Unseen real `MAP25–32`:** **`+3.42` (POSITIVE!) vs `+0.10`** (`Δ = +3.31 ± 2.28`; `MAP25 +5.00`, `MAP28 +17.83`, `MAP30 +30.02`, `MAP32 +8.72`) — **PASS**
    - **`ship_rule.py`: `SHIP RULE: PASS`!**
  - **Throwaway Linux 1-CPU CUDA `validate.py`:** **PASSED** (`two_rooms` exit in `124` steps, `p50 9.1 ms`, `p99 12.9 ms`, `1596 MB` RAM, `130 MB` VRAM).
  - **`t2r7a_0446` (`t2r7a_0446_s04.csv`, `~2.64M` R7 steps):** `44.66` (`T1 10/10`, `T2 8/10`, `T3 7/10`, `T4 0/10 (17.74)`, `MAP25–32 +3.47`). Note to Core/T3: T4 holdout exits remain `0/10` on both lines so far, though `t2r7b_0447` lifted T2+T3 to `19/20` and `MAP25–32` to `+3.42`!
  - **Resuming both trainers until `05:24 IST`** (decayed LRs `1e-4 / 8e-5`) ahead of the `05:30 IST` final cutoff; ready to push one more update if it beats `51.57` / `58`. **Also noted `fdbffe1` (`t2r7a_0420` i3 FAIL) — please check `t2r7b_0447_score51_57.pt` on cloud & i3!**
- **UPDATE 04:35 (NEW Round-7 CANDIDATE `t2r7a_0420_score45_58.pt` PUSHED TO `origin/candidates` — 5-seed CUDA `SHIP RULE: PASS` vs `v8`!):**
  - **Paired deterministic 5-seed CUDA pre-screen (`seeds 0..4`, `v8` default head temps; `t2r7a_0420_s04.csv` vs `cuda_v8_seeded_s04.csv`, `~1.4M` R7 steps from `t2h_0037`):**
    - **(1) Overall MEAN:** **`45.58` vs `42.65`** (**`Δ = +2.93 ± 4.76`**, `worse_seeds = 1/5`) — **PASS**
    - **(2) Exits per tier:** **`T1 10/10 (101.86)`**, **`T2 8/10 (94.14 vs 7/10 76.70)`**, **`T3 8/10 (120.99 vs 7/10 109.38)`**, **`T4 0/10 (25.46 vs 20.12 — T4_1 25.26, T4_2 25.66, 0.2 keys on both)`** → **`T2+T3+T4` exits `16/30 vs 14/30`**, **`exit_points = 50 vs 45 (+5)`** — **PASS**
    - **(3) Unseen real `MAP25–32`:** `-2.50` vs `+0.10` (`Δ = -2.60 ± 4.01`, limit `-8.03`) — **PASS**
    - **`ship_rule.py`: `SHIP RULE: PASS`!**
  - **`t2r7b_0420` (`t2r7b_0420_s04.csv`, `~0.9M` R7 steps, `lr 1.0e-4`):** `40.13` (`T1 10/10`, `T2 7/10`, `T3 5/10`, `T4 20.60`, `MAP25–32 +0.59`) — not pushed.
  - **Pushed to `origin/candidates` (Git LFS):** `t2r7a_0420_score45_58.pt`. Both Round-7 trainers continuing (`~1,300` combined fps); next 5-seed eval at **`04:50 IST`** and final push **before `05:15 IST`**.
- **UPDATE 03:50 (Acknowledged `06:30 IST` extension + Round 7 T4 reverse-curriculum training LIVE from `t2h_0037`!):**
  - Pulled `1ecb652` (`maps/exam2/` untouched — will NEVER evaluate on it), verified `check_warp_curriculum.py` → **`OK round 7`** + `check_geodesic.py` → **`OK`**.
  - Launched **dual Round-7 trainers from `t2h_0037` (`v8` weights, `2797f55f...`)** on RTX 3050 CUDA:
    - **`t2r7a` (`24` envs, seed 101, `lr 1.5e-4`):** Core's exact T4-heavy mix (`two_rooms` + 20 compact Freedoom + `2×gen1 + 10×gen2 + 12×gen3 + 20×gen4`).
    - **`t2r7b` (`12` envs, seed 102, `lr 1.0e-4`):** Conservative T4-heavy fine-tune (`two_rooms` + 10 compact Freedoom + `2×gen1 + 8×gen2 + 10×gen3 + 16×gen4`) to hedge against T2/T3 drift while absorbing `T7/T8` T4 reverse-curriculum exits.
  - **Eval schedule:** Deterministic 5-seed CUDA paired checks vs `cuda_v8_seeded_s04.csv` at **`04:20`**, **`04:50`**, and final push to `origin/candidates` **before `05:15 IST`**; standing by for `[core] FINAL` (`~06:15 IST`) for final Linux + Docker Task F.
- **UPDATE 02:30 (`shipped-v10` = `v8` / `t2h_0037` + turn/strafe `0.5` PASSED BOTH Linux CUDA + official Docker `validate.py` — 0 warnings!):**
  - Pulled `af88b72` (`shipped-v10`, `2797f55f...`, `3.0 MB` LFS verified) — huge win seeing `v8` (`210` exit_pts, `T1 100% / T2 77.5% / T3 90%`) and `t2h_0122` (`197` exit_pts, `49.4` mean, `T3 97.5%`, `+5.9` unseen real) take #1 and #2 on the untouched `maps/fresh/` final exam!
  - **Official Docker `validate.py --docker --image devforge-doom --no-gpu` on `shipped-v10`:** **PASSED (0 warnings)** — `5.0 MB` repo, `init 6.0s`, `two_rooms` exit in **`147` steps** (`0.9s / 21.0s`), **`first 15 ms`, `p50 5.9 ms`, `p99 6.6 ms`, `899 MB` RAM**.
  - **Linux 1-CPU CUDA `validate.py --device cuda` on `shipped-v10`:** **PASSED (0 warnings)** — `two_rooms` exit in `113` steps (`1.2s / 21.0s`), `p50 9.0 ms`, `p99 13.3 ms`, `1598 MB` RAM, `130 MB` VRAM.
  - Standing by on 10-min sync for **`[core] FINAL`** to execute final **Task F** (Linux 1-CPU CUDA + Docker `validate.py` + `FINAL` `eval_zeroshot.py`).
- **UPDATE 02:22 (`shipped-v9` CONFIRMED on second machine — Linux RTX 3050 CUDA 5-seed `SHIP RULE: PASS` + `validate.py` PASS!):**
  - **Paired deterministic 5-seed CUDA check (`seeds 0..4`, `v8/v9` default head temps; `cuda_v9_seeded_s04.csv` vs `cuda_v8_seeded_s04.csv`):**
    - **(1) Overall MEAN:** **`46.20` vs `42.65`** (**`Δ = +3.56 ± 3.95`**, `worse_seeds = 1/5`) — **PASS**
    - **(2) Exits per tier:** **`T1 10/10 (100.94)`**, **`T2 9/10 (90%, 103.72 vs 7/10 76.70)`**, **`T3 8/10 (80%, 123.57 vs 7/10 109.38)`**, `T4 0/10 (19.12 vs 20.12)` → **`T2+T3+T4` exits `17/30 vs 14/30`**, **`exit_points = 52 vs 45 (+7)`** — **PASS**
    - **(3) Unseen real `MAP25–32`:** `-2.31` vs `+0.10` (`Δ = -2.41 ± 1.91`, limit `-3.82`) — **PASS**
    - **`ship_rule.py`: `SHIP RULE: PASS`! Keep `v9`.**
  - **Linux 1-CPU CUDA `validate.py` on `shipped-v9` (`7b2f37c9...`):** **PASSED (0 warnings)** — `two_rooms` exit in **59 steps** (`0.8s / 21.0s`), `p50 11.1 ms`, `p99 15.8 ms`, `1602 MB` RAM, `130 MB` VRAM, `4.6 MB` repo.
  - **Pre-cutoff `02:15` wrap-up:** Paired `t2h_0122` at `turn T=0.5` vs `v7` (`t2h_0122_tt05_s04.csv` vs `cuda_v7_seeded_s04.csv`) scored `46.60 vs 42.74` (`+3.86 ± 3.66`, `MAP25–32 +3.15 vs -3.13`, `exit_points 47 vs 49`, `SHIP RULE: FAIL` on exit count); later `01:48`/`02:00` snapshots did not beat `t2h_0122`. Stopped local trainers after the `02:15` cutoff (`9.7 GiB` RAM free) — standing by for **`[core] FINAL`** to run **Task F** (Linux 1-CPU CUDA + official Docker `validate.py` + `eval_zeroshot.py`).
- **UPDATE 01:51 (NEW CANDIDATE `t2h_0122_score49_23.pt` PUSHED TO `origin/candidates` — CRUSHING DETERMINISTIC 5-SEED `SHIP RULE: PASS`, `0/5` WORSE SEEDS!):**
  - **Deterministic seeded 5-seed paired CUDA comparison (`seeds 0..4`, after pulling `6f8bff9`; `t2h_0122_seeded_s04.csv` vs `cuda_v6_seeded_s04.csv`):**
    - **(1) Overall MEAN:** **`49.23` vs `40.95`** (**`Δ = +8.28 ± 3.89`**, `t = 2.13`, **`worse_seeds = 0/5`** — beat `shipped-v6` on all 5 seeds!) — **PASS**
    - **(2) Exits per tier:** **`T1 10/10 (100%, 101.21)`**, **`T2 9/10 (90%, 103.78)`**, **`T3 9/10 (90%, 134.77 vs 5/10 87.37)`**, `T4 0/10 (15.57 vs 12.96)` → **`T2+T3+T4` exits `18/30 vs 15/30`**, **`exit_points = 55 vs 45 (+10)`** — **PASS**
    - **(3) Unseen real `MAP25–32`:** **`+2.18` (POSITIVE!) vs `-5.10`** (**`Δ = +7.28 ± 6.08`**, limit `-12.17`; `MAP32 +43.79` with `27.8` kills, `4.0` doors, `104.8` cells vs `-5.77`) — **PASS**
    - **`ship_rule.py`: `SHIP RULE: PASS`!**
  - **Throwaway Linux 1-CPU CUDA `validate.py`:** **PASSED** (`two_rooms` exit in **55 steps**, `130 MB` VRAM, `1618 MB` RAM).
  - **Pushed to `origin/candidates` (Git LFS):** `t2h_0122_score49_23.pt` (`~12.4M` steps, Round 6 Geodesic + `256/192` warp) — ready for T3/Core verification (`shipped-v7`?) and the `02:30` Vast `maps/fresh/` final exam!
  - **Next (ahead of `02:15 IST` hard deadline):** Already running deterministic 5-seed CUDA eval of our **`01:48` snapshot (`t2h_0148_seeded_s04`)** — live training just hit **`T2 100%`, `T3 62%/50`** at `662 fps` (`~13.5M` steps); will push `t2h_0148` ONLY if it beats `t2h_0122` (`49.23` / `55` exit_pts).
- **UPDATE 01:36 (`core_vA_0035` CUDA FAIL + `t2h_0122` hits `53.14` with `100%` T1/T2/T3 exits + `+6.36` `MAP25–32`!):**
  - **`core_vA_0035` 5-seed CUDA pre-screen (`t2_core_vA_0035.csv` vs `t2h_0037_s04.csv`):** **SHIP RULE: FAIL** — `40.24` vs `44.22`, `T2+T3+T4` exits `14 vs 17` (`T1 10/10`, `T2 7/10 (82.78)`, `T3 7/10 (112.66)`, `T4 0/10 (1.75)`), `MAP25–32 -2.98` (`+0.50 ± 2.29`). Confirms T3's `FAIL` verdict (`52439dc`).
  - **`t2h_0122` (3-seed preview, `~12.4M` steps, Round 6 + `256/192` warp):** **`53.14` MEAN (`41%` exits!)** — **`T1 6/6 (100%, 100.93)`**, **`T2 6/6 (100%, 113.83)`**, **`T3 6/6 (100%, 142.88)`**, `T4 0/6 (14.59)`, **`MAP25–32 +6.36`!** (`MAP32 +76.65` with `45.7` kills, `5.0` doors, `149.7` cells).
  - **Now running:** Pulled `6f8bff9` (deterministic `torch.manual_seed` eval) and running **paired 5-seed deterministic CUDA check** of `t2h_0122` vs `shipped-v6` (`t2h_0122_seeded_s04.csv` vs `cuda_v6_seeded_s04.csv`) + throwaway `validate.py` before pushing to `origin/candidates`.
  - **Training live:** `t2h` (`673 fps`, `T5: 30–33%`!), `t2w` (`435 fps`, `T2 68%`, `T5: 32–43%`!).
- **UPDATE 00:57 (Official Organizer Track-1 Notice relayed from Akash — answers T3 `00:45` QUESTION):**
  - **Evaluation format:** **6 unseen maps across 4 tiers** (`two_rooms.wad` is verification only).
  - **Compliance audit against organizer checklist — ALL PASS:** (1) `submission/agent.py` loads real LFS weights (`3.0 MB`, `4.1 MB` repo ≤ 1 GB); (2) synced to latest template limits (`2 GB` VRAM, `8 GB` RAM, `1 CPU`), self-contained `submission/` (zero `training/` imports); (3) **pure custom RL** (explicitly preferred over heuristic/rule-based models; `training/` kept in repo); (4) **`validate.py --docker` PASSED** (`00:53 IST`, `p50 5.9 ms`, `797 MB` RAM); (5) repo visibility — Akash will flip repo to **PUBLIC** before submission deadline.
  - **DOOM engine quirks flagged by organizers:** (a) **`USE` only triggers on a fresh `0→1` press, and pressing it on a moving door reverses the door!** (Note: our `22:55` audit showed `overall_use = 32–48%` of steps, so continuous `1→1` holds don't trigger and rapid toggling can reverse opening doors); (b) killing all enemies does not end the episode (exit/death/tick cap only); (c) holding down pistol fire spreads shots.
- **UPDATE 00:54 (TASK F0 DOCKER PASS + NEW CANDIDATE `t2h_0037_score44_22.pt` PUSHED TO `candidates`!):**
  - **Task F0 Official Docker Validation (`validate.py --docker --image devforge-doom --no-gpu`):** **PASSED (0 warnings)!** Exact scorer sandbox (`--network none`, `--read-only`, `--cap-drop ALL`, 1-CPU pin): `init 6.0s`, `two_rooms` exit in `190` steps (`1.1s / 21.0s`), **`first 13 ms`, `p50 5.9 ms`, `p99 6.4 ms`, `797 MB` RAM**. Full output pasted below under `## validate.py --docker (official image)`.
  - **NEW CANDIDATE `t2h_0037_score44_22.pt` pushed to `origin/candidates` (Round 6 Geodesic + `256/192` warp, `~11.7M` steps):** **5-seed CUDA `SHIP RULE: PASS`** vs `cuda_v3_s04.csv` (`t2h_0037_s04.csv`, unmasked):
    - **Overall:** **`44.22` vs `43.11`** (`+1.11 ± 4.32`, worse 2/5) — **PASS**
    - **Exits per tier (per T3 `MESSAGE 00:55`):** **`T1 10/10 (100%)`**, **`T2 9/10 (90%, 99.21, +17.13 ± 12.00)`**, **`T3 8/10 (80%, 120.35, +9.82 ± 22.06)`**, `T4 0/10 (14.44)` → **`T2+T3+T4` exits `17/30` vs `14/30`**, **`exit_points = 52 vs 45 (+7)`** — **PASS**
    - **`MAP25–32`:** `-3.49` vs `-2.10` (`-1.38 ± 2.31`, limit `-4.62`) — **PASS**
    - **Throwaway Linux 1-CPU CUDA `validate.py`:** **PASSED** (`two_rooms` exit in **60 steps**, `p50 9.0 ms`, `p99 13.1 ms`, `130 MB` VRAM).
    - **Training live (`t2h`):** `T4: 2%/42` (first Tier-4 training exit!), `T5: 19%/42`, `539 fps`; continuing training toward `02:15 IST`.
- **UPDATE 00:50 (Round-6 `t2h` surging — `43.35` holdout + FIRST Tier-4 exit `T4: 2%/42` + `T5: 19%`!):**
  - **`00:37` Round-6 + `256/192` warp 3-seed evals (unmasked, CUDA):**
    - **`t2h_0037`:** **`43.35` MEAN** (`37.3%` exits; `T1 101.00 (100%)`, `T2 88.82 (83%)`, **`T3 123.80 (83%)`**, `T4 7.27`, `MAP25–32 -1.78`) — up **+5.75** from `00:06` (`37.60`)! Even better: at `00:49` (`upd 125`, `768k` post-fix steps, `539 fps`), **`t2h` just logged our first Tier-4 two-lock exit (`T4: 2%/42`)** alongside `T5: 19%/42`, `T2: 100%`, `T3: 46%`!
    - **`t2w_0037`:** `23.18` MEAN (`T2 50%`, `T3 0%`, `T4 16.43`, `MAP25–32 -6.50`; `T5 7–15%` in training). `t2h`'s `2×gen2+6×gen3+6×gen4` mix is clearly outperforming `t2w` so far.
  - **Task F0 Docker (`devforge-doom:latest`, `6.78 GB`):** Build **COMPLETE**! Running `validate.py --docker --image devforge-doom --no-gpu` right now; will paste full container output as soon as it exits.
  - **Next:** Ready to 5-seed CUDA pre-screen `core_vA..E` as soon as Core pushes them to `origin/candidates` (`~01:30 IST`), plus next `t2h`/`t2w` 5-seed eval at `01:17 IST` ahead of the `02:15 IST` hard candidate deadline.
- **UPDATE 00:25 (Warp radius `256/192` live + Linux 1-CPU `MAP32` timing PASS):**
  - **Pulled `04e63fb` (`WARP_R0=256 / WARP_RMIN=192`)** and restarted **both `t2h` & `t2w`** from their latest Round-6 checkpoints (`t2h` ~`768k` R6 steps, `t2w` ~`491k` R6 steps).
  - **Closed T3 `X 00:25` `MAP32` 1-CPU timing risk (Linux agent-only 1-CPU pin, `validate.py --device cpu --map freedoom2.wad:MAP32 --ticks 4200`):** **PASSED** with **0 warnings** (`init 2.6s`, **`p50 8.3 ms`**, **`p99 28.0 ms`**, `first 64 ms`, `0.9s / 42.0s` agent budget, `690 MB` RAM) — `4.8×` under the `40 ms` average budget even while training! Will re-run on `[core] FINAL` at `03:30 IST`.
  - **Task F0 `docker build`:** finished downloading `torch` + `cudnn` + `cublas`; continuing `stage 6/9`.
- **UPDATE 00:14 (Warp-spot dead-end audit COMPLETE — `training/results/t2_warp_audit.csv`):**
  - Audited all **`907` exit-warp spots `<= 1,500u`** across all 20 `REAL_TRAIN_MAPS` × 3 episodes = **`2,721` episodes** with `shipped-v3` (`2,100` tics).
  - **Huge finding:** **`874 / 907` (`96.4%`) have `0/3` exits**; only **`33 / 907` (`3.6%`)** ever reached an exit! Nine entire maps (`MAP02, 04, 07, 08, 09, 10, 17, 20, 24`) had **0 working spots (`100%` dead)**, while `MAP05` (`10` working) and `MAP14` (`7` working, all `3/3`) carry over half the working spots. Full 20-map table + 33 working spots listed below; all 874 zero-exit `(map, x, y, dist)` rows are in `training/results/t2_warp_audit.csv`.
  - **Acknowledged `62c186c` / `d14d890`:** `02:15 IST` hard candidate deadline armed; repo visibility flagged to Akash (will switch to public before deadline). `t2h` + `t2w` running fixed Round 6 now with all 8 audit workers freed; `00:06` 3-seed eval (`t2w_0006` + `t2h_0006`) finishing now; Task F0 `docker build` still on `stage 6/9`.
- **UPDATE 00:03 (Task M2 `--mask fist` complete):**
  - **`t2_v3_mask_fist.csv` vs `cuda_v3_s04.csv` (5 seeds CUDA, 17 holdout maps):** **SHIP RULE: FAIL (purely neutral)** — overall `43.09` vs `43.11` (**Δ `-0.03 ± 4.53`**, worse 3/5); `T0_all -0.69 ± 2.67`; **`MAP25–32 -0.81 ± 3.08`** (`-2.91` vs `-2.10`); `T1 -0.30 ± 0.44` (`10 vs 10`); `T2 +11.13 ± 30.05` (`8 vs 7`); `T3 -2.40 ± 18.18` (`7 vs 7`); `T4 -5.59 ± 7.86` (`20.35` vs `25.94` — recovered ~10 pts vs full mask's `-15.25`, confirming the weapon-switch rule was what hurt T4); `T2+T3+T4` exits `15 vs 14`. Recommendation: **keep masks OFF (`shipped-v5`)**.
  - **Warp-spot dead-end audit (`task-996`):** `850/907` (`94%` done, final ~2 maps left!) — will push `t2_warp_audit.csv` immediately on completion. **Task F0 `docker build`** still on `stage 6/9` (`Downloaded triton`).
- **UPDATE 23:59 (Round-6 fixed live + 3-hour win plan for T3 & Core review):**
  - Pulled `2cde54b` (`shipped-v5` masks off + both Round-6 anti-farming fixes), verified `check_geodesic.py` (`MAP08` included → `OK`), and **restarted both `t2h` & `t2w` onto fixed Round 6** (`t2h` resumed from `522k` R6 steps, `t2w` from `322k` R6 steps).
  - **Task M2 (`--mask fist`, 5 seeds CUDA)** is past `MAP32` (~2 min left); **907-spot warp dead-end audit (`t2_warp_audit.csv`)** is at **`827/907` (`91%` done, ~2 min left!)** and already shows **>90% of Euclidean near-exit warp spots are `0/3`-exit dead ends** (`MAP04/07/09/17/24` = 100% dead; `MAP05` 10 working, `MAP14` 7 working, `MAP13` 3 working).
  - **Proposal for 00:00–03:00 IST final push (asking Ayush + Havish's opinion):**
    1. **BFS-filtered warp spots (3-line fix in `rewards.py`, zero changes to `submission/`):** Since `training/geodesic.py` already computes exact BFS walking distances on every map, sort/filter `warp_spots()` by **BFS walking distance** instead of Euclidean straight-line distance — instantly eliminates walled-off / behind-locked-door warp dead ends for *both* `T5` (exits) and `T6` (keys), boosting real-map reverse-curriculum signal ~5–10× on top of our `t2_warp_audit.csv` blacklist!
    2. **Scale fixed Round 6 (`10–12M+` more Geodesic steps on RTX 3050 before `03:00`)** once `task-996` finishes and frees 8 workers + GPU, plus Task M2 `--mask fist` ONLY if it clears the paired ship rule without hurting T4.
    3. **Final judge package (`~02:15–03:45 IST`):** Pick final ship on untouched `maps/fresh/`, run Task F Linux 1-CPU + Task F0 Docker `validate.py`, and record a final champion gameplay + scrolling EgoMap showcase on `origin/demos` showing every tier solved live.

**QUESTION(t3 & core):** What's your opinion on this 3-hour plan — especially **#1 (filtering `warp_spots()` by `geodesic.py` BFS distance instead of Euclidean distance)** once `t2_warp_audit.csv` lands in ~2 min? Any blind spots or anything higher-leverage we should prioritize before `03:00 IST`? (Short reply fine for T3 low-token mode!)

- **UPDATE 23:34 (Task M + Round 6 Geodesic live):**
  - **Task M (`t2_v3_mask.csv` vs `cuda_v3_s04.csv`, 5 seeds CUDA, 17 holdout maps):** **SHIP RULE: FAIL** on CUDA (`38.85` vs `43.11`, **Δ overall `-4.26 ± 3.74`**, worse 3/5; `T2+T3+T4` exits `11 vs 14`; `T1 -0.21 ± 0.39`, `T2 -20.11 ± 20.49`, `T3 -15.97 ± 15.26`, `T4 -15.25 ± 5.80`). **However, `MAP25–32` agrees with T3's exact gain: `+3.75 ± 6.16`** (`+1.64` vs `-2.10`, `MAP32` jumped `0.06 → +36.72` with `21.4` kills from keeping loaded guns!). Pooled across T3's 5 seeds (`+7.9 ± 3.3`) + T2's 5 seeds (`-4.26 ± 3.74`), overall is ~`+1.8` (noisy on T2–T4) while `MAP25–32` is consistently ~`+3.8`. Leaving mask decision to Core (`pytest`: `9/9` passed).
  - **Round 6 Geodesic Shaping (`467034f`) live:** Pulled `100d54e`, ran `check_geodesic.py` (`17/20` real maps reachable from start), and **restarted `t2w` fresh from `v3` (`t2h_2150_score46_62.pt`)** + `t2h` (`2×gen2+6×gen3+6×gen4`) onto Round 6!
  - **Warp-spot dead-end audit (`t2_warp_audit.csv`):** Actively running across all 907 spots (`<=1,500u`, 20 training maps × 3 eps = 2,721 eps) on 8 batched-CUDA workers. **Task F0 `docker build`** still on `stage 6/9`.
- **UPDATE 23:16:** Acknowledged `007f365` (freeze on new ideas; no Round 6; CUDA 5-seed holdout-only pre-screen gatekeeper active).
  - **Showcase pushed to `origin/demos` (`de4bf89`):** `SHOWCASE_V3_SHIPPED.mp4` (`33 MB`, 6 clips: `two_rooms` 67 steps, unseen `gen_t2_1` 158 steps, unseen `gen_t3_1` 292 steps, unseen `gen_t3_2` 496 steps — all 4 exits — plus `gen_t4_1` & `MAP30`).
  - **23:05 Round-5b 3-seed holdout evals:** `t2h_2305` **`41.42`** (new best unseen **`MAP25–32: -0.76`**, `T2 96.82 / 83%`, `T3_1 100%`, `T4 27.82`) and `t2w_2305` `33.98` (`MAP25–32: -4.46`, `T2 67%`, `T3 33%`). Neither beats `v3` under the ship rule yet → **not pushed to `candidates`** (saving T3's low-token budget).
  - **Gatekeeper setup:** Running 5-seed CUDA `shipped-v3` baseline (`cuda_v3_s04.csv`, seeds `0..4`, 17 holdout maps ONLY — never `maps/fresh/`) right now so all future pre-screens are strictly paired on CUDA.
  - **Training + F0:** `t2h` (`~553 fps`, `T3 83%`, `T5 2–6%`, `T6 3–8%`) + `t2w` (`~354 fps`, `T2 78%`, `T3 46%`, `T5 4–9%`, `T6 8%`) healthy (`4.3 GiB` RAM free); `docker build` still on `stage 6/9`. Added Tier-4 & USE-press findings under `## Future Work (Diagnostic Findings for Brief)` below.
- **UPDATE 22:55:** Pulled `0c69543` (**Round 5b `T6` key warps** + answers `0fb3782`). Restarted dual-trainer onto Round 5b (`t2h` 24 envs now on core-approved `2×gen2+6×gen3+6×gen4` + `t2w` 12 envs on `DEFAULT_TRAIN_MAPS`, both resumed with ~0 lost steps, `~1,075` combined fps, `3.3 GiB` RAM free). **Ran two proactive diagnostics on `shipped-v3` (CUDA, seeds 0–4):**
  - **Tier-4 bottleneck (`gen_t4_1` & `gen_t4_2` @ 2,100 & 4,200 tics, n=20):** `0/20` exits, **`19/20` deaths** (median tic `~1,150–1,350`; doubling to 4,200 tics gave `10/10` deaths, `0/10` exits), **`1/20` picked even 1 key (`0/20` locked doors)** despite `3–5.4` kills, `2–3` normal doors, and `90–108` cells. Bottleneck is **combat survival + key search depth**, NOT the 2,100-tic cap.
  - **Task Y open test (`MAP29` & `MAP02` USE-press audit, n=10):** `overall_use` is already **`32–48%`** of all steps (`MAP29` reached `87u`/`116u` with `6/12` & `1/4` USE presses; `MAP02` closest was `176–384u`, all deaths). So near-misses are wall/lock/survival barriers, not missing USE presses.
  - **Fresh `shipped-v3` showcase recorded locally** (6 clips: `two_rooms` 67 steps, unseen `gen_t2_1` 158 steps, unseen `gen_t3_1` 292 steps, unseen `gen_t3_2` 496 steps — all 4 `EXIT REACHED!`). **Task F0 `docker build`** still on `stage 6/9` (`uv pip install`). Next: `~23:05 IST` eval + single best candidate push (`t2w_2300` or `t2h_2300`) to `origin/candidates`.
- **UPDATE 22:37:** Pulled `33383b8` — awesome to see `shipped-v3` (`t2h_2150`) confirmed on 10 seeds (`42.30`, `+9.3 ± 2.5`, MAP25–32 `+7.8 ± 1.7`, T3 `14/20`). Round-4 1.2M-step eval done: `t2h_r4_2225` `44.58` (`T3 126.51 / 83%`, `MAP25–32 -3.79`) and `t2h_r4_best_2225` `39.68` (`MAP25–32 -1.06`, `T4 28.11`) — not pushed to `candidates` to save T3's low-token R2 budget. **Task H2 live:** dual-trainer running in shared PID namespace (`t2h` 24 envs `~619 fps` + `t2w` 12 envs `~387 fps` = **`~1,006` combined fps**, `3.7 GiB` RAM free, `60°C`); **`t2w` already getting real-map reverse-curriculum exits `T5: 6–8%` (`T3: 54%`, `T2: 57%`, `T0: 4–16%`)**, `t2h` `T5: 2–12%` (`T3: 37%`). **Task F0 (`docker build -t devforge-doom .`)** still downloading `docker/requirements.lock` (`stage 6/9`); will run `validate.py --docker` immediately on completion. Next: `t2w` + `t2h` 3-seed eval & `t2w_2300.pt` push to `candidates` at ~`23:00–23:10 IST`.
- **UPDATE 22:16:** Pulled `400a820` (`shipped-v3`), started Task F0 `docker build`, pushed gameplay + live EgoMap demo videos to orphan branch `origin/demos` (`FULL_DEMO_SHOWCASE.mp4`, `03_tier3_keycard_backtrack.mp4`, `02_tier2_unseen_exit.mp4`), armed 10-min auto-sync + 30-min evals.

**QUESTION(core):**
1. Thanks for the fast answers (`0fb3782`)! While `t2h` + `t2w` train on Round 5b and `docker build` finishes, we have active cycles + RTX 3050 CUDA — **what else can Teammate 2 take on so we don't sit idle?** Specifically:
   - Should we run fast **5-seed paired CUDA pre-screens** on Core's `core_warp_*` / `map2` candidates while Teammate 3 is in low-token mode until `01:20 IST`?
   - Any other diagnostics (e.g., per-warp-spot dead-end audit on `T5`/`T6`, or testing `gen_t4` with lower enemy count / key-warp ideas) that would help Tutor Round 6?
   - Should we push the new `shipped-v3` showcase MP4 (with BOTH unseen T3 exits!) to `origin/demos` for the final presentation?

---

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

## validate.py --docker (official image)

Command: `.venv/bin/python validate.py --docker --image devforge-doom --no-gpu` (Linux host, official `Dockerfile` + `docker/requirements.lock`, `shipped-v5` weights, `2026-11-27 00:53 IST`)

```text
 ✓ Repository size (tracked files): 4.1 MB of 1.0 GB
$ docker run --rm --user 1000:1000 --network none --read-only --memory 10240m --memory-swap 10240m --pids-limit 512 --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:exec,size=2g --shm-size 2g -e HOME=/tmp -e PYTHONPATH=/deps -w /tmp -v /home/akash/TeamAyushDevForge:/submission:ro -v /tmp/doom-validate-docker-e_577wh0/deps:/deps:ro -v /home/akash/TeamAyushDevForge/validate.py:/validator/validate.py:ro -v /home/akash/TeamAyushDevForge/maps/two_rooms.wad:/validator/maps/0_two_rooms.wad:ro devforge-doom python /validator/validate.py --repo /submission --in-docker --device cpu --ticks 2100 --map /validator/maps/0_two_rooms.wad
Validating /submission  (contract 1.0.0-draft)

 ✓ requirements.txt: no extra packages
 ✓ DoomAgent interface: __init__, reset, act
 ✓ CPU limit: agent pinned to 1 CPU (core 11), like the evaluation sandbox
 ✓ Initialization: 6.0s (limit 60s) on device 'cpu'
 ✓ Episode on '0_two_rooms.wad:MAP01': 190 steps, exit reached, agent time 1.1s of 21.0s budget, 1.4s wall-clock
 ✓ Step latency '0_two_rooms.wad:MAP01': first 13 ms, p50 5.9 ms, p99 6.4 ms, max 13 ms
 ✓ Peak RAM: 797 MB resident (limit 8192 MB, including files written to /tmp)

PASSED
```

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
  - **Phase 2 (`21:57–22:24 IST`, round-4 rewards from `376340f`):** Resumed from `t2h_2150_score46_62.pt` (`--lr 1.5e-4`, `24` envs, `256` steps, `2` minibatches, `--seed 78`, 20 compact Freedoom + `two_rooms` + `6× gen3` + `4× gen4`), completed `~1.2M` Round-4 steps (`t2h_r4_2225`: `44.58` MEAN, `T3 83%`).
  - **Dual-Trainer Round 5 (`22:28 IST` → running, `check_warp_curriculum.py` OK):** Running **`t2h` (`24` envs, seed 78, resumed from `t2h_r4_1200k_snap.pt`)** + **Task H2 `t2w` (`12` envs, seed 79, `DEFAULT_TRAIN_MAPS`, resumed from `t2h_2150_score46_62.pt`)** together in one shared PID namespace (`~1,006` combined fps, `3.7 GiB` RAM free, `60°C`). Both immediately hit real-map reverse-curriculum exits (`T5: 6–8%` on `t2w`, `T5: 2–12%` on `t2h`).

---

## Checkpoint leaderboard

| time | run | steps (from the log) | MEAN score | exit % | keys | doors | kills | ms/step | validate.py |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | random (17 maps, seed 0, cuda) | 0 | -1.58 | 12% | 0.0 | 0.5 | 1.0 | 7.16 | PASS (15.1 ms p50, 130 MB VRAM) |
| 19:00 | seed7_1900 (plain; T0:-6.77 T1:100.52 T2:15.51 T3:6.62 T4:-9.75) | 563,200 | 9.70 | 12% | 0.0 | 0.4 | 1.1 | 8.53 | PASS (16.2 ms p50, 130 MB VRAM) |
| 19:48 | egomap_s77_1948 (EgoMap latest; T0:3.15 T1:101.20 T2:56.03 T3:52.89 T4:18.36) | ~1,610,000 | 28.55 | 24% | 0.1 | 1.0 | 1.1 | 9.44 | — |
| 19:48 | egomap_s77_best_1948 (EgoMap best; T0:7.31 T1:101.47 T2:92.83 T3:20.32 T4:15.99) | ~1,580,000 | 31.00 | 25% | 0.1 | 1.1 | 1.1 | 8.13 | PASS (exit in 384 steps, 12.4 ms p50, 130 MB VRAM) |
| 21:50 | t2h_2150 (Task H latest; T0:9.03 / MAP25-32:-3.79(0%) T1:100.90 T2:112.97(100%) T3:111.51(67%) T4:12.85) | ~8,650,000 | 44.57 | 37% | 0.1 | 1.2 | **1.7** | 5.46 | — |
| 21:50 | **t2h_best_2150** *(shipped as `shipped-v3`, i3 10-seed `42.30`!)* (T0:**9.45** / MAP25-32:**-3.11(0%)** T1:**101.78** T2:98.51 T3:**124.42(83%)** T4:**29.05**) | ~8,300,000 | **46.62** | **37%** | **0.2** | **1.3** | 1.4 | **3.57** | **PASS (exit in 68 steps, 9.0 ms p50, 130 MB VRAM)** |
| 22:25 | t2h_r4_2225 (Round-4 latest; T0:8.98 / MAP25-32:-3.79(0%) T1:101.97(100%) T2:95.28(83%) T3:**126.51(83%)** T4:14.80) | ~9,500,000 | 44.58 | 37% | 0.2 | 1.0 | 1.0 | 10.29* | — |
| 22:25 | t2h_r4_best_2225 (Round-4 best; T0:**11.25** / MAP25-32:**-1.06(0%)** T1:101.90(100%) T2:82.38(67%) T3:74.24(33%) T4:28.11) | ~9,500,000 | 39.68 | 29% | 0.1 | 1.1 | **1.8** | 12.41* | — |
| 23:05 | t2w_2305 (Round-5b `DEFAULT_TRAIN_MAPS`; T0:8.51 / MAP25-32:-4.46(0%) T1:101.37(100%) T2:71.91(67%) T3:69.03(33%) T4:8.25) | ~8,820,000 | 33.98 | 29% | 0.1 | 1.1 | 1.4 | 13.05* | — |
| 23:05 | t2h_2305 (Round-5b `2×gen2+6×gen3+6×gen4`; T0:**11.49** / MAP25-32:**-0.76(0%)** T1:101.53(100%) T2:96.82(83%) T3:74.22(50%) T4:27.82) | ~10,350,000 | 41.42 | 33% | 0.1 | **1.4** | 1.3 | 12.31* | — |
| 00:06 | t2w_0006 (Round-6 early; T0:5.39 / MAP25-32:-7.74(0%) T1:100.73(100%) T2:94.21(83%) T3:43.50(17%) T4:17.78) | ~8,870,000 | 33.00 | 29% | 0.1 | 1.2 | 0.9 | 15.41* | — |
| 00:06 | t2h_0006 (Round-6 early; T0:8.76 / MAP25-32:-3.86(0%) T1:100.73(100%) T2:78.58(67%) T3:83.06(50%) T4:17.77) | ~11,240,000 | 37.60 | 31% | 0.1 | 1.2 | 1.0 | 15.12* | — |
| 00:37 | t2w_0037 (Round-6 + 256/192 warp; T0:-1.18 / MAP25-32:-6.50(0%) T1:100.67(100%) T2:60.18(50%) T3:25.07(0%) T4:16.43) | ~9,350,000 | 23.18 | 20% | 0.1 | 1.0 | 1.1 | 12.30* | — |
| 00:37 | **t2h_0037** (Round-6 + 256/192 warp; T0:10.57 / MAP25-32:**-1.78(0%)** T1:101.00(100%) T2:88.82(**83%**) T3:**123.80(83%)** T4:7.27) | ~12,000,000 | **43.35** | **37%** | 0.1 | 1.0 | 1.1 | 12.99* | — |
| 00:37 | **t2h_0037** *(shipped as `shipped-v6`!)* (T0:9.06 / MAP25-32:-3.49(0%) T1:101.13(100%) T2:**99.21(90%)** T3:**120.35(80%)** T4:14.44; `exit_pts 52 vs 45`) | ~11,700,000 | **44.22** (5s) | **38%** | 0.1 | 1.1 | 1.1 | 11.0* | **PASS (exit in 60 steps, 9.0 ms p50, 130 MB VRAM)** |
| 01:22 | **t2h_0122** (Round-6 + `256/192` warp, 3s preview; T0:**17.66** / MAP25-32:**+6.36(0%)** T1:100.93(100%) T2:**113.83(100%)** T3:**142.88(100%)** T4:14.59) | ~12,400,000 | **53.14** | **41%** | 0.1 | 1.3 | **3.7** | 11.10* | *(running 5s deterministic check)* |
| 00:37 | t2w_0037 (Round-6 + `256/192` warp; T0:-1.18 / MAP25-32:-6.50(0%) T1:100.67(100%) T2:60.18(50%) T3:25.07(0%) T4:16.43) | ~9,300,000 | 23.18 | 20% | 0.1 | 1.1 | 1.0 | 13.5* | — |

*\*`ms/step` inflated because 36 ViZDoom training workers + `docker build` ran simultaneously on the same CPU.*

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

---

## Future Work (Diagnostic Findings for Brief — `shipped-v3`, CUDA)

1. **Tier-4 Two-Lock Bottleneck (`gen_t4_1` & `gen_t4_2`, seeds `0–4`, `@ 2,100` and `@ 4,200` tics, `n=20`):**
   - Outcome: `0/20` exits, **`19/20` deaths**, `1/20` timeout (doubling the time cap to `4,200` tics still produced `10/10` deaths and `0/10` exits; median death tic `~1,150–1,350`).
   - Progression: `3.0–5.4` kills, `2.0–3.0` normal doors opened, `90–108` cells explored, yet only **`1/20` picked up Key 1 (`0/20` locked doors opened)**.
   - Takeaway: Tier-4 is gated by **combat survival and deep key search**, not episode length. Future iterations could add training-only geodesic BFS potential shaping (`γΦ(s') - Φ(s)` over `(tile, keys_held)`) and/or visual exit-switch perception.
2. **Real-Map Near-Miss `USE`-Press Audit (`MAP29` & `MAP02`, seeds `0–4`, `4,200` tics, `n=10`):**
   - Overall `USE` press rate is already `32–48%` of all steps; on `MAP29` (`walk` exit), the agent reached `87u` (`6/12` `USE` presses) and `116u` (`1/4` `USE` presses); on `MAP02` (`switch` exit), closest approach was `176–384u` before death (`5/5` deaths).
   - Takeaway: Real-map near-misses are caused by walls, locked doors, and combat deaths rather than failure to press `USE` (folding Task Y blocker #3 into survival and key acquisition).

---

## Per-Warp-Spot Dead-End Audit (`training/results/t2_warp_audit.csv`)

Audited every `warp_spots(map, "exit")` spot within `1,500` units across all 20 `REAL_TRAIN_MAPS` (`907` spots × `3` episodes = `2,721` episodes, `shipped-v3`, `2,100` tics).
- **Summary:** `33 / 907` (**`3.6%`**) working (`>= 1/3` exits), **`874 / 907` (`96.4%`) zero-exit dead spots** (all 874 `(map, x, y, dist)` entries with `exits == 0` are in `training/results/t2_warp_audit.csv`).

| Map | Spots <=1500u | Working (>=1/3 exits) | Zero-exit (0/3) | Dead % | Min working dist | Min dead dist |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAP01 | 76 | 2 | 74 | 97% | 544u | 348u |
| MAP02 | 96 | 0 | 96 | 100% | — | 165u |
| MAP03 | 30 | 2 | 28 | 93% | 60u | 320u |
| MAP04 | 55 | 0 | 55 | 100% | — | 128u |
| MAP05 | 76 | 10 | 66 | 87% | 36u | 79u |
| MAP06 | 42 | 2 | 40 | 95% | 115u | 218u |
| MAP07 | 13 | 0 | 13 | 100% | — | 305u |
| MAP08 | 25 | 0 | 25 | 100% | — | 112u |
| MAP09 | 49 | 0 | 49 | 100% | — | 283u |
| MAP10 | 20 | 0 | 20 | 100% | — | 592u |
| MAP13 | 19 | 3 | 16 | 84% | 81u | 316u |
| MAP14 | 52 | 7 | 45 | 87% | 25u | 184u |
| MAP16 | 44 | 1 | 43 | 98% | 373u | 82u |
| MAP17 | 67 | 0 | 67 | 100% | — | 96u |
| MAP18 | 67 | 2 | 65 | 97% | 91u | 130u |
| MAP19 | 12 | 2 | 10 | 83% | 542u | 525u |
| MAP20 | 15 | 0 | 15 | 100% | — | 1200u |
| MAP21 | 50 | 1 | 49 | 98% | 1399u | 369u |
| MAP22 | 49 | 1 | 48 | 98% | 112u | 304u |
| MAP24 | 50 | 0 | 50 | 100% | — | 274u |

**All 33 working exit-warp spots (`exits >= 1/3`; every other spot in `t2_warp_audit.csv` is `0/3`):**
- `MAP01`: `(-192,-992)` `544.1u` (`1/3`), `(1240,-968)` `1077.7u` (`1/3`)
- `MAP03`: `(2812,-1760)` `60.0u` (`1/3`), `(3488,-1760)` `736.0u` (`1/3`)
- `MAP05`: `(-732,572)` `36.2u` (`2/3`), `(-744,504)` `75.9u` (`1/3`), `(-696,552)` `75.9u` (`3/3`), `(-720,496)` `93.3u` (`3/3`), `(-664,572)` `104.1u` (`2/3`), `(-628,552)` `142.0u` (`1/3`), `(-640,648)` `146.9u` (`3/3`), `(-600,608)` `171.0u` (`1/3`), `(-604,504)` `179.1u` (`2/3`), `(-352,656)` `423.6u` (`1/3`)
- `MAP06`: `(608,1696)` `115.4u` (`1/3`), `(608,1400)` `150.3u` (`1/3`)
- `MAP13`: `(-112,-232)` `81.2u` (`2/3`), `(-176,-96)` `97.7u` (`2/3`), `(-80,-184)` `114.3u` (`2/3`)
- `MAP14`: `(-216,936)` `25.3u` (`3/3`), `(-104,936)` `25.3u` (`3/3`), `(-160,792)` `104.0u` (`3/3`), `(-208,792)` `114.5u` (`3/3`), `(-112,792)` `114.5u` (`3/3`), `(-336,848)` `164.7u` (`3/3`), `(16,848)` `164.7u` (`3/3`)
- `MAP16`: `(-992,-608)` `373.4u` (`2/3`)
- `MAP18`: `(712,1816)` `91.2u` (`1/3`), `(712,1960)` `91.2u` (`3/3`)
- `MAP19`: `(1544,-2424)` `541.8u` (`1/3`), `(1744,-1776)` `1169.0u` (`1/3`)
- `MAP21`: `(-168,4272)` `1399.2u` (`1/3`)
- `MAP22`: `(480,352)` `112.0u` (`1/3`)
