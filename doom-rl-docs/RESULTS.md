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

## Seed-7 training run (RTX 3050 CUDA)

- **Smoke test (`two_rooms`, 4 envs, 20k steps):** PASSED (`19,968` steps, `142–146 fps` on CUDA, zero crashes).
- **Main Seed-7 run:**
  - **Command:** `.venv/bin/python -u training/train_ppo.py --num-envs 8 --seed 7 --total-steps 20000000 --save-path training/checkpoints/seed7.pt`
  - **Start time:** `2026-11-26 18:16 IST` (on commit `5a9bca3` map mix: `two_rooms` + Freedoom `MAP01–24` + infinite procedural `gen1–gen4`, no `--use-map`)
  - **10-min log check (`18:27 IST`):** `upd 125/19531 | step 128000 | fps 207 | ret 3.68 (peak 8.00) | exit 12% | doors 1.1 | keys 0.1 | cells 48.2 | T0:6% T1:94% T2:17% T3:3% T4:0%`
  - **Steady-state throughput:** **`207 fps`** on RTX 3050 CUDA (`8` envs)

---

## Checkpoint leaderboard

| time | run | steps (from the log) | MEAN score | exit % | keys | doors | kills | ms/step | validate.py |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | random (17 maps, seed 0, cuda) | 0 | -1.58 | 12% | 0.0 | 0.5 | 1.0 | 7.16 | PASS (15.1 ms p50, 130 MB VRAM) |
| 19:00 | seed7_1900 (17 maps, seeds 0-2, cuda; T0:-6.77 T1:100.52 T2:15.51 T3:6.62 T4:-9.75) | 563,200 | **9.70** | 12% | 0.0 | 0.4 | 1.1 | 8.53 | PASS (16.2 ms p50, 130 MB VRAM) |
