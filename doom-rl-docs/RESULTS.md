# Teammate 2 (RTX 3050) — Validation, Holdout Baselines & Leaderboard

Environment: Linux (12 CPU threads, 15 GB RAM, RTX 3050 6 GB), Python 3.13.18, `torch==2.15.0+cpu` (temporary CPU wheel while CUDA download timed out on slow Wi-Fi; `doom_env` contract `1.0.0-draft`).
Docker validation: skipped locally (running `validate.py` directly on Linux with 1-CPU pinning).

---

## Baseline validate.py (random weights)

Command: `.venv/bin/python validate.py`

```text
Validating /home/akash/TeamAyushDevForge  (contract 1.0.0-draft)

 ✓ Repository size (tracked files): 346.7 KB of 1.0 GB
 ✓ requirements.txt: no extra packages
 ✓ DoomAgent interface: __init__, reset, act
 ✓ CPU limit: agent pinned to 1 CPU (core 11), like the evaluation sandbox
 ✓ Initialization: 3.0s (limit 60s) on device 'cpu'
 ✓ Episode on 'two_rooms.wad:MAP01': 525 steps, tick cap, agent time 7.5s of 21.0s budget, 8.6s wall-clock
 ✓ Step latency 'two_rooms.wad:MAP01': first 3950 ms, p50 6.7 ms, p99 8.5 ms, max 3950 ms
 ✓ Peak RAM: 355 MB resident (limit 8192 MB, including files written to /tmp)

PASSED
```

- **Status:** PASSED (0 warnings, 0 failures)
- **Latency:** p50 `6.7 ms/step`, p99 `8.5 ms/step`, first `3950 ms` (on 1 pinned CPU core)
- **Memory:** Peak RAM `355 MB` / `8192 MB` (VRAM N/A on CPU torch)

---

## Random-weights holdout baseline

Command: `.venv/bin/python training/eval_zeroshot.py --name random --seeds 0 --device cpu`
CSV: `training/results/random.csv`

```text
two_rooms.wad:MAP01        | score    4.30 | exit    0% | keys 0.0 | doors  1.0 | kills  0.0 | cells  25.0 | hp   0.0 | 10.61 ms/step
freedoom2.wad:MAP25        | score   -8.10 | exit    0% | keys 0.0 | doors  2.0 | kills  3.0 | cells  33.0 | hp   0.0 | 9.51 ms/step
freedoom2.wad:MAP26        | score  -22.70 | exit    0% | keys 0.0 | doors  0.0 | kills  1.0 | cells   9.0 | hp   0.0 | 10.58 ms/step
freedoom2.wad:MAP27        | score  -22.60 | exit    0% | keys 0.0 | doors  0.0 | kills  0.0 | cells  17.0 | hp   0.0 | 6.72 ms/step
freedoom2.wad:MAP28        | score   -5.90 | exit    0% | keys 0.0 | doors  2.0 | kills  3.0 | cells  31.0 | hp   0.0 | 8.86 ms/step
freedoom2.wad:MAP29        | score  -22.85 | exit    0% | keys 0.0 | doors  0.0 | kills  0.0 | cells  12.0 | hp   0.0 | 9.36 ms/step
freedoom2.wad:MAP30        | score    4.35 | exit    0% | keys 0.0 | doors  1.0 | kills  0.0 | cells  52.0 | hp   0.0 | 8.77 ms/step
freedoom2.wad:MAP31        | score  -27.80 | exit    0% | keys 0.0 | doors  0.0 | kills  0.0 | cells   1.0 | hp   0.0 | 8.96 ms/step
freedoom2.wad:MAP32        | score  -17.15 | exit    0% | keys 0.0 | doors  1.0 | kills  0.0 | cells  16.0 | hp   0.0 | 10.92 ms/step
--------------------------------------------------------------------------------------------------------------------
MEAN                       | score  -13.16 | exit    0% | keys 0.0 | doors  0.8 | kills  0.8 | cells  21.8 | 9.37 ms/step
MEAN score: -13.16
```

---

## Checkpoint leaderboard

| time | run | steps (from the log) | MEAN score | exit % | keys | doors | kills | ms/step | validate.py |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| baseline | random (seed 0) | 0 | -13.16 | 0% | 0.0 | 0.8 | 0.8 | 9.37 | PASS (6.7 ms p50) |
