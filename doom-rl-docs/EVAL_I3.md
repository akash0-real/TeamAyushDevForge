# Low-End CPU Validation + 5-Seed Holdout Eval (teammate 3)

**Machine:** our slowest laptop, the best stand-in for a weak evaluation CPU.

| | |
|---|---|
| CPU | Intel Core i3-1215U |
| RAM | 8 GB |
| GPU | none (CPU only) |
| OS | Windows 11 |
| Python | 3.13.15 |
| Torch | `torch==2.15.0+cpu` |

**Weights:** `submission/weights/best_agent.pt`, 3,051,736 bytes. That's the real LFS object, not a pointer; this is
the shipped `map_phase1` checkpoint. Run at commit `6cfa81c`.

## 1. `validate.py --device cpu` (19:25 IST)

```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run python validate.py --device cpu
```

```text
Validating C:\Users\ayush\Downloads\Devforge\TeamAyushDevForge  (contract 1.0.0-draft)

 ✓ Repository size (tracked files): 3.6 MB of 1.0 GB
 ✓ requirements.txt: no extra packages
 ✓ DoomAgent interface: __init__, reset, act
 ! CPU limit: could not pin the agent to 1 CPU here; evaluation gives it one CPU, so timings may look better than they will be
 ✓ Initialization: 15.0s (limit 60s) on device 'cpu'
 ✓ Episode on 'two_rooms.wad:MAP01': 79 steps, exit reached, agent time 0.7s of 21.0s budget, 2.9s wall-clock
 ✓ Step latency 'two_rooms.wad:MAP01': first 46 ms, p50 7.1 ms, p99 17.6 ms, max 46 ms
 ! Peak RAM: could not be measured here (needs Linux /proc); the limit is 8192 MB

PASSED with 2 warning(s).
```

| Check | i3 (this run) | Limit | Status |
|---|---:|---:|---|
| Initialization | 15.0 s | 60 s | OK, 4× headroom |
| First step | 46 ms | 10 s | OK |
| p50 / p99 step | 7.1 / 17.6 ms | 40 ms avg budget (alarm: p99 > 30 ms) | OK |
| Agent time on the test map | 0.7 s | 21.0 s | OK; it exited in 79 steps |

**Caveats:**
- Windows can't pin to 1 CPU, so these timings may be slightly optimistic.
- Peak RAM needs Linux. Teammate 2's Linux run with 1 pinned CPU measured 355 MB.

## 2. 5-seed holdout eval

```powershell
uv run python training/eval_zeroshot.py --weights submission/weights/best_agent.pt --name i3_shipped --seeds 0 1 2 3 4 --device cpu
```

19:25–19:49 IST, 24 minutes, at commit `6cfa81c`. 17 holdout maps × 5 seeds. CSV:
`training/results/i3_shipped.csv`.

```text
[T0] two_rooms.wad:MAP01        | score  112.67 | exit  100% | keys 0.0 | doors  1.0 | kills  0.8 | cells  36.0 | hp  94.6 | 11.78 ms/step
[T0] freedoom2.wad:MAP25        | score   -9.70 | exit    0% | keys 0.0 | doors  1.2 | kills  1.2 | cells  43.2 | hp   0.0 | 12.26 ms/step
[T0] freedoom2.wad:MAP26        | score  -13.59 | exit    0% | keys 0.0 | doors  0.0 | kills  4.4 | cells  19.6 | hp   0.0 | 10.58 ms/step
[T0] freedoom2.wad:MAP27        | score    6.80 | exit    0% | keys 0.0 | doors  1.0 | kills  2.6 | cells  62.4 | hp   2.6 | 9.98 ms/step
[T0] freedoom2.wad:MAP28        | score    8.29 | exit    0% | keys 0.0 | doors  1.0 | kills  1.0 | cells  54.8 | hp   8.0 | 8.42 ms/step
[T0] freedoom2.wad:MAP29        | score   -8.32 | exit    0% | keys 0.0 | doors  1.0 | kills  0.2 | cells  65.2 | hp   0.0 | 9.25 ms/step
[T0] freedoom2.wad:MAP30        | score    4.75 | exit    0% | keys 0.0 | doors  0.8 | kills  2.2 | cells 119.6 | hp   0.0 | 10.76 ms/step
[T0] freedoom2.wad:MAP31        | score  -26.20 | exit    0% | keys 0.0 | doors  0.0 | kills  0.0 | cells   1.0 | hp   0.0 | 9.96 ms/step
[T0] freedoom2.wad:MAP32        | score  -13.01 | exit    0% | keys 0.0 | doors  1.2 | kills  1.6 | cells  36.0 | hp   0.0 | 15.44 ms/step
[T1] gen_t1_1.wad:MAP01         | score  100.76 | exit  100% | keys 0.0 | doors  0.0 | kills  0.0 | cells   3.8 | hp 100.0 | 16.37 ms/step
[T1] gen_t1_2.wad:MAP01         | score  103.30 | exit  100% | keys 0.0 | doors  0.0 | kills  0.8 | cells  14.6 | hp  91.6 | 9.38 ms/step
[T2] gen_t2_1.wad:MAP01         | score   59.27 | exit   40% | keys 0.0 | doors  0.8 | kills  1.6 | cells  76.0 | hp  85.4 | 10.45 ms/step
[T2] gen_t2_2.wad:MAP01         | score   56.65 | exit   40% | keys 0.0 | doors  1.8 | kills  3.2 | cells  58.0 | hp  44.4 | 8.44 ms/step
[T3] gen_t3_1.wad:MAP01         | score  100.75 | exit   60% | keys 1.0 | doors  1.0 | kills  0.8 | cells  60.8 | hp  98.2 | 8.51 ms/step
[T3] gen_t3_2.wad:MAP01         | score   33.55 | exit    0% | keys 0.8 | doors  1.0 | kills  4.0 | cells  81.4 | hp  53.4 | 14.71 ms/step
[T4] gen_t4_1.wad:MAP01         | score   13.46 | exit    0% | keys 0.2 | doors  0.8 | kills  3.6 | cells  70.8 | hp  34.6 | 20.51 ms/step
[T4] gen_t4_2.wad:MAP01         | score    0.24 | exit    0% | keys 0.0 | doors  1.0 | kills  3.8 | cells  43.2 | hp   3.2 | 17.70 ms/step
-------------------------------------------------------------------------------------------------------------------------
MEAN                            | score   31.16 | exit   26% | keys 0.1 | doors  0.8 | kills  1.9 | cells  49.8 | 12.03 ms/step
MEAN score: 31.16
T0 mean score:    6.85 (n=9, exit=  11%)
T1 mean score:  102.03 (n=2, exit= 100%)
T2 mean score:   57.96 (n=2, exit=  40%)
T3 mean score:   67.15 (n=2, exit=  30%)
T4 mean score:    6.85 (n=2, exit=   0%)
```

| Tier | Mean score | Exits |
|---|---:|---:|
| **All 17 maps** | **31.16** | **26%** |
| T0: real Freedoom MAP25–32 + `two_rooms` | 6.85 | 11% |
| T1: combat | 102.03 | 100% |
| T2: exploration | 57.96 | 40% |
| T3: key → locked door → exit | 67.15 | 30% (`gen_t3_1`: 60%, keys 1.0) |
| T4: two locks | 6.85 | 0% |

**ms/step is not a latency measurement.** These are wall-clock numbers on a busy laptop: my audit edge test also ran
during part of this eval. Use §1 for latency.

### ⚠ Finding: 2-seed scores are noisy enough to flip checkpoint decisions
Same shipped weights, same eval, same seeds 0 1:

| Run | Seeds | Mean score |
|---|---|---:|
| Core owner (`CHECKPOINTS.md`) | 0 1 | 27.7 |
| This run, seeds 0 1 only (from the CSV) | 0 1 | **36.43** |
| This run, all 5 seeds | 0–4 | 31.16 |

Identical weights and seeds differ by **8.7 points**. `act()` *samples* its actions (`deterministic=False`, see
`submission/agent.py`), and torch's RNG isn't seeded, so reruns aren't reproducible.

That spread is bigger than the gaps used to choose checkpoints: shipped 27.7 vs. `map_1970k` 27.0 vs. `map_eval3`
22.5. **Suggestion for the core owner:**
- Use 5 or more seeds for the final pick (the plan says 3).
- Treat differences under about 9 points as ties.
- Optionally, seed torch in `eval_zeroshot.py` so reruns are comparable.

## 3. Setup notes
- `uv python install 3.13` failed on this machine with a uv error ("Missing expected target directory for Python
  minor version link"). It isn't needed when Python 3.13 is already installed, so I skipped it; `uv sync` then
  worked.
- Behind this network's TLS proxy, uv needs `$env:UV_SYSTEM_CERTS = "1"`.

## 4. Candidate checks (Task R2)

**Setup:** 17 holdout maps × seeds 0–4, CPU, i3. Both candidates ran in parallel at 20:23–20:43 IST, at temperature
1.0. Baseline = the shipped weights from §2 (same maps, same seeds).

**CSVs:** `training/results/i3_core_r1_2025.csv`, `i3_core_r2_2025.csv`.

| Checkpoint | MEAN | exit % | T0 real | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| shipped `best_agent.pt` (§2) | 31.16 | 26% | 6.9 (11%) | 102.0 (100%) | 58.0 (40%) | 67.2 (30%) | 6.9 (0%) |
| `core_r1_2025` | 31.29 | 28% | −0.3 (9%) | 103.4 (100%) | 91.6 (70%) | 62.3 (30%) | 9.8 (0%) |
| **`core_r2_2025`** | **34.85** | **29%** | 4.2 (9%) | 102.3 (100%) | **98.6 (80%)** | 64.4 (30%) | **12.2 (0%)** |

**Paired comparison with the shipped weights.** Same seed against same seed, which removes most of the
seed-to-seed noise:

| Checkpoint | Δ MEAN per seed (0..4) | Δ MEAN ± SE | Δ T0 ± SE |
|---|---|---:|---:|
| `core_r1_2025` | −4.0, +3.3, +11.8, +3.7, −14.1 | +0.1 ± 4.4 | **−7.1 ± 2.9** (worse on 5/5 seeds) |
| `core_r2_2025` | +1.5, +3.5, +4.0, +1.9, +7.6 | **+3.7 ± 1.1** (better on 5/5 seeds) | −2.7 ± 3.6 |

**Verdicts:**
- **`core_r2_2025`: best candidate so far.** It's consistently better overall: +3.7 on every seed, mostly from tier 2
  (98.6 vs 58.0, 80% exits). Its lower T0 point estimate (4.2 vs 6.9) is **within noise** (−2.7 ± 3.6). By the strict
  shipping rule ("not worse on real maps") it's a tie on T0, not a loss.
  - **Recommendation:** worth shipping if `validate.py` passes on a clean clone.
  - If the core owner wants more certainty on T0, a 10-seed run of T0 maps only would settle it. I can run that.
- **`core_r1_2025`: don't ship.** No real overall gain (+0.1 ± 4.4), and it's **genuinely worse on real maps**
  (−7.1 ± 2.9, lower on all 5 seeds).
