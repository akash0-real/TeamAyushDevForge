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

## 5. Action-temperature test (Task T)

**Setup:** shipped weights, 17 holdout maps × seeds 0–2, CPU, i3. The three runs ran in parallel at 20:44–21:00 IST,
on code `376340f` (`agent.temperature`, `eval --temperature`). The T = 1.0 row is the same seeds 0–2 from
`i3_shipped.csv`. Δ = paired per-seed difference vs T = 1.0.

**CSVs:** `training/results/i3_T03.csv`, `i3_T05.csv`, `i3_T07.csv`.

| T | MEAN | T0 real | T1 | T2 | T3 | T4 | exit % | Δ per seed (0, 1, 2) | Δ mean |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1.0 | 30.88 | 6.4 (11%) | 102.2 (100%) | 55.7 (33%) | 69.5 (33%) | 6.3 (0%) | 25% | — | — |
| 0.3 | 30.71 | 2.4 (4%) | 101.6 (100%) | 78.6 (67%) | 47.7 (17%) | 22.3 (0%) | 24% | −4.3, −3.1, +6.9 | −0.2 |
| **0.5** | 33.37 | 5.2 (11%) | 102.4 (100%) | 79.6 (67%) | **73.2 (33%)** | 5.1 (0%) | **29%** | −4.8, +2.2, +10.1 | +2.5 |
| 0.7 | **34.32** | **11.8 (11%)** | 101.6 (100%) | **101.6 (83%)** | 18.4 (**0%**) | 17.2 (0%) | 27% | +1.5, −5.6, +14.4 | +3.4 |

**Reading it:**
- **No temperature is a clear winner on 3 seeds.** The per-seed Δ ranges from −5.6 to +14.4, so a mean gain of +2.5 or
  +3.4 is inside the noise.
- **T = 0.3 is too greedy:** no gain, and worse on real maps and tier 3.
- **T = 0.7** has the best mean, real maps and tier 2, but **tier 3 collapses: 0% exits** vs 33% at T = 1.0. Key → locked
  door → exit is our headline skill and a heavy evaluation tier, so this is a real trade-off, not a free win.
- **T = 0.5** is the safest improvement. It's equal or better than T = 1.0 on T1, T2, T3 and overall exits, and only T0
  and T4 dip, both within noise.

**Recommendation:**
- If a value must be set now, use **T = 0.5**. Not 0.7, because of the tier-3 collapse.
- To make it a real decision, I'm extending T = 0.5 and T = 0.7 to seeds 3–4. That gives a paired 5-seed comparison with
  `i3_shipped.csv`, and the result goes in §5b.
- The best checkpoint (`core_r2_2025`, §4) should also be re-checked at the chosen T before shipping.

**`validate.py --device cpu` on `376340f` (round-4 `submission/`), i3, 20:44: PASSED.** Init 9.5 s, first step
35 ms, p50 11.9 ms, p99 20.2 ms, test map exited in 510 steps. It ran while 2 evals were finishing, so latency is
slightly inflated.

### 5b. Temperature test extended to 5 seeds (21:12 IST)

T = 0.5 and T = 0.7 also ran on seeds 3–4 (`i3_T05_s34.csv`, `i3_T07_s34.csv`). Combined with seeds 0–2, every row now
covers seeds 0–4 and is paired with `i3_shipped.csv` (T = 1.0). Δ = mean paired per-seed difference ± standard error.

| T | MEAN | T0 real | T1 | T2 | T3 | T4 | exit % | Δ MEAN | Δ T0 | Δ T3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 31.16 | 6.9 (11%) | 102.0 (100%) | 58.0 (40%) | 67.2 (30%) | 6.9 (0%) | 26% | — | — | — |
| 0.5 | 33.29 | 6.5 (9%) | 102.3 (100%) | 85.8 (70%) | 60.1 (20%) | 5.7 (0%) | 27% | +2.1 ± 2.4 | −0.4 ± 3.0 | −7.1 ± 11.1 |
| 0.7 | 32.97 | 12.1 (11%) | 101.8 (100%) | 95.2 (80%) | 18.1 (0%) | 10.7 (0%) | 27% | +1.8 ± 4.7 | **+5.3 ± 0.9** | **−49.1 ± 14.3** |

**Updated verdict.** This supersedes the 3-seed recommendation in §5.
- **T = 0.5 shows no real gain.** Every difference (+2.1 overall, −0.4 real maps, −7.1 tier 3) is inside its noise.
  It's neither better nor worse than T = 1.0.
- **T = 0.7 is a real trade-off, not a win.**
  - It's significantly better on real Freedoom maps: +5.3 ± 0.9, higher on all 5 seeds.
  - It's significantly worse on tier 3: −49 ± 14, with **0% exits** on key → locked door → exit vs 30% at T = 1.0.
  - Which matters more depends on the hidden tier weights, which we don't know.
- **Recommendation: keep T = 1.0 in `submission/agent.py`.** There's no evidence-based gain from changing it.
  - If the core owner wants to bet on real-map-like hidden maps, T = 0.7 is the only setting with a proven gain there,
    but it gives up tier 3.
  - Cheaper alternative: re-check the trade-off on the final checkpoint, because a checkpoint that's stronger on
    tier 3 may tolerate T = 0.7 better.

### 5c. Best candidate at T = 0.5 (21:19 IST)

`core_r2_2025` at T = 0.5, seeds 0–4 (`i3_core_r2_2025_T05.csv`).

| Run | MEAN | T0 real | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|
| `core_r2_2025`, T = 1.0 (§4) | 34.85 | 4.2 | 102.3 | 98.6 | 64.4 | 12.2 |
| `core_r2_2025`, T = 0.5 | 36.64 | 7.7 | 102.9 | 89.9 | 50.0 | 33.9 |

Paired per-seed differences:

| Comparison | Δ MEAN | Δ T0 | Δ T3 |
|---|---:|---:|---:|
| `core_r2` T = 0.5 vs `core_r2` T = 1.0 | +1.8 ± 3.1 | +3.5 ± 5.9 | −14.4 ± 22.1 |
| `core_r2` T = 0.5 vs shipped T = 1.0 | +5.5 ± 2.1 | +0.9 ± 2.9 | −17.2 ± 10.3 |

**Verdict:** lowering the temperature adds nothing measurable on top of `core_r2` (+1.8 ± 3.1). The tier-3 dip shows up
again, as in §5b. It's not significant here, but it points the same way. **Recommendation unchanged: ship
`core_r2_2025` at T = 1.0** (§4: +3.7 ± 1.1 vs the shipped weights, better on 5/5 seeds).

### 4b. `core_r2_2025` on real maps, 10 seeds (21:34 IST)

The shipping rule is "not worse on T0", so I ran 5 more seeds (5–9) on the 9 T0 maps, `two_rooms` + MAP25–32, for both
the shipped weights and `core_r2_2025`. CSVs: `i3_shipped_T0_s59.csv`, `i3_core_r2_2025_T0_s59.csv`. Combined with
§2/§4, that gives seeds 0–9 at T = 1.0.

| Checkpoint | T0 mean (10 seeds) | T0 exits |
|---|---:|---:|
| shipped | 6.19 | 9% |
| `core_r2_2025` | 4.29 | 10% |

**Paired per seed (0..9):** −3.4, +5.2, −15.2, −3.3, +3.3, +5.8, −2.2, +0.6, −3.3, −6.7. **Δ T0 = −1.9 ± 2.0**: within
noise, and the exit rate is the same.

**Final verdict for `core_r2_2025`:** it's better overall (+3.7 ± 1.1, 5/5 seeds) and **not measurably worse on real
maps** (−1.9 ± 2.0 over 10 seeds). It passes the shipping rule. **Ship it at T = 1.0 if `validate.py` passes on a
clean clone.**


### 4c. Candidate `core_r2_2155` (round 4), 5 seeds (21:53–22:02 IST)

Seeds split across 3 parallel processes, then merged into `training/results/i3_core_r2_2155.csv` (85 rows = 17 maps × 5 seeds). Baseline = **shipped v2** (`i3_core_r2_2025.csv`), paired per seed. "MAP25–32" = truly unseen real maps, excluding `two_rooms` (see Strict review item 4 in `AUDIT.md`).

| Slice | `core_r2_2155` | exits | Δ vs shipped v2 ± SE | seeds worse | beyond 2 SE? |
|---|---:|---:|---:|---:|---|
| overall | 39.55 | 34% | +4.69 ± 7.07 | 2/5 | noise |
| T0 incl two_rooms | 3.46 | 11% | -0.73 ± 5.20 | 3/5 | noise |
| MAP25-32 | -10.16 | 0% | -3.44 ± 3.84 | 3/5 | noise |
| T1 | 102.67 | 100% | +0.39 ± 0.46 | 2/5 | noise |
| T2 | 98.56 | 80% | -0.04 ± 29.90 | 2/5 | noise |
| T3 | 102.27 | 60% | +37.88 ± 21.20 | 1/5 | noise |
| T4 | 17.11 | 0% | +4.95 ± 7.08 | 2/5 | noise |

**Verdict:** **promising, but not yet proven better than shipped v2.**
- Overall it scores 39.55 vs 34.85 (+4.7), but that gain is noise (±7.1, worse on 2 of 5 seeds).
- Tier 3 jumps to **60% exits** (+37.9 ± 21.2, 1.8 SE), which is close to significant.
- Real maps: no exits on MAP25–32 (0/40), and the score difference there is noise.
- **Recommendation:** don't ship it over v2 on this evidence alone. Teammate 2's `t2h_2150` candidate is being scored next with the same protocol.

### 4d. Candidate `t2h_2150_score46_62` (teammate 2), 5 seeds (22:03–22:11 IST)

Seeds split across 3 parallel processes, then merged into `training/results/i3_t2h_2150.csv` (85 rows). Baseline = **shipped v2** (`i3_core_r2_2025.csv`), paired per seed.

| Slice | `t2h_2150` | exits | Δ vs shipped v2 ± SE | seeds worse | beyond 2 SE? |
|---|---:|---:|---:|---:|---|
| overall | 40.53 | 33% | +5.68 ± 3.46 | 2/5 | noise |
| T0 incl two_rooms | 11.48 | 11% | +7.29 ± 4.59 | 1/5 | noise |
| MAP25-32 | -1.03 | 0% | +5.69 ± 3.12 | 1/5 | noise |
| T1 | 101.03 | 100% | -1.26 ± 0.38 | 5/5 | SIG WORSE |
| T2 | 69.78 | 60% | -28.81 ± 22.86 | 4/5 | noise |
| T3 | 105.66 | 70% | +41.27 ± 24.23 | 2/5 | noise |
| T4 | 16.43 | 0% | +4.26 ± 7.06 | 2/5 | noise |

**Reproducibility of the 46.62 claim.**
- On the same seeds 0–2, the i3 gets **40.38** vs teammate 2's 46.62 (`RESULTS.md`, `t2h_best_2150.csv`).
- My per-seed means are 33.5 / 49.1 / 38.6 / 40.3 / 41.2.
- A 6-point gap sits inside the ~9-point rerun noise of unseeded sampling (§2 finding). It's **not evidence of a
  reporting error**; it's why single 3-seed runs can't rank checkpoints.

**Verdict: no measurable difference from shipped v2 overall** (+5.7 ± 3.5, 1.6 SE). It's also level with
`core_r2_2155`: +1.0 ± 6.7.
- **Strongest real-map trend of any candidate:** MAP25–32 −1.0 vs −7.8 for v2, Δ +5.7 ± 3.1 (1.8 SE, better on 4/5
  seeds). Not significant yet.
- **T3: 70% exits** (+41 ± 24, not significant). **T2 trends lower** (−29 ± 23, not significant).
- **T1 is significantly lower (−1.26 ± 0.38, 5/5 seeds).** Both checkpoints exit 100% of T1 maps; the difference is
  only the HP/time bonus.
  - IMPACT on hidden score: **LOW**. About 1 point on the easiest tier, with the same exit rate.
- **Recommendation:** don't ship on this alone. It and `core_r2_2155` both trend above v2 without clearing 2 SE.
  - If the core owner wants to decide between the three, the cheapest decisive test is 5 more seeds (5–9) of all 17
    maps for `t2h_2150` and v2. That doubles n and halves the noise.
  - IMPACT of the decision: **MEDIUM**. The possible gain is about 5 points of mean score, mostly on T3/real maps.

### 4e. COORDINATION ship rule applied (22:18 IST)

The rule is from `CHECKPOINTS.md` (22:15). Computed with `training/results/t3_checks/ship_rule.py` from the 5-seed i3
CSVs. It asserts both CSVs have the same seed set.

```text
== i3_t2h_2150 vs shipped v2 (i3_core_r2_2025)
(1) MEAN 40.53 vs 34.85: PASS
(2) T2+T3+T4 exits 13 vs 11 (of 30 episodes): PASS
(3) MAP25-32 delta +5.69 +/- 3.12 (limit -2 SE = -6.23): PASS
SHIP RULE: PASS
== i3_core_r2_2155 vs shipped v2
(1) MEAN 39.55 vs 34.85: PASS
(2) T2+T3+T4 exits 14 vs 11 (of 30 episodes): PASS
(3) MAP25-32 delta -3.44 +/- 3.84 (limit -2 SE = -7.68): PASS
SHIP RULE: PASS
== head-to-head: i3_t2h_2150 vs i3_core_r2_2155
(1) MEAN 40.53 vs 39.55: PASS
(2) T2+T3+T4 exits 13 vs 14 (of 30 episodes): FAIL
(3) MAP25-32 delta +9.13 +/- 1.04 (limit -2 SE = -2.07): PASS
SHIP RULE: FAIL
```

**Reading:**
- **Both candidates pass the ship rule against shipped v2.** Rules (1) and (2) compare point estimates, and the mean
  gains (+5.7, +4.7) are within noise (§4c, §4d). By the rule, either may ship.
- **Between the two:**
  - `t2h_2150` is **significantly better on unseen real maps**: MAP25–32 **+9.1 ± 1.0**, better on 5/5 seeds.
  - It's level on mean score (+1.0 ± 6.7).
  - It has **one fewer T2–T4 exit** (13 vs 14 of 30). A one-exit difference is noise.
- **Recommendation:** if one ships now, ship **`t2h_2150_score46_62.pt`**. It passes the rule against v2, and it's the
  only candidate with a large, consistent edge somewhere (real maps).
  - IMPACT: **MEDIUM**. A likely gain of a few points of mean score, mainly on real-map-like and T3 maps.
- **Tooling:** `ship_rule.py` takes any two CSV stems, so the core owner can rerun it on future candidates.

### 4f. Ship confirmation: v3 (`t2h_2150`) vs v2, 10 seeds (22:40 IST)
**CONFIRMED: v3 is better. Keep v3.** Seeds 0–9, 17 maps, paired (`i3_t2h_2150*.csv` vs `i3_core_r2_2025*.csv`, `*_s59` = seeds 5–9):
- **overall** 42.30 vs 33.01: **+9.3 ± 2.5** (3.8 SE, worse on 2/10 seeds).
- **MAP25–32** −0.9 vs −8.7: **+7.8 ± 1.7** (worse on 1/10).
- **T3** 107.9 vs 51.1: **+56.8 ± 15.6**, exits **14/20 vs 4/20**.
- **T2–T4 exits:** 28 vs 18 (of 60).
- **No measurable difference:** T2 −10.4 ± 16.2, T4 −2.6 ± 4.8.
- **T1 −0.9 ± 0.4** (significant). IMPACT **LOW**: both exit 100% of T1 maps; the gap is only the HP/time bonus.

### 6. Action masking (`78b0544`): v3 with `--mask` vs v3, 5 seeds (23:22–23:28 IST)
**SHIP RULE: PASS. Recommend turning `action_mask = True` on.** Holdout, seeds 0–4, paired with `i3_t2h_2150.csv`; new
CSV `i3_v3_mask.csv`; checked with `ship_rule.py`.
- **overall** 48.39 vs 40.53: **+7.9 ± 3.3**, significant (2.4 SE), worse on 1/5 seeds.
- **T2–T4 exits:** 17 vs 13 (of 30). T3 exits 90%.
- **MAP25–32:** +3.8 ± 2.2 (no measurable difference, trending up). **T4:** +5.5 ± 3.5 (no measurable difference).
- **Caveats:**
  - This is the same 17 holdout maps used for every ship, but a fixed mask isn't tuned on them, so selection bias is
    smaller than for a checkpoint pick.
  - The final word is still the 03:00 `maps/fresh/` test.
  - The rocket-launcher HYPOTHESIS (the mask stops switching away from point-blank rockets) shows no harm here: T4 is
    up.
- **Also checked:** with the mask off, `validate.py` PASSED (p99 19.2 ms) and pytest **9 passed** (core's new mask test
  included).

### 6b. CORRECTION: action masking, pooled i3 + CUDA, 10 seed pairs (23:45 IST)
**CORRECTION to §6:** my "+7.9 ± 3.3, significant" was 5 seeds on one machine. Teammate 2's independent CUDA run
(`t2_v3_mask.csv` vs `cuda_v3_s04.csv`) gave −4.3 ± 3.7. Pooling both sets of within-device paired seed differences
(n = 10):
- **overall +1.8 ± 3.1: no measurable difference.**
- MAP25–32: +3.8 ± 3.1, the same sign on both devices (i3 +3.8, CUDA +3.7).
- T2: +0.8 ± 13.3. T3: +4.2 ± 13.2. T4: −4.9 ± 4.7 (i3 +5.5, CUDA −15.2). T4 exits are 0 for both.
- T2–T4 exits: 28 vs 27 (of 60).

**Verdict:** masking is **about neutral**. By point estimates it still passes the ship rule (+1.8 mean, 28 ≥ 27 exits,
MAP25–32 not worse), but there's no evidence it helps, and T4 trends worse, which fits the rocket-launcher HYPOTHESIS
(no switching away from point-blank rockets).
- IMPACT: **LOW–MEDIUM**.
- **Options for core:**
  - keep v4 (neutral, slight real-map edge); or
  - drop only the weapon-switch rule, keep the fist rule, and re-check.

### 7. Candidate `t2h_0037_score44_22` (i3, 5 seeds, vs v5): SHIP RULE PASS, modest gain
`i3_t2h_0037.csv` vs `i3_t2h_2150.csv`, seeds 0–4, paired:
- **Weighted exits** (1/2/3/4): **47 vs 43**. T2 8/10 vs 6/10; T1, T3 and T4 equal (10, 7, 0).
- **Mean:** 41.12 vs 40.53 (+0.6 ± 3.6, no measurable difference).
- **MAP25–32:** −4.3 ± 2.6, trending worse (1.7 SE; teammate 2's CUDA: −3.5).

**Both machines agree on more T2 exits** (i3 +4 weighted, CUDA 52 vs 45) and on a slight real-map dip. The ship rule
passes on both. **Verdict:** a real candidate for the 03:00 fresh-map pick. The gain is exits on T2, and the risk is
real maps. Let `maps/fresh/` (10 seeds) decide between it and v5.
