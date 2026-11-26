# Checkpoint Leaderboard (core-owned; all times IST)

Which weights exist, how each scored on the holdout maps, and which one is shipped in `submission/weights/best_agent.pt`.
Updated by the core owner after every evaluation. Teammate 2's own runs go in `RESULTS.md`.

**Holdout** = maps never trained on: Freedoom MAP25–32 (T0 = real maps), `two_rooms`, 8 generated maps
(`maps/holdout/gen_t{1-4}_{1,2}.wad`). **Score** = proxy score from `training/eval_zeroshot.py` (higher is better).
Unless noted, scores use the current eval (`[t2]` version: generated maps at 2,100 tics) with seeds 0 1.

## Team status board (updated 21:55 IST)

**Roles from 20:15 IST:** Core = orchestrator (lessons/rewards, final pick, validation, shipping) + 2 training runs.
Teammate 2 = **primary trainer** (fastest machine, Linux, ~974 steps/s), hands candidates over via the `candidates`
branch (Task H). Teammate 3 = **continuous reviewer**, every 30 min + independent 5-seed checks of candidates (Tasks R, R2).

| Who | Doing now | Next |
|---|---|---|
| Core (i7 + RTX 4050) | Shipped `core_r2_2025` (21:55). Run 2 (round 4) improving: T3 exits 35–46% in training; run 1 has 0% T3 and real maps falling | Push run 2's next checkpoint to `candidates` for teammate 3 |
| Teammate 2 (3050, Linux) | **No push since 20:17.** Expected: Task H (round-4 training, hand checkpoints over on `candidates`) | Push a candidate or a status note |
| Teammate 3 (i3) | Candidate checks + temperature test done (EVAL_I3 §4–5c, 4b) | Keep scoring new candidates (R2), 10-min reviews (R) |

**Plan check:** on schedule. A trained, validated agent is on `main`; final pick ~03:00, `[core] FINAL` ~03:30,
Linux validation + brief PDF re-export by 04:00 IST.

**Fixed in `3cfa239` (from the audit + profiling):** torch loaded in every env worker (~5 GB wasted RAM), GRU fed one
timestep at a time in the PPO update (GPU idle), memory grid smearing on big maps, `act()` freezing on an exception.

## Currently shipped

**`core_r2_2025`** (core run 2, round 3 lessons). Shipped 21:55, temperature stays 1.0. `validate.py` PASSED on a clean
clone (p50 8.1 ms, p99 21.6 ms while training ran). **5-seed score on the i3 (teammate 3): 34.85**, +3.7 ± 1.1 vs the
old weights (better on 5/5 seeds); T0 4.3 vs 6.2 over 10 seeds (−1.9 ± 2.0, within noise; same exit rate).
**34.85 is the bar a new checkpoint must beat.** Previous: `map_phase1` (31.2), commit `d37c9e4`.

**Temperature (teammate 3, Task T):** T = 0.5 no real gain; T = 0.7 +5.3 on real maps but 0% tier-3 exits → keep 1.0.

## Leaderboard

| Checkpoint | Model | Total steps | Mean score | T0 real | T1 | T2 | T3 | T4 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| random | plain | 0 | −13.2* | | | | | | *9 real maps only, 1 seed (teammate 2) |
| base_190k | plain | 184k | 9.9† | | | | | | †old eval (gen maps 4,200 tics); 18% exits |
| map_phase1 (shipped until 21:55) | EgoMap | 645k | 27.7 / 31.2 (i3, 5 seeds) | **7.9** (11% exits) | 101 (100%) | 59 (50%) | 30 | 9.8 | old eval: 34.1, 29% exits |
| map_eval3 | EgoMap | 1.29M | 22.5 | −2.1 (6%) | 104 (100%) | 68 (50%) | 21 | 8.2 | tutor round 2 (T3 curriculum); worse on real maps → not shipped |
| **core_r2_2025 (shipped)** | EgoMap | ~2.0M | **34.85** (i3, 5 seeds) | 4.2 (9%) | 102 (100%) | 99 (80%) | 64 (30%) | 12 | +3.7 ± 1.1 vs map_phase1 |
| core_r1_2025 | EgoMap | ~2.0M | 31.3 (i3, 5 seeds) | −0.3 (9%) | 103 (100%) | 92 (70%) | 62 (30%) | 9.8 | worse on real maps on 5/5 seeds → not shipped |
| map_1970k | EgoMap | 1.97M | 27.0 | 2.2 (11%) | 102 (100%) | 97 (75%) | 29 | −8.9 | big T2 gain, but real maps down again → not shipped |

## Training-log snapshot (19:15, round-2 run, last ~200 episodes)
Tier 1 exits 95%, Tier 2 58%, Tier 3 0%, Tier 4 0%, real Freedoom 0%; ~258 env steps/s.

## Tutor round 3 (19:25)
Two checkpoints in a row improved on generated maps while real Freedoom maps got worse (7.9 → −2.1 → 2.2): the
agent was specializing in our generator's style. Changes: real Freedoom maps doubled to ~52% of map loads,
generated tier 1 cut (6 → 3 slots), learning rate halved to 1.5e-4 (each restart had reset it to 3e-4).
Resumed from the 1.97M-step weights (best tier 2).

## Overfitting check (shipped agent)
Training maps Freedoom MAP01–08: score 14.3, 0% exits. Unseen MAP25–32: score 7.9, 11% exits. A modest gap,
mostly difficulty (early Freedoom maps are easier), and it doesn't finish the maps it trained on either, so it
didn't memorize routes.

## Rules for shipping a new checkpoint
1. It must beat the shipped mean score **and** not be worse on T0 (real maps): those look most like the hidden maps.
2. `validate.py` must pass on a clean clone before pushing.
3. Final pick at ~03:00 IST uses seeds 0 1 2; final push `[core] FINAL` at ~03:30 IST.
