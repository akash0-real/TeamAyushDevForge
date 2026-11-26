# Checkpoint Leaderboard (core-owned; all times IST)

Which weights exist, how each scored on the holdout maps, and which one is shipped in `submission/weights/best_agent.pt`.
Updated by the core owner after every evaluation. Teammate 2's own runs go in `RESULTS.md`.

**Holdout** = maps never trained on: Freedoom MAP25–32 (T0 = real maps), `two_rooms`, 8 generated maps
(`maps/holdout/gen_t{1-4}_{1,2}.wad`). **Score** = proxy score from `training/eval_zeroshot.py` (higher is better).
Unless noted, scores use the current eval (`[t2]` version: generated maps at 2,100 tics) with seeds 0 1.

## Team status board (updated 19:55 IST)

**Roles from 20:15 IST:** Core = orchestrator (lessons/rewards, final pick, validation, shipping) + 2 training runs.
Teammate 2 = **primary trainer** (fastest machine, Linux, ~974 steps/s), hands candidates over via the `candidates`
branch (Task H). Teammate 3 = **continuous reviewer**, every 30 min + independent 5-seed checks of candidates (Tasks R, R2).

| Who | Doing now | Next |
|---|---|---|
| Core (i7 + RTX 4050) | **Two** memory-grid runs in parallel (run 1: 20 envs, run 2: 16 envs, different seeds). PPO update 3 s → 0.6 s; the single-threaded game loop is the remaining limit, so a 2nd process uses the idle CPU/GPU/RAM | Score new checkpoints every ~40 min; ship only if it beats the shipped one incl. real maps |
| Teammate 2 (3050, Linux) | Seed-7 plain run (scores 9.7) | **Task G now:** switch to the memory-grid model from the shipped weights with the fixes |
| Teammate 3 (i3) | Brief, README, docs, audit, demo script done | Task E (i3 validation + 5-seed score), then Task D2 (re-check audit) |

**Plan check:** on schedule. A trained, validated agent is on `main`; final pick ~03:00, `[core] FINAL` ~03:30,
Linux validation + brief PDF re-export by 04:00 IST.

**Fixed in `3cfa239` (from the audit + profiling):** torch loaded in every env worker (~5 GB wasted RAM), GRU fed one
timestep at a time in the PPO update (GPU idle), memory grid smearing on big maps, `act()` freezing on an exception.

## Currently shipped

**`map_phase1`**: EgoMap model, 645k total training steps (184k plain + 460k EgoMap). Shipped 18:03, commit `d37c9e4`.
`validate.py` PASSED (p50 4.3 ms/step). **5-seed score on the i3 (teammate 3): 31.2**, T0 6.9, T3 map 1 exits 60%
at 2,100 tics; i3 validation PASSED (p50 7.1 ms, p99 17.6 ms). **31.2 is the bar a new checkpoint must beat.**

## Leaderboard

| Checkpoint | Model | Total steps | Mean score | T0 real | T1 | T2 | T3 | T4 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| random | plain | 0 | −13.2* | | | | | | *9 real maps only, 1 seed (teammate 2) |
| base_190k | plain | 184k | 9.9† | | | | | | †old eval (gen maps 4,200 tics); 18% exits |
| **map_phase1 (shipped)** | EgoMap | 645k | **27.7** | **7.9** (11% exits) | 101 (100%) | 59 (50%) | 30 | 9.8 | old eval: 34.1, 29% exits |
| map_eval3 | EgoMap | 1.29M | 22.5 | −2.1 (6%) | 104 (100%) | 68 (50%) | 21 | 8.2 | tutor round 2 (T3 curriculum); worse on real maps → not shipped |
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
