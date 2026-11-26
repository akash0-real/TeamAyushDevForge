# Checkpoint Leaderboard (core-owned; all times IST)

Which weights exist, how each scored on the holdout maps, and which one is shipped in `submission/weights/best_agent.pt`.
Updated by the core owner after every evaluation. Teammate 2's own runs go in `RESULTS.md`.

**Holdout** = maps never trained on: Freedoom MAP25–32 and 8 generated maps (`maps/holdout/gen_t{1-4}_{1,2}.wad`).
**Correction (22:20, teammate 3's strict review):** the "T0" column also includes `two_rooms`, which is a *training*
map, and every T0 exit came from it. On MAP25–32 alone both shipped agents have 0% exits (v1 −4.6, v2 −7.8, 10 seeds).
From now on, real-map comparisons use **MAP25–32 only**. **Score** = proxy score from `training/eval_zeroshot.py` (higher is better).
Unless noted, scores use the current eval (`[t2]` version: generated maps at 2,100 tics) with seeds 0 1.

## COORDINATION (22:15 IST): everyone reads this every cycle

**Goal = the most competition points** (`docs/EVALUATION.md`). Reaching the exit is the largest award (plus a small
bonus for health and time left), harder tiers weigh more, and partial progress counts. So we pick checkpoints by
**exits on tiers 2–4 first**, then overall score.

**Ship rule (core applies it, from teammate 3's 5-seed paired check):**
1. paired 5-seed MEAN is higher than the shipped one;
2. T2 + T3 + T4 exits are not fewer than the shipped one's (count over the 5 seeds);
3. MAP25–32 mean is not worse by more than 2 SE.

**Final pick (03:00):** same rule, but on the untouched `maps/fresh/` set (see its README).
Then core runs `validate.py` on a clean clone, ships, tags it `shipped-vN`, and updates this board.

| Time (IST) | Teammate 2 (3050, trainer) | Teammate 3 (i3, reviewer) | Core (orchestrator) |
|---|---|---|---|
| now | Task F0 (docker validation), keep phase-2 training | confirm the ship: seeds 5–9, all 17 maps, for shipped `t2h_2150` **and** v2 (`core_r2_2025`) | shipped `t2h_2150` (22:20) |
| every ~45 min | push your best to `candidates` + a 1-line README note (score, steps) | R2 each new candidate within ~20 min; Task X every 30 min; Task R every 10 min | push core candidates; ship winners; reply to every QUESTION(core) |
| 02:15 | **push your final candidate** (deadline for the 02:30 final test) | R2 the last candidates | Vast final-pick test at 02:30 |
| 03:00 | — | check core's final-pick numbers from the CSVs | final pick |
| 03:30 | Task F (Linux 1-CPU validation of `[core] FINAL`) | last review of `[core] FINAL` | push `[core] FINAL` |
| 04:00 | done | BRIEF.pdf re-export | done |

**Talking to each other:** questions go in your own file as `QUESTION(core): …` (teammate 3: `AUDIT.md`; teammate 2:
`RESULTS.md`); core answers in your task file's top block within one cycle. Anything that could cost points
(validate fails, a crash, a rules risk) also goes to your human immediately.

## CORE UPDATES (core → everyone; newest first, one line each, per the 22:32 team-channel rule)
- **UPDATE 00:05: a rented 192-core machine (Vast.ai) is running 3 core runs until 03:30 (auto-destroy).** A = round 6 from v3 (56 envs),
  B = round 6 from core's `warp` run (48), **C = bigger model** (GRU 512, encoders copied from v3, round 6, 56). Bigger-model support is in
  `0438dfc` (`load_policy` reads the GRU size; validated). **New timeline:** every candidate on `candidates` by **02:15**; at **02:30** the
  Vast machine runs the final-pick test (`maps/fresh/`, 10 seeds, all candidates in parallel, plus v3 as the baseline); **03:00** pick; **03:30**
  `[core] FINAL`. ⚠ **The GitHub repo is PRIVATE**, and `docs/SUBMISSION.md` requires a **public** repo whose URL reaches the organizers
  before the deadline. **Akash (owner): please switch it to public** (Settings → General → Danger Zone → Change visibility) and confirm here.
- **UPDATE 23:50:** round 6, teammate 3's one-line fix applied (keep the last defined Phi through unreachable cells); `check_geodesic.py` OK. **Teammate 2:** pull this before you restart `t2w`.
- **UPDATE 23:43: CORRECTION, masks OFF again (tag `shipped-v5`, same behaviour as v3).** Teammate 3 pooled both machines (EVAL_I3 §6b):
  +1.8 ± 3.1, no measurable gain; T4 −4.9 ± 4.7 (CUDA −15). I shipped v4 on one 5-seed run that didn't replicate; that's on me. The rules
  are now selectable (`--mask fist`, `--mask weapon`). **Bar back to v3: 42.30 (10 seeds).** Round 6 fixed: no per-step cap, so teleport loops
  sum to 0 (teammate 3 RISK). **Teammate 2:** pull and restart `t2w` (round 6 changed), then Task M2. `validate.py` PASSED on a clean clone.
- **UPDATE 23:32:** **Shipped v4 = v3 weights + action masking ON** (tag `shipped-v4`). Teammate 3 EVAL_I3 §6: +7.9 ± 3.3 overall, T2–T4 exits 17 vs 13,
  MAP25–32 +3.8. `validate.py` PASSED on a clean clone (p50 7.0 ms, p99 12.4 ms). Rollback to no-mask: `shipped-v3`. **Bar: 48.39 (5 seeds, masked).**
  Candidates are now scored **with `--mask`**, since that's how they would ship. Teammate 2: Task M is done (teammate 3 ran it); skip it.
- **UPDATE 23:31:** **Round 6 is live** (`training/geodesic.py`: geodesic progress shaping toward the next missing key, then the exit; teammate 3's
  proposal item 1). Self-check `python training/check_geodesic.py` → `OK`. **Teammate 2: restart `t2w` on round 6 from v3 now** (Task H2 step 3).
  Action masks (item 3) are in `agent.py`, OFF by default, waiting on teammate 2's Task M score. Teammate 3: please review round 6 (`training/geodesic.py`,
  `rewards.py` §4b) on your next R cycle.
- **UPDATE 23:20, decision on teammate 3's PROPOSAL 23:20 (the human says adopt it; this lifts the 23:10 freeze for these items only):**
  - **(3) action masking: YES.** Core builds it in `agent.py` now; inference only, no retraining. It ships only if teammate 3's 5-seed
    paired check on the holdout passes the ship rule.
  - **(1) geodesic potential shaping: YES.** Core builds it in `training/rewards.py` as round 6, tested on real routes first. Teammate 2's
    fast machine trains it, from v3.
  - **(2) auxiliary heads: NO.** It would change `submission/model.py` and checkpoint loading this late.
  - Also live: **round 5c**, an early-death penalty scaled by time left (teammate 3 blocker #1), on core run 1 since 23:17.
- **UPDATE 23:10:** Freeze on new ideas (human's call): no tutor round 6. Teammate 2 now pre-screens candidates on CUDA (holdout only); teammate 3 confirms. A Vast.ai CPU box may join for training + the 03:00 final-pick evals (10 seeds on `maps/fresh/`); core will post when it's live.
- **UPDATE 22:50, sanity check of v3 on untouched maps** (`maps/fresh/`: 8 new-seed generated + 8 Freedoom-1 maps, 3 seeds,
  `training/results/untouched_v{2,3}.csv`). Overall v3 − v2 = **+0.9 ± 5.4**, so most of the +9.3 on our 17 holdout maps was
  selection bias. The pattern holds, though: T3 +36.8 ± 22 (exits 2 vs 0), T4 +10.4 (v3 picks up keys), **T2 −24.7 ± 23**,
  Freedoom 1 −3.7 ± 5.6. **Keep v3** (harder tiers weigh more). Actions:
  (1) **final pick at 03:00 uses `maps/fresh/` as the deciding test**; nobody evaluates on it before then;
  (2) T2 is v3's weak spot, so keep gen2 in every training mix.
  Round 5b pushed: warped starts now also land near keys (T6 in the log); core run 1 restarted with it from `warp_2250`.
- **UPDATE 22:36:** v3 ship **confirmed** by teammate 3 over 10 seeds (+9.3 ± 2.5 overall, MAP25–32 +7.8 ± 1.7, T3 exits
  14/20 vs 4/20). **New bar = 42.30 (v3, 10 seeds)**; candidates are compared to v3 on the same seeds 0–9 (or 0–4: 40.53). Core runs:
  `warp` T5:3%/38 near-exit starts finished; `map2` T3:35%/23.
- **UPDATE 22:31:** Shipped = v3 `t2h_2150` (tag `shipped-v3`; rollback `shipped-v2`/`v1`). Round 5 (reverse curriculum) runs
  on core run 1 (`warp`, from v3), T5:5%/39 of near-exit starts finished so far. Run 2 is round 4, unchanged. Renamed the
  self-check to `training/check_warp_curriculum.py` (teammate 3 NIT). Next: a `core_warp_*` candidate once T5 is steady. Waiting on
  teammate 3's seeds 5–9 confirmation and teammate 2's F0 (docker) / H2 (round-5 run).

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

**`t2h_2150`** (teammate 2's primary run, ~8.3M steps). Shipped 22:20, tag `shipped-v3`, temperature 1.0.
`validate.py` PASSED on a clean clone (init 10.9 s, p50 7.1 ms, p99 15.5 ms). **i3 5-seed (teammate 3, EVAL_I3 §4d):
40.53** vs v2 34.85 (+5.7 ± 3.5, 1.6 SE). Passes the points-first rule: T2–T4 exits 13 vs 11 (T3 7/10 vs 3/10,
T2 6/10 vs 8/10), MAP25–32 −1.0 vs −7.8. Not yet significant overall → teammate 3 confirms with seeds 5–9.
**Bar: 42.30 over 10 seeds (40.53 over seeds 0–4); confirmed +9.3 ± 2.5 vs v2 (EVAL_I3 §4f).** Rollback: `shipped-v2` (core_r2_2025, 34.85), `shipped-v1` (map_phase1, 31.2).

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
Training maps Freedoom MAP01–08: score 14.3, 0% exits. Unseen MAP25–32: score −4.6, 0% exits (corrected; the old
7.9 / 11% included `two_rooms`). A real gap, partly difficulty (early Freedoom maps are easier), and it doesn't finish the maps it trained on either, so it
didn't memorize routes.

## Rules for shipping a new checkpoint
1. It must beat the shipped mean score **and** not be worse on T0 (real maps): those look most like the hidden maps.
2. `validate.py` must pass on a clean clone before pushing.
3. Final pick at ~03:00 IST uses seeds 0 1 2; final push `[core] FINAL` at ~03:30 IST.
