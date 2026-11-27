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

**Final pick (03:00):** on the untouched `maps/fresh/` set (**20 seeds**, seeded eval), rank by **exit_points = 1·T1 + 2·T2 + 3·T3 + 4·T4 exits** (teammate 3's 00:55 ask), then by mean score; then the ship rule.
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
- **UPDATE 06:21 (core): NEW DEADLINE 08:00 (human, final). Tier-4 root cause found: weapon cycling.** v11 on T4 presses SELECT_NEXT_WEAPON on **49% of steps**
  and holds the **fist 45%** of the time (pistol → next wraps to fist; switching can't fire). New default-off mask rule `"gun"` in `agent.py`: no weapon switch while
  holding a loaded pistol or better (the network still chooses everything else). On T4 it cuts switches 834 → 96, fist steps 764 → 134, shotgun steps 51 → 391.
  20-seed holdout (cloud) running now. Combat training from v11 running: core laptop `COMBAT_W=3`, cloud `COMBAT_W=2` and `=3`.
  **New untouched exam `maps/exam3/`** (seeds 4e9 + 100·t + k, Freedoom 1 E1M3 E1M6 E2M3 E2M6 E3M3 E3M6 E4M3 E4M6): nobody evaluates on it before ~07:20.
  **Timeline:** train → 07:05; holdout screen (with and without `gun`) → 07:20; exam3 once → 07:35; ship → 07:40; FINAL validate → 07:50.
  **Teammate 2:** train `COMBAT_W=3` from v11 (see 06:10 note), push your best by **07:05**, seeded 5s vs v11 **with `--mask gun` and without**.
  **Teammate 3:** R2 the `gun` rule on the i3 (v11 with vs without, 5 seeds, per tier + T4 deaths), and review `agent.py` for rule-compliance (§1 masking).
- **URGENT 06:10 (core → t2, t3): tier 4 dies in combat.** v11 on T4: dies in 75-85% of runs (3 kills, ~110 damage taken) vs 0-20% on T1-T3.
  **Teammate 2: train NOW on your GPU from shipped v11** (`submission/weights/best_agent.pt`) with `COMBAT_W=3` (new env knob in `training/rewards.py`: kill + damage-dealt
  reward ×3), lr 1e-4, mix `two_rooms MAP01 03 05 07 09 13 17 21 gen1×8 gen2×6 gen3×6 gen4×16`. **Push a candidate by 06:20 sharp** with seeded 5s vs v11 (T4 deaths!).
  **Teammate 3:** list any other combat/survival mechanic we're missing (ammo, weapon pickups, health use, strafing) in AUDIT.md by 06:18.
  Ship only if it beats v11 on the holdout (exits per tier, T4 deaths) by 06:25; otherwise v11 stays.
- **[core] FINAL v11 (06:01).** Grand test on exam 1 (`maps/fresh`, 20 seeds): v11 vs v8 **+4.9 ± 1.5, unseen-real +5.9 ± 2.3, exit_points 216 vs 210** →
  v11 beats v8 on all three sets (holdout, exam 1, exam 2). Thanks t2 for the Docker PASS and t3 for the verify PASS. Temperature sweep on v11 (turn 0.4/0.65,
  strafe 0.35, move 0.7, use 0.7): none better → settings unchanged. `"use"` cooldown stays off (v8 −4.0 ± 1.8, v11 +0.1). BRIEF.md/.pdf refreshed (2 pages).
  Cloud box destroyed 05:59. Repo PUBLIC. **No more changes to `submission/` unless something is broken.**
- **SHIPPED v11 (05:46, tag `shipped-v11`, commit ccb86e7): `t2r7b_0447` (teammate 2, round 7), sha 8ab22c0c.** Exam 2 (untouched, 20 seeded seeds, used once) vs v8:
  **+3.8 ± 2.1, unseen-real +2.7 ± 1.0, exit_points 188 vs 184, T2-T4 exits 60 vs 59, T4 1 vs 0 (first T4 exit)**; T1 33 vs 29, T2 26 vs 22, T3 33 vs 37 (`ex2_*.csv`).
  `core_r7best` failed the exam (162 vs 184). Rollback: `shipped-v10` / `final-submission` (v8). Great work t2, and t3 for the pooled call.
  **Teammate 2: please run `python validate.py --docker` on a fresh pull of main now and post PASS/FAIL.** Core runs the fresh-clone validate too.
  Next (inference-only, default off): t3's `"use"` cooldown (no USE re-press for 9 steps; door reversal) is in `agent.py`, being tested on the 20-seed holdout.
- **UPDATE 05:43 (core): exam 2 is running** (`maps/exam2`, 20 seeded seeds, used once): v8 vs `t2r7b_0447` vs `core_r7best_0350`, as t3 recommended.
  Cloud holdout, 20 seeds vs v8: **`t2r7b_0447` +3.2 ± 1.8, unseen-real +4.8 ± 2.0, exit_points 212 vs 210, T2-T4 exits 69 vs 68 (PASS)**; `core_r7best` +3.1 ± 2.4,
  207 vs 210. My retrains from r7best (bal/hvy/laptop) all lost T2/T3 exits (190-204) → dropped. Ship decision at ~05:55; FINAL clone check ~06:10.
  Teammate 2: please stand by for one Docker/Linux validate run of the shipped weights after ~06:00.
- **UPDATE 05:12 (core): cloud holdout, 20 seeded seeds vs v8 (`training/results/h7_*.csv`).** `t2r7a_0420`: exit_points 93 vs 105 on 10 seeds, T3 13 vs 17 → **agrees with
  t3's FAIL, dropped.** Core round 7: `core_r7best_0350` (1.33M steps) **+3.1 ± 2.4, unseen-real +4.9 ± 2.0, T4 score +7.3 ± 3.5**, exit_points 207 vs 210; later snapshots
  drift worse (2.97M: −0.5, 205; 3.58M: −1.6, 190, T3 30 vs 34), so the T4-heavy mix erodes T2/T3 with time. All runs restarted from r7best at lr 1e-4 (balanced + heavy mix).
  **Teammate 3 (and/or 2):** please run the 2nd-machine check of `core_r7best_0350` (on `candidates`) vs v8, seeded 5+ seeds, exits per tier + MAP25-32, by 05:45.
  **Teammate 2:** stop the t2r7a line; if you have a spare GPU/CPU, train from `core_r7best_0350` at lr 1e-4 and push by 05:30.
- **UPDATE 05:02 (core): final stretch.** Thanks t2: `t2r7a_0420` (sha 0d9f061e) is in the pre-screen. A 88-core cloud box now runs a 10-seed seeded holdout of
  v8 / t2r7a_0420 / core r7 (laptop) / r7.best, and continues training both r7 lines until ~05:35. **Teammate 2:** push your best t2r7a/t2r7b checkpoint to `candidates`
  **by 05:30** (last call). **Teammate 3:** R2 of `t2r7a_0420` vs v8 on the i3 (5 seeds, per tier + MAP25-32) if your budget allows; do NOT touch `maps/exam2`.
  Plan: 05:35 holdout pick of the top 2 → 05:45 `maps/exam2` once (20 seeds, vs v8) → ship only if it beats v8 on both, with a rollback tag → 06:15 FINAL.
- **UPDATE 03:05: NEW DEADLINE 06:30 IST (the human wants us to aim higher).** v8 stays submitted unless something beats it on **both** the holdout and a **new
  untouched exam** (`maps/exam2/`: 8 new-seed generated maps + 8 *other* Freedoom 1 maps; nobody evaluates on it before ~05:45).
  **Target: T4** (0/40 exits, the heaviest tier), then T2. **Tutor round 7** (`8b0bab9`): generated T3/T4 maps also start near the exit or a key (T7/T8 in the log).
  - **Teammate 2:** pull, check `python training/check_warp_curriculum.py` → `OK round 7`, then train **from `t2h_0037`** (v8 weights) with a **T4-heavy mix**:
    `--maps two_rooms MAP01..MAP24 (no 11/12/15/23) gen1 ×2 gen2 ×10 gen3 ×12 gen4 ×20`. Push your best to `candidates` every 30 min, **last one by 05:15**, and
    pre-screen it seeded vs v8 (exits per tier).
  - **Teammate 3:** review round 7 when you're back; R2 at ~05:20 on the top candidates (seeded, vs v8).
  - **Timeline:** training until 05:15 → holdout pre-screen 05:15–05:35 → exam2 05:35–05:55 → `[core] FINAL` 06:15.
- **UPDATE 02:51:** the Vast machine is destroyed (about $1.53 total); final checkpoints and results were copied to the core laptop first. Core training is stopped. The submission stays at `[core] FINAL` (v8).
- **UPDATE 02:45: [core] FINAL = v8 (`shipped-v10`)**: weights `t2h_0037` (sha `2797f55f…`) + `turn_temperature 0.5` + strafe 0.5. **No further changes to `submission/`.**
  - **Last check of the Vast runs** (holdout, 10 seeded seeds vs v8): none clearly better. vH +1.1 ± 1.0 went to the exam and **failed** there
    (exit_points 179 vs 210, T3 27 vs 36).
  - **Fresh clone of the GitHub repo:** real LFS weights (3,051,736 bytes), v8 settings, `validate.py` **PASSED** (p50 6.8 ms), repo 5.0 MB.
  - **Also passed:** teammate 2's Linux CUDA + official Docker validation, 0 warnings; teammate 3's i3 validation; brief reviewed (2 pages).
  - **Teammate 2:** Task F is already covered by your 02:30 Docker/Linux pass on this exact state. No need to rerun unless you want to.
  - **Repo visibility:** must be **public** before the deadline (owner action).
- **UPDATE 02:27: final exam by tier** (`maps/fresh/`, 20 seeded seeds; exits / attempts, then mean score):
  | version | T1 | T2 | T3 | T4 | Freedoom-1 big real | exit_points | mean |
  |---|---|---|---|---|---|---|---|
  | **v8 (live)** | 40/40, 102 | **31/40, 95** | 36/40, 132 | 0/40, 15 | 0/160, 12 | **210** | 48.9 |
  | t2h_0122 | 40/40, 102 | 20/40, 71 | **39/40, 141** | 0/40, 12 | 0/160, 17 | 197 | 49.4 |
  | v7 | 40/40, 102 | 26/40, 80 | 33/40, 126 | 0/40, 9 | 0/160, 12 | 191 | 45.3 |
  | soupAFGH | 40/40, 102 | 26/40, 80 | 22/40, 99 | 0/40, 21 | 0/160, 16 | 158 | 45.4 |
  | v9 | 40/40, 102 | 25/40, 81 | 22/40, 99 | 0/40, 13 | 0/160, 13 | 156 | 43.5 |
  T4 and big real maps: 0 exits for every version (known limitation, goes in the brief).
- **UPDATE 02:25: FINAL EXAM result → rolled back to v8** (tag `shipped-v10` = v8 state: `t2h_0037` weights + turn 0.5 + strafe 0.5). `maps/fresh/`, 16 untouched maps × 20 seeded
  seeds (`training/results/fin_*.csv`):
  | | exit_points | mean | exits [T1,T2,T3,T4] | unseen-real vs v7 |
  |---|---|---|---|---|
  | **v8** | **210** | 48.9 | 40,31,36,0 | +0.3 |
  | t2h_0122 | 197 | **49.4** | 40,20,39,0 | **+5.9** |
  | v7 | 191 | 45.3 | 40,26,33,0 | — |
  | soupAFGH | 158 | 45.4 | 40,26,22,0 | +4.1 |
  | **v9** | **156** | 43.5 | 40,25,22,0 | +1.8 |
  **v9's holdout gain was selection fit** (the soup was picked on the 17 holdout maps; on untouched maps it loses 11 T3 exits vs v7). By the agreed rule (exit_points first),
  **v8 wins**. Teammate 3's i3 confirmation of v9 was on the holdout, so it doesn't override the exam. `validate.py` PASSED on a clean clone. Plan: **`[core] FINAL` = this state**,
  unless a new candidate beats v8 on **both** the holdout and the exam before 03:00.
- **UPDATE 02:18:** round-2 sweep vs v9 (20 seeds): soup ratios 70/30 and 30/70, turn 0.7, strafe 0.3 and the 5-way soup were all equal or worse, so **v9 stays**. **Final `maps/fresh/`
  test started early** (first use of that set): v9, v8, v7, soupAFGH, t2h_0122 (v9 settings) × 20 seeded seeds × 16 maps. Results around 02:40, then a decision and `[core] FINAL` by 03:30.
- **UPDATE 02:09: shipped v9 = `core_soupA_0200` weights (average of v6 + Vast run A) + v8 settings** (tag `shipped-v9`, sha `7b2f37c9…`; rollback `shipped-v8`).
  Seeded, **20 seeds** × 17 holdout vs v7: **+3.9 ± 1.0**, exit_points **215 vs 199**, unseen-real **+4.4 ± 1.6**; consistent on seeds 0–9 (+4.3) and 10–19 (+3.5). `validate.py` PASSED
  on a clean clone. **Teammate 3: your second-machine check now compares `core_soupA_0200` (= v9) vs v8.** If the i3 shows a loss, core rolls back to v8.
- **UPDATE 02:07: shipped v8 = v7 + strafe temperature 0.5** (tag `shipped-v8`; rollback `shipped-v7`). Held up on fresh seeds: pooled 20 seeds × 17 holdout,
  +2.0 ± 1.6, exit_points 210 vs 199, unseen-real +2.0. `validate.py` PASSED on a clean clone. **Strongest result so far: `core_soupA_0200.pt` on `candidates`**
  (weight average v6 + Vast run A) with v8's settings. On seeds 10–19 (not used to pick it): **+3.5 ± 1.4 (significant)**, exit_points 106 vs 101, unseen-real **+5.9 ± 2.5**.
  **Teammate 3 (and teammate 2 if free): second-machine check of `core_soupA_0200.pt` vs v8**, seeded, 5 seeds (pull first: v8's head temperatures are the
  default now, so just pass `--weights`). It ships as v9 if it passes on your machine too.
- **UPDATE 02:01:** **v7 confirmed on both machines** (teammate 3's i3: +6.3 ± 2.9, T3 exits 9/10 vs 5/10). **Knob sweep vs v7** (seeded, 10 seeds × 17 holdout,
  `training/sweep_check.py`, CSVs `sw_*`): strafe T 0.5 **+3.5 ± 2.3 / 105 exit_points** (v7 98); soup(v6+vA) +2.6 / 106; soup(v6+vA+vD) +2.8 / 104; move, attack, use,
  weapon and fist about neutral or worse. **Teammate 2's `t2h_0122` vs v7: −1.8 ± 2.0 / 89**, so it isn't better than v7 (its +8 was against v6 at turn T 1.0).
  Not shipped. Teammate 2: compare future candidates against **v7** (pull: `turn_temperature 0.5` is the default now). Winner's-curse guard: the top 3 + soupA+strafe
  are being re-tested on **new seeds 10–19** now; anything that holds up ships before 03:00.
- **UPDATE 01:51: shipped v7 = v6 weights + `turn_temperature 0.5`** (tag `shipped-v7`; rollback `shipped-v6`). The human's call: ship small gains, keep rollbacks.
  Evidence (seeded, 10 seeds × 17 holdout): exit_points 98 vs 89 (+0.9 ± 1.0 per seed, better on 6/10 seeds), T3 exits 16 vs 13, mean +0.7 ± 3.3, two_rooms identical.
  `validate.py` PASSED on a clean clone. **Teammate 3:** your i3 check of T 0.5 still counts. If it shows a loss, core rolls back to v6.
- **UPDATE 01:39:** jitter check (the human asked why the agent twitches). New `agent.turn_temperature` (turn head only; default 1.0 = unchanged). Seeded
  test of v6, 10 seeds × 17 holdout maps: turn T 0.7 **+1.5 ± 3.0**, T 0.5 **+0.7 ± 3.3** (exit_points 95 / 98 vs 89; T 0.5 unseen-real −2.2). **No measurable
  gain, so not enabled.** Both go into the 03:00 `maps/fresh/` test as `v6_tt07` / `v6_tt05` and ship only if they clearly win there. CSVs: `training/results/tt*_v6_*.csv`.
- **UPDATE 01:25:** teammate 3's i3 check of `core_vA_0035` = **FAIL** (32.8 vs v6 41.1, exit_points 37 vs 47). That contradicts core's Vast result for the
  same file (46.1). **Root cause found:** `eval_zeroshot.py` seeded the map but not the agent's action sampling, so one file on one seed played differently
  every run. **Fixed** (`torch.manual_seed(seed)` per episode; verified that two reruns are identical). **Everyone: pull before your next eval**; old unseeded
  CSVs are noisier than they looked. vA is not shipped. **Final test raised to 20 seeds** per candidate on `maps/fresh/` (Vast makes it cheap).
- **UPDATE 01:12: shipped v6 = `t2h_0037`** (tag `shipped-v6`, sha `2797f55f…`), per the 01:15 message. The ship rule passes on **both** machines:
  weighted exits i3 47 vs 43, CUDA 52 vs 45; mean +0.6 ± 3.6; MAP25–32 −4.3 ± 2.6 (i3) / −3.5 (CUDA), so a **real-map dip on both machines**. Watch it.
  `validate.py` PASSED on a clean clone (p50 5.6 ms). Rollback: `shipped-v5`. Timeline per the 01:15 message: ship any rule-passing gain right away until 03:00;
  03:00–03:15 `maps/fresh/` check of the top 2 (v6 vs the best of core vA/F/G/H and teammate 2's latest) by exit_points; 03:30 `[core] FINAL`.
- **UPDATE 01:08: direction + overfitting check** (`training/direction_check.py`; raw CSVs `training/results/chk_*`, `trn_*`; Vast snapshots
  at 00:44, 17 holdout maps × 5 seeds, paired with v3). Exits by tier [T1,T2,T3,T4] and exit_points = 1·T1 + 2·T2 + 3·T3 + 4·T4 (teammate 3's 00:55 ask):
  | run | Δ vs v3 | exits | exit_points | train−unseen gap |
  |---|---|---|---|---|
  | v3 | — | 10,7,7,0 | 45 | 10.1 |
  | **vA** (round 6 from v3) | **+3.3 ± 2.6**, MAP25–32 +1.8 | 10,9,7,0 | **49** | **6.0** |
  | vD (same, other seed) | −0.2 ± 3.6 | 10,8,7,0 | 47 | 10.9 |
  | vB (from core warp run) | −4.2 ± 2.7 | 10,7,5,0 | 39 | 12.7 |
  | vC / vE (GRU 512) | **−15.0 / −16.4 (significantly worse)** | 10,6,0,0 / 10,1,4,0 | 22 / 24 | — |
  **Read:** round 6 is the right direction but not yet proven (vA +3.3, within noise), and there's **no overfitting** (vA's train/unseen gap shrank
  from 10 to 6). The bigger model can't catch up by 02:15, so it's **dropped**. T4 is still 0 exits everywhere. Actions: stopped B/C/E; started F/G/H
  as branches of vA. **`core_vA_0035.pt` is on `candidates`. Teammate 2: please run an independent CUDA 5-seed check of it (paired with your v3 CSV),
  with exits per tier.** The final pick ranks by exit_points on `maps/fresh/`, then mean.
- **UPDATE 00:19:** thanks, teammate 2: the warp audit showed the spots aren't dead ends. v3 succeeds from 25/64 spots within 250 units
  but 8/843 beyond, and the 874 failures are mostly deaths. So the curriculum radius now starts at **256** (floor **192**; was 768/512), `04e63fb`.
  The Vast runs and core's laptop run were restarted from their own checkpoints. **Teammate 2: pull `04e63fb` and restart `t2w`/`t2h` from their latest checkpoints.**
- **UPDATE 00:14:** teammate 3 is offline from 01:15 (token budget). Plan change:
  - **teammate 2 pre-screens** the core Vast candidates (`core_vA`…`core_vE`) on CUDA, 5 seeds holdout, `--mask` off;
  - the **02:30 final test on Vast** (`maps/fresh/`, 10 seeds) decides;
  - **core** checks its own final numbers by script and posts the CSVs, then re-exports `BRIEF.pdf` after `[core] FINAL` with
    teammate 3's `export_brief.py` (reviewed: local-only, asserts ≤ 2 pages).
- **UPDATE 00:09:** repo visibility: the human confirms the team will make it **public before the deadline** (owner action; tracked, not blocking). Vast runs A–E training; candidates from them go to `candidates` around 01:30.
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
