# Tasks: Teammate 3 (no GPU) — Project Brief, Judge-Facing README, Docs That Match the Code

> **To the AI assistant reading this:** follow this file exactly. You are writing documentation only.
> Do not edit, refactor, or "improve" any code. Use **only the facts in section 4**, the repo files named here,
> and `git log`. **Never invent numbers, features, or results.** If a fact you need is missing, write
> `TODO(core): <what is missing>` in the text and tell your human; don't guess.

> **Every cycle, first read the `COORDINATION` block at the top of `doom-rl-docs/CHECKPOINTS.md`**: shared timeline, ship rule, how to ask questions.

## CURRENT ORDERS (22:25 IST) — read this block first, every cycle. It replaces the 22:10 block.

Task S was excellent: it found the `two_rooms` leak that nobody else saw. **Answer to your QUESTION(core): yes.**
Correct `BRIEF.md`, `04_PROCEDURAL_MAPS.md` and `DEMO_SCRIPT.md`:
- report unseen real maps as **MAP25–32 only** (0% exits);
- call `two_rooms` a training map;
- commit `[t3] docs: two_rooms is a training map`.

The core owner has already fixed `REVIEW_CHEATSHEET.md` and `CHECKPOINTS.md`.

From now on, in every R2 table, also give **MAP25–32 separately** (mean and exits) and use that, not T0, for the
"not worse on real maps" check.

### Task Y — Research: what stops the agent from reaching the exit on real maps? (NEW 22:45, priority after R2)
The human wants us to **finish real Freedoom maps**, not explain why we can't. Core just added a reverse curriculum
(tutor round 5, `training/rewards.py`, `exit_spots()`): half of the real-map training episodes start near the exit and
move further back as the agent succeeds. Your job is to find what *else* blocks exits, with evidence.
1. **Map facts**, by script from the WAD, for MAP01–29, 31, 32. MAP30 is a boss map with no exit line.
   - Exit type: switch (11/51), walk-over (52/124), or shoot (197/198). Use `exit_spots()` and the special numbers in
     `EXIT_SPECIALS`.
   - Straight-line distance from the player start (thing type 1) to the nearest exit.
   - Number of keys (thing types 5, 6, 13, 38, 39, 40) and of locked doors (`load_mapinfo(...).lock`).
2. **Where the shipped agent fails.** Write a script under `training/results/t3_checks/` that runs `submission/agent.py`
   (`DoomAgent`, shipped weights) through `doom_env.make_env` at 4,200 tics. Per episode, log from
   `env.privileged_state()`: the closest approach to the exit; the tic of the last new 64-unit cell; and death vs
   timeout. Run MAP25–29, 31, 32 plus MAP02, 05, 08 (training maps), 2 seeds each.
3. **Rank the blockers by impact**, using the numbers from 1 and 2. Examples:
   - dies on the way;
   - stops exploring (no new cell for over 1,000 tics);
   - the exit needs a key;
   - the exit is a switch it walks past;
   - it gets close (under 256 units) but never presses USE.

   For each blocker: `CONFIRMED`/`SUSPECTED`, an `IMPACT` line, and **what training change would fix it** (map mix,
   reward, curriculum).

Write `### Y HH:MM — exit blockers` in `AUDIT.md`, commit `[t3] research: exit blockers`, and tell your human the top 3.
Later, core's reverse-curriculum checkpoints (`core_warp_*.pt` on `candidates`) get your normal R2, with MAP25–32
mean **and closest approach** as the headline numbers.

### Your loop until 03:30 IST, in priority order
1. **Task R2**, immediately, for each new file on `candidates`. Right now that's `t2h_2150_score46_62.pt` (teammate 2
   claims 46.62) and `core_r2_2155.pt`.
   - 5 seeds each, paired against the shipped `core_r2_2025` (`i3_core_r2_2025.csv`).
   - Report Δ ± SE overall, per tier, and for MAP25–32.
2. **Task R** reviews, every 10 min, on new commits.
3. **Task X — explore failure outcomes**, every 30 min, one topic per cycle (list below). Run it whenever the CPU has
   room (≤ 3 evals at once, keep ≥ 1.5 GB RAM free).

You may now also write new small check scripts, but **only under `training/results/t3_checks/`**. Never touch the
files in hard rule 1.

### Task X topics: "what could go wrong on the hidden maps that we haven't seen?"
Pick the next unchecked topic each cycle. Write `### X HH:MM — <topic>` in `AUDIT.md`.
1. **Official image.** Python 3.13 in the scorer: does `submission/` import and run with the torch and numpy versions
   in `uv.lock`? Is anything Windows-only or CUDA-only?
2. **1 CPU thread.** Run `validate.py` with the process pinned to one core (PowerShell
   `(Get-Process -Id $pid).ProcessorAffinity = 1` in that window before launching).
   - Check that the per-map budget holds on a 4,200-tick map, not only on `two_rooms`.
3. **Long and huge maps.** Eval on MAP11/12/15/23 (excluded from training). Check:
   - Does the EgoMap scroll correctly?
   - Is any step slow?
   - Does the agent get stuck, i.e. no new cells for over 500 steps?
4. **Map features we never generate.** Find which of these appear in MAP25–32 and how the agent behaves there:
   - skull keys (vs keycards);
   - lifts and switches;
   - exits behind a switch;
   - teleporters;
   - dark sectors;
   - the agent starting while facing a wall.
5. **Edge cases in `act()`.**
   - Observations with 0 entities, or with all 32 entity slots filled.
   - Vitals at extreme values.
   - `reset()` called twice in a row.
   - `act()` called after the episode ends.
   - These are synthetic-obs tests only; don't change the code.
6. **Seed and device stability.** Does the same checkpoint score differently on CPU vs CUDA, or between runs with the
   same seed?
7. **Anything else you think of.** Add it to this list in your AUDIT block first, then check it.

### Accuracy rules: being wrong is worse than saying nothing
- **CONFIRMED vs SUSPECTED.**
  - A finding is `CONFIRMED` only if you reproduced it, with the command and its output quoted.
  - Reasoning without a reproduction is `SUSPECTED`.
  - Only `CONFIRMED` findings may be labelled `BLOCKER` or `RISK`. `SUSPECTED` ones are `HYPOTHESIS`, with how to test
    them.
- **Re-read the exact line at the current `HEAD` before citing it** (`git show HEAD:<file> | sed -n 'a,bp'`). Line
  numbers move.
- **Numbers.**
  - Compute them with a script from the CSVs, never by hand.
  - Always give n (seeds) and ± SE.
  - Never call a difference within 2 SE "better" or "worse"; call it "no measurable difference".
- **Corrections.** If an earlier claim of yours turns out wrong, put a `CORRECTION:` line at the top of your next AUDIT
  block (as you did at 20:30).
- **Before you tell your human "BLOCKER", run the check a second time.**
- **Report only.** Never edit code, never roll back, and never ship. The core owner does that.
- **Every finding gets an impact line:** `IMPACT on hidden score: HIGH / MEDIUM / LOW`, with one sentence explaining
  why. Use `docs/EVALUATION.md` as the reference:
  - the hidden maps are purpose-built tier maps (its example cap is 2,100 tics), not whole Freedoom campaign levels;
  - partial progress counts (exploration, doors, keys, kills);
  - harder tiers weigh more.
  - Example: "0/80 exits on MAP25–32" is true, but those are full campaign levels that can't realistically be finished in
    4,200 tics. Report that MEDIUM, with the MAP25–32 *mean score* (partial points) as the number that matters.
  - Tell your human only about HIGH items right away; the rest go in `AUDIT.md`.

---

## 0. Hard rules

1. **Never edit**: anything in `submission/`, `training/`, `maps/`, `env/`, `tests/`, `docs/` (official), `validate.py`,
   `Dockerfile`, `docker/`, `pyproject.toml`, `uv.lock`, `.gitattributes`, `.gitignore`, `.github/`.
2. Never commit `.pt`/weights files. Never `git push --force`. Never rewrite history.
3. Don't claim anything the code doesn't do. Planned-but-not-built items must be labelled **"not implemented"**.
4. Don't copy text from other teams' repos or papers. Ideas can be cited by name (e.g. "PPO (Schulman et al., 2017)").

## Files you own (the ONLY files you may create or edit)

- `doom-rl-docs/BRIEF.md` (new) and `doom-rl-docs/BRIEF.pdf` (new)
- `README.md`: **only** add a new section at the very top, `## Team Ayush — Our Agent`. Keep the template text below it unchanged.
- `doom-rl-docs/01_PROBLEM_AND_CONTRACT.md` … `05_EXECUTION_AND_HARDWARE.md` and `AGENTS.md` (Task C only)

The other teammates own everything else: the core owner owns code + training (`[core]` commits) and teammate 2
owns eval/results (`[t2]` commits, `doom-rl-docs/RESULTS.md`). Read their files; don't edit them.

---

## 1. Setup (Windows, i3 + 8 GB RAM is fine; no GPU needed)

1. Install **Git** (https://git-scm.com/download/win). Then:
   ```powershell
   git clone https://github.com/akash0-real/TeamAyushDevForge.git
   cd TeamAyushDevForge
   ```
2. You don't need Python, ViZDoom or PyTorch for these tasks. Don't install or run training.
3. Read, in this order: `docs/RULES.md`, `docs/EVALUATION.md`, `training/README.md`, `doom-rl-docs/RESULTS.md`,
   `submission/model.py` (skim the class names), `maps/gen_maps.py` (top docstring), then `git log --oneline`
   (each `[core]` commit message explains what changed and why).

---

## 2. Task A — The 2-page Project Brief (`doom-rl-docs/BRIEF.md` → `BRIEF.pdf`) — do this first

Hard limit: **2 pages** as PDF (A4, normal font size ~10–11 pt). Structure:

**Page 1 — What we built**
1. Title + team name + one-sentence pitch: *A recurrent PPO agent with a self-built spatial memory map, trained on
   an infinite stream of procedurally generated DOOM levels plus real Freedoom maps.*
2. **USP (3 bullets):** EgoMap spatial memory built only from the agent's own sensors and actions (heading tracked
   with 0.0° error); infinite procedural map generator with provably solvable key/lock levels; "tutor" training loop
   (we read training logs, diagnose the failing skill, adjust rewards/curriculum, resume from the same weights).
3. **Architecture diagram** (a simple box diagram: rays / entities / vitals / EgoMap → encoders → GRU → 6 action
   heads + value). Draw it in Markdown/ASCII or as an image you make yourself (e.g. draw.io, export PNG into
   `doom-rl-docs/`). Use the exact sizes in section 4.
4. **How Tier 3 (key → locked door → exit) is solved without global coordinates:** EgoMap remembers where the
   locked door was, the GRU carries "I have the key now", and the agent backtracks. Use the facts in section 4.

**Page 2 — How we trained it and how well it works**
1. Training pipeline: custom recurrent PPO (not Stable-Baselines3), reward shaping table (short), map mix, curriculum.
2. Procedural map factory: tiers 1–4, solvability proof, reserved holdout seeds.
3. **Results table**. Put it between the markers below so the core owner can refresh the numbers in the final hour:
   ```
   <!-- RESULTS:START -->
   ...table built from section 4.5...
   <!-- RESULTS:END -->
   ```
4. Latency / memory footprint (section 4.6).
5. Honest failure analysis (section 4.7) + what we'd do with more time.

Export to PDF: open `BRIEF.md` in VS Code with a Markdown-to-PDF extension, or paste the rendered Markdown into
Word/Google Docs → **File → Download → PDF**. Check it's ≤ 2 pages. Commit both files.
Commit message: `[t3] project brief v1`.

---

## 3. Task B — Judge-facing README section (~30 min)

Add `## Team Ayush — Our Agent` at the very top of `README.md` (above the template text), max ~40 lines:
- 3-sentence summary, the USP bullets, a link to `doom-rl-docs/BRIEF.pdf`.
- "Where things are": `submission/` (agent + model + EgoMap), `training/` (PPO, rewards, eval), `maps/gen_maps.py`
  (generator), `tests/test_egomap.py`, `doom-rl-docs/` (design docs, results).
- How to reproduce: copy the "Setup" and "Run" commands from `training/README.md` (don't invent new ones).
- Rule compliance line: decisions come only from trained network weights; privileged engine state is used only in
  `training/` for rewards; no pretrained/downloaded weights.

Commit: `[t3] README: team section for judges`.

## 4. Task C — Make the design docs match what was built (~45 min)

`doom-rl-docs/01`–`05` and `AGENTS.md` were written as a *plan*; some of it was cut or changed. Judges will compare
docs and code. For each item below, edit the docs so they describe the **built** system, and move cut items into a
short "Not implemented (future work)" list at the end of the relevant file. Don't delete the reasoning.

| Planned in docs | What was actually built |
|---|---|
| GRU hidden 384, input 384 | GRU hidden **256**; input 320 (plain) / 384 (with EgoMap) |
| Dual value heads (extrinsic + RND intrinsic) | **Single** value head; **RND not implemented** |
| Behavior cloning warm-start from human demos | **Not implemented** (cut: too little demo data for 10 h) |
| Prioritized Level Replay (PLR) | **Not implemented**; fixed weighted map mix + manual curriculum ("tutor rounds") instead |
| Auxiliary heads | **Not implemented** |
| Obsidian-generated tier-4 maps | **Not implemented**; our own generator makes tiers 1–4 |
| Truncated BPTT `seq_len 64` | BPTT over each full 128-step rollout, hidden state reset at episode boundaries |
| EgoMap 21×21 crop | Two **15×15** heading-up crops (fine 1 cell/px + coarse 8 cells/px) of a 128×128×6 grid |
| UDMF generator tiers 1–3 + Obsidian tier 4 | `maps/gen_maps.py`, tiers 1–4, node-less UDMF, self-test |
| 12-map holdout of our own maps | Holdout = Freedoom MAP25–32 + `two_rooms` + 8 generated maps (reserved seeds ≥ 1e9) |

Commit: `[t3] docs: match the built system`.

---

## 4. Facts you may use (verified by the core owner; numbers as of 18:55 IST)

### 4.1 Observation → network (`submission/model.py`, `submission/preprocess.py`)
- **Rays:** 64 rays over 360° (distance + 6 hit types) → circular 1D CNN (3 conv layers 7→32→64→64 channels,
  stride 2) → 128-d.
- **Entities:** up to 32 visible entities, 5 features + 16-d learned type embedding → MLP → 4-head self-attention →
  2 learned-query attention pooling → 128-d. Safe when 0 entities are visible.
- **Vitals:** 17 normalized vitals + previous action (9) + 4 front-cone summary features = 30 → MLP → 64-d.
- **EgoMap (optional input, used by the shipped agent):** 128×128 grid, 6 channels (visited, free space seen,
  wall, door, locked door, key), 64 map units per cell (8192×8192 units around spawn). Built **only from the
  observation + the agent's own previous actions**, so no engine coordinates. Network sees two 15×15 heading-up crops
  (fine + 8× coarse) → small CNN → 64-d.
- **Pose tracking (dead reckoning):** heading from our own turn presses, using Doom's turning rules measured per tic
  (slow-turn ramp; turn input ignored during the first 7 tics of an episode). **Heading error vs. the engine: 0.0°
  over 600 steps** on three maps. Position from the observed player-frame velocity: **47–160 map units drift
  after 4,500–7,300 units travelled** (1–2 grid cells).
- **Core:** GRU (hidden 256) → LayerNorm → actor + critic. Actor = **6 factored categorical heads** (move 3,
  strafe 3, turn 3, attack 2, use 2, next-weapon 2 = 15 logits) mapped to the 9-button `MultiBinary(9)` action,
  so contradictory buttons (forward+back) are impossible.
- **Parameters:** 596,624 (plain) / **759,056 (with EgoMap, shipped)**.
- The EgoMap is an *input* to the network; it never picks actions (allowed by `docs/RULES.md`: "learned memory
  modules", "preprocessing and encoding observations").

### 4.2 Training (`training/train_ppo.py`, `training/rewards.py`)
- Custom recurrent PPO in PyTorch: 12 parallel envs, 128-step rollouts, 4 epochs × 4 per-env minibatches,
  γ 0.995, GAE λ 0.95, clip 0.2, entropy 0.015, value coef 0.5, Adam lr 3e-4 → 5e-5 linear, grad-norm clip 0.5.
- Rewards (training only, from the engine's privileged state; the agent never sees them): exit +25 (+2 × health
  fraction), locked door opened +4, door opened +0.8, key +8, first key sighting +0.5, new 64-unit cell +0.04
  (cap 600/episode), kill +1, damage dealt +0.01/hp, item +0.15, health gained while hurt +0.01/hp,
  damage taken −0.015/hp, death −3. No per-step penalty (avoids learning to suicide).
- Episode length random 2,100–4,200 tics (like evaluation).
- Hardware: RTX 4050 laptop (i7-13620H) for the main run; RTX 3050 laptop for eval + validation.
- Windows speed fix: several ViZDoom games stall each other on Windows' 15.6 ms timer; raising it to 1 ms gave
  **7–10× more env steps/s** (4 games: 51 → 380 steps/s; 12 envs ≈ 1,400 steps/s on small maps).

### 4.3 Maps (`maps/gen_maps.py`)
- **Infinite procedural generator:** every map load creates a brand-new map. Rooms on a grid joined by
  2-cell-wide corridors, doors, enemies, pickups. Tier 1 combat (1–2 rooms); tier 2 exploration (4–7 rooms,
  loops); tier 3 the exit room is behind a locked keycard door, the key in another branch; tier 4 two locks
  (key 1 itself behind lock 2), 7–11 rooms.
- **Solvability proof in the self-test:** every key is reachable, and the exit is reachable only after unlocking.
  Generated maps load in the official `doom_env` with exactly the doors and locks the generator placed.
- Training mix per map load: 25 real slots (Freedoom 2 MAP01–24 + `two_rooms`) vs 48 generated slots
  (tier 1: 6, tier 2: 12, tier 3: 18, tier 4: 12).
- Holdout (never trained on): Freedoom MAP25–32, `two_rooms`, 8 generated maps from reserved seeds (≥ 1e9).

### 4.4 Tutor rounds (`git log` messages starting with `[core] tutor`)
1. Memory-grid model at 460k steps: tier 1 exits 83–92%, tier 2 30–55%, tier 3 stuck (~0.1 keys/episode) →
   key reward 3 → 8, +0.5 first key sighting, tier-3 share raised.
2. Tier 3 still at 0% → curriculum: 35% of tier-3 maps are small (3 rooms). Also found and fixed a generator bug
   (chain layouts produced tier-3 maps with no lock).
3. Model switch: plain model trained first (184k steps), then the EgoMap model was warm-started from it with the new
   map inputs initialised to zero (identical behaviour at the switch; proven by a test).

### 4.5 Results (holdout; proxy score defined in `training/eval_zeroshot.py`; higher = better)
| Agent | Mean proxy score | Exit % | Notes |
|---|---:|---:|---|
| Random weights | −13.2 | 0% | Freedoom + two_rooms holdout (teammate 2) |
| Plain model (184k steps) | 9.9 | 18% | holdout v1 |
| **EgoMap model, shipped (184k plain + 460k EgoMap)** | **34.1** | **29%** | holdout v1 |
| Shipped, current eval (gen maps 2,100 tics, 2 seeds) | 27.7 | — | per tier: real Freedoom 7.9 (11% exits), T1 101 (100%), T2 59 (50%), T3 30, T4 9.8 |
- The shipped agent solves the key → locked door → exit sequence on generated tier-3 holdout maps
  (frozen-set eval at 4,200 tics: 100% and 50% exits, keys picked up) and reaches the exit of the official test
  map in 137 steps on its own.
- The final numbers will change; the core owner refreshes the `RESULTS` block in the final hour.

### 4.6 Footprint (official `validate.py`)
- PASSED. Step latency p50 **4.3 ms**, p99 8.1 ms, first step 23 ms (after warm-up in `__init__`); 0.6 s of the
  21 s per-map compute budget on the test map; repo 3.4 MB; weights 3.05 MB via Git LFS.
- Linux run pinned to 1 CPU (teammate 2, random weights): RAM 355 MB of the 8,192 MB limit.

### 4.7 Honest limitations (use in the failure analysis)
- Tier 4 (two locks) still 0% exits; real Freedoom maps are big and hard (≈ 11% exits).
- At the shortest time cap (2,100 tics) tier-3 maps often time out: the backtracking works but is slow.
- Position dead reckoning drifts 1–2 cells over long episodes (heading is exact).
- Only ~0.7M training steps by the time of writing (≈ 250–300 env steps/s on one laptop).

---

## 4b. Timeline (team target: everything finished by 04:00 IST)

- **Brief v1 (Task A) pushed by ~23:00 IST**, README section (Task B) by ~23:30, docs (Task C) by ~01:00.
- ~03:30 IST the core owner pushes a commit starting with `[core] FINAL` and refreshes the numbers between the
  `RESULTS` markers in `BRIEF.md`. After that commit: `git pull --rebase`, re-export `BRIEF.pdf` (still ≤ 2 pages),
  commit `[t3] brief PDF final`, push **before 04:00 IST**.

---

## 4c. Round 2 tasks (after A, B, C) — do them in this order

You may also create/edit: `doom-rl-docs/AUDIT.md`, `doom-rl-docs/EVAL_I3.md`, `doom-rl-docs/DEMO_SCRIPT.md`,
`training/results/i3_*.csv`. Everything in section 0 still applies: **no code edits**.

### Task D — Independent audit (read-only), `doom-rl-docs/AUDIT.md` (~60 min)
Read `docs/RULES.md`, `docs/CONTRACT.md`, `docs/EVALUATION.md`, then every file in `submission/` and `training/`,
plus `maps/gen_maps.py`. Report, each with `file:line` and one sentence of evidence:
1. **Rule risks:** anything in `submission/` that chooses actions without the network (rule trees, pathfinding,
   aiming with `atan2` on entities to *turn*, hard-coded button sequences), any network/file access at eval,
   anything written outside `/tmp`, anything `submission/` imports that isn't in the official image
   (`docker/requirements.lock`). Note: `atan2` inside `EgoMap.update` is only used to draw a key into the memory map
   (encoding), which is allowed; say whether you agree after reading it.
2. **Crash risks at evaluation:** what happens with 0 visible entities, a key that's never seen, a 4,200-tic map,
   `device="cuda"` on a machine without CUDA, missing weights file, an observation exactly at the limits
   (distance 1.0, health 0).
3. **Contract mismatches:** the action returned by `act()` (shape `(9,)`, values in {0,1}), `reset()` clearing all
   memory (GRU state *and* EgoMap), paths relative to `submission/`.
4. Severity for each: `BLOCKER` (could disqualify or crash), `RISK`, `NIT`.
Commit `[t3] audit` and **tell your human right away about any BLOCKER**. The core owner fixes code.

### Task E — Low-end CPU validation + 5-seed score, `doom-rl-docs/EVAL_I3.md` (~60 min, mostly waiting)
1. Install Python tooling (8 GB RAM is enough):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # then reopen PowerShell
   git lfs install
   git pull --rebase ; git lfs pull
   uv python install 3.13
   uv sync
   uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
   ```
   Check `submission/weights/best_agent.pt` is ~3 MB (not a ~130-byte LFS pointer).
2. `$env:PYTHONIOENCODING = "utf-8"` then `uv run python validate.py --device cpu`. Close other apps first.
   Paste the full output into `EVAL_I3.md`. The latency lines (first, p50, p99, agent time vs budget) matter most:
   your i3 is our slowest machine, so it's the best stand-in for a weak evaluation CPU.
3. `uv run python training/eval_zeroshot.py --weights submission/weights/best_agent.pt --name i3_shipped --seeds 0 1 2 3 4 --device cpu`
   (≈ 20–40 min). Paste the MEAN line and the per-tier `T0`–`T4` lines into `EVAL_I3.md`.
4. Commit `[t3] i3 validation + 5-seed eval` (include `training/results/i3_shipped.csv`).
   **Tell your human immediately if validate.py fails** or p99 latency is above 30 ms.

### Task D2 — Re-run your edge-case audit on the fixed code (~15 min, after Task E)
Commit `3cfa239` acted on your audit: `act()` now returns a no-op instead of freezing on an exception (your 2.1), and
the EgoMap grid scrolls on big maps instead of smearing at the edge (your 2.2). `git pull --rebase`, re-run your
`edge_test.py`, and add one case: monkeypatch `agent._act` to raise, then check `act()` returns zeros of shape `(9,)`.
Also re-check the long 1,050-step run: the pose should end inside the grid now (`a.ego_map.ox` is non-zero after
scrolling). Append the results to `AUDIT.md` under `## Re-check after 3cfa239` and commit `[t3] audit re-check`.

### Speed-up (20:30 IST): use your whole i3, review faster
- **Review cycle every 10 min** (not 30). Skip work that can't have changed: run `pytest` only if a new commit touched
  `submission/` or `training/`; run `validate.py` only if it touched `submission/`. Docs-only commits: read the diff, done.
- **Run evaluations in parallel.** Your i3 has 8 threads and one eval uses about 1 core and ~600 MB, so run up to
  **3 evals at once** in separate PowerShell windows (keep ≥ 1.5 GB RAM free; close the browser if needed).
- Order of work: **Task T first**, then Task R2 for every file on the `candidates` branch, with Task R reviews in between.

### Task T — Action-temperature test (PRIORITY, ~30 min with 3 parallel evals)
Teammate 2's replays showed twitchy turns and avoidable deaths. The agent samples its buttons at temperature 1.0;
lower temperatures make it pick its most likely action more often. Find the best value on the shipped weights:
```powershell
git pull --rebase ; git lfs pull
# three windows at once:
uv run python training/eval_zeroshot.py --weights submission/weights/best_agent.pt --temperature 0.3 --name i3_T03 --seeds 0 1 2 --device cpu
uv run python training/eval_zeroshot.py --weights submission/weights/best_agent.pt --temperature 0.5 --name i3_T05 --seeds 0 1 2 --device cpu
uv run python training/eval_zeroshot.py --weights submission/weights/best_agent.pt --temperature 0.7 --name i3_T07 --seeds 0 1 2 --device cpu
```
Compare with T = 1.0 using **the same seeds 0 1 2** from your `training/results/i3_shipped.csv` (average those rows
per map). Post a table `T | MEAN | T0 | T1 | T2 | T3 | T4 | exit %` in `EVAL_I3.md`, commit `[t3] temperature test`, and
tell your human which T wins. The core owner then sets it in `submission/agent.py`.

### Task R — Continuous reviewer (until 03:30 IST)
You are now the team's code reviewer, so the core owner can spend its time on training. Use the token budget freely.
Each cycle:
1. `git pull --rebase && git lfs pull`. Review **every new commit** since your last cycle (`git log --oneline <last>..HEAD`,
   `git show <hash>`), from all three of us (`[core]`, `[t2]`, `[t3]`).
2. Check each against: `docs/RULES.md` (no hand-coded control in `submission/`, no downloaded weights, no network/disk
   writes at eval), train/eval consistency (`training/rewards.py` and `submission/agent.py` must call `EgoMap` the same
   way), crash risks, wrong numbers in docs, `.pt` files accidentally on `main`, anything that breaks
   `docs/CONTRACT.md`.
3. Run `uv run pytest tests/test_egomap.py tests/test_submission.py -q` and
   `$env:PYTHONIOENCODING="utf-8"; uv run python validate.py --device cpu`. Record pass/fail + p50/p99.
4. Append a block to `AUDIT.md` under `## Continuous review log`:
   `### HH:MM — commits <a>..<b>` with findings (`BLOCKER` / `RISK` / `NIT`, `file:line`, one sentence each) and test results.
5. Commit `[t3] review HH:MM`. **Tell your human immediately about any BLOCKER.** Suggest fixes in the text, but
   don't edit code; the core owner fixes it.

### Task R2 — Independent check of candidate checkpoints
When a new `.pt` appears on the `candidates` branch (`git fetch origin && git ls-tree -r --name-only origin/candidates`),
check it out into a separate folder (`git worktree add ../cand origin/candidates`, then `git lfs pull` inside it) and run
the 5-seed eval you did in Task E on it:
`uv run python training/eval_zeroshot.py --weights ../cand/<file>.pt --name i3_<file> --seeds 0 1 2 3 4 --device cpu`.
Post MEAN + T0–T4 in `EVAL_I3.md` and commit `[t3] candidate check <file>`. Your 5-seed numbers are the tie-breaker for the
final pick.

### Task F — 3-minute live-demo script, `doom-rl-docs/DEMO_SCRIPT.md` (~30 min)
For the review (judges may ask for a live demo). Use only facts from section 4 and `doom-rl-docs/REVIEW_CHEATSHEET.md`.
- 0:00–0:30 pitch; 0:30–1:30 what's on screen (the demo video: left game view, right the agent's memory map;
  point out the red locked door being remembered and the yellow key); 1:30–2:30 the 3 USPs with one number each;
  2:30–3:00 limitations + what's next.
- Add the exact command to run a live game if asked:
  `uv run python training/eval_zeroshot.py --weights submission/weights/best_agent.pt --maps two_rooms --seeds 0 --device cpu`
Commit `[t3] demo script`.

---

## 5. How to commit and push (every time)

```powershell
git pull --rebase
git status            # must show ONLY files you own
git add doom-rl-docs/BRIEF.md doom-rl-docs/BRIEF.pdf README.md doom-rl-docs/0*.md AGENTS.md
git commit -m "[t3] <what changed>"
git push
```
If `git status` shows any other changed file, don't add it; report it. If `git pull --rebase` conflicts on a
file you don't own, run `git rebase --abort` and report it.

## Done = all of these are true
- [ ] `BRIEF.md` + `BRIEF.pdf` (≤ 2 pages) pushed, with the results table between the `RESULTS` markers.
- [ ] README team section pushed.
- [ ] Docs 01–05 + AGENTS.md match the built system; cut items listed as "Not implemented".
- [ ] No `TODO(core)` left unreported to your human.
