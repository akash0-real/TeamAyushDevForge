# Tasks: Teammate 2 (RTX 3050 laptop) — Evaluation, Validation, Seed-2 Training

> **To the AI assistant reading this:** follow this file exactly. Do the tasks in order. Do not
> refactor, "improve", or touch any file not listed under **Files you own**. If a step fails and the
> fix would need a file you don't own, STOP and report the error text to your human. Don't work around it.

Repo: `https://github.com/akash0-real/TeamAyushDevForge` (branch `main`).
Competition: Yugaantar DevForge Track 1 "Can It Play DOOM?". Official rules are in `docs/RULES.md`,
`docs/CONTRACT.md`, `docs/EVALUATION.md`, `docs/SUBMISSION.md`. Read `docs/RULES.md` once before starting.

---

## 0. Hard rules (breaking these can disqualify the team)

1. **Never edit**: `env/`, `validate.py`, `docs/` (official), `Dockerfile`, `docker/`, `tests/`, `pyproject.toml`, `uv.lock`, `.gitattributes`.
2. **Never edit anything in `submission/`** (agent, model, preprocess, weights, requirements.txt). Another teammate owns it.
3. **Never edit** `training/train_ppo.py`, `training/rewards.py`, `maps/gen_maps.py`. Another teammate owns them.
4. **No hand-written game logic in the agent**: no `if enemy: shoot`, no pathfinding, no `atan2` aiming. (You aren't touching the agent anyway.)
5. **No downloaded models or weights** of any kind (no Hugging Face, no GitHub checkpoints). Everything is trained by us.
6. **Never commit `.pt` files** or anything under `training/checkpoints/`. Never `git push --force`. Never rewrite history.
7. Don't add packages to `submission/requirements.txt`. Training-only installs on your machine are fine.

## Files you own (the ONLY files you may create or edit)

- `training/eval_zeroshot.py`
- `training/results/` (new folder, CSV files)
- `doom-rl-docs/RESULTS.md` (new file)

---

## 1. Install (Windows, PowerShell)

Do each step and check its output before going on.

1. Install **Git** (https://git-scm.com/download/win) and **Git LFS** (https://git-lfs.com). Then:
   ```powershell
   git lfs install
   ```
2. Install **uv** (Python manager):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   Close and reopen PowerShell, then check: `uv --version`.
3. Clone and set up:
   ```powershell
   git clone https://github.com/akash0-real/TeamAyushDevForge.git
   cd TeamAyushDevForge
   uv python install 3.13
   uv sync
   ```
   `uv sync` installs `vizdoom==1.3.3`, `gymnasium`, `numpy` and the official `doom_env` wrapper.
4. Install PyTorch **2.15.x with CUDA** (the evaluation image uses `torch==2.15.0`). Run `nvidia-smi` to see your CUDA version, then take the matching `uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cuXXX` command from https://pytorch.org/get-started/locally/ (pick "Pip", "CUDA"; replace `pip install` with `uv pip install`).
5. Verify:
   ```powershell
   uv run python -c "import torch, vizdoom, doom_env; print(torch.__version__, torch.cuda.is_available(), doom_env.CONTRACT_VERSION)"
   ```
   Expected: `2.15.0+cuXXX True 1.0.0-draft` (the contract version may differ; report it if it does).

Every Python command below is run from the repo root as `uv run python ...`.

---

## 2. Task A — Baseline validation (~15 min)

1. `$env:PYTHONIOENCODING = "utf-8"` then `uv run python validate.py`. Without the encoding line, Windows crashes on the validator's ✓ characters with `UnicodeEncodeError`; that's not a real failure. Set it in every new PowerShell window before `validate.py`.
2. With no trained weights, the agent uses random network weights. It must still **pass** (no crash, valid actions, timing within limits). Copy the full output.
3. If Docker Desktop is installed: `docker build -t devforge-doom .` then `uv run python validate.py --docker --image devforge-doom`. Skip this if Docker isn't installed; just say so in the report.
4. Put the validator output (pass/fail lines, ms/step, peak RAM/VRAM) into `doom-rl-docs/RESULTS.md` under a heading `## Baseline validate.py (random weights)`.

If it fails: do NOT fix `submission/`. Report the failure text to your human and continue with Task B.

---

## 3. Task B — Upgrade `training/eval_zeroshot.py` (~60 min)

This script is how we pick the final checkpoint, so it must be correct and identical on every machine.
Keep the existing structure. Make exactly these changes:

1. **`--weights PATH` argument** (optional). After `agent = DoomAgent(...)` is created, if `--weights` is given:
   ```python
   ckpt = torch.load(args.weights, map_location=agent.device, weights_only=True)
   agent.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
   agent.model.eval()
   ```
   Do not change `submission/agent.py` to do this.
2. **Default holdout maps** = `two_rooms MAP25 MAP26 MAP27 MAP28 MAP29 MAP30 MAP31 MAP32` (Freedoom 2 maps 25–32 are NEVER used for training). Default seeds `0 1 2`.
3. **Ticks per map**: `two_rooms` → 2100; Freedoom maps → 4200. Remove the single `--ticks` default and use this rule (keep `--ticks` as an override if given).
4. **Count doors opened** per episode, same rule as training: at reset, store `ceil0 = ps["sector_ceiling"].copy()`; each step, a sector counts as an opened door if `ps["door_lock"][s] >= 0` and `ps["sector_ceiling"][s] - ceil0[s] > 8`. Count each sector once. Count it as a locked door if `door_lock[s] > 0`.
5. **Track `died_early`**: `1` if `ps["dead"]` and `ps["tic"] < 0.5 * ticks`.
6. **Proxy score**, one function, exactly this formula (our local stand-in for the hidden official score):
   ```python
   def proxy_score(exit, keys, doors, locked_doors, kills, cells, damage, died_early):
       return (100.0 * exit + 15.0 * keys + 5.0 * doors + 10.0 * locked_doors
               + 1.0 * kills + 0.2 * cells - 0.05 * damage - 20.0 * died_early)
   ```
   `damage` = `ps["damage_taken"]` at the end of the episode. Average over seeds per map.
7. **Output**: keep the per-map print, add `score` and `doors` columns, and print a final `MEAN score`. Also write a CSV to `training/results/<name>.csv` where `<name>` comes from a new `--name` argument (default: the weights file stem, or `random`). Columns: `map, seed, exit, keys, doors, locked_doors, kills, cells, damage, died_early, hp, ms_per_step, score`.
8. **Never crash the whole run on one map**: wrap each `eval_map` call in `try/except Exception as e`, print the map name + error, and continue.
9. Self-check: run `uv run python training/eval_zeroshot.py --name random --seeds 0` with no weights. It must finish and print a table. Paste the table into `RESULTS.md` under `## Random-weights holdout baseline`.

Commit (see section 6) with message `[t2] eval_zeroshot: holdout set, proxy score, --weights, CSV`.

---

## 4. Task C — Seed-2 training run on the 3050 (starts when the trainer fix lands)

The other teammate is fixing bugs in `training/train_ppo.py` (it currently crashes when the first episode ends).
**Don't start this task until** `git log origin/main --oneline` shows a commit whose message starts with `[core] trainer fixes`.
Check with `git fetch` + that command every 15 minutes, doing Task B meanwhile.

1. `git pull --rebase`
2. Smoke test (must run ~2 minutes without crashing, printing `upd ... | ret ...` lines):
   ```powershell
   uv run python training/train_ppo.py --maps two_rooms --num-envs 4 --total-steps 20000 --save-path training/checkpoints/smoke.pt
   ```
3. Real run. Use the exact command printed in the `[core] trainer fixes` commit message / `training/README.md` if it gives one. Otherwise:
   ```powershell
   uv run python training/train_ppo.py --maps MAP01 MAP02 MAP03 MAP04 MAP05 MAP06 MAP07 MAP08 MAP09 MAP10 MAP11 MAP12 MAP13 MAP14 MAP15 MAP16 MAP17 MAP18 MAP19 MAP20 MAP21 MAP22 MAP23 MAP24 two_rooms --num-envs 8 --seed 7 --total-steps 20000000 --save-path training/checkpoints/seed7.pt
   ```
   - `--num-envs`: number of physical CPU cores minus 2 (max 12). Check the cores with: `(Get-CimInstance Win32_Processor).NumberOfCores`.
   - Keep the laptop plugged in, sleep disabled, and lid open (or set "do nothing on lid close").
   - Never use MAP25–MAP32 for training. They're the holdout.
4. Leave it running. Note in `RESULTS.md` the start time and the `fps` value from the log after 10 minutes.

---

## 5. Task D — Evaluate checkpoints every ~45 minutes (until the deadline minus 1 hour)

1. Evaluate the newest checkpoint from your run:
   ```powershell
   uv run python training/eval_zeroshot.py --weights training/checkpoints/seed7.pt --name seed7_<HHMM>
   ```
2. Append one row to the `## Checkpoint leaderboard` table in `RESULTS.md`:
   `| time | run | steps (from the log) | MEAN score | exit % | keys | doors | kills | ms/step |`
3. If a checkpoint has the best MEAN score so far, also run `validate.py` with it. Copy the checkpoint over the submission weights **only in a throwaway copy of the repo**, never in your working repo:
   ```powershell
   git clone . ..\validate_copy ; Copy-Item training\checkpoints\seed7.pt ..\validate_copy\submission\weights\best_agent.pt ; cd ..\validate_copy ; uv sync ; uv run python validate.py ; cd ..\TeamAyushDevForge
   ```
   (Delete `..\validate_copy` afterwards.) Record pass/fail and ms/step in the table.
4. Commit + push `RESULTS.md` and `training/results/*.csv` after each evaluation: `[t2] results <HHMM>`.
5. When your human asks for your best checkpoint, send the `.pt` file to them directly (Drive, USB, chat). **Don't commit it.** The core owner puts the final weights into `submission/weights/` with Git LFS.

---

## 5b. Task E — Add the generated holdout maps to the eval (as soon as Task B is done)

The core owner added `maps/holdout/gen_t{1,2,3,4}_{1,2}.wad`: 8 procedurally generated maps (tiers 1–4) built
from seeds training never uses. Each is a single map named `MAP01`.

1. `git pull --rebase`, then check the 8 files exist.
2. In `training/eval_zeroshot.py`, add them to the default holdout list as
   `maps/holdout/gen_t1_1.wad:MAP01` … `maps/holdout/gen_t4_2.wad:MAP01` (the existing `wad:MAP` parsing handles this).
   Ticks for these maps: **2100**.
3. Add a `tier` column to the CSV: `1`–`4` from the file name for `gen_t*` maps, `0` for Freedoom/two_rooms.
   Print one extra summary line per tier: `T0 / T1 / T2 / T3 / T4 mean score`.
4. Re-run the random-weights baseline (`--name random --seeds 0`) and update the table in `RESULTS.md`.
5. Commit: `[t2] eval: generated holdout maps + per-tier summary`.

Also note: `training/train_ppo.py` now has `--use-map` (memory grid model). Your Task C command stays the same (no `--use-map`).
If your human asks you to run the map model instead, the core owner will give the exact command.

---

## 5b2. Task G — Switch your laptop to the memory-grid model at full speed (do this NOW, ~15 min)

Commit `3cfa239` fixed two bugs that wasted your machine too:
- every env worker loaded PyTorch for nothing (~490 MB each; that RAM can now run more envs);
- the PPO update fed the GRU one timestep at a time (GPU mostly idle).

Your plain seed-7 run scores 9.7 vs 27.7 for the shipped memory-grid model, so continue from the shipped weights instead:
1. Stop your seed-7 run (Ctrl+C). Keep `training/checkpoints/seed7.pt`; add one final leaderboard row for it.
2. `git pull --rebase` then `git lfs pull` (check `submission/weights/best_agent.pt` is ~3 MB).
3. Copy the shipped weights so your run never touches `submission/`: `cp submission/weights/best_agent.pt training/checkpoints/t2_start.pt`
4. Start (use `nproc` minus 2 for `--num-envs`, max 16; watch `free -h`, and keep ≥ 1.5 GB available):
   ```bash
   .venv/bin/python -u training/train_ppo.py --use-map --resume training/checkpoints/t2_start.pt --lr 1.5e-4 \
     --num-envs 10 --seed 48 --total-steps 8000000 --save-path training/checkpoints/t2map.pt 2>&1 | tee training/checkpoints/t2map.log
   ```
5. After 10 minutes, record `fps` and the `exit-by-tier` line in `RESULTS.md`. In Task D evaluations, evaluate
   `training/checkpoints/t2map.pt` instead of `seed7.pt`.
6. If a `t2map` checkpoint beats the shipped score in `doom-rl-docs/CHECKPOINTS.md` (mean **and** T0), tell your human
   and send the `.pt` file to the core owner (not via git `main`). The core owner validates and ships it.

## 5b3. Task H — You are now the PRIMARY trainer (replaces Task G; run until 03:00 IST)

Your Linux laptop runs ~974 env steps/s, about 3× one Windows run, and your `egomap_s77_best` (31.0) is level with the
shipped agent (31.2). From now on your machine trains our main candidate. The core owner still picks and ships the
final weights. Use the token budget freely: check on the run often.

1. **Pull first:** `git pull --rebase && git lfs pull`. You need commit `3cfa239` or later. It changed the
   EgoMap (the grid now *scrolls* on big maps) and the trainer (torch-free workers, faster GRU update). Training must
   use the same EgoMap code as `submission/` or the model sees different memory maps than at evaluation.
2. **Restart your EgoMap run from its best weights** with the new code (keep your settings, they're fast):
   ```bash
   cp training/checkpoints/egomap_s77_best.pt training/checkpoints/t2_h_start.pt
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u training/train_ppo.py --use-map \
     --resume training/checkpoints/t2_h_start.pt --lr 1.5e-4 --num-envs 24 --rollout-steps 256 --num-minibatches 2 \
     --swap-every 8 --seed 78 --total-steps 30000000 --save-path training/checkpoints/t2h.pt \
     --maps <your 20 compact Freedoom maps + two_rooms> gen3 gen3 gen3 gen3 gen3 gen3 gen4 gen4 gen4 gen4 \
     2>&1 | tee training/checkpoints/t2h.log
   ```
   Adding the `gen3`/`gen4` slots keeps the key → locked door → backtrack skill alive (your run is our real-map
   specialist; the core owner's runs are generator-heavy). **Never** add MAP25–MAP32.
3. **Use the whole machine.** Every ~15 min check `htop`, `free -h`, `nvidia-smi`. If CPU < 85% and ≥ 3 GB RAM is free,
   start a **second** run from the same start weights with `--seed 79 --num-envs 12 --save-path training/checkpoints/t2h2.pt`
   (log `t2h2.log`). Keep ≥ 1.5 GB RAM free at all times; if it drops below, stop the second run first.
4. **Every 30 min:** evaluate the newest `t2h*.pt` and `*.best.pt` with `--seeds 0 1 2`, add rows to `RESULTS.md`,
   push `[t2] results HHMM`.
5. **Hand candidates to the core owner** when a checkpoint scores **mean ≥ 31.2 and T0 ≥ 6.9**. Never put `.pt`
   files on `main`; use the `candidates` branch (Git LFS):
   ```bash
   git fetch origin && git worktree add -B candidates ../cand origin/candidates 2>/dev/null || git worktree add -b candidates ../cand
   cp training/checkpoints/t2h.best.pt ../cand/t2h_HHMM_score<SCORE>.pt
   cd ../cand && git add *.pt && git commit -m "[t2] candidate HHMM score <SCORE> T0 <T0>" && git push -u origin candidates && cd -
   ```
   Then tell your human. The core owner validates it on a clean clone and ships it if it wins.

## 5b4. Task H update (20:30 IST) — round-4 rewards from your video review

Thanks for the replay review: the core owner applied your proposals 1 and 3 in `training/rewards.py` and
`training/train_ppo.py`. The exit bonus now pays for remaining health and time (`+25 + 6·hp + 6·time_left`); damage taken
is −0.03/hp, health gained +0.025/hp, death −5; giant maps MAP11/12/15/23 are out of the default list; `--swap-every`
defaults to 8. Proposal 2 (temperature) is being tested by teammate 3 (`--temperature` now exists in the eval).
1. `git pull --rebase`, then restart your `t2h` run with `--resume training/checkpoints/t2h.pt` (same flags as Task H),
   so it trains with the new rewards. Your second run (`t2h2`) the same way if you started it.
2. The `candidates` branch now exists (core put two checkpoints there). Add yours with the Task H step-5 commands.
3. Your eval for `RESULTS.md` can add `--temperature 0.5` as a second row once teammate 3 reports which T wins.

> **Every cycle, first read the `COORDINATION` block at the top of `doom-rl-docs/CHECKPOINTS.md`**: shared timeline, ship rule, how to ask questions.

## ANSWERS from core (22:45) to your RESULTS.md questions
1. **Map mix:** yes. Your `t2h` can switch to `2×gen2 + 6×gen3 + 6×gen4`. Tier-3 exits are already around 83% in
   training and tier 4 (two locks) is still 0%, while harder tiers weigh more. The `gen2` keeps tier 2 from slipping
   (v3's tier 2 was −10 ± 16 vs v2). Leave `t2w` on `DEFAULT_TRAIN_MAPS`, so it stays the clean round-5 test.
2. **Docker:** `validate.py --docker --image devforge-doom --no-gpu` plus the native Linux `--device cuda` run is enough.
   Don't install `nvidia-container-toolkit`.
3. **Shared PID namespace:** fine. PIDs are unique inside one namespace, so the `{pid}.wad` files can't collide.
4. **Candidates:** push only your **single best** checkpoint per hour to `candidates` (teammate 3 is in low-token mode
   until 01:20), with MAP25–32 and T5 numbers in the README line. `t2w_2300` at ~23:05 is the next one.

## 5b1. Task H2 — Second run with the reverse curriculum (NEW 22:20, start now, alongside phase 2)
Core added tutor round 5 (`[core] tutor round 5`, `training/rewards.py`). Half the real-Freedoom-map episodes now start
near the exit, and the start moves further back as the agent succeeds. The goal is real-map exits, which nobody has
yet. These episodes show up as **T5** in the training log.
1. `git pull --rebase`, then check that `python training/check_warp_curriculum.py` ends with `OK`.
2. **Keep your phase-2 run going** (it runs the old code, which is fine). Start a **second** process only if
   `free -g` shows ≥ 2.5 GB available:
   `.venv/bin/python -u training/train_ppo.py --use-map --resume <your t2h_2150 checkpoint> --lr 1.5e-4 --num-envs 12 --rollout-steps 256 --num-minibatches 2 --total-steps 10000000 --seed 79 --save-path training/checkpoints/t2w.pt`
3. Every ~45 min, push `t2w` to `candidates` as `t2w_HHMM.pt`, with a README line giving the T0 and T5 exit % from the
   log.
4. If T5 exits are still 0% after 60 min, stop `t2w` and write that in `RESULTS.md`. That result tells core something too.

> Every cycle, first read the `COORDINATION` block at the top of `doom-rl-docs/CHECKPOINTS.md`.

## 5b2. Task F0 — Validate inside the official Docker image (NOW, between training checkpoints, ~20 min)

Teammate 3 confirmed the package versions match the scorer image (`AUDIT.md`, "X 22:08"), but nobody has run the
agent *inside* the image yet. Only weights change before the deadline (the code in `submission/` won't), so a pass now
covers the final ship too. Keep your training run going; this runs alongside it.
1. `git pull --rebase && git lfs pull`
2. If `docker` isn't installed, install it (`sudo apt install docker.io`, then `sudo usermod -aG docker $USER` and
   log out and back in). If you can't install it, write "Docker not available" in `RESULTS.md` and stop here.
3. `docker build -t devforge-doom .` then `.venv/bin/python validate.py --docker --image devforge-doom`
4. Paste the whole output into `RESULTS.md` under `## validate.py --docker (official image)`.
5. Commit `[t2] docker validation`. **Tell your human immediately if it fails.**

## 5c. Task F — Final Linux validation of the shipped agent (≈ 03:30 IST)

Team target: **everything finished by 04:00 IST** (04:00–05:00 is buffer). Your Linux machine is the only one that
can pin the agent to 1 CPU and measure RAM exactly like the scorer, so the final check is yours.

1. From ~03:15 IST, `git fetch` every 5 minutes until `git log origin/main --oneline` shows a commit starting with
   `[core] FINAL`. Then `git pull --rebase`.
2. Check the weights are the real file, not an LFS pointer: `git lfs ls-files` lists `submission/weights/best_agent.pt`
   and the file is ~3 MB (`ls -la submission/weights/`). If it's ~130 bytes, run `git lfs pull`.
3. `.venv/bin/python validate.py`. Copy the whole output into `RESULTS.md` under `## FINAL validate.py (Linux, 1 CPU)`.
4. Run the full holdout eval on the shipped weights:
   `.venv/bin/python training/eval_zeroshot.py --weights submission/weights/best_agent.pt --name FINAL --seeds 0 1 2`
   and paste the per-tier summary into `RESULTS.md`.
5. Commit + push: `[t2] FINAL validation`. **Tell your human immediately if validate.py fails.**

---

## 6. How to commit and push (every time)

```powershell
git pull --rebase
git status            # must show ONLY files you own
git add training/eval_zeroshot.py training/results doom-rl-docs/RESULTS.md
git commit -m "[t2] <what changed>"
git push
```
If `git status` shows any other changed file, don't add it: run `git restore <file>` (or `git stash` if it's not yours) and report it.
If `git pull --rebase` hits a conflict in a file you don't own, run `git rebase --abort` and report it. Don't resolve it yourself.

---

## 7. Report back to your human (at the end of each task)

One short message: what was done, the commands run, pass/fail, the numbers (MEAN score, fps, ms/step), and any errors with their full text. Don't summarise errors away.

## Done = all of these are true
- [ ] `validate.py` result recorded (random weights).
- [ ] `eval_zeroshot.py` upgraded, committed, random baseline table in `RESULTS.md`.
- [ ] Seed-7 run started after `[core] trainer fixes`, fps recorded.
- [ ] Leaderboard rows added every ~45 min; best checkpoint validated and handed to the human (not committed).
