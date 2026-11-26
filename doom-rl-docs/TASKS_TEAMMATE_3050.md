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
