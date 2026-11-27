# Independent Audit (teammate 3, read-only)

**Scope:** `docs/RULES.md`, `docs/CONTRACT.md`, `docs/EVALUATION.md`, `docs/SUBMISSION.md`, every file in
`submission/` and `training/`, and `maps/gen_maps.py`. Audited at commit `6cfa81c` (19:20 IST).

**Method:**
- A static read of every file.
- A dynamic edge-case run of the real `DoomAgent` on synthetic extreme observations (§2; script in the appendix).

**Severity:**
- **BLOCKER**: could disqualify us or crash the agent.
- **RISK**: could cost points.
- **NIT**: cosmetic.

## Summary
**No BLOCKERs found.** RISK items: **2** (§2.1, §2.2). NITs: **3**.

---

## 1. Rule risks (`docs/RULES.md` §1, §4)

| # | Finding | Evidence | Severity |
|---|---|---|---|
| 1.1 | **Actions come only from the network.** `act()` builds the observation encoding, runs the model, samples the 6 factored heads and maps them to 9 buttons. No rule trees, pathfinding, aiming or scripted button sequences anywhere in `submission/`. | `submission/agent.py:53-65` | OK |
| 1.2 | **The `atan2` in EgoMap is encoding, not aiming.** I agree with the core owner. `np.arctan2` converts a visible key's `sin/cos` into an angle, and the result is only used to set a cell in the key channel of the memory grid. It never touches an action. | `submission/preprocess.py:176-181` | OK |
| 1.3 | **Imports are only `numpy` and `torch`,** plus our own `model`/`preprocess`. Both are pinned in the official image (`numpy==2.5.5`, `torch==2.15.0`). `submission/requirements.txt` is empty. | `submission/agent.py:9-14`, `submission/model.py:15-21`, `docker/requirements.lock:154,333` | OK |
| 1.4 | **No network access, no file writes, no engine access** in `submission/`. The only file read is the weights, through a path relative to `__file__`. Privileged engine state (`privileged_state()`, `DAMAGECOUNT`) is used only in `training/rewards.py`, which `submission/` never imports. | `submission/agent.py:16-17`, `training/rewards.py:118,152` | OK |
| 1.5 | **Weights load with `weights_only=True`**, so a tampered checkpoint can't run code. | `submission/model.py:214` | OK |

## 2. Crash risks at evaluation

| # | Case | Result | Evidence | Severity |
|---|---|---|---|---|
| 2.1 | **Any exception inside `act()`.** There is no `try/except`, so an unexpected observation or a bug makes the player **stand still for the rest of that map** (`docs/EVALUATION.md` §"When your agent stops"). No test case below triggered one. A fallback that returns a no-op `np.zeros(9, np.int8)` on exception would only cap the damage, and would need the core owner's call. | see §4 | `submission/agent.py:53-65` | RISK |
| 2.2 | **Maps larger than the memory grid.** The grid covers ±4,096 units around spawn. Further out, `_cells()` clips positions to the edge, so the map smears along the border, and the crops show only what's still inside. **No crash** (see the long-run test), but memory degrades on very large real maps. | see §4 | `submission/preprocess.py:105-106,134-137,194-196` | RISK |
| 2.3 | 0 visible entities | handled: slot 0 is unmasked to avoid a NaN softmax, then the output is zeroed | `submission/model.py:84-98` | OK |
| 2.4 | A key that's never seen | the key channel just stays empty | `submission/preprocess.py:176-177` | OK |
| 2.5 | A 4,200-tic map (1,050 steps) | fixed-size grid, constant cost per step | see §4 | OK |
| 2.6 | `device="cuda"` without CUDA | falls back to CPU | `submission/agent.py:24` | OK |
| 2.7 | Missing weights file | silently uses an **untrained** plain model: no crash, but it would play randomly. Weights are in Git LFS, and the scorer fetches LFS objects (`docs/SUBMISSION.md:28`). | `submission/agent.py:26,30` | NIT |
| 2.8 | Observation at the limits (distance 1.0 / 0.0, health 0) | inputs are clipped; no-hit rays (`dist ≥ 1023` units) aren't drawn as walls | `submission/preprocess.py:46,63,172-173` | OK |

## 3. Contract checks (`docs/CONTRACT.md`)

| # | Check | Evidence | Severity |
|---|---|---|---|
| 3.1 | `act()` returns an `np.ndarray` of shape `(9,)`, `int8`, values in `{0,1}`, and contradictory buttons are impossible by construction | `submission/preprocess.py:23-36`, `submission/agent.py:63-65` | OK |
| 3.2 | `reset()` clears the GRU state, the previous action, the first-step flag and the EgoMap (grid + pose + turn counter) | `submission/agent.py:45-50`, `submission/preprocess.py:127-132` | OK |
| 3.3 | The weights path is relative to `submission/` (`Path(__file__).resolve().parent`) | `submission/agent.py:16-17` | OK |
| 3.4 | Warm-up runs in `__init__` (60 s limit), not on the first `act()` (10 s limit) | `submission/agent.py:33-42` | OK |
| 3.5 | Training and evaluation call EgoMap in the same order (update with no action on the first step, then with the previous action), so there's no train/eval mismatch | `training/rewards.py:140-148,241-242`, `submission/agent.py:55-59` | OK |

## Other NITs
- **`--seq-len` does nothing.** `train_ppo.py` parses it, but BPTT always spans the full rollout. Either remove it or
  note it in `--help` (docs 03 already says so). `training/train_ppo.py:81`
- **Thread count is left to the environment.** `submission/` doesn't call `torch.set_num_threads(1)`. That's fine at
  evaluation, because the scorer sets `OMP_NUM_THREADS=1` etc. (`docs/EVALUATION.md:15-16`). It only matters for
  local timing runs.

## 4. Dynamic edge-case results
Run on the teammate-3 laptop (i3-1215U, 8 GB, CPU only) with the shipped `best_agent.pt`, at 19:27 IST. **All
8 cases pass.** Every returned action is an `int8` array of shape `(9,)` with values in `{0,1}`.

```text
PASS  device='cuda' without CUDA -> cpu, use_map=True
PASS  0 visible entities, all rays no-hit (distance 1.0): shape=(9,) dtype=int8
PASS  all rays distance 0.0 + locked_door type: shape=(9,) dtype=int8
PASS  health 0: shape=(9,) dtype=int8
PASS  32 visible keys: shape=(9,) dtype=int8
PASS  1,050 steps (4,200 tics) moving fast, never sees a key: shape=(9,) dtype=int8
      avg 11.1 ms/step; agent pose drifted to x=6506
PASS  reset(): h zero=True, ego grid zero=True, pose=(0.0,0.0,0.0)
PASS  missing weights file (falls back to random plain model): shape=(9,) dtype=int8
```

- The long run ends at x = 6,506 units, **past the ±4,096-unit grid edge**. That confirms §2.2 degrades the memory
  but doesn't crash.
- The 11.1 ms/step was measured while the 5-seed eval was running on the same CPU. Use `validate.py` for latency
  (`EVAL_I3.md`).

## Appendix: how to rerun
The edge-case script builds synthetic observations from `doom_env.spaces.observation_space`, calls the real
`DoomAgent`, and checks every returned action against the contract.
Save it anywhere, then run it from the repo root: `uv run python edge_test.py .`

```python
"""Task D edge cases: run the real DoomAgent on synthetic extreme observations. Usage: python edge_test.py <repo>"""
import sys, time, shutil, tempfile
from pathlib import Path
import numpy as np

repo = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(repo / "submission"), str(repo / "env")]
import agent as A  # noqa: E402
from doom_env.spaces import observation_space as _obs_fn, action_space as _act_fn  # noqa: E402
observation_space, action_space = _obs_fn(), _act_fn()


def obs(dist=1.0, hit=0, n_ent=0, ent_type=9, health=100.0, vel=0.0):
    o = observation_space.sample()
    o["rays"]["distance"][:] = dist
    o["rays"]["hit_type"][:] = hit
    m = np.zeros(32, np.int8); m[:n_ent] = 1
    o["visible_entities"]["mask"][:] = m
    o["visible_entities"]["entity_type"][:] = np.where(m, ent_type, 0)
    o["visible_entities"]["features"][:] = 0
    o["visible_entities"]["features"][:n_ent] = [0.5, 0.0, 1.0, 0.0, 0.0]
    o["vitals"][:] = 0; o["vitals"][0] = health; o["vitals"][14] = vel
    return o


def check(a, name, o, steps=1):
    for _ in range(steps):
        act = a.act(o)
    ok = isinstance(act, np.ndarray) and act.shape == (9,) and set(np.unique(act)) <= {0, 1}
    print(f"{'PASS' if ok else 'FAIL'}  {name}: shape={getattr(act, 'shape', None)} dtype={getattr(act, 'dtype', None)}")


a = A.DoomAgent(observation_space, action_space, "cuda")  # no CUDA on this laptop -> must fall back to CPU
print(f"PASS  device='cuda' without CUDA -> {a.device}, use_map={a.model.use_map}")
check(a, "0 visible entities, all rays no-hit (distance 1.0)", obs())
check(a, "all rays distance 0.0 + locked_door type", obs(dist=0.0, hit=3))
check(a, "health 0", obs(health=0.0))
check(a, "32 visible keys", obs(n_ent=32, ent_type=9))
a.reset()
t = time.perf_counter()
check(a, "1,050 steps (4,200 tics) moving fast, never sees a key", obs(vel=30.0, dist=0.3, hit=1), steps=1050)
print(f"      avg {(time.perf_counter() - t) / 1050 * 1000:.1f} ms/step; agent pose drifted to x={a.ego_map.x:.0f}")
a.reset()
print(f"PASS  reset(): h zero={bool((a.h == 0).all())}, ego grid zero={not a.ego_map.grid.any()}, pose=({a.ego_map.x},{a.ego_map.y},{a.ego_map.theta})")

# missing weights file: copy submission/ without weights, import the copy
tmp = Path(tempfile.mkdtemp())
shutil.copytree(repo / "submission", tmp / "submission", ignore=shutil.ignore_patterns("*.pt"))
for m in ("agent", "model", "preprocess"):
    sys.modules.pop(m, None)
sys.path[0] = str(tmp / "submission")
import agent as B  # noqa: E402
b = B.DoomAgent(observation_space, action_space, "cpu")
check(b, "missing weights file (falls back to random plain model)", obs())
shutil.rmtree(tmp, ignore_errors=True)
```

## Re-check after 3cfa239

Re-run on the teammate-3 i3 at commit `bf9ee12` (which includes `3cfa239`), 19:50 IST, with the shipped
`best_agent.pt`. The edge-case script above gained two cases: a monkeypatched `_act` that raises, and a check that
the grid scrolls.

**Code review of the fixes:**
- **2.1 → FIXED.** `act()` wraps `_act()` in `try/except Exception`. On error it prints the error and returns
  `np.zeros(9, int8)` (press nothing) for that step only, and resets `prev_action`. Actions still come only from the
  network. The safety net never chooses a button. (`submission/agent.py`, `act` / `_act`)
- **2.2 → FIXED.** `EgoMap._recenter()` scrolls the grid by multiples of 8 cells when the agent gets within ¼ of an
  edge. The origin shift `ox`/`oy` is applied in `_cells()` and in both crops, and `reset()` zeroes it. I checked the
  shift with a positive and a negative example: kept memory lands where it should, and the coarse view stays
  aligned. Maps within ±2,048 units of spawn behave exactly as before. (`submission/preprocess.py`, `_recenter`,
  `_cells`, `crops`, `reset`)
- **NIT `--seq-len` → FIXED** (flag removed).

**Dynamic results: 11/11 PASS.**

```text
PASS  device='cuda' without CUDA -> cpu, use_map=True
PASS  0 visible entities, all rays no-hit (distance 1.0): shape=(9,) dtype=int8
PASS  all rays distance 0.0 + locked_door type: shape=(9,) dtype=int8
PASS  health 0: shape=(9,) dtype=int8
PASS  32 visible keys: shape=(9,) dtype=int8
PASS  1,050 steps (4,200 tics) moving fast, never sees a key: shape=(9,) dtype=int8
      avg 9.8 ms/step; agent pose drifted to x=-3079
PASS  long run stays inside the grid after scrolling: ox=-40, oy=80, agent cell=(55,73), visited cells kept=613
PASS  reset(): h zero=True, ego grid zero=True, pose=(0.0,0.0,0.0), origin=(0,0)
DoomAgent.act error, no-op this step: RuntimeError('injected failure')
PASS  _act raises -> act() returns no-op zeros: shape=(9,) dtype=int8 sum=0
PASS  next step after the injected failure plays normally: shape=(9,) dtype=int8
PASS  missing weights file (falls back to random plain model): shape=(9,) dtype=int8
```

**`validate.py --device cpu` on the fixed code (i3): PASSED.**

| | Before (`6cfa81c`) | After (`bf9ee12`) |
|---|---:|---:|
| Initialization | 15.0 s | 12.3 s |
| First step | 46 ms | 73 ms |
| p50 / p99 | 7.1 / 17.6 ms | 9.3 / 21.1 ms |
| Test map | exit in 79 steps | exit in 388 steps |

Both runs exit the test map. Step counts differ because the policy samples its actions (see `EVAL_I3.md`). p99 stays
under the 30 ms alarm line.

**Updated summary: no BLOCKERs, no open RISKs. 1 NIT still open (§2.7: missing weights silently fall back to random
weights; covered by Git LFS).** The thread-count NIT needs no change.

The two new script cases, appended to the edge-case script above:

```python
e = a.ego_map
if hasattr(e, "ox"):  # 3cfa239+: grid scrolls instead of clipping at the edge
    cx = int(np.floor(e.x / 64.0)) + 64 - e.ox
    cy = int(np.floor(e.y / 64.0)) + 64 - e.oy
    print("long run inside grid:", 0 <= cx < 128 and 0 <= cy < 128, "ox/oy:", e.ox, e.oy)
if hasattr(a, "_act"):  # 3cfa239+: act() safety net
    def boom(_obs):
        raise RuntimeError("injected failure")
    real, a._act = a._act, boom
    act = a.act(obs())  # must be np.zeros((9,), int8)
    a._act = real
```

## Continuous review log

### 20:20 — commits 3a9433d..c403d14
Reviewed `b6b0ac1` [t3], `7e44134` [core], `82a7cca` [t2], `f12eaca` [core], `c403d14` [t2].

**Findings:**
- **RISK: Task H's start command copies a file that doesn't exist, and the trainer then silently trains from scratch.**
  - Task H (`TASKS_TEAMMATE_3050.md` §5b3, step 2) runs `cp training/checkpoints/egomap_s77_best.pt …/t2_h_start.pt`.
  - The run was saved with `--save-path training/checkpoints/egomap_s77.pt` (`RESULTS.md:75`), so its best file is
    `egomap_s77.best.pt` (`training/train_ppo.py:302`). The backup is `egomap_s77_1948_score31.pt` (`RESULTS.md:79`).
  - If the `cp` fails, `--resume` silently ignores the missing file (`training/train_ppo.py:124`: `if args.resume and
    Path(args.resume).is_file()`), and the primary run starts from **random weights**.
  - **Fix:**
    - Teammate 2: copy `egomap_s77_1948_score31.pt` (or `egomap_s77.best.pt`), and check that `t2h.log` prints
      `Resumed weights from …`.
    - Core: make a missing `--resume` file a hard error.
- **NIT: Akash's suggested exit-time bonus wouldn't run as written.** It reads `ps["episode_ticks"]`, but
  `privileged_state()` has no such key. The episode length is the `ticks` that `ShapedDoomEnv` passes to `make_env`
  (`training/rewards.py:106`), so a version that works would use that value. The idea itself (a time bonus paid only on
  exit, no per-step penalty) is sound.
- **NIT: Akash's README note doesn't hold up.** `uv run python …` after `uv pip install torch` does **not** uninstall
  torch: `uv run` syncs inexactly. I tested it on the i3; torch 2.15.0+cpu was still installed afterwards. Only
  `uv sync` removes packages that aren't in the lockfile. No README change needed.
- **NIT: the PID-named WAD cache** (`training/rewards.py:103`, `GEN_CACHE / f"{os.getpid()}.wad"`) can't collide on one
  machine, since PIDs are unique there, including across Havish's two parallel runs. The cache lives in the repo's
  `training/gen_cache/` (`training/rewards.py:37`), not `/tmp`, so separate machines can't share it either. No action.
- **OK:**
  - `7e44134` only adds timing logs to `train_ppo.py`; no behaviour change.
  - Akash's numbers match his CSVs exactly. I recomputed: `egomap_s77_best_1948` mean 31.00, T0 7.31, 25% exits over
    seeds 0–2; `egomap_s77_1948` 28.55.
  - Only `submission/weights/best_agent.pt` is on `main`, through LFS (`git lfs ls-files`).
  - No changes to `submission/` since `3cfa239`, so train/eval EgoMap call order is unchanged.
- **Note on the candidate bar:** "mean ≥ 31.2 and T0 ≥ 6.9" (Task H step 5) is judged on 3 seeds, and same-seed
  reruns differ by about 9 points (`EVAL_I3.md`). Treat it as a filter, not a verdict; Task R2's 5-seed check decides.
- **Brief footprint (Akash's §4 note):** I'll refresh `BRIEF.md`'s footprint table from the final checkpoint's
  `validate.py` run at `[core] FINAL`, not from a candidate's numbers.

**Tests (i3, CPU):**
- `pytest tests/test_egomap.py tests/test_submission.py`: **8 passed**.
- `validate.py --device cpu`: **PASSED**. Init 11.3 s; first step 22 ms; p50 9.5 ms; p99 19.8 ms; test map exited in
  137 steps.

**Candidates branch:** doesn't exist yet. Nothing for R2.

### 20:30 — commit 376340f (tutor round 4)

**Correction to my 20:20 review:** my NIT that `ps["episode_ticks"]` "doesn't exist" was **wrong**.
`privileged_state()` returns `"episode_ticks": self.episode_ticks` (`env/doom_env/wrapper.py:210`), and `376340f` didn't
change `env/`. Teammate 2's suggestion was valid as written, and round 4 uses it correctly.

**Findings:**
- **OK: `submission/agent.py` temperature.** `self.temperature = 1.0` and sampling from `logits / T`
  (`submission/agent.py:33,74`). At T = 1.0 this behaves exactly as before. Sampling temperature is a normal inference
  setting, and actions still come only from the network, so rules are fine.
- **NIT: a temperature of 0 would freeze the agent.** `T <= 0` makes `logits / T` infinite or NaN. `Categorical` then
  raises, and the `act()` safety net returns a no-op **every step**, so the agent would stand still for the whole map.
  If anyone wants "greedy", use `sample_and_logprob(..., deterministic=True)`, or guard with `max(T, 1e-3)`.
  (`submission/agent.py:74`)
- **OK: `training/rewards.py` round 4.**
  - The exit reward is `25 + 6·hp_frac + 6·time_left`, with `time_left = 1 − tic / episode_ticks`, clamped at ≥ 0.
    Its maximum (37) stays above the exploration cap (+24).
  - Damage −0.03/hp, health +0.025/hp, death −5.
  - There's still no per-step penalty, so dying early still never pays.
  - Train/eval consistency is unaffected, since rewards are training-only.
- **Watch: giant maps now dropped from training.** The default training maps drop MAP11/12/15/23, the 539–863-sector
  maps (`training/train_ppo.py`, `REAL_TRAIN_MAPS`). That's good for speed, but the hidden maps may include large
  ones. Keep an eye on T0 (real holdout maps MAP25–32) in the next candidate checks.
- **OK: `training/README.md` now uses `uv run --no-sync`.** It's harmless but not required: plain `uv run` kept torch
  installed in my 20:20 test.
- **OK: eval `--temperature`** only overrides `agent.temperature` after the weights load (`training/eval_zeroshot.py`).

**Tests (i3):**
- `pytest tests/test_egomap.py tests/test_submission.py`: **8 passed**.
- `validate.py`: **deferred**. Two 5-seed candidate evals (R2) are using the CPU right now, which would inflate
  latency. It runs on the round-4 `submission/` as soon as they finish.

**Work queue:**
- R2: `core_r1_2025`, `core_r2_2025` (5 seeds each), started 20:24.
- Then Task T (T = 0.3 / 0.5 / 0.7, seeds 0–2).
- **RAM:** only 0.71 GB is free on the i3 (the desktop apps use most of it), so I'm not stacking a third eval on top
  of R2.

### 21:52 — commits 376340f..6e68142 (`core_r2_2025` shipped)
Reviewed `0258015` [core] (ship) and `6e68142` [core] (board). The teammate-3 commits in between are my own.

**Findings:**
- **OK: the shipped weights are exactly the checkpoint I evaluated.** `sha256(submission/weights/best_agent.pt)` =
  `sha256(candidates:core_r2_2025.pt)` = `a5eea925…08bd`. It's still a Git LFS object (`git lfs ls-files`), and it's
  the only `.pt` on `main`. The commit message's numbers match `EVAL_I3.md` §4/§4b. T stays 1.0, as recommended (§5b).
- **RISK (latency alarm rule): p99 went above 30 ms on the i3 in 2 of 3 `validate.py` runs.** I'm logging it because
  Task E says to flag p99 > 30 ms. I don't think it's a real risk:
  - **Same compute per step.** The architecture is unchanged (only weights changed, and `submission/*.py` is untouched
    since `376340f`), so each step costs what it did with the old weights.
  - **Short episodes inflate p99.** The new agent exits `two_rooms` in **60–110 steps** (the old one took 79–510), so
    "p99" is really the 1–2 slowest steps, which is OS jitter on a busy laptop.
  - **The budget is far away.** Agent time is 0.7–0.8 s of the 21 s per-map budget (about 4%), and the average step is
    about 11 ms against the 40 ms limit.
  - **To confirm:** teammate 2's Linux run pinned to 1 CPU on `0258015` would settle it.

| Run | Steps | First | p50 | p99 | max | Agent time / budget |
|---|---:|---:|---:|---:|---:|---|
| 1 | 60 | 34 ms | 9.8 ms | **34.4 ms** | 36 ms | 0.7 s / 21 s |
| 2 | 110 | 18 ms | 5.5 ms | **33.6 ms** | 40 ms | 0.8 s / 21 s |
| 3 | 67 | 19 ms | 9.7 ms | 21.1 ms | 32 ms | 0.7 s / 21 s |

All three runs: **PASSED**, init 9.7 s.

**Tests (i3):** `pytest tests/test_egomap.py tests/test_submission.py`: **8 passed**.

## Strict review 22:10 (Task S)
Independent of commit messages and `CHECKPOINTS.md`: every number below was recomputed from teammate-3 CSVs by script,
or by a command whose output is quoted. Code at `e794b14`.

**1. Rules sweep of `submission/`: PASS.** I read all three files line by line, checking every branch and loop.
- **`agent.py:53-77`.** Action selection is only `model.forward_step` → `sample_and_logprob(logits / T)` →
  `factored_to_multibinary`. The only `if` in the decision path is `None if self.first_step else self.prev_action`
  (`:69`), which picks the EgoMap input, not a button. The `except` (`:59-62`) returns a no-op on error and never
  chooses a button. The only I/O is `print` to stdout (`:60`): no file writes, no network.
- **`preprocess.py`.** Every branch is encoding or EgoMap bookkeeping:
  - `:70` prev-action default.
  - `:146` grid scroll.
  - `:158-168` heading dead-reckoning from our own turn bits.
  - `:190` ray → wall/door/locked-door channels.
  - `:195-197` key projection. `np.arctan2` here only computes a grid cell ("encoding only, never used to aim").
  - `:206` crops.
  - The front-cone summary (`:75-89`: nearest distance, door/locked door ahead, enemy visible) is a hand-built *input
    feature*. That's allowed ("preprocessing and encoding observations", `docs/RULES.md` §1), but expect a judge to
    ask about it. No planner, pathfinding or BFS anywhere: `grep -n "deque|heap|bfs|plan|path"` finds nothing in
    `preprocess.py`.
- **`model.py`.** Branches are init and plumbing only (`:25-27`, `:87` empty-entity mask, `:128,140,157` map
  variant, `:188` GRU chunks, `:202-206` sample/argmax, `:213-230` loading). Weights load with
  `weights_only=True` (`:215`).

**2. Shipped and backup weights are the right files: PASS.**

```text
$ git show shipped-v2:submission/weights/best_agent.pt | grep oid      # tag -> 0258015
oid sha256:a5eea925bf49dd35b7c569f92b3a853d7b253d653215742baee4594474b408bd
$ git show origin/candidates:core_r2_2025.pt | grep oid
oid sha256:a5eea925bf49dd35b7c569f92b3a853d7b253d653215742baee4594474b408bd
$ git show shipped-v1:submission/weights/best_agent.pt | grep oid      # tag -> d37c9e4
oid sha256:980f05c69b823dc694e365fb44c8edf5eb8a21f5449443a7671db37471963af0
$ sha256sum submission/weights/best_agent.pt                           # after git lfs pull
a5eea925bf49dd35b7c569f92b3a853d7b253d653215742baee4594474b408bd
```

`git lfs fetch origin shipped-v1` succeeds, and the v1 object (3,051,736 bytes) is in `.git/lfs/objects/98/0f/`, so
a rollback would work.

**3. Is shipping `core_r2_2025` justified? PASS: verdict `KEEP v2`.** Paired per-seed differences, v2 minus v1, at
T = 1.0. CSVs: `i3_shipped`, `i3_core_r2_2025`, `i3_shipped_T0_s59`, `i3_core_r2_2025_T0_s59`.

| Slice | n seeds | v1 | v2 | Δ ± SE | seeds worse | beyond 2 SE? |
|---|---:|---:|---:|---:|---:|---|
| overall (17 maps) | 5 | 31.16 | 34.85 | **+3.70 ± 1.08** | 0/5 | **better** |
| T0 incl. `two_rooms` | 10 | 6.19 | 4.29 | −1.90 ± 1.98 | 6/10 | no |
| **MAP25–32 only** | 10 | −4.60 | −7.77 | −3.17 ± 2.20 | 7/10 | no (1.4 SE) |
| T1 | 5 | 102.03 | 102.29 | +0.26 ± 0.51 | 2/5 | no |
| T2 | 5 | 57.96 | 98.60 | **+40.64 ± 16.09** | 0/5 | **better** |
| T3 | 5 | 67.15 | 64.39 | −2.77 ± 16.38 | 2/5 | no |
| T4 | 5 | 6.85 | 12.16 | +5.31 ± 5.90 | 1/5 | no |

**No slice is significantly worse; overall and T2 are significantly better. `KEEP v2`.**
- **Caveat:** on the truly unseen real maps (MAP25–32), v2 trends lower (−3.2 ± 2.2, worse on 7 of 10 seeds). It's
  not significant, but it's the one slice to watch when picking the final checkpoint.
- **Caveat:** T3 has only 2 maps, so its standard error (±16) is too wide to detect anything smaller than a collapse.

**4. Leakage: RISK. `two_rooms` is in both the training set and the holdout.**
- **Training:** `training/train_ppo.py:43`, `DEFAULT_TRAIN_MAPS = (["two_rooms"] + REAL_TRAIN_MAPS * 2 ...`.
- **Holdout:** `training/eval_zeroshot.py:24` lists `"two_rooms"` first in `DEFAULT_HOLDOUT_MAPS`, and the docs say
  it's never trained on.
- **Clean:**
  - MAP25–32 aren't in training: `REAL_TRAIN_MAPS = range(1, 25)` minus 11/12/15/23 (`train_ppo.py:42`).
  - Generated training seeds are `rng.randrange(HOLDOUT_SEED_BASE)`, so always below 1e9 (`training/rewards.py:104`,
    `maps/gen_maps.py:28`).
  - `maps/holdout/*` isn't referenced by the trainer.
  - The gen_maps self-test generates holdout seeds (`maps/gen_maps.py:377`), but it doesn't train on them.
- **Impact, measured over 10 seeds (`i3_*_T0_s59` + §2/§4 CSVs):**

  | Checkpoint | MAP25–32 exits | `two_rooms` exits | "T0" exits as reported |
  |---|---:|---:|---:|
  | v1 | **0% (0/80)** | 80% (8/10) | 9% |
  | v2 | **0% (0/80)** | 90% (9/10) | 10% |

  **Every exit we report on "real Freedoom maps" is `two_rooms`, a training map.** On the truly unseen MAP25–32, the
  agent has never reached an exit. The "T0" score is also inflated: `two_rooms` averages 92–101 points, while
  MAP25–32 average −4.6 (v1) / −7.8 (v2).
- **Not a rule violation.** `two_rooms` is the official test map, not an evaluation map, and training on it is
  allowed. But these docs claim "about 11% exits on real maps" and "two_rooms … never trained on", which is wrong:
  `BRIEF.md` §5/§7, `04_PROCEDURAL_MAPS.md` §3, `DEMO_SCRIPT.md`, and `REVIEW_CHEATSHEET.md`. A judge comparing the docs
  with `train_ppo.py:43` would catch it.

**5. Reward hacking, tutor round 4 (`training/rewards.py`): PASS.** Every term is either paid once or monotonic and
bounded:
- cells: a set capped at 600 (`:165`);
- doors: once per sector via `_opened_doors` (`:177`);
- first key sighting: once (`:188`);
- keys: the delta of sticky flags (`:191`);
- kills and items: monotonic counters (`:197`, `:202`);
- damage dealt: a monotonic `DAMAGECOUNT` delta (`:208-209`), bounded by total enemy HP;
- **healing can't be farmed:** taking damage costs 0.03/hp (`:219`) and healing it back earns 0.025/hp (`:215`), so a
  hurt-then-heal cycle loses 0.005/hp;
- exit: paid once on the terminal step (`:223-227`);
- no per-step penalty, so dying early never pays (`:229-230`).
- **Side effect, not a hack:** death now costs 5 plus up to 3 for the damage taken. Watch for an over-passive agent
  (fewer kills, lower T1).

**6. Train/eval consistency: PASS.** The same sequence on both sides:

| Step | Training | Agent |
|---|---|---|
| Reset | `_prev_action = 0` (`rewards.py:120`); `ego_map.reset(); update(obs, None)` (`:141-142`) | `prev_action = 0` (`agent.py:48`); `update(obs, None)` on the first step (`:69`) |
| Each step | `_prev_action = mb_action` (`:236`); `update(obs, mb_action)` (`:244`); `preprocess_obs(obs, _prev_action)` + `crops()` (`:146-148`) | `preprocess_obs(obs, prev_action)` (`:66`); `update(obs, prev_action)` + `crops()` (`:69-70`) |
| Action conversion | `factored_to_multibinary` (`rewards.py:156`) | `factored_to_multibinary` (`agent.py:75`) |

**Summary:** 5 PASS, 1 RISK (item 4), **no BLOCKER. Item-3 verdict: `KEEP v2`.**

QUESTION(core): `two_rooms` is a training map (`train_ppo.py:43`) but counted in the T0 holdout
(`eval_zeroshot.py:24`), and it accounts for all of our "real-map" exits (MAP25–32: 0/80 exits). Should I correct
my docs (`BRIEF.md`, `04_PROCEDURAL_MAPS.md`, `DEMO_SCRIPT.md`) to report MAP25–32 separately? They're outside my
scope lock, so I won't touch them until you say so.

### X 22:08 — Official image (Task X topic 1)

**Verdict: PASS (CONFIRMED). The submission runs on the official image's package versions.**

**Versions.** The local venv that ran every `validate.py` and eval in this log has the same versions as the image:

```text
$ grep -n "^torch==\|^numpy==\|^gymnasium==\|^vizdoom==" docker/requirements.lock
101:gymnasium==1.4.0
154:numpy==2.5.5
333:torch==2.15.0
354:vizdoom==1.3.3
$ grep -n "FROM\|python 3.13" Dockerfile
8:FROM ubuntu:24.04
26:RUN uv venv --python 3.13 /opt/venv
$ uv run --no-sync python -c "import sys,torch,numpy,gymnasium,vizdoom; print(...)"
3.13.15 torch 2.15.0+cpu numpy 2.5.5 gym 1.4.0 vizdoom 1.3.3
```

**Imports.** `submission/` imports only `numpy`, `torch`, `math`, `pathlib`, `__future__`, plus its own `model` and
`preprocess`:

```text
$ grep -hn "^import\|^from" submission/*.py | sort -u
```

All of these are in `docker/requirements.lock` or the standard library, and `submission/requirements.txt` is empty.

**Platform code.**
- No Windows-only code in `submission/`. `grep -n "win32\|windows\|ctypes\|msvcrt" submission/*.py` finds nothing.
  The Windows timer code (`ctypes`) lives only in `training/rewards.py`, which `submission/` never imports.
- No CUDA-only code. The one CUDA line is the fallback
  `torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")` (`submission/agent.py:24`).

**Both devices are covered.**
- **CPU:** `validate.py --device cpu` passed on Windows here at 19:25, 19:50, 20:20, 20:44 and 21:52.
- **CUDA on Linux:** teammate 2's runs passed (`RESULTS.md`, rows with `PASS … 130 MB VRAM`).

**Not covered (HYPOTHESIS):** an actual `validate.py --docker` run inside the image. To test: `docker build -t
devforge-doom .` then `python validate.py --docker --image devforge-doom`. That's a large download plus a build, so it
doesn't fit on the i3 (8 GB RAM) while evals run. A better fit for teammate 2's or the core owner's machine.

### 22:17 — commits b82e22e..a330b51
Reviewed `aae31c8`, `2b2f75a`, `a330b51` [core] (task-file and board updates only: no `submission/` or `training/*.py`
changes, so no pytest or validate needed) and `c3d4e83` [t2] (candidates branch only).

**Impact lines for my earlier findings, per the new rule (`TASKS_TEAMMATE_DOCS.md`, 22:06):**

| Finding | IMPACT on hidden score | Why |
|---|---|---|
| Strict review item 4, `two_rooms` leak | **MEDIUM** | It doesn't change what the agent does on hidden maps. It changes how we *select* checkpoints: T0 is inflated by a training map, so the MAP25–32 mean must be used. Fixed in the docs (`cd51760`) and in `CHECKPOINTS.md` (`aae31c8`). |
| 21:52, p99 > 30 ms on the i3 | **LOW** | Agent time was 0.7–0.8 s of a 21 s per-map budget. |
| X 22:08, official image | **LOW** | PASS. The only gap is an actual `--docker` run, now assigned to teammate 2 (Task F0, `a330b51`). |
| 20:30, temperature ≤ 0 freezes `act()` | **LOW** | T stays 1.0. It only matters if someone sets T = 0. |

**CONFIRMED: Task F0 (`a330b51`) closes the Docker gap from X 22:08.** The task text quotes my HYPOTHESIS and tells
teammate 2 to run `validate.py --docker`.

### 22:27 — commits a330b51..fc5e0b5 (`t2h_2150` shipped as v3)
Reviewed `189b9d5` [t2], `6075137` [core], `2af54a5` [core] (ship), `400a820` [core] and `fc5e0b5` [t2].

**`2af54a5` (ship v3): PASS (CONFIRMED).**

```text
$ git show shipped-v3:submission/weights/best_agent.pt | grep oid      # tag -> 2af54a5
oid sha256:5833380514a6cb82d571b6f3824994a0d12caf5f4f202b35217e33760c32a237
$ git show origin/candidates:t2h_2150_score46_62.pt | grep oid
oid sha256:5833380514a6cb82d571b6f3824994a0d12caf5f4f202b35217e33760c32a237
$ sha256sum submission/weights/best_agent.pt
5833380514a6cb82d571b6f3824994a0d12caf5f4f202b35217e33760c32a237
```

- Git LFS tracks it, and it's the only `.pt` on `main` (`git lfs ls-files`).
- The commit message's numbers match `EVAL_I3.md` §4d/§4e.
- Tests: `pytest tests/test_egomap.py tests/test_submission.py` **8 passed**.
- `validate.py --device cpu` on the i3: **PASSED**. Init 11.8 s; first step 111 ms; p50 8.7 ms; p99 15.1 ms; test map
  exited in 206 steps; agent time 1.9 s of 21 s.
- IMPACT: **LOW** (it passes).

**`189b9d5` (`training/eval_zeroshot.py`): PASS.**
- It adds a print-only "T0 unseen Freedoom only" line, filtering on `r["map"].startswith("freedoom2.wad:")`. That
  matches the map labels the eval prints (e.g. `freedoom2.wad:MAP25`).
- `python -m py_compile training/eval_zeroshot.py`: OK.
- pytest: 8 passed.
- IMPACT: **LOW**. It makes the `two_rooms` fix permanent in the eval output.

**`fc5e0b5` (`RESULTS.md`): docs only.**
- **CONFIRMED:** the demo videos exist. `git ls-tree -r -l origin/demos` lists `FULL_DEMO_SHOWCASE.mp4` (17.2 MB),
  `03_tier3_keycard_backtrack.mp4` (7.6 MB) and `02_tier2_unseen_exit.mp4` (3.8 MB), committed `5ac6f48` at 20:22.
- **NIT:** they're plain git blobs, not LFS, so 28.5 MB sits in the repo history.
  - They're off `main`, so `validate.py`'s size check and the scorer's evaluated commit aren't affected.
  - IMPACT: **LOW**.
- Teammate 2's QUESTION(core) items (T4 map mix; Docker without the NVIDIA toolkit) are for the core owner.
  - Reviewer's note, as a HYPOTHESIS: a `--no-gpu` Docker run still checks the image, the imports and the sandbox
    limits, and `agent.py:24` falls back to CPU when CUDA is missing. Only GPU memory inside the container would go
    untested.

QUESTION(core): `DEMO_SCRIPT.md` still has a `TODO(core)` for `demo/agent_demo.mp4` (cheat-sheet numbers: 246 / 364
steps). The real videos are on `origin/demos` (teammate 2: Tier 2 exit in 123 steps, Tier 3 in 516 steps). May I point
the script at `FULL_DEMO_SHOWCASE.mp4` and use those step counts?

### 22:37 — commits fc5e0b5..621f17b (tutor round 5: reverse curriculum)
Reviewed `936b9d1` [core] (`training/rewards.py`, new `training/test_warp_curriculum.py`) and `621f17b` [core]
(task files). `submission/` is unchanged (`git show --stat 936b9d1` lists training files only), so no `validate.py` is
needed.

**`936b9d1`: PASS (CONFIRMED), with 1 NIT and 1 HYPOTHESIS.**
- **Rules:** the warp uses `sv_cheats 1` + `warp x y` in `ShapedDoomEnv.reset()`. That's training only; `RULES.md` §2
  allows any engine change in training. `submission/` doesn't import `rewards.py`.
- **Train/eval consistency: OK.**
  - The warp and its single no-op step happen *before* `ego_map.reset(); ego_map.update(raw_obs, None)`, so the EgoMap
    starts at the warped spot exactly as the agent's EgoMap starts at spawn.
  - `ego_map.tic = self._warp_tics` keeps the turn dead-time model aligned with the game tic.
  - The GRU state is reset by the trainer at the episode boundary as before.
- **Self-check passes:** `uv run --no-sync python training/test_warp_curriculum.py` → `all 32 maps have exits + spawn
  spots` … `OK 2 of 12 warped` (20 s).
- **Warp rate matches `WARP_PROB = 0.5`.** The self-check's 2/12 looked low (about a 2% chance at p = 0.5), so I
  measured it with `training/results/t3_checks/warp_rate.py`:
  `seed 7: 17/30 warped (57%)`, `seed 11: 17/30 warped (57%)`. That's 34/60, consistent with 0.5; the 2/12 was
  seed-3 chance.
- **NIT (CONFIRMED): plain `pytest` runs the script.** `training/test_warp_curriculum.py` matches pytest's `test_*.py`
  pattern and runs at module level, so plain `pytest` executes the 20 s environment script during collection:
  `pytest --collect-only -q -s training/test_warp_curriculum.py` → prints `OK 2 of 12 warped` … `no tests collected
  in 22.46s`. CI only runs `tests/test_submission.py` (`.github/workflows/ci.yml:39`), so CI isn't affected.
  - **Fix:** wrap the body in `if __name__ == "__main__":`, or rename it `check_warp_curriculum.py`.
  - IMPACT: **LOW**.
- **HYPOTHESIS (not tested): some warp spots may be dead ends.**
  - Spawn spots are any item or player start within the radius of an exit line (`exit_spots()`), with no check that
    the exit is reachable *from* that spot without a key the agent doesn't hold.
  - On maps where the exit sits behind a locked door, a warped start beyond that door would be unwinnable, or one in
    a closed secret area. The radius would shrink there (×0.95 per failure, floor 512).
  - **Test:** log `exit_reached` per warp spot over about 100 warped episodes per map, and flag spots that never exit.
  - IMPACT: **LOW–MEDIUM**. It wastes some training episodes but can't hurt evaluation.

**`621f17b`:** a new Task Y for teammate 3 (exit-blocker research) and Task H2 for teammate 2. Task Y starts once
the v3 confirmation evals (seeds 5–9) finish, under the 3-eval limit.

## TEAM CHANNEL (teammate 3 → core + teammate 2), from 22:32 IST
**MESSAGE from our human (Ayush):** all communication goes through GitHub from now on.
- **Core (Havish) and teammate 2 (Akash):** please put *everything* in commits: decisions, questions, results, status,
  and anything you'd otherwise say in chat. Use your own files (`CHECKPOINTS.md` / task files for core, `RESULTS.md` for
  teammate 2), one short `UPDATE HH:MM` line each time.
- **Teammate 3 (me):** every update I give our human is also posted here as `### UPDATE HH:MM`. Questions stay
  `QUESTION(core): …`.

### UPDATE 22:32
- v3 (`t2h_2150`) is verified byte-identical to the scored candidate. `validate.py` PASSED on the i3 (p99 15.1 ms).
  The seeds 5–9 confirmation (v3 vs v2, all 17 maps) is running.
- Round 5 (reverse curriculum) review: PASS; warp rate 34/60. Low-token mode acknowledged.
- Task Y has started: `training/results/t3_checks/exit_blockers.py`. Part 1 (map facts) works; part 2 runs after the
  confirmation evals.

### UPDATE 22:40
- **v3 ship CONFIRMED over 10 seeds:** +9.3 ± 2.5 overall, +7.8 ± 1.7 on MAP25–32, T3 exits 14/20 vs 4/20
  (`EVAL_I3.md` §4f). The new bar is the v3 10-seed mean, 42.30.
- Task Y part 2 (exit-blocker run, 10 maps × 2 seeds at 4,200 tics) has started.

### Y 22:45 — exit blockers (shipped v3, 10 real maps × 2 seeds, 4,200 tics)
Script: `training/results/t3_checks/exit_blockers.py`; CSV: `exit_blockers.csv` (n = 20).
**Outcomes:** 0 exits, 18 deaths, 2 timeouts. The maps are MAP25–29, 31, 32 plus MAP02, 05, 08 (training maps), and
those also give 0/6 exits.
1. **Dies early. CONFIRMED.** 18/20 episodes end in death, at a median tic of **920** (22% of the time limit); 17 of the
   18 deaths come before tic 2,100.
   - IMPACT: **MEDIUM**. Measured on campaign maps, not on tier maps. Death ends progress and costs the early-death
     penalty.
   - **Fix:** more survival pressure on real maps: round-4 rewards (v3 trained mostly on round 3), more real-map
     episodes with monsters, and possibly health pickups placed near warp starts.
2. **Never picks up a key on real maps. CONFIRMED.** 0 keys in 20 episodes. Every probed map has 2–4 keys and 1–6
   locked doors (map facts, part 1).
   - IMPACT: **MEDIUM**. Key → door progression is the main source of points, and exits on these maps are likely
     behind locks (SUSPECTED; not checked per map).
   - **Fix:** extend round 5 to also warp near *keys*, not only near exits.
3. **Gets close but doesn't finish. SUSPECTED.** 4/20 episodes came within 256 units of an exit line: MAP29 at 119 and
   84 (its start is only 224 away), MAP02 at 184 and 80 (a switch exit). The distance is a straight line, so a wall or
   a locked door may separate them.
   - IMPACT: **MEDIUM**.
   - **Test:** replay MAP29 and MAP02 and log USE presses within 128 units of the exit line.
   - **Fix:** round 5 (near-exit starts) targets exactly this.
- **Not a blocker:** stalled exploration. Only 1/20 episodes went over 1,000 tics without a new cell.

### UPDATE 22:45
- Task Y: **0/20 exits on real maps.** Top blockers: dying early (median tic 920), never picking up keys (0/20), and
  near-misses at the exit (4/20). Suggestion for core: warp near keys too in round 5.

### 22:50 — commit 5abd953 (round 5b: also warp near keys): PASS (CONFIRMED)
- **Code:**
  - Key warps spawn on item or player-start spots near a key, never on the key itself: `KEY_THINGS` (5/6/13/38/39/40)
    is disjoint from the spawn types (2000–2049, 1, 8, 11).
  - Success means `_stats["keys"] > 0`.
  - Exit and key warps keep separate radii (`"<map>:<kind>"`).
  - `_warped` is now `False`/`"exit"`/`"key"`, still truthy for the `ego_map.tic` fix.
- **Tests:** `check_warp_curriculum.py` → `OK 10 of 30 warped (exit + key)`; pytest **8 passed**.
- **Warp rate is still about 0.5** (re-measured, since 10/30 looked low): `warp_rate.py` → seed 7 16/30 and seed 11
  20/30, so 36/60 = 60%.
  - My script needed `bool(env._warped)` now that `_warped` is a string. Fixed in this commit.
- IMPACT: **MEDIUM (positive)**. It targets Task Y blocker #2: 0 keys in 20 real-map episodes.

### UPDATE 22:50
- Round 5b (key warps) reviewed: PASS. No new candidates yet, so the next R2 is on teammate 2's `t2w_2300` when it lands.

### 23:00 — commit f25bd1d (`maps/fresh/` final-exam set): PASS
- **No leakage (CONFIRMED):**
  - The fresh generated seeds are 2,000,000,000 + 100·tier + k (`maps/fresh/README.md`). That's disjoint from training
    (seeds below 1e9, `training/rewards.py:104`) and from the holdout seeds (1e9 + k), and it's below 2³¹.
  - The Freedoom 1 maps are a different IWAD and appear in no training map list.
- **Recomputed from core's CSVs** (`untouched_v{2,3}.csv`, 16 maps × seeds 0–2): v3 37.62 vs v2 36.71, Δ **+0.90**,
  which matches core. The paired-per-seed SE is ±3.2 (n = 3) vs core's ±5.4, probably a per-map pairing. Same
  conclusion: **no measurable difference on untouched maps.**
- **This confirms the selection-bias risk:** the 17 holdout maps have now picked three ships.
- **Teammate 3 won't evaluate anything on `maps/fresh/` before 03:00.**

### UPDATE 23:00
- Core's untouched-map check: v3 is about level with v2 there (+0.9). The +9.3 was mostly holdout selection bias, but
  T3/T4 still favour v3. The final pick at 03:00 uses `maps/fresh/`.

### UPDATE 23:10
- **CORRECTION to Y 22:45 blocker #3** ("gets close but doesn't finish"). Teammate 2 tested it (`RESULTS.md`, UPDATE
  22:55, n = 10). The agent *does* press USE near those exits: MAP29 at 87/116 units, with 6/12 and 1/4 USE presses;
  overall USE is 32–48% of steps. So #3 isn't a missing USE: those near-misses are walls, locks or deaths, which folds
  it into blockers #1 and #2.
- **Teammate 2's tier-4 diagnostic** agrees with Task Y: 19/20 deaths and 1/20 keys on `gen_t4_*` at both 2,100 and
  4,200 tics. Survival and key search are the top two blockers.
- Teammate 2 offered a per-warp-spot dead-end audit. That's my 22:37 HYPOTHESIS, so it's a good use of the 3050.

### PROPOSAL 23:20 — three proven techniques to beat the field (decision needed from all three)
**From our human (Ayush):** we want to win outright. I researched what published Doom/navigation work uses that we
don't, and each item below targets a blocker we *measured* (Y 22:45: dies early; 0 keys on real maps). This respects
core's 23:10 freeze: it's a decision request, not a change. Items are ordered by gain per hour of work.
1. **Geodesic potential-based shaping** (Ng et al. 1999 — shaping of this form provably keeps the optimal policy
   unchanged; geodesic progress rewards are standard in embodied navigation).
   - Training-only reward `γΦ(s') − Φ(s)`, where Φ = −(walking distance to the next missing key, else to the exit).
   - Distance comes from a BFS over the map's blocking linedefs, plus the room graph `gen_maps.py` already has.
   - It can't be farmed by looping, and it attacks both blockers.
   - **Cost:** about 1–2 h of code + round 6 training. **Conflicts with the freeze.**
2. **Auxiliary game-feature heads** (Lample & Chaplot, AAAI 2017 "Arnold": predicting enemy/item presence from
   privileged state "dramatically" sped up learning).
   - Training-only heads on the GRU output: enemy near, key visible, damage next step. They're dropped in `act()`.
   - This was planned in doc 02 and never built.
   - **Cost:** about 1 h + training. **Conflicts with the freeze.**
3. **Action masking from observation fields** (explicitly allowed, `docs/RULES.md` §1).
   - Never switch away from a loaded gun, and never fire fists with no enemy within melee range.
   - It's an inference-only change in `agent.py`, so no retraining.
   - **Cost:** about 20 min + my 5-seed check. **Compatible with the freeze**, provided it passes the ship rule on the
     holdout.
- **Already have:** reverse curriculum (Florensa et al. 2017), curiosity-free exploration rewards, the T = 1.0 choice.
- **Teammate 3 can take:** a prototype BFS distance field under `training/results/t3_checks/` (no code edits) and R2
  on any masked or round-6 candidate.

QUESTION(core): given the freeze, do we do (3) only tonight, (3) + (1) on teammate 2's fast machine, or nothing new?
Teammate 2, please weigh in in `RESULTS.md`.

### 23:25 — commits a11809a (round 5c), 2120b55: PASS
- **Round 5c (`training/rewards.py`): PASS.** The death reward is `−(5 + 10·time_left)`: −15 for an immediate death, −5
  at the cap. It reuses `time_left` from round 4, and there's still no per-step penalty, so dying never pays.
  `py_compile` is OK, and pytest **8 passed**.
- **Watch:** a larger early-death cost can make the policy passive early in an episode. Check that kills and cells don't
  drop in the next candidate (IMPACT **LOW–MEDIUM**).
- `2120b55`: masking and geodesic shaping adopted, aux heads declined. I'll R2 `[core] action masking` and review round 6
  when they land.

### UPDATE 23:25
- Round 5c reviewed: PASS. Waiting for the action-masking build (R2) and round 6 (review).

### UPDATE 23:30
- **Action masking passes the ship rule on the i3:** +7.9 ± 3.3 overall (significant), T2–T4 exits 17 vs 13, MAP25–32
  +3.8 (`EVAL_I3.md` §6). **Recommend that core turns `action_mask = True` on.**
- Code review of `78b0544`: allowed by `RULES.md` §1 (masking from observation fields). It never masks every option of
  a head. Default OFF until switched on.

### 23:40 — commit 467034f (round 6, geodesic shaping): PASS, with 1 RISK to monitor
- **Checks:** `check_geodesic.py` → OK (exiting `gen3` episodes collect +2.29/+2.34); pytest **9 passed**. It's
  training-only, since `submission/` doesn't import `training/geodesic.py`.
- **OK: γ = 1 (not PPO's 0.995).**
  - Standing still pays 0, and loops in normal terrain telescope to 0.
  - Dying forfeits future progress reward, so there's no suicide incentive.
  - It isn't strictly Ng-invariant, but the only bias is a mild preference for early progress.
- **RISK (SUSPECTED, not reproduced): teleporter farming.** The BFS has no teleport edges (`training/geodesic.py`,
  `blocks()` only), while the per-step shaping is clipped to ±0.1 (`rewards.py`, `GEO_STEP_CAP`).
  - A teleport *away* from the goal costs at most 0.1, or 0 if it lands where `phi` is `None`. Walking back pays the
    full Φ difference, so a teleport loop could net positive reward every cycle.
  - **18/20 default training maps have player-teleport lines** (script: count of specials 39/97/Boom teleports, e.g.
    MAP08: 70, MAP24: 60).
  - **Test:** log per-episode `geo` and flag any episode where `geo > −Φ(start) + 0.5`. Without farming it can't exceed
    the start distance.
  - **Fix if seen:** don't clip negative steps, or set Φ to `None` on teleport lines.
  - IMPACT: **MEDIUM**. It could waste round-6 training; R2 would catch a worse candidate.

### UPDATE 23:40
- Round 6 (geodesic shaping) reviewed: PASS, plus one RISK to watch (possible teleporter reward farming). Teammate 2:
  please log `geo` per episode in your round-6 runs and post any episode with geo above its start distance.

### UPDATE 23:45
- **CORRECTION:** action masking isn't a proven gain. Pooled i3 + CUDA (10 seed pairs): **+1.8 ± 3.1 overall (no
  measurable difference)**, MAP25–32 +3.8 ± 3.1, T4 −4.9 ± 4.7, T2–T4 exits 28 vs 27 (`EVAL_I3.md` §6b).
  - v4 (mask on) still passes the ship rule by point estimates, so it's not harmful as measured.
  - Core's call: keep, or drop only the weapon-switch rule (T4 / rocket HYPOTHESIS).
- Round 6 review (23:40) stands: PASS plus the teleporter-farming RISK to monitor.

### 23:55 — commit 49b85fd (masks off = v5, round-6 cap removed): PASS, 1 residual RISK
- **v5 `submission/agent.py`:** `action_mask = frozenset()` (off), and the rules are now selectable. pytest **9 passed**;
  `validate.py` **PASSED** on the i3 (p50 5.2 ms, p99 11.2 ms).
- **Round 6 fix CONFIRMED.** With no per-step cap, `f = phi - self._phi` (`rewards.py:314`), so loops between cells
  where Φ is defined telescope to 0. `check_geodesic.py` (now with MAP08 and an anti-farming bound) → **OK**.
- **Residual RISK (SUSPECTED): cells the BFS can't reach still break telescoping.**
  - When `phi is None`, the step pays 0 (`:311`) and `self._phi = phi` sets the previous value to `None` (`:317`).
  - A teleport from A into a BFS-unreachable area (common in Doom: areas reachable only by teleport), then out to a
    spot B farther from the goal, costs 0, after which walking B → A pays `d_B − d_A`. That loop can repeat.
  - The self-check's random-action episodes wouldn't find it.
  - **One-line fix:** at `:317`, only update `self._phi` when `phi is not None` (keep the last defined Φ). Then
    A → U → B pays `Φ_B − Φ_A`, and every loop telescopes again.
  - IMPACT: **LOW–MEDIUM**. It's training only; a farming policy would show up as `geo` above the bound in the logs.

### UPDATE 23:55
- v5 (masks off) reviewed: PASS (tests 9/9, validate p99 11.2 ms). The teleport-cap fix is CONFIRMED. One residual
  farming path remains via Φ = None cells, with a one-line fix suggested at `rewards.py:317`.

### 00:00 — commit 069f601: PASS (CONFIRMED). The 23:55 residual RISK is closed.
- `rewards.py:318-319` only updates `self._phi` when `phi is not None`, so a detour A → U → B pays `Φ_B − Φ_A`, and every
  loop telescopes. `check_geodesic.py` → **OK**.
- A goal switch from a cell unreachable for the new goal leaves `_phi = None` for one step, which pays 0. That's harmless.

### UPDATE 00:00
- Round-6 farming fix verified: PASS. No open RISKs on round 6.

### UPDATE 00:05 — teammate 3 switches to minimal mode (token budget)
- From now on teammate 3 only (1) reviews and validates commits that touch `submission/`, (2) runs R2 on new
  `candidates` files, and (3) re-exports `BRIEF.pdf` after `[core] FINAL`. Routine reviews of training/docs commits
  are paused.
- Originality scan (`git grep`): no other team names in the repo. The only "copied" mention is the official template,
  disclosed in `AGENTS.md`.

### UPDATE 00:10 — teammate 3 goes offline at 01:15 IST (token budget)
- Until 01:15: minimal mode (`submission/` reviews, R2 on candidates).
- **After `[core] FINAL` (about 03:30), whoever is awake re-exports the brief with:**
  `python training/results/t3_checks/export_brief.py`. It needs the `markdown` package plus Chrome or Edge, and
  asserts ≤ 2 pages.
  Then commit `[t3] brief PDF final` (`doom-rl-docs/BRIEF.pdf`).
- Also refresh the `RESULTS` block in `BRIEF.md` with the final `maps/fresh/` numbers. Report MAP25–32 separately (not
  `two_rooms`).

### X 00:25 — 1-CPU timing on long real maps (topic 2): PASS, 1 RISK for Linux to confirm
**Method:** Windows process affinity = 1 core (it also pins the *engine*, which is more pessimistic than the scorer),
`validate.py --device cpu --map freedoom2.wad:<MAP> --ticks 4200`.

| Map | Pinned p50 / p99 | Unpinned p50 / p99 | Validator |
|---|---|---|---|
| MAP27 | 6.5 / 15.2 ms | — | PASSED |
| MAP32 | **27.3 / 38.9 ms, reproduced as 27.1 / 40.4** | 9.0 / 15.1 ms | PASSED, with a "Speed margin" warning |

- **RISK (CONFIRMED on Windows, not under scorer conditions):** MAP32 leaves thin headroom on one core: p50 27 ms
  against the 40 ms average budget.
- Agent time stayed far below budget (2.5–3.2 s of 42 s), because the agent died early.
- **Ask, teammate 2 (Task F, Linux 1-CPU agent pin):** please also run
  `validate.py --device cpu --map <freedoom2.wad>:MAP32 --ticks 4200`.
- IMPACT: **MEDIUM if it reproduces there.** Big busy maps could overrun the per-map budget.

### X 00:35 — MAP32 timing RISK resolved: not a risk (CONFIRMED)
`training/results/t3_checks/agent_only_timing.py` records 300 real observations with the engine running, then times
only `DoomAgent.act()` with the process pinned to 1 core (`SetProcessAffinityMask`, `torch.set_num_threads(1)`):
- **MAP32: p50 5.1 ms, p99 12.0 ms, max 17.3 ms.**
- MAP27: p50 5.6 ms, p99 11.9 ms.

The agent's cost doesn't depend on the map. The 27 ms at 00:25 came from the engine sharing the single pinned core,
which the scorer doesn't do; it pins the agent only. **Teammate 2: the MAP32 run in Task F is optional now.**
IMPACT: **LOW**. There's about 3× headroom against the 40 ms budget.

### X 00:45 — scorer categories (`docs/EVALUATION.md:40-49`) vs our training rewards: all covered

| Scored | Our training reward (`training/rewards.py`) |
|---|---|
| Exit (largest) + HP/time bonus | +25 + 6·hp + 6·time_left (round 4) |
| Keycards, doors, locked doors | +8 key, +0.8 door, +4 locked door; key warps (round 5b); geodesic to key/exit (round 6) |
| Unique areas visited | +0.04 per new 64-unit cell. ⚠ The scorer may count sector areas (doc 01); cells are a close proxy. |
| Kills | +1 per kill, +0.01 per hp of damage dealt |
| Self-damage and early death | −0.03 per hp of damage taken (includes self-damage); death −5…−15 by time left (round 5c) |
| Compute budget (the only "draft" value: 40 ms × steps) | agent-only p50 about 5 ms on 1 core (X 00:35) |

**Winning lever left: the 03:00 pick.** Harder tiers weigh more (`:49`), so rank candidates on `maps/fresh/` by T3/T4
exits and score first, as in the ship rule. Use ≥ 10 seeds, because 5-seed runs flipped sign tonight (§6b).

QUESTION(core): has the event page or the organizers announced anything beyond `docs/EVALUATION.md`, such as tier
weights, per-tier tick caps, map count or skill level? If so, please post it in `CHECKPOINTS.md` so the final pick can
weight tiers to match.

### MESSAGE 00:55 from our human (Ayush) → core + teammate 2: EXITS FIRST, on every tier
**Why:** `docs/EVALUATION.md:40-49` says the exit is the largest award per map, and harder tiers carry more weight.
Partial progress (keys, doors, cells, kills) is small change next to exits.
**Where we are** (shipped v5 = `t2h_2150`, 10 seeds, `i3_t2h_2150*.csv`):

| Tier | Exits |
|---|---|
| T1 | 100% |
| T2 | 14/20 (70%) |
| T3 | 14/20 (70%) |
| **T4** | **0/20** (dies every run, HP 0 at the end; 2 keys in 20 runs) |
| MAP25–32 | 0% |

**Asks:**
1. **Training (core, teammate 2):** spend the remaining hours on what creates exits: survival (round 5c), key finding
   (5b + the warp fix), and the route to key → exit (round 6). Keep gen2 and gen4 in every mix: T2 must not drop below
   70%, and T4 is the biggest gap.
2. **03:00 pick (core):** rank candidates on `maps/fresh/` (≥ 10 seeds) by a **tier-weighted exit score**, *then* by mean
   score:
   `exit_points = 1·T1_exits + 2·T2_exits + 3·T3_exits + 4·T4_exits`
   The weights are our stand-in for the hidden "harder tiers weigh more".
   - A candidate with even 1–2 T4 exits and no T2/T3 loss should win.
   - Then apply the ship rule and validate on a clean clone.
3. Post `UPDATE` lines on exits per tier for every candidate, not just mean score.

### UPDATE 01:00
- **Teammate 2's `t2h_0037`** (`t2h_0037.csv`, CUDA, seeds 0–2): exits T1 6/6, **T2 5/6, T3 5/6**, T4 0/6, so a
  weighted exit score of **31**. v5 on the same seeds (i3) gets T1 6/6, T2 4/6, T3 4/6, T4 0/6 = **26**.
  - That's a promising direction, but it's 3 seeds on different devices: no measurable difference yet.
  - **Teammate 2: please push it to `candidates`.** The final pick should compare it with v5 on `maps/fresh/`, 10 seeds.
- The first T4 exit came in *training* (2% of 42 T4 episodes); the eval T4 is still 0/6. Round 5b/5c/6 is starting to
  move T4.
- **Teammate 3 offline from 01:15.** Core runs the 03:00 pick (tier-weighted exits first) and `export_brief.py` after
  FINAL.

### UPDATE 01:00 — teammate 3 stays online (the 01:15 offline note is cancelled)
- Teammate 3 keeps running through FINAL: R2 on candidates, reviews of `submission/` changes, and the `BRIEF.pdf`
  re-export after `[core] FINAL`.
- The R2 check of `t2h_0037` (i3, 5 seeds, weighted exits vs v5) is running now.

### PROPOSAL 01:05 → core: a "use" mask rule against door reversal (inference only, measurable)
**Evidence:**
- **Organizer notice** (relayed in `RESULTS.md`, UPDATE 00:57): USE triggers only on a fresh 0→1 press, and **pressing
  USE on a moving door reverses it**.
- **Our agent presses USE on 32–48% of steps** (teammate 2's 22:55 audit). A press, then a release, then another press
  while the door is still moving would close the door it just opened. That's a SUSPECTED exit blocker, worst on T2–T4
  where doors gate the exit.
- The eval is **6 maps**, so each lost exit costs a lot.

**Proposed rule:** add a `"use"` rule to `DoomAgent._mask`: after a USE press, mask USE = 1 for the next *k* ≈ 9 steps
(about 1 s, one door cycle), tracked from `prev_action` history.
- It removes only re-presses that can reverse a moving door. The network still decides every first press.
- Fits `RULES.md` §1 (action masking; previous actions are the agent's own input). Default OFF, like the others.

**Test (teammate 3, R2):** `eval --mask use` vs v5 and vs the best candidate, 5 seeds on the i3 plus teammate 2's CUDA
run, compared on weighted exits and doors opened. Ship only if both machines agree.

### MESSAGE 01:15 from our human → core + teammate 2: ship faster, keep improving until about 03:15
1. **Core: ship `t2h_0037_score44_22` now as v6.** It passes the ship rule on both machines (i3 weighted exits 47 vs
   43, CUDA 52 vs 45; `EVAL_I3.md` §7). Keep v5 as rollback. Holding it until 03:00 gains nothing.
2. **Teammate 2: push a candidate every 30 min** (not hourly). Teammate 3 runs R2 within about 10 min of each push
   (3 parallel evals on the i3, ship rule plus weighted exits).
3. **Timeline:**
   - training and shipping continue until **03:00**;
   - **03:00–03:15** final `maps/fresh/` check of the top 2;
   - **03:30** `[core] FINAL`, validate, brief.
   Every improvement that passes the rule ships immediately; we don't batch them up.

### UPDATE 01:32
- **v6 highlight reel recorded:** the shipped agent on unseen holdout maps, one exiting run per tier, captioned, 2×.
  Tier 1 exit in 194 steps, tier 2 in 735, tier 3 (key → locked door → exit) in 585.
  - Rebuild it with `python training/results/t3_checks/demo_reel.py out.mp4` (needs ffmpeg).
  - The video (40 MB) isn't committed to `main`. Teammate 2 can add it to `origin/demos` if wanted.

### PROPOSAL 01:40 → core: turn-only temperature (smoother replays, same exits?)
- **Human's review of the v6 reel:** the turning looks twitchy, and the agent fights nothing on T1.
- **Data** (`i3_t2h_0037.csv`, T1, n = 10): 10/10 exits, HP 98 at the end, **0 kills**, 8 cells. It rushes the exit.
  That's fine for exit points, but it skips kill points and looks jittery in replays (`RULES.md` §5: organizers watch
  the top teams' replays).
- **Cause of the twitching:** every head is sampled at T = 1.0. A global T = 0.7 smooths everything but killed T3 exits
  (§5b).
- **Proposal:** a *per-head* temperature. `turn` at about 0.5, every other head at 1.0. That's inference-only, a
  2-line change in `agent._act`: `logits[2] / T_turn`.
- **Test (R2, both machines):** the ship rule plus weighted exits plus kills vs v6, seeded eval.
  Ship only if the exits don't drop.

### UPDATE 01:55
- **v7 confirmed on a second machine:** +6.3 ± 2.9 overall, T3 exits 9/10 vs 5/10, weighted exits 51 vs 37
  (`EVAL_I3.md` §9). Keep v7.

### UPDATE 02:22
- **v9 confirmed on the i3:** it passes the ship rule vs v8. T3 exits 10/10, T4 score +13.2 ± 3.1 (significant), weighted
  exits 54 vs 53 (`EVAL_I3.md` §10).
- The v9 weights are byte-identical to `candidates:core_soupA_0200.pt` (sha256 `7b2f37c9…`).

### UPDATE 02:30: v10 (rollback to v8) verified
- **Weights:** sha256 `2797f55f…`, identical to `t2h_0037` (v6/v8).
- **Agent settings:** `turn_temperature = 0.5`, `head_temperature = [1, 0.5, 1, 1, 1, 1]` (strafe 0.5), which is v8.
- **`validate.py` on the i3:** PASSED (test map exited in 80 steps, p50 8.7 ms).
- **Lesson:** the 17-map holdout picked v9, and the untouched 16-map exam ranked it last (T3 exits 22 vs 33). Keep
  using `maps/fresh/` only for this final call.

### UPDATE 02:40: brief (74c987e) reviewed: PASS
- `BRIEF.pdf` is **2 pages**.
- The RESULTS block (v8: exam T1–T3 40/31/36 of 40, T4 0/40, unseen big real 0/160) matches core's exam numbers.
  `two_rooms` is correctly called a training map.
- The tier-4 limitation ("dies about halfway, after finding keys") is checked against `fin_v8_s*.csv`: on the exam's
  40 T4 runs the agent picks up ≥ 1 key in 24 and dies early in 24, and opened 1 locked door in total. Accurate.
- No changes needed from teammate 3. Waiting for `[core] FINAL`.

### FINAL review 02:50 (teammate 3): PASS
- **`[core] FINAL` (`5cde5d9`) is docs/results only.** It doesn't touch `submission/`.
- **Weights:** sha256 `2797f55f…`, which is v8 (`t2h_0037`). The only `.pt` on `main`, through Git LFS.
- **`validate.py --device cpu` on the i3: PASSED.** Repo 5.1 MB, no extra requirements, init 8.9 s, test map exited in
  80 steps, p50 12.8 ms.
- **Brief:** `BRIEF.md` and `BRIEF.pdf` were last changed together in `74c987e` (2 pages, reviewed 02:40), so they're in
  sync and no re-export is needed.
- **Teammate 3's tasks are complete.**

### MESSAGE 02:55 from our human (Ayush) → core + teammate 2: keep improving until 06:30 IST (extended)
The human extends our internal deadline to **06:30 IST**. **v10 (= v8) stays shipped as the safe fallback.** The plan,
by expected points:
1. **T4 exits (0/40, highest weight).** Exam: a key is found in 24/40 T4 runs, but there are 24/40 early deaths
   (`fin_v8_s*.csv`).
   - Extend the round-5b near-key/near-exit warps to **generated T4 maps** (today they're real maps only).
   - Train gen4-heavy on teammate 2's machine and the Vast boxes, from v8.
2. **T2 (31/40).** Keep gen2 in every mix, and consider a short T2-weighted fine-tune.
3. **Door reversal.** Test the default-off `"use"` mask (PROPOSAL 01:05) with seeded R2 on two machines.
4. **T1 kills (0).** Optional: a small kill bonus on gen1.
5. **⚠ A NEW untouched exam is required.** `maps/fresh/` picked v8, so it's no longer untouched.
   - Core: generate a second set now (seeds ≥ 3e9 + other Freedoom 1 maps).
   - Use it only for one final pick at about **06:00**, 20 seeds, ranked by tier-weighted exits.
   - **Nothing replaces v8 unless it beats v8 there.**
- `[core] FINAL` and the brief move to about 06:15. Teammate 3 keeps R2 and reviews running.

### 03:55 — round 7 (8b0bab9, 0960de0): PASS, 1 HYPOTHESIS
- Warps now also run on generated T3/T4 maps: `GEN_WARP_PROB = 0.4`, spawn spots parsed from the map's `TEXTMAP`,
  per-tier radius `gen<tier>:<kind>`, logged as T7/T8. It's training-only (`submission/` untouched).
  Teammate 2 reports `check_warp_curriculum` OK.
- **HYPOTHESIS:** a T4 *exit* warp can spawn inside the exit room, past both locks, so its exit rate (T7) can overstate
  real T4 skill. Judge round 7 only by the new untouched exam's T4 exits.

### UPDATE 04:56
- **Round-7 candidate fails on the i3:** −5.1 ± 2.1 overall, T3 exits 6 vs 9, T4 still 0 (`EVAL_I3.md` §11). The CUDA
  pass isn't confirmed. **v8 stays.**

### UPDATE 05:20
- `core_r7best_0350` on the i3: ship rule FAIL by one exit (16 vs 17; weighted 50 vs 53), mean +2.8 ± 4.7. Level with
  v8. `t2r7b_0447` result about 05:32.

### UPDATE 05:25: `core_r7best_0350`, pooled i3 + CUDA (10 seed pairs, seeded) vs v8
- overall **+4.4 ± 2.5**, MAP25–32 **+5.5 ± 2.9**, T4 +5.0 ± 6.8; weighted exits 100 vs 98.
- The i3 (fails by 1 exit) and CUDA (passes) disagree individually. Pooled, it trends better and is near-significant on
  real maps.
- **Recommendation: take it into exam 2 as a finalist against v8, and let the untouched set decide.**
  `t2r7b_0447` (i3) result about 05:32.

### REQUEST 05:35 → core: test the `"use"` mask (PROPOSAL 01:05) before exam 2
- It's still untested. It's inference-only (no re-press of USE for about 9 steps after a press), against the
  organizer-flagged door reversal. Our agent presses USE on 32–48% of steps.
- If it's built as a default-off rule, run it on v8 and the top finalist in the 05:45 exam (seeded). Keep it only if it
  adds exits.

### RESEARCH 05:40: techniques common in published Track-1-style work vs ours (ideas only; build our own)
**Same as us:** circular ray CNN, masked entity attention, vitals MLP, GRU, factored or macro action heads,
procedural UDMF maps with solvability checks, recurrent PPO. Our core design is in line with the field.

**Where others differ** (ranked by likely value for us):
1. **Exit visibility.** Exit switches and pads appear in neither rays nor entities. Others add a tiny downsampled
   RGB(+depth) CNN (about 30×40 to 80×60) just to spot exits.
   - **Ours:** no camera input, and 0 exits on big real maps.
   - **Future:** a zero-init camera branch grafted onto v8, then fine-tuned.
2. **USE edge semantics.** Others pulse USE or add a refractory lockout so a held or repeated USE can't reverse a
   door.
   - **Ours:** the `"use"` mask proposal (01:05) is untested. **Cheapest win: test it now.**
3. **Teacher bootstrapping.** An observation-limited scripted teacher, then BC/DAgger, then PPO with a KL anchor to the
   BC policy.
   - **Ours:** pure PPO. Mixed reports: one team found DAgger worse than plain PPO.
4. **Augmentation.** Left–right mirroring (rays, entities, map and actions flipped together) and domain randomization
   (textures, light, ceilings). **Ours:** none.
5. **Asymmetric critic.** The critic sees privileged stats during training only. **Ours:** a plain critic.
6. **Memory without odometry.** A rotation-invariant place signature (FFT magnitude of the ray ring plus a hit-type
   histogram) for novelty and familiarity. **Ours:** dead-reckoning EgoMap (exact heading, 1–2 cells of drift).
7. **Robustness.** Plain-git backup weights in case LFS isn't fetched. **Ours:** LFS only, validated from a fresh clone
   (low risk).

**Caveat:** others' results are self-reported on *their own* maps. Their "tier 4" isn't ours (ours: two chained locks,
key 1 behind lock 2). The official T4 is "multiple locks, ambushes, scarce resources" (`docs/EVALUATION.md`). Don't
compare numbers directly.

**Plan:** (2) now if time allows. Then (1), (4) and continued T4 training if the deadline allows more time. Leave
(3), (5) and (6) as future work.

### UPDATE 05:50: v11 (`t2r7b_0447`) verified: PASS
- **Weights:** sha256 `8ab22c0c…`, identical to `candidates:t2r7b_0447_score51_57.pt`; LFS.
- **`agent.py`:** the `"use"` cooldown (9 steps) masks re-presses only when `"use"` is in `action_mask`, which defaults
  to off, so behaviour is unchanged.
- **Tests:** pytest 11/11.
- **`validate.py` on the i3: PASSED** (test map exited in 52 steps, p50 8.0 ms).
- **Exam 2** (core): +3.8 ± 2.1, unseen real +2.7 ± 1.0, first T4 exit. Agrees with the pooled i3 + CUDA +4.9 ± 2.1
  (`EVAL_I3.md` §13).

### MESSAGE 06:06 from our human (Ayush) → core + teammate 2: everything done by 08:00 (hard limit 08:30 IST)
- v11 stays shipped and verified (teammate 3, 05:50; Docker PASS by teammate 2; repo public).
- Any further improvement must beat v11 on an untouched set before about 07:45, plus validate. **`[core] FINAL` by
  08:00.**

### MESSAGE 06:12 from our human (Ayush) → teammate 2 (free now) + core: final push until 08:30, with proper testing
**Teammate 2, in priority order:**
1. **Demo videos of v11** (for the live demo and judges): side-by-side game + EgoMap, one exiting run per tier T1–T3,
   plus a T4 attempt, on unseen holdout maps, captioned. Push to `origin/demos` with a README line listing steps per
   clip. Done by about 07:00.
2. **Round 7 from v11 → v12 candidate** (T4-heavy mix, warps on), both runs until about 07:20.
   - Push your best to `candidates` at about 07:25.
3. **Exam 3, a new untouched set.** Generate it *now*, but don't evaluate on it before 07:30:
   - 8 generated maps, seeds 2,140,000,000 + 100·tier + k (below 2³¹; disjoint from `fresh` and `exam2`);
   - plus 4 unused Freedoom 1 maps;
   - commit it as `maps/exam3/` with a README of the seeds.
4. **At about 07:30:** run v11 vs your candidate on `maps/exam3`, 20 seeded seeds, using `ship_rule.py` plus weighted
   exits. Post the result.
   - **Ship (core) only if it beats v11 there and Docker validate passes, by 08:00.**

**Teammate 3 (i3):** a crash/robustness soak of v11 now on CPU: all 13 `tests/fixtures` WADs plus long maps, seeded.
Then an i3 second-machine R2 on the v12 candidate at about 07:25.

**Core:** v11 stays FINAL unless v12 wins exam 3. Please confirm or veto this plan in `CHECKPOINTS.md`.

### 06:16 → core: combat/survival mechanics we don't model yet (for the T4-death problem)
1. **Pistol tap-fire.** Organizer notice: holding fire with the pistol spreads shots. The policy samples ATTACK every
   step, so it effectively holds fire.
   - Fix (inference, default off): a `"tapfire"` mask rule; when `vitals[2]` is the pistol slot, no ATTACK on two
     consecutive steps.
   - Training alternative: none needed.
2. **Weapon upgrades.** There's no reward for picking up or switching to the shotgun or chaingun. We train the network
   only on the `next_weapon` head, and the `"weapon"` mask was neutral or harmful.
   - Fix (training): a small bonus when the selected slot is at least 3 with ammo while enemies are visible.
3. **Ammo starvation.** We pay no attention to ammo running out on T4 ("scarce resources"); the agent falls back to
   fists. Ammo pickups only pay the generic +0.15 item reward.
   - Fix (training): a +0.1 bonus for ammo pickups when ammo is under 20%.
4. **Barrels.** `entity_type` 10 explodes and causes self-damage, which the scorer penalizes. We have no signal to keep
   away from barrels or to shoot them only from a distance.
5. **Strafe temperature 0.5 (v8+)** lowers random strafing, which may reduce dodging under fire.
   - **Test, cheap:** on v11 with strafe 1.0 vs 0.5, compare **T4 deaths only** (seeded, 5 seeds), not the overall mean.

**Cheapest to test before 06:25:** #1 (inference rule) and #5 (one eval flag). #2–#4 need training.

### 06:24: gap #5 tested, refuted. Strafe temperature doesn't cause the T4 deaths.
v11 on `gen_t4_1` and `gen_t4_2`, seeds 0–4 (n = 10 each), `--head-temps` with strafe 0.5 vs 1.0
(`i3_t4_s05.csv`, `i3_t4_s10.csv`):

| Strafe | T4 deaths | Exits | Mean score |
|---|---|---|---|
| 0.5 | 8/10 | 0 | 26.3 |
| 1.0 | 8/10 | 0 | 15.9 |

**Keep 0.5.** The T4 deaths come from combat itself, so round 8 (`COMBAT_W`) plus pistol tap-fire (#1) are the right
levers.

### 06:26: v11 robustness soak (CPU, seeded): PASS, 0 problems in 38 episodes
`training/results/t3_checks/soak.py` covers all 13 `tests/fixtures` WADs, the giant MAP11/12/15/23 and Freedoom 1
E2M2/E3M3, at 2 seeds each and 4,200 tics:
- **0 exceptions, 0 safety-net triggers, 0 invalid actions.**
- Worst single step: 93.8 ms (one-off; the 5 s per-act limit is far away). Typical p99 is 12–19 ms.

### 06:30 — `gun` (fa63c7a) and `aim` / `WASTE_W` (7f132c8): rules PASS, 1 HYPOTHESIS
- **Rules: PASS.** `gun` and `aim` read only `vitals[2:4]` and the `visible_entities` mask/type, and remove one option
  each (`RULES.md` §1 allows "action masking from observation fields"). `WASTE_W` is training-only and off by default.
- **HYPOTHESIS: `aim` and `WASTE_W` can block gun-triggered lines.** Doom has *shoot-to-activate* exits (specials 197/198,
  listed in `EXIT_SPECIALS`) and gun-activated doors.
  - `aim` masks ATTACK whenever no enemy is in view, so the agent could never shoot such a switch.
  - `WASTE_W` teaches the same.
  - Generated maps don't use them, so the holdout can't detect the risk; hidden or real maps might.
  - **Suggestion:** ship `gun` alone if it wins. Ship `aim` only if exam 3's real maps show no exit loss.
- The i3 R2 of `gun` on v11 (seeded, 5 seeds) finishes at about 06:35.

## ⚠ ACTION FOR TEAMMATE 2 (Akash), 06:35, from our human. You've been silent since 05:56. Ack in RESULTS.md.
1. **NOW, on your GPU:** train 2 runs from shipped v11 (`submission/weights/best_agent.pt`), round 8:
   - run A: `COMBAT_W=3 WASTE_W=1`
   - run B: `COMBAT_W=2 WASTE_W=2`
   - lr 1e-4, the 06:10 mix from `CHECKPOINTS.md`.
2. **07:05:** push your best to `candidates` with a seeded 5-seed result vs v11: weighted exits plus **T4 deaths**.
3. **While it trains:** v11 demo videos (game + EgoMap, T1–T3 exits plus a T4 attempt) to `origin/demos`, with a README
   line per clip.
4. **07:45:** `validate.py --docker` on the final `main` and post PASS/FAIL. That's the last gate before the 08:00
   deadline.

### FINAL review 07:15 (teammate 3): PASS. Teammate 3 is done.
- **`submission/` is identical to tag `shipped-v11`** (`git diff shipped-v11 HEAD -- submission/` is empty). Weights
  sha256 `8ab22c0c…`.
- **`BRIEF.md` / `BRIEF.pdf`** were updated together in `e27e66d`, so they're in sync, and it's **2 pages**.
  - The RESULTS block is v11, and every number matches the core commits: exam 2 +3.8 ± 2.1 (real +2.7 ± 1.0), exam 1
    +4.9 ± 1.5 (real +5.9 ± 2.3), holdout +3.2 ± 1.8.
  - The T4 limitation reads "1 exit in 40", which is correct. No stale v8 claims remain.
- Earlier evidence for v11: i3 verify (05:50), robustness soak with 0 problems in 38 episodes (06:26), Docker PASS
  (teammate 2).
