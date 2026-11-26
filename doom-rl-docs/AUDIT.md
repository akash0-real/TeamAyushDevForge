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
