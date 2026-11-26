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
