# 04 — Procedural Map Factory & Zero-Shot Holdout Suite (as built)

## 1. Why Maps Are Half the Battle
The template ships one test map (`maps/two_rooms.wad`: one door, one monster, an exit) plus 13 tiny grid-layout WADs
in `tests/fixtures/`. None of these are evaluation maps; they're good for smoke tests, not for training. An RL agent
trained on 1–5 maps memorizes the geometry and collapses on hidden maps.

So we train on **two sources**:
1. **Our own infinite procedural generator** (`maps/gen_maps.py`) for tiers 1–4. Every map load creates a brand-new
   map.
2. **Real Freedoom 2 maps** (MAP01–24) plus `two_rooms`, which give realistic, varied layouts.

**Hard constraint for every map:** ViZDoom reports at most **2,048 sectors**. Beyond that, rays and line of sight miss
walls and doors (the wrapper warns). Evaluation maps stay under the limit.

---

## 2. Procedural Generator (`maps/gen_maps.py`)
- **Format:** node-less UDMF WADs (namespace `zdoom`), the same format as the template's own test maps. Packing uses
  the official `doom_env.wad` writer.
- **Layout:** rooms on a grid joined by 2-cell-wide corridors, plus doors, enemies and pickups. Enemies are Freedoom 2
  zombiemen, shotgun guys and imps, plus demons in harder maps. Pickups are stimpacks, medikits, clips, shells and
  green armor.

### Tiers
- **Tier 1 (combat):** 1–2 rooms, several enemies, and an exit. *Goal:* aiming, dodging and exiting alive.
- **Tier 2 (exploration):** 4–7 rooms in a tree with loops, doors, enemies in some rooms, and the exit in the farthest
  room. *Goal:* opening doors, exploring branches systematically, avoiding revisits.
- **Tier 3 (keys):** like tier 2, but the exit room is behind a locked keycard door and its key is in another branch,
  which forces backtracking. Tutor round 2 made 35% of tier-3 maps small (3 rooms).
- **Tier 4 (gauntlet):** 7–11 rooms, two locks (key 1 is itself behind lock 2), more enemies and fewer pickups.

Keys are **keycards only** (red, blue, yellow; ZDoom locks 1–3). Skull keys are untested in `doom_env`, so they're
left out.

### Solvability proof
`python maps/gen_maps.py --selftest` checks, for every generated map:
- every key is reachable;
- the exit is reachable **only after unlocking**;
- the map loads in the official `doom_env` with exactly the doors and locks the generator placed.

Tutor round 2 found and fixed a bug where chain layouts produced tier-3 maps with no lock.

```powershell
python maps/gen_maps.py --tier 3 --seed 1 --out my.wad     # one map
python maps/gen_maps.py --selftest                         # solvability + loads in doom_env
```

---

## 3. Strictly Held-Out Zero-Shot Validation Suite
None of these maps is ever trained on:
- **Freedoom 2 MAP25–32**: real maps. Training uses MAP01–24.
- **8 generated maps** in `maps/holdout/` (`gen_t{1..4}_{1,2}.wad`), from **reserved seeds ≥ 1e9**. Training only
  ever uses seeds below 1e9.

**`two_rooms` is not held out.** The official test map is in the training mix (`training/train_ppo.py`,
`DEFAULT_TRAIN_MAPS`), but `eval_zeroshot.py` also plays it, and its T0 tier mixes it with MAP25–32. On MAP25–32
alone, both shipped checkpoints have 0% exits over 10 seeds, so every T0 exit comes from `two_rooms`. Always report
MAP25–32 separately as the truly unseen real maps (`internal/AUDIT.md`, Strict review item 4).

`training/eval_zeroshot.py` runs the real `DoomAgent` from `submission/agent.py` on these maps, with no training
rewards or hints. It reports a **proxy score** per map:

```text
proxy = 100·exit + 15·keys + 5·doors + 10·locked_doors + 1·kills + 0.2·cells − 0.05·damage − 20·died_early
```

It also prints per-tier means. The official weights and per-category points are hidden; this is our guess at their
shape. Results are in `doom-rl-docs/RESULTS.md` and the brief.

---

## Not implemented (future work)
- **Obsidian CLI pipeline** (`maps/gen_obsidian.py`) for richer tier-4 maps. Our own generator makes tiers 1–4, and
  real Freedoom maps supply realistic layouts.
- **Separate `gen_udmf.py` + `verify_wads.py` scripts.** Both jobs are in `maps/gen_maps.py`: generation, plus a
  self-test that checks solvability and that maps load.
- **The planned 12-map holdout of our own maps** (3 per tier, in `maps/eval_holdout/`). We replaced it with Freedoom
  MAP25–32 + 8 generated maps (plus `two_rooms`, which is a training map), because real maps are a better stand-in for the hidden ones.
- **A tier-weighted local score** (`1.0·T1 + 1.5·T2 + 2.5·T3 + 3.0·T4`). `eval_zeroshot.py` reports the unweighted
  mean plus per-tier means.
- **Skull keys** (locks 129–131) in generated maps.
- **Cover pillars and traps/ambushes** in generated maps.
