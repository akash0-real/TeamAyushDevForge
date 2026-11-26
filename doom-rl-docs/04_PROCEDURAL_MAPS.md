# 04 — Procedural Map Factory & Zero-Shot Holdout Suite

## 1. Why Maps Are Half the Battle
The starter template ships one test map (`maps/two_rooms.wad`: one door, one monster, an exit) plus 13 tiny grid-layout WADs in `tests/fixtures/`. None of these are evaluation maps. They're good for smoke tests, not for training. An RL agent trained on 1–5 maps will memorize geometry and collapse on hidden maps.

**Hard constraint for every generated map:** ViZDoom reports at most **2,048 sectors**; beyond that, rays and line-of-sight miss walls/doors (the wrapper warns). Evaluation maps stay under the limit, so reject any training map over it.

We combine **two complementary map sources** (all compatible with Freedoom 2 IWAD at skill 3):
1. **Custom Python UDMF Generator (`maps/gen_udmf.py`)** — Controlled, fast, guaranteed-solvable Tier 1–3 curriculum maps.
2. **Obsidian CLI (`maps/gen_obsidian.py`)** — Rich, varied, realistic Doom multi-room layouts for Tier 4 gauntlets.

---

## 2. Custom Python UDMF Generator (`maps/gen_udmf.py`)
UDMF (Universal Doom Map Format) is plain text and supported by ViZDoom / `doom_env`. Template `CONTRACT.md` §7: `make_env` accepts any ViZDoom-compatible WAD, and door / locked-door ray types are detected from linedef specials in Doom, Boom, Hexen and UDMF formats. Pack with `omgifol` (or a self-contained UDMF-to-WAD packer, plus ZDBSP if ViZDoom won't build nodes itself). Verify with `doom_env` on 1 generated map first: check that rays report `door`/`locked_door` on your door linedefs.

### Parametric Tier Generators
Every generated map is randomized in **room sizes, corridor widths, lighting, textures, enemy placements, ammo/health pickups, and orientation**:

- **Tier 1 (`combat_survival`):**
  - 1–2 medium rooms, cover pillars, 2–6 Freedoom enemies (zombiemen, imps, shotgunners), medkits/ammo, and an exit switch/linedef.
  - *Goal:* Learn strafing, aiming, projectile dodging, and exiting alive.
- **Tier 2 (`multi_room_explore`):**
  - 4–8 interconnected rooms (tree/loop graph via randomized room-grid / BSP), standard push-doors (`Door_Raise`), enemies in 50% of rooms, exit placed in a room at max BFS distance from spawn.
  - *Goal:* Learn door opening, systematic branch exploration, and avoiding revisits.
- **Tier 3 (`keycard_backtrack`):**
  - Branching layout with **Wing A** (contains a **Locked Door** blocking the Exit) and **Wing B** (contains the matching **Key**: randomize over all 6 the vitals track, i.e. blue/yellow/red **cards and skulls**).
  - Crucially: Spawn the player near the junction so they frequently *sight the locked door first*, must explore Wing B to grab the keycard, and **backtrack** to Wing A to unlock the exit!
  - Start with small 3-room Tier 3 maps, scale up to 8-room Tier 3 maps.
- **Tier 4 (`gauntlet`):**
  - Larger 8–14 room layouts with 1–2 keycards, multiple enemy types, traps/ambushes, ammo/health management, plus **Obsidian-generated** Freedoom 2 levels.

---

## 3. Obsidian CLI Integration (`maps/gen_obsidian.py`)
- Repo: `https://github.com/obsidian-level-maker/Obsidian`
- Run Obsidian in batch mode with `--batch` and config files targeting **Freedoom 2**, small-to-medium map sizes, single-level output, varied seeds. *(CLI flags and per-map generation time are unverified; time 5 maps before promising 100+.)*
- Generate **100+ diverse WADs** into `maps/train_pool/obsidian/`.
- `maps/verify_wads.py` filters out maps that fail to load in `doom_env`, exceed 2,048 sectors, or have no exit linedef. A smoke test can't prove an exit is *reachable* (a random agent won't find it). For UDMF maps the generator already knows the room graph; for Obsidian maps, flag maps where the PPO agent's exit rate stays at 0 and drop them from PLR.

---

## 4. Strictly Held-Out Zero-Shot Validation Suite (`maps/eval_holdout/`)
Create **12 fixed maps** with dedicated seeds that are **NEVER** seen during training:
- `holdout_t1_{01..03}.wad` — 3 unseen Combat & Survival maps
- `holdout_t2_{01..03}.wad` — 3 unseen Multi-Room Exploration maps
- `holdout_t3_{01..03}.wad` — 3 unseen Keycard → Locked Door → Exit maps
- `holdout_t4_{01..03}.wad` — 3 unseen Obsidian Gauntlet maps

`training/eval_zeroshot.py` runs `DoomAgent` (the exact submission class, no training rewards/hints) on all 12 maps + the official template test map, computing a local proxy of the official tier-weighted leaderboard score. The official weights and per-category points are hidden; these are our guesses:
```text
Per-map score = exit (largest) + keys/doors + sectors visited + kills − damage − early-death penalty
Local Score   = 1.0 * Avg(T1) + 1.5 * Avg(T2) + 2.5 * Avg(T3) + 3.0 * Avg(T4)
```
Also run the same checkpoint with `python validate.py --map maps/eval_holdout/<map>.wad:MAP01` to catch latency/memory failures on big maps.
Only promote a checkpoint to `submission/weights/` when its **12-map zero-shot score** improves AND `python validate.py` passes!
