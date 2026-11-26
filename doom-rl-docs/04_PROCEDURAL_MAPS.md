# 04 — Procedural Map Factory & Zero-Shot Holdout Suite

## 1. Why Maps Are Half the Battle
The starter template ships only **one test map** (not used in evaluation). An RL agent trained on 1–5 maps will memorize geometry and collapse on hidden maps.

We combine **two complementary map sources** (all compatible with Freedoom 2 IWAD at skill 3):
1. **Custom Python UDMF Generator (`maps/gen_udmf.py`)** — Controlled, fast, guaranteed-solvable Tier 1–3 curriculum maps.
2. **Obsidian CLI (`maps/gen_obsidian.py`)** — Rich, varied, realistic Doom multi-room layouts for Tier 4 gauntlets.

---

## 2. Custom Python UDMF Generator (`maps/gen_udmf.py`)
UDMF (Universal Doom Map Format) is plain text and directly supported by ViZDoom / `doom_env` (Section 4 of PS explicitly confirms: *"The wrapper loads any Doom-format or UDMF WAD"*). Using `wadlib` or `omgifol` (or a clean self-contained UDMF-to-WAD packer + ZDBSP nodebuilder if required by ViZDoom—verify with `doom_env` on 1 generated map first!).

### Parametric Tier Generators
Every generated map is randomized in **room sizes, corridor widths, lighting, textures, enemy placements, ammo/health pickups, and orientation**:

- **Tier 1 (`combat_survival`):**
  - 1–2 medium rooms, cover pillars, 2–6 Freedoom enemies (zombiemen, imps, shotgunners), medkits/ammo, and an exit switch/linedef.
  - *Goal:* Learn strafing, aiming, projectile dodging, and exiting alive.
- **Tier 2 (`multi_room_explore`):**
  - 4–8 interconnected rooms (tree/loop graph via randomized room-grid / BSP), standard push-doors (`Door_Raise`), enemies in 50% of rooms, exit placed in a room at max BFS distance from spawn.
  - *Goal:* Learn door opening, systematic branch exploration, and avoiding revisits.
- **Tier 3 (`keycard_backtrack`):**
  - Branching layout with **Wing A** (contains a **Locked Door** blocking the Exit) and **Wing B** (contains the matching **Keycard** — Red/Blue/Yellow randomized).
  - Crucially: Spawn the player near the junction so they frequently *sight the locked door first*, must explore Wing B to grab the keycard, and **backtrack** to Wing A to unlock the exit!
  - Start with small 3-room Tier 3 maps, scale up to 8-room Tier 3 maps.
- **Tier 4 (`gauntlet`):**
  - Larger 8–14 room layouts with 1–2 keycards, multiple enemy types, traps/ambushes, ammo/health management, plus **Obsidian-generated** Freedoom 2 levels.

---

## 3. Obsidian CLI Integration (`maps/gen_obsidian.py`)
- Repo: `https://github.com/obsidian-level-maker/Obsidian`
- Run Obsidian in batch mode with `--batch` and config files targeting **Freedoom 2**, small-to-medium map sizes, skill 3, varied seeds.
- Generate **100+ diverse WADs** in < 2 minutes into `maps/train_pool/obsidian/`.
- Filter out any map that fails to load or has unreachable exits via a quick automated smoke-test script (`maps/verify_wads.py`).

---

## 4. Strictly Held-Out Zero-Shot Validation Suite (`maps/eval_holdout/`)
Create **12 fixed maps** with dedicated seeds that are **NEVER** seen during training:
- `holdout_t1_{01..03}.wad` — 3 unseen Combat & Survival maps
- `holdout_t2_{01..03}.wad` — 3 unseen Multi-Room Exploration maps
- `holdout_t3_{01..03}.wad` — 3 unseen Keycard → Locked Door → Exit maps
- `holdout_t4_{01..03}.wad` — 3 unseen Obsidian Gauntlet maps

`training/eval_zeroshot.py` runs `DoomAgent` (the exact submission class, no training rewards/hints) on all 12 maps + the official template test map, computing a local proxy of the official tier-weighted leaderboard score:
```text
Local Score = 1.0 * Avg(T1) + 1.5 * Avg(T2) + 2.5 * Avg(T3) + 3.0 * Avg(T4)
```
Only promote a checkpoint to `submission/weights/` when its **12-map zero-shot score** improves AND `python validate.py` passes!
