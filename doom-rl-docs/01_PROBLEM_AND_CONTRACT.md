# 01 — Problem Statement, Scoring & Official Contract

## 1. Environment & Observation Contract (`doom_env`)
- **Engine:** ViZDoom (`vizdoom==1.3.3`) running **Freedoom 2 IWAD** at **skill 3**.
- **Decision Rate:** 1 action every **4 game ticks** (`frame_skip = 4`).
- **Episode Cap:** ~2,100–4,200 ticks per map (~525–1,050 agent decisions per map), fixed random seed per evaluation map.
- **No Global Coordinates:** The observation dictionary NEVER contains global `(x, y)` map coordinates.

### Observation Dict (verified against template `docs/CONTRACT.md` v`1.0.0-draft` + `env/doom_env/spaces.py`)
| Key | Shape / Space | What it is |
| :--- | :--- | :--- |
| `rays.distance` | `Box(0,1,(64,))` | Hit distance / 1024 (already normalized; `1.0` = nothing hit) |
| `rays.hit_type` | `MultiDiscrete([6]*64)` | `0` none, `1` wall, `2` door, `3` locked_door, `4` enemy, `5` item |
| `visible_entities.features` | `(32, 5) float32` | `[dist/1024, sin(rel_angle), cos(rel_angle), rel_vel_fwd, rel_vel_right]` |
| `visible_entities.entity_type` | `MultiDiscrete([12]*32)` | `pad, enemy_melee, enemy_hitscan, enemy_projectile, projectile, health, armor, ammo, weapon, key, barrel, other` |
| `visible_entities.mask` | `MultiBinary(32)` | `1` = real entity; sorted nearest first |
| `vitals` | `(17,) float32` | see below |
| `depth` | `(120,160) uint8` | Depth buffer *(optional — fuse pooled depth only if needed)* |
| `rgb` | `(120,160,3) uint8` | HUD hidden *(don't use unless strictly needed — latency)* |

Facts that change our design:
- **Rays cover 360°**: index 0 = straight ahead, counter-clockwise (16 = left, 32 = behind), 5.625° apart. Rays stop at anything that blocks *movement* (walls, ledges > 24 high, enemies, items); open doors, corpses and projectiles are transparent.
- **Entities only cover the 90° forward FOV** with line of sight, up to 32. There are **no `(dx, dy)` coords** — only distance + angle (sin/cos) + relative velocity. Enemies behind you show up only as `hit_type == 4` rays.
- **Key colour is not observed** on entities or rays (`key`, `locked_door` only). Which keys we hold *is* observed (vitals 8–13).
- **`vitals` (17):** `0 health (0–200), 1 armor (0–200), 2 selected_weapon (slot 0–7), 3 selected_weapon_ammo, 4–7 ammo bullets/shells/rockets/cells, 8–13 keys (blue/yellow/red card, blue/yellow/red skull; 0/1, sticky), 14–16 vel_forward/vel_right/vel_up` (player frame, units/tic). **There is no yaw / heading / angular velocity.**

### Action Space
`MultiBinary(9)`: `0 MOVE_FORWARD, 1 MOVE_BACKWARD, 2 MOVE_LEFT, 3 MOVE_RIGHT, 4 TURN_LEFT, 5 TURN_RIGHT, 6 ATTACK, 7 USE, 8 SELECT_NEXT_WEAPON`. `act()` must return something convertible to an `np.ndarray` of shape `(9,)` with values in `{0,1}`; anything else ends the episode.
- **`USE` and `SELECT_NEXT_WEAPON` fire on button-down only.** Holding `USE` does nothing extra; the policy must output `USE=0` for ≥1 step to press again.
- **`USE` on a moving door reverses it** — spamming USE in a doorway keeps it bouncing. Press once, wait ~1–2 s.

### Training-only signals
`env.privileged_state()` → `x, y, z, angle, health, kills, items, damage_taken, keys, dead, timeout, exit_reached, tic, sector_floor, sector_ceiling, door_lock` (verified). `env.sector_polygons()` gives sector geometry (the scorer uses it for coverage). Not available at eval. `make_env(..., render_buffers=False)` zeroes `rgb`/`depth` (it did **not** speed things up in our benchmark — rays dominate).

---

## 2. Required Agent Entry Point (`submission/agent.py`)
Must expose a stateful class `DoomAgent`:

```python
class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Load model weights ONCE onto `device` ("cuda" or "cpu"). Must finish within 60s.
        Load from paths relative to Path(__file__).parent; repo is read-only (scratch: /tmp only)."""
        ...

    def reset(self) -> None:
        """Clear ALL recurrent/spatial memory before every new map. Must finish within 10s."""
        ...

    def act(self, observation: dict) -> np.ndarray:
        """Return np.ndarray shape (9,), values in {0,1}. Per-map budget: 40 ms x decision steps."""
        ...
```

> **CRITICAL:** If `act()` crashes, times out, or returns an invalid action, the player stands still for the rest of the map (keeping points earned so far, still taking death/damage penalties). Always wrap inference in `torch.inference_mode()`, validate output shape/type, and guarantee clean `reset()`. Anything printed to stdout/stderr is readable by organizers.

---

## 3. Evaluation Tiers & Scoring Formula
Automated evaluation runs on a hidden suite of unseen maps across 4 tiers (increasing difficulty & weight):
- **Tier 1:** Combat & Survival
- **Tier 2:** Multi-Room Exploration
- **Tier 3:** Long-Horizon Progression (`Keycard → Locked Door → Exit`)
- **Tier 4:** Full Gauntlet

### Non-Binary Per-Map Score Components
1. **Reaching the Exit:** **Largest single reward** + small bonus for remaining health and time.
2. **Progression:** Keycards picked up, doors opened, locked doors unlocked.
3. **Exploration:** Unique map areas visited (spinning in place earns nothing). The scorer computes coverage from **sector polygons** (`env.sector_polygons()`), so our exploration reward should be sector/area-based too.
4. **Combat:** Enemies killed, **minus** small penalties for damage taken and dying early.

**Leaderboard Score:** Tier-weighted sum across all hidden maps (weights are hidden). Maps are played in order of difficulty with fixed seeds. Evaluated code = last commit on the default branch at the deadline.
- Organizers manually review code + replays of the **top 4–5** teams for rule violations (per template `docs/RULES.md`; if the event page says top 10 for Stage 2, follow the event page and note the discrepancy).
- Top 2 teams → final judging, which rewards **innovative, custom-engineered architectures** (custom spatial memory, curiosity, hierarchical control).

---

## 4. Sandbox & Resource Constraints
| Resource / Limit | Official Cap | Our Target Budget |
| :--- | :--- | :--- |
| GPU VRAM | 2 GB (RTX 3090) | **< 400 MB** |
| System RAM | 8 GB | **< 1 GB** |
| CPU Threads | 1 thread | **1 thread (`torch.set_num_threads(1)`)** |
| `__init__` Timeout | 60 s | **< 3 s** |
| `reset()` Timeout | 10 s | **< 5 ms** |
| First `act()` (warm-up) | 10 s | **< 50 ms** |
| Subsequent `act()` | 5 s per call + per-map budget 40 ms × decision steps | **< 5 ms avg** |
| Repo Size (incl. Git LFS) | ≤ 1 GB | **< 100 MB** |
| Network / Filesystem | Offline, repo mounted read-only | Zero network calls, zero disk writes in `act()` |

---

## 5. Base Image Dependencies (scorer uses the organizers' image)
Pre-installed in organizers' evaluation image (exact versions: template `docker/requirements.lock`):
- Python 3.13, `vizdoom==1.3.3`, `gymnasium`, `torch`, `jax`/`flax`, `numpy`, `scipy`, `opencv-python-headless`, `doom_env`.
- Keep `submission/requirements.txt` empty or minimal: `package==version` lines only, max 50, no URLs/`-r`/`-e`. A bad file or failed install = **0 on every map**. Training-only packages (`omgifol`, `tensorboard`, etc.) do NOT go in it.
- Keep the template's `Dockerfile` — it is only used by `validate.py --docker` to mimic the official image.
