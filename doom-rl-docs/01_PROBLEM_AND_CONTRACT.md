# 01 — Problem Statement, Scoring & Official Contract

## 1. Environment & Observation Contract (`doom_env`)
- **Engine:** ViZDoom (`vizdoom==1.3.3`) running **Freedoom 2 IWAD** at **skill 3**.
- **Decision Rate:** 1 action every **4 game ticks** (`frame_skip = 4`).
- **Episode Cap:** ~2,100–4,200 ticks per map (~525–1,050 agent decisions per map), fixed random seed per evaluation map.
- **No Global Coordinates:** The observation dictionary NEVER contains global `(x, y)` map coordinates.

### Signals Provided by `doom_env` on Every Step
*(Verify exact dictionary keys, shapes, and dtypes in `devforge-doom-template/docs/CONTRACT.md` once cloned!)*
1. **64 Ego-Centric Raycasts:** Distance + hit type (`wall`, `door`, `locked door`, `enemy`, `item`).
2. **Visible Entities List:** Variable-length list of visible entities in relative ego-centric coordinates (enemies, projectiles, items, keycards).
3. **Vitals Vector:** Health, armor, ammo, keycards held, velocity.
4. **Depth Buffer:** Ego-centric depth map *(optional — start with pure spatial vectors first for max FPS; fuse pooled depth only if needed)*.
5. **Low-Resolution RGB Frame:** *(DO NOT USE unless strictly needed — wastes GPU compute and hurts latency).*

---

## 2. Required Agent Entry Point (`submission/agent.py`)
Must expose a stateful class `DoomAgent`:

```python
class DoomAgent:
    def __init__(self, observation_space, action_space, device):
        """Load model weights ONCE onto `device`. Must finish within 60s."""
        ...

    def reset(self):
        """Clear ALL recurrent/spatial memory before every new map. Must finish within 10s."""
        ...

    def act(self, observation):
        """Return a valid action in `action_space`. Avg <= 40 ms/decision."""
        ...
```

> **CRITICAL:** If `act()` crashes, times out, or returns an invalid action, the player freezes for the rest of the map (keeping points earned so far, still taking death/damage penalties). Always wrap inference in `torch.no_grad()`, validate output shape/type, and guarantee clean `reset()`.

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
3. **Exploration:** Unique map areas visited (tracked privately by the scorer).
4. **Combat:** Enemies killed, **minus** small penalties for damage taken and dying early.

**Leaderboard Score:** Tier-weighted sum across all hidden maps.
- Top 10 teams (after code/rule compliance audit) → Stage 2 Desk Review & Live Demo.
- Top 2 teams → Round 3 Final Judging (heavy weight on **innovative, custom-engineered architectures**).

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
| Subsequent `act()` | 5 s hard / ~40 ms avg | **< 5 ms avg** |
| Repo Size (incl. Git LFS) | ≤ 1 GB | **< 100 MB** |
| Network / Filesystem | Offline, repo mounted read-only | Zero network calls, zero disk writes in `act()` |

---

## 5. Base Image Dependencies (Do NOT submit a Dockerfile)
Pre-installed in organizers' evaluation image:
- Python 3.13, `vizdoom==1.3.3`, `gymnasium`, `torch`, `jax`/`flax`, `numpy`, `scipy`, `opencv`, common ML utilities.
- Keep `submission/requirements.txt` empty or minimal (only pure-Python lightweight packages if truly needed at inference). Training-only packages (`wadlib`, `omgifol`, `tensorboard`, etc.) do NOT need to be in `submission/requirements.txt`.
