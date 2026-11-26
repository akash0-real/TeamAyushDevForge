"""Training-only reward shaping and multi-map wrapper around `doom_env`.

Uses `env.privileged_state()` strictly during training to compute dense
survival-aware progression rewards aligned with docs/EVALUATION.md.
Never imported by `submission/agent.py`.
"""

from __future__ import annotations

import math
import os
import random
import struct
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import vizdoom as vd

if sys.platform == "win32":
    # Windows' default 15.6 ms timer makes several ViZDoom games stall each other (measured 51 -> 380 steps/s
    # for 4 games in one process). Every env worker imports this module, so each process gets 1 ms timers.
    import ctypes

    ctypes.windll.winmm.timeBeginPeriod(1)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "submission") not in sys.path:
    sys.path.insert(0, str(ROOT / "submission"))

if str(ROOT / "maps") not in sys.path:
    sys.path.insert(0, str(ROOT / "maps"))

from doom_env import make_env
from doom_env.wad import map_lumps, read_wad
from doom_env.wrapper import IWAD
from gen_maps import HOLDOUT_SEED_BASE, write_map
from geodesic import GeoField
from preprocess import MAP_CHANNELS, MAP_CROP, NUM_ACTIONS, EgoMap, factored_to_multibinary, preprocess_obs

GEN_CACHE = ROOT / "training" / "gen_cache"
GEN_CACHE.mkdir(parents=True, exist_ok=True)

# Reverse curriculum (training only): on real Freedoom maps, sometimes start the player near the exit and move the
# start further back as it succeeds, so the agent experiences finishing big maps it can't yet cross from the start.
WARP_PROB = 0.5
EXIT_SPECIALS = (11, 51, 52, 124, 197, 198)  # Doom-format exit line specials (switch / walk-over / shoot)
KEY_WARP_PROB = 0.4  # of warped episodes on maps with keys: start near a key instead (teammate 3 Task Y: 0 keys on real maps)
KEY_THINGS = (5, 6, 13, 38, 39, 40)  # blue/yellow/red keycards and skull keys
_WARP_SPOTS: dict[tuple[str, str], tuple[list, list]] = {}


def warp_spots(map_name: str, kind: str = "exit") -> tuple[list, list]:
    """(targets, spawn spots sorted by distance to the nearest target) for an IWAD map; targets are exit line
    midpoints (kind "exit") or key positions (kind "key"). Spawn spots are item and player-start positions: always on
    a reachable floor, never inside a monster."""
    if (map_name, kind) not in _WARP_SPOTS:
        L = map_lumps(read_wad(IWAD), map_name)
        things = [struct.unpack_from("<hhhhh", L["THINGS"], i) for i in range(0, len(L["THINGS"]), 10)]
        targets = []
        if kind == "key":
            targets = [(x, y) for x, y, _a, t, _f in things if t in KEY_THINGS]
        else:
            vx = [struct.unpack_from("<hh", L["VERTEXES"], i) for i in range(0, len(L["VERTEXES"]), 4)]
            for i in range(0, len(L["LINEDEFS"]), 14):
                v1, v2, _fl, sp, _tag, _f, _b = struct.unpack_from("<HHHHhHH", L["LINEDEFS"], i)
                if sp in EXIT_SPECIALS:
                    targets.append(((vx[v1][0] + vx[v2][0]) / 2, (vx[v1][1] + vx[v2][1]) / 2))
        spots = sorted(
            (min(math.hypot(x - tx, y - ty) for tx, ty in targets), x, y)
            for x, y, _a, t, _f in things
            if targets and (2000 <= t <= 2049 or t in (1, 8, 11))
        )
        _WARP_SPOTS[(map_name, kind)] = (targets, spots)
    return _WARP_SPOTS[(map_name, kind)]


def exit_spots(map_name: str) -> tuple[list, list]:
    return warp_spots(map_name, "exit")


class ShapedDoomEnv(gym.Env):
    """Multi-map training wrapper returning preprocessed spatial tensors and shaped rewards."""

    def __init__(
        self,
        map_specs: list[tuple[str, str, int]],
        swap_every: int = 4,
        cell_size: float = 64.0,
        seed: int = 0,
        use_map: bool = False,
    ):
        """
        map_specs: list of (wad_path, map_name, episode_ticks).
        swap_every: number of episodes before switching to another map in `map_specs`.
        """
        super().__init__()
        self.map_specs = list(map_specs)
        self.swap_every = max(1, int(swap_every))
        self.cell_size = float(cell_size)
        self.rng = random.Random(seed)

        self._env = None
        self._warp_radius: dict[str, float] = {}  # per real map: max start distance from the exit (grows on success)
        self._warped = False
        self._geo: GeoField | None = None
        self._geo_failed: set[str] = set()
        self._cur_spec: tuple[str, str, int] | None = None
        self._ep_on_cur = 0
        self._ep_count = 0

        self.ego_map = EgoMap() if use_map else None  # same code + call order as submission/agent.py
        self.observation_space = gym.spaces.Dict(
            {
                "rays": gym.spaces.Box(0.0, 1.0, (7, 64), np.float32),
                "ent_feat": gym.spaces.Box(-1.0, 1.0, (32, 5), np.float32),
                "ent_type": gym.spaces.Box(0, 11, (32,), np.int64),
                "ent_mask": gym.spaces.Box(0, 1, (32,), np.bool_),
                "vitals": gym.spaces.Box(-10.0, 10.0, (30,), np.float32),
                **({"map": gym.spaces.Box(0.0, 1.0, (2 * MAP_CHANNELS, MAP_CROP, MAP_CROP), np.float32)} if use_map else {}),
            }
        )
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3, 2, 2, 2])

        # Per-episode trackers
        self._prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self._prev_ps: dict | None = None
        self._visited_cells: set[tuple[int, int]] = set()
        self._opened_doors: set[int] = set()
        self._init_ceil: np.ndarray | None = None
        self._stats: dict[str, float] = {}

    def _ensure_env(self) -> None:
        multi = len(self.map_specs) > 1 or self.map_specs[0][0].startswith("gen")
        need_swap = self._env is None or (multi and self._ep_on_cur >= self.swap_every)
        if not need_swap:
            return
        spec = self.rng.choice(self.map_specs)
        if self._env is not None and spec == self._cur_spec and not spec[0].startswith("gen"):
            self._ep_on_cur = 0
            return
        if self._env is not None:
            self._env.close()
            self._env = None
        wad, map_name, ticks = spec
        self._tier = int(wad[3]) if wad.startswith("gen") else 0  # 0 = real map (Freedoom / template)
        if self._tier:  # brand-new procedural map; seeds < 1e9 (>= 1e9 reserved for holdout)
            wad = str(GEN_CACHE / f"{os.getpid()}.wad")
            write_map(self._tier, self.rng.randrange(HOLDOUT_SEED_BASE), wad)
        ticks = ticks or self.rng.randint(2100, 4200)  # evaluation caps vary per map in this range
        self._env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=False)
        self._geo = None  # round 6: rebuilt for the new level at the next reset
        self._cur_spec = spec
        self._ep_on_cur = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng.seed(seed)
        self._ensure_env()
        self._ep_on_cur += 1
        self._ep_count += 1
        ep_seed = self.rng.randint(0, 2**30 - 1)
        raw_obs, info = self._env.reset(seed=ep_seed)
        self._warped = False  # False, "exit" or "key"
        wad, map_name, _ = self._cur_spec
        if wad == IWAD and self.rng.random() < WARP_PROB:
            kind = "key" if warp_spots(map_name, "key")[1] and self.rng.random() < KEY_WARP_PROB else "exit"
            _targets, spots = warp_spots(map_name, kind)
            if spots:
                radius = self._warp_radius.setdefault(f"{map_name}:{kind}", 768.0)
                near = [sp for sp in spots if sp[0] <= radius] or spots[:3]
                _d, x, y = self.rng.choice(near)
                g = self._env._game
                tic0 = int(info["tic"])
                g.send_game_command("sv_cheats 1")
                g.send_game_command(f"warp {x} {y}")
                raw_obs, _, term, trunc, info = self._env.step(np.zeros(NUM_ACTIONS, dtype=np.int8))  # 1 no-op step
                if term or trunc:  # shouldn't happen; fall back to a normal start
                    raw_obs, info = self._env.reset(seed=ep_seed)
                else:
                    self._warped = kind
                    self._warp_tics = int(info["tic"]) - tic0
        ps = self._env.privileged_state()

        self._prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self._prev_ps = ps
        self._prev_dmg_dealt = self._damage_dealt()
        self._visited_cells = {(int(ps["x"] // self.cell_size), int(ps["y"] // self.cell_size))}
        self._opened_doors = set()
        self._key_seen = False
        self._init_ceil = None if ps["sector_ceiling"] is None else ps["sector_ceiling"].copy()
        self._stats = {
            "exit_reached": 0.0,
            "dead": 0.0,
            "kills": 0.0,
            "keys": 0.0,
            "doors": 0.0,
            "locked_doors": 0.0,
            "cells": 1.0,
            "damage_taken": 0.0,
            "steps": 0.0,
            "return": 0.0,
            # T5 / T6 in the log = real map started near the exit / near a key
            "tier": {"exit": 5.0, "key": 6.0}.get(self._warped, float(self._tier)),
            "geo": 0.0,
        }
        # round 6: geodesic progress shaping (training/geodesic.py); a level it can't parse just gets no shaping
        level = f"{self._env.wad}:{self._cur_spec[1]}"
        if self._geo is None and level not in self._geo_failed and ps["sector_floor"] is not None:
            try:
                self._geo = GeoField(self._env.wad, self._cur_spec[1], ps["sector_floor"], ps["sector_ceiling"])
            except Exception:  # noqa: BLE001
                if self._tier == 0:  # generated maps reuse one file name per worker; only remember real-map failures
                    self._geo_failed.add(level)
        if self._geo is not None:
            self._geo_src = self._geo.target(ps["keys"])
            self._phi = self._geo.phi(ps["x"], ps["y"], self._geo_src)
        if self.ego_map is not None:
            self.ego_map.reset()
            self.ego_map.update(raw_obs, None)
            if self._warped:
                self.ego_map.tic = self._warp_tics  # the no-op step used game tics; keep the turn dead-time model exact
        return self._encode(raw_obs), info

    def _encode(self, raw_obs: dict) -> dict:
        enc = preprocess_obs(raw_obs, self._prev_action)
        if self.ego_map is not None:
            enc["map"] = self.ego_map.crops()
        return enc

    def _damage_dealt(self) -> float:
        # Training-only: privileged_state() has no damage-dealt field, so read the engine counter directly.
        return float(self._env._game.get_game_variable(vd.GameVariable.DAMAGECOUNT))

    def step(self, action):
        mb_action = factored_to_multibinary(action)
        raw_obs, _, terminated, truncated, info = self._env.step(mb_action)
        ps = self._env.privileged_state()
        prev = self._prev_ps

        reward = 0.0

        # 1. Exploration: unique 64x64 grid cells visited (capped at 600 cells/ep -> max +24, below the exit)
        cell = (int(ps["x"] // self.cell_size), int(ps["y"] // self.cell_size))
        if cell not in self._visited_cells and len(self._visited_cells) < 600:
            self._visited_cells.add(cell)
            reward += 0.04
            self._stats["cells"] += 1.0

        # 2. Door opening & locked door unlocking
        if self._init_ceil is not None and ps["sector_ceiling"] is not None and ps["door_lock"] is not None:
            ceil_delta = ps["sector_ceiling"] - self._init_ceil
            door_lock = ps["door_lock"]
            opened = np.where((door_lock >= 0) & (ceil_delta > 8.0))[0]
            for sid in opened:
                sid_int = int(sid)
                if sid_int not in self._opened_doors:
                    self._opened_doors.add(sid_int)
                    if int(door_lock[sid_int]) > 0:
                        reward += 4.0
                        self._stats["locked_doors"] += 1.0
                    else:
                        reward += 0.8
                        self._stats["doors"] += 1.0

        # 3. Keycard pickup (+ a one-off nudge the first time a key is visible in the official observation)
        ent = raw_obs["visible_entities"]
        if not self._key_seen and np.any((np.asarray(ent["mask"]) > 0) & (np.asarray(ent["entity_type"]) == 9)):
            self._key_seen = True
            reward += 0.5
        new_keys = int(np.sum(ps["keys"]) - np.sum(prev["keys"]))
        if new_keys > 0:
            reward += 8.0 * new_keys  # tutor: was 3.0; T3 stalled at ~0.1 keys/episode
            self._stats["keys"] += float(new_keys)

        # 4. Combat & survival
        new_kills = max(0, int(ps["kills"] - prev["kills"]))
        if new_kills > 0:
            reward += 1.0 * new_kills
            self._stats["kills"] += float(new_kills)

        new_items = max(0, int(ps["items"] - prev["items"]))
        if new_items > 0:
            reward += 0.15 * new_items

        # Damage dealt to enemies: dense aiming signal (imp = 60 hp -> +0.6 total)
        if not (terminated or truncated):  # engine state is gone after the episode ends
            dealt = self._damage_dealt()
            reward += 0.01 * max(0.0, dealt - self._prev_dmg_dealt)
            self._prev_dmg_dealt = dealt

        # Health pickups while hurt (the scorer's exit bonus counts remaining health)
        hp_gain = float(ps["health"] - prev["health"])
        if hp_gain > 0.0 and prev["health"] < 100.0:
            reward += 0.025 * hp_gain  # tutor round 4: was 0.01

        dmg = max(0.0, float(ps["damage_taken"] - prev["damage_taken"]))
        if dmg > 0.0:
            reward -= 0.03 * dmg  # tutor round 4: was 0.015 (a kill paid more than tanking its hits cost)
            self._stats["damage_taken"] += dmg

        # 4b. Round 6 geodesic progress: Phi(s') - Phi(s), Phi = -walking distance to the next missing key / exit.
        # When a key pickup switches the goal, Phi(s) is re-evaluated for the new goal, so the switch itself pays 0.
        if self._geo is not None:
            src = self._geo.target(ps["keys"])
            if src != self._geo_src:
                self._geo_src = src
                self._phi = self._geo.phi(prev["x"], prev["y"], src)
            phi = self._geo.phi(ps["x"], ps["y"], src)
            if phi is not None and self._phi is not None:
                # no per-step cap: Phi is a function of position, so any loop (teleporters included) sums to exactly 0
                # (teammate 3 review 23:40: a cap let a teleport away cost <= 0.1 while walking back paid in full)
                f = phi - self._phi
                reward += f
                self._stats["geo"] += f
            if phi is not None:  # keep the last defined Phi through BFS-unreachable cells, so A -> U -> B pays Phi_B - Phi_A
                self._phi = phi  # (teammate 3 review 23:55: resetting to None let a teleport detour skip the loss)

        # 5. Terminal outcomes (no negative step penalty -> prevents early suicide)
        if ps["exit_reached"]:
            # tutor round 4: the scorer's exit bonus counts remaining health and time, so reward fast, healthy exits
            hp_frac = max(0.0, min(1.0, ps["health"] / 100.0))
            time_left = max(0.0, 1.0 - ps["tic"] / ps["episode_ticks"])
            reward += 25.0 + 6.0 * hp_frac + 6.0 * time_left
            self._stats["exit_reached"] = 1.0
        elif ps["dead"]:
            # tutor round 5c (teammate 3 Task Y: 18/20 real-map runs die, median tic 920): the scorer penalises dying
            # *early*, so an early death costs up to -15, a late one -5. No per-step penalty, so dying never pays.
            time_left = max(0.0, 1.0 - ps["tic"] / ps["episode_ticks"])
            reward -= 5.0 + 10.0 * time_left
            self._stats["dead"] = 1.0

        self._stats["steps"] += 1.0
        self._stats["return"] += reward
        self._prev_ps = ps
        self._prev_action = mb_action

        if terminated or truncated:
            if self._warped:  # move the start further back after a success, closer after a failure
                m = f"{self._cur_spec[1]}:{self._warped}"
                won = ps["exit_reached"] if self._warped == "exit" else self._stats["keys"] > 0
                r = self._warp_radius[m]
                self._warp_radius[m] = min(r * 1.25, 12000.0) if won else max(r * 0.95, 512.0)
            info = dict(info)
            info["episode_stats"] = dict(self._stats)
            info["map"] = f"{Path(self._cur_spec[0]).name}:{self._cur_spec[1]}"

        if self.ego_map is not None and not (terminated or truncated):
            self.ego_map.update(raw_obs, mb_action)
        return self._encode(raw_obs), float(reward), terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
