"""Training-only reward shaping and multi-map wrapper around `doom_env`.

Uses `env.privileged_state()` strictly during training to compute dense
survival-aware progression rewards aligned with docs/EVALUATION.md.
Never imported by `submission/agent.py`.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "submission") not in sys.path:
    sys.path.insert(0, str(ROOT / "submission"))

from doom_env import make_env
from preprocess import NUM_ACTIONS, factored_to_multibinary, preprocess_obs


class ShapedDoomEnv(gym.Env):
    """Multi-map training wrapper returning preprocessed spatial tensors and shaped rewards."""

    def __init__(
        self,
        map_specs: list[tuple[str, str, int]],
        swap_every: int = 4,
        cell_size: float = 64.0,
        seed: int = 0,
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
        self._cur_spec: tuple[str, str, int] | None = None
        self._ep_on_cur = 0
        self._ep_count = 0

        self.observation_space = gym.spaces.Dict(
            {
                "rays": gym.spaces.Box(0.0, 1.0, (7, 64), np.float32),
                "ent_feat": gym.spaces.Box(-1.0, 1.0, (32, 5), np.float32),
                "ent_type": gym.spaces.Box(0, 11, (32,), np.int64),
                "ent_mask": gym.spaces.Box(0, 1, (32,), np.bool_),
                "vitals": gym.spaces.Box(-10.0, 10.0, (30,), np.float32),
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
        need_swap = self._env is None or (len(self.map_specs) > 1 and self._ep_on_cur >= self.swap_every)
        if not need_swap:
            return
        spec = self.rng.choice(self.map_specs)
        if self._env is not None and spec == self._cur_spec:
            self._ep_on_cur = 0
            return
        if self._env is not None:
            self._env.close()
        wad, map_name, ticks = spec
        self._env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=False)
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
        ps = self._env.privileged_state()

        self._prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self._prev_ps = ps
        self._visited_cells = {(int(ps["x"] // self.cell_size), int(ps["y"] // self.cell_size))}
        self._opened_doors = set()
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
        }
        return preprocess_obs(raw_obs, self._prev_action), info

    def step(self, action):
        mb_action = factored_to_multibinary(action)
        raw_obs, _, terminated, truncated, info = self._env.step(mb_action)
        ps = self._env.privileged_state()
        prev = self._prev_ps

        reward = 0.0

        # 1. Exploration: unique 64x64 grid cells visited (capped at 200 cells/ep)
        cell = (int(ps["x"] // self.cell_size), int(ps["y"] // self.cell_size))
        if cell not in self._visited_cells and len(self._visited_cells) < 200:
            self._visited_cells.add(cell)
            reward += 0.08
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

        # 3. Keycard pickup
        new_keys = int(np.sum(ps["keys"]) - np.sum(prev["keys"]))
        if new_keys > 0:
            reward += 3.0 * new_keys
            self._stats["keys"] += float(new_keys)

        # 4. Combat & survival
        new_kills = max(0, int(ps["kills"] - prev["kills"]))
        if new_kills > 0:
            reward += 1.0 * new_kills
            self._stats["kills"] += float(new_kills)

        new_items = max(0, int(ps["items"] - prev["items"]))
        if new_items > 0:
            reward += 0.15 * new_items

        dmg = max(0.0, float(ps["damage_taken"] - prev["damage_taken"]))
        if dmg > 0.0:
            reward -= 0.015 * dmg
            self._stats["damage_taken"] += dmg

        # 5. Terminal outcomes (no negative step penalty -> prevents early suicide)
        if ps["exit_reached"]:
            hp_bonus = 2.0 * max(0.0, min(1.0, ps["health"] / 100.0))
            reward += 15.0 + hp_bonus
            self._stats["exit_reached"] = 1.0
        elif ps["dead"]:
            reward -= 3.0
            self._stats["dead"] = 1.0

        self._stats["steps"] += 1.0
        self._stats["return"] += reward
        self._prev_ps = ps
        self._prev_action = mb_action

        if terminated or truncated:
            info = dict(info)
            info["episode_stats"] = dict(self._stats)
            info["map"] = f"{Path(self._cur_spec[0]).name}:{self._cur_spec[1]}"

        return preprocess_obs(raw_obs, self._prev_action), float(reward), terminated, truncated, info

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
