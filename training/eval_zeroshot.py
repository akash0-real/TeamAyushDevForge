"""Zero-shot holdout evaluator using the exact submission DoomAgent interface."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import vizdoom as vd
from doom_env import make_env

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))

from agent import DoomAgent

IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")


def eval_map(agent: DoomAgent, wad: str, map_name: str, ticks: int, seeds: list[int]) -> dict:
    exits, kills_l, keys_l, cells_l, hp_l, ms_l = [], [], [], [], [], []
    for seed in seeds:
        env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=False)
        obs, _ = env.reset(seed=seed)
        agent.reset()
        ps = env.privileged_state()
        cells = {(int(ps["x"] // 64.0), int(ps["y"] // 64.0))}
        t_act = 0.0
        steps = 0
        while True:
            t0 = time.perf_counter()
            act = agent.act(obs)
            t_act += time.perf_counter() - t0
            obs, _, term, trunc, info = env.step(act)
            ps = env.privileged_state()
            cells.add((int(ps["x"] // 64.0), int(ps["y"] // 64.0)))
            steps += 1
            if term or trunc:
                break
        exits.append(1.0 if ps["exit_reached"] else 0.0)
        kills_l.append(float(ps["kills"]))
        keys_l.append(float(np.sum(ps["keys"])))
        cells_l.append(float(len(cells)))
        hp_l.append(max(0.0, float(ps["health"])))
        ms_l.append(1000.0 * t_act / max(1, steps))
        env.close()
    return {
        "map": f"{Path(wad).name}:{map_name}",
        "exit": float(np.mean(exits)),
        "kills": float(np.mean(kills_l)),
        "keys": float(np.mean(keys_l)),
        "cells": float(np.mean(cells_l)),
        "hp": float(np.mean(hp_l)),
        "ms": float(np.mean(ms_l)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=["two_rooms", "MAP17", "MAP18", "MAP19", "MAP20"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--ticks", type=int, default=2100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dummy_env = make_env(str(ROOT / "maps" / "two_rooms.wad"), "MAP01", episode_ticks=args.ticks, render_buffers=False)
    agent = DoomAgent(dummy_env.observation_space, dummy_env.action_space, args.device)
    dummy_env.close()

    rows = []
    for item in args.maps:
        if item == "two_rooms":
            wad, m = str(ROOT / "maps" / "two_rooms.wad"), "MAP01"
        elif ":" in item:
            w, m = item.split(":", 1)
            wad = str(Path(w) if Path(w).is_absolute() else ROOT / w)
        else:
            wad, m = IWAD, item.upper()
        r = eval_map(agent, wad, m, args.ticks, args.seeds)
        rows.append(r)
        print(
            f"{r['map']:26s} | exit {r['exit']*100:4.0f}% | kills {r['kills']:4.1f} | "
            f"keys {r['keys']:3.1f} | cells {r['cells']:5.1f} | hp {r['hp']:5.1f} | {r['ms']:.2f} ms/step",
            flush=True,
        )
    print("-" * 92)
    print(
        f"{'MEAN':26s} | exit {np.mean([r['exit'] for r in rows])*100:4.0f}% | "
        f"kills {np.mean([r['kills'] for r in rows]):4.1f} | keys {np.mean([r['keys'] for r in rows]):3.1f} | "
        f"cells {np.mean([r['cells'] for r in rows]):5.1f} | {np.mean([r['ms'] for r in rows]):.2f} ms/step"
    )


if __name__ == "__main__":
    main()
