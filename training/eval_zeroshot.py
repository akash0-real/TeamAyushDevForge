"""Zero-shot holdout evaluator using the exact submission DoomAgent interface."""

from __future__ import annotations

import argparse
import csv
import os
import re
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
DEFAULT_HOLDOUT_MAPS = [
    "two_rooms",
    "MAP25",
    "MAP26",
    "MAP27",
    "MAP28",
    "MAP29",
    "MAP30",
    "MAP31",
    "MAP32",
    "maps/holdout/gen_t1_1.wad:MAP01",
    "maps/holdout/gen_t1_2.wad:MAP01",
    "maps/holdout/gen_t2_1.wad:MAP01",
    "maps/holdout/gen_t2_2.wad:MAP01",
    "maps/holdout/gen_t3_1.wad:MAP01",
    "maps/holdout/gen_t3_2.wad:MAP01",
    "maps/holdout/gen_t4_1.wad:MAP01",
    "maps/holdout/gen_t4_2.wad:MAP01",
]


def infer_tier(wad_path: str) -> int:
    m = re.search(r"gen_t([1-4])_", Path(wad_path).name)
    return int(m.group(1)) if m else 0


def proxy_score(exit, keys, doors, locked_doors, kills, cells, damage, died_early):
    return (
        100.0 * exit
        + 15.0 * keys
        + 5.0 * doors
        + 10.0 * locked_doors
        + 1.0 * kills
        + 0.2 * cells
        - 0.05 * damage
        - 20.0 * died_early
    )


def eval_map(
    agent: DoomAgent, wad: str, map_name: str, ticks: int, seeds: list[int], tier: int
) -> tuple[dict, list[dict]]:
    per_seed: list[dict] = []
    map_label = f"{Path(wad).name}:{map_name}"

    for seed in seeds:
        env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=False)
        try:
            obs, _ = env.reset(seed=seed)
            agent.reset()
            ps = env.privileged_state()
            cells = {(int(ps["x"] // 64.0), int(ps["y"] // 64.0))}
            ceil0 = None if ps["sector_ceiling"] is None else ps["sector_ceiling"].copy()
            opened_doors: set[int] = set()
            doors = 0
            locked_doors = 0
            t_act = 0.0
            steps = 0

            while True:
                t0 = time.perf_counter()
                act = agent.act(obs)
                t_act += time.perf_counter() - t0
                obs, _, term, trunc, _ = env.step(act)
                ps = env.privileged_state()
                cells.add((int(ps["x"] // 64.0), int(ps["y"] // 64.0)))

                if ceil0 is not None and ps["sector_ceiling"] is not None and ps["door_lock"] is not None:
                    ceil_delta = ps["sector_ceiling"] - ceil0
                    door_lock = ps["door_lock"]
                    opened = np.where((door_lock >= 0) & (ceil_delta > 8.0))[0]
                    for sid in opened:
                        sid_int = int(sid)
                        if sid_int not in opened_doors:
                            opened_doors.add(sid_int)
                            if int(door_lock[sid_int]) > 0:
                                locked_doors += 1
                            else:
                                doors += 1

                steps += 1
                if term or trunc:
                    break

            ex = 1.0 if ps["exit_reached"] else 0.0
            keys = float(np.sum(ps["keys"]))
            kills = float(ps["kills"])
            n_cells = float(len(cells))
            damage = float(ps["damage_taken"])
            died_early = 1.0 if (ps["dead"] and ps["tic"] < 0.5 * ticks) else 0.0
            hp = max(0.0, float(ps["health"]))
            ms_per_step = 1000.0 * t_act / max(1, steps)
            score = float(proxy_score(ex, keys, doors, locked_doors, kills, n_cells, damage, died_early))

            per_seed.append(
                {
                    "map": map_label,
                    "tier": tier,
                    "seed": seed,
                    "exit": ex,
                    "keys": keys,
                    "doors": float(doors),
                    "locked_doors": float(locked_doors),
                    "kills": kills,
                    "cells": n_cells,
                    "damage": damage,
                    "died_early": died_early,
                    "hp": hp,
                    "ms_per_step": ms_per_step,
                    "score": score,
                }
            )
        finally:
            env.close()

    agg = {
        "map": map_label,
        "tier": tier,
        "exit": float(np.mean([r["exit"] for r in per_seed])),
        "keys": float(np.mean([r["keys"] for r in per_seed])),
        "doors": float(np.mean([r["doors"] for r in per_seed])),
        "locked_doors": float(np.mean([r["locked_doors"] for r in per_seed])),
        "kills": float(np.mean([r["kills"] for r in per_seed])),
        "cells": float(np.mean([r["cells"] for r in per_seed])),
        "damage": float(np.mean([r["damage"] for r in per_seed])),
        "died_early": float(np.mean([r["died_early"] for r in per_seed])),
        "hp": float(np.mean([r["hp"] for r in per_seed])),
        "ms": float(np.mean([r["ms_per_step"] for r in per_seed])),
        "score": float(np.mean([r["score"] for r in per_seed])),
    }
    return agg, per_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=DEFAULT_HOLDOUT_MAPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ticks", type=int, default=None, help="Override episode ticks for all maps")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", type=str, default=None, help="Optional checkpoint path to load into agent.model")
    ap.add_argument("--temperature", type=float, default=None, help="[core] override agent action temperature")
    ap.add_argument("--name", type=str, default=None, help="Result name for CSV (default: weights stem or 'random')")
    args = ap.parse_args()

    run_name = args.name or (Path(args.weights).stem if args.weights else "random")

    dummy_env = make_env(str(ROOT / "maps" / "two_rooms.wad"), "MAP01", episode_ticks=2100, render_buffers=False)
    agent = DoomAgent(dummy_env.observation_space, dummy_env.action_space, args.device)
    dummy_env.close()

    if args.weights:
        agent.load_weights(args.weights)  # [core] handles plain and EgoMap (--use-map) checkpoints
    if args.temperature is not None:
        agent.temperature = args.temperature

    rows: list[dict] = []
    csv_rows: list[dict] = []
    for item in args.maps:
        if item == "two_rooms":
            wad, m = str(ROOT / "maps" / "two_rooms.wad"), "MAP01"
            tier = 0
            default_ticks = 2100
        elif ":" in item:
            w, m = item.split(":", 1)
            wad = str(Path(w) if Path(w).is_absolute() else ROOT / w)
            tier = infer_tier(wad)
            default_ticks = 2100 if tier > 0 else 4200
        else:
            wad, m = IWAD, item.upper()
            tier = 0
            default_ticks = 4200
        ticks = args.ticks if args.ticks is not None else default_ticks

        try:
            agg, per_seed = eval_map(agent, wad, m, ticks, args.seeds, tier)
            rows.append(agg)
            csv_rows.extend(per_seed)
            print(
                f"[T{tier}] {agg['map']:26s} | score {agg['score']:7.2f} | exit {agg['exit']*100:4.0f}% | "
                f"keys {agg['keys']:3.1f} | doors {agg['doors']:4.1f} | kills {agg['kills']:4.1f} | "
                f"cells {agg['cells']:5.1f} | hp {agg['hp']:5.1f} | {agg['ms']:.2f} ms/step",
                flush=True,
            )
        except Exception as e:
            print(f"[T{tier}] {item:26s} | ERROR: {e}", flush=True)

    print("-" * 121)
    if rows:
        mean_score = float(np.mean([r["score"] for r in rows]))
        print(
            f"{'MEAN':31s} | score {mean_score:7.2f} | exit {np.mean([r['exit'] for r in rows])*100:4.0f}% | "
            f"keys {np.mean([r['keys'] for r in rows]):3.1f} | doors {np.mean([r['doors'] for r in rows]):4.1f} | "
            f"kills {np.mean([r['kills'] for r in rows]):4.1f} | cells {np.mean([r['cells'] for r in rows]):5.1f} | "
            f"{np.mean([r['ms'] for r in rows]):.2f} ms/step",
            flush=True,
        )
        print(f"MEAN score: {mean_score:.2f}", flush=True)
        for t in range(5):
            t_rows = [r for r in rows if r["tier"] == t]
            if t_rows:
                t_score = float(np.mean([r["score"] for r in t_rows]))
                t_exit = float(np.mean([r["exit"] for r in t_rows])) * 100.0
                print(f"T{t} mean score: {t_score:7.2f} (n={len(t_rows)}, exit={t_exit:4.0f}%)", flush=True)
        fd_rows = [r for r in rows if r["map"].startswith("freedoom2.wad:")]
        if fd_rows:
            fd_score = float(np.mean([r["score"] for r in fd_rows]))
            fd_exit = float(np.mean([r["exit"] for r in fd_rows])) * 100.0
            print(f"T0 unseen Freedoom only: {fd_score:7.2f} (n={len(fd_rows)}, exit={fd_exit:4.0f}%)", flush=True)
    else:
        print("MEAN score: N/A (all maps failed)", flush=True)

    results_dir = ROOT / "training" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{run_name}.csv"
    fieldnames = [
        "map", "tier", "seed", "exit", "keys", "doors", "locked_doors",
        "kills", "cells", "damage", "died_early", "hp", "ms_per_step", "score",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved CSV to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
