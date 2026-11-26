"""Inspect Freedoom 2 maps + template maps and benchmark real doom_env throughput."""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import vizdoom as vd
from doom_env import make_env
from doom_env.mapinfo import load_mapinfo

IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
ROOT = Path(__file__).resolve().parent.parent


def inspect_map(wad: str, map_name: str) -> dict | None:
    try:
        mi = load_mapinfo(wad, map_name)
    except Exception as e:
        return {"wad": Path(wad).name, "map": map_name, "error": str(e)}
    if mi is None:
        return None
    n_doors = sum(1 for v in mi.door_sectors.values() if v == 0)
    n_locked = sum(1 for v in mi.door_sectors.values() if v > 0)
    return {
        "wad": Path(wad).name,
        "map": map_name,
        "sectors": mi.num_sectors,
        "doors": n_doors,
        "locked_doors": n_locked,
        "under_limit": mi.num_sectors <= 2048,
    }


def benchmark_fps(wad: str, map_name: str, steps: int = 500) -> float:
    env = make_env(wad, map_name, episode_ticks=2100, render_buffers=False)
    obs, _ = env.reset(seed=0)
    act = np.zeros(9, dtype=np.int8)
    act[0] = 1
    t0 = time.perf_counter()
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            obs, _ = env.reset(seed=0)
    dt = time.perf_counter() - t0
    env.close()
    return steps / dt


def main() -> None:
    warnings.simplefilter("ignore", RuntimeWarning)
    print(f"Freedoom 2 IWAD: {IWAD}\n")

    print("=== Template Map ===")
    two = str(ROOT / "maps" / "two_rooms.wad")
    print(inspect_map(two, "MAP01"))
    fps = benchmark_fps(two, "MAP01", steps=500)
    print(f"Single-env FPS (two_rooms, render_buffers=False): {fps:.1f} steps/sec\n")

    print("=== Freedoom 2 Built-in Maps (MAP01..MAP32) ===")
    good = []
    for i in range(1, 33):
        m = f"MAP{i:02d}"
        info = inspect_map(IWAD, m)
        if info and not info.get("error"):
            tag = "OK" if info["under_limit"] else "TOO_BIG"
            print(f"  {m}: sectors={info['sectors']:4d}  doors={info['doors']:2d}  locked={info['locked_doors']:2d}  [{tag}]")
            if info["under_limit"]:
                good.append(m)
    print(f"\nUsable Freedoom 2 maps (<= 2048 sectors): {len(good)}/32 -> {good}")
    if good:
        fps_fd = benchmark_fps(IWAD, good[0], steps=500)
        print(f"Single-env FPS (freedoom2 {good[0]}, render_buffers=False): {fps_fd:.1f} steps/sec")


if __name__ == "__main__":
    main()
