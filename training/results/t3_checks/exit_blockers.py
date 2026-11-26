"""Task Y (teammate 3): what stops the shipped agent from reaching exits on real Freedoom maps?
Part 1: map facts from the IWAD. Part 2: shipped DoomAgent at 4,200 tics, per-episode exit-approach diagnostics.
usage: python training/results/t3_checks/exit_blockers.py [seeds...]   (default seeds 0 1)
Writes training/results/t3_checks/exit_blockers.csv and prints a summary."""
import csv, math, struct, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "env"), str(ROOT / "training")]
import numpy as np  # noqa: E402
from doom_env import make_env  # noqa: E402
from doom_env.mapinfo import load_mapinfo  # noqa: E402
from doom_env.wad import map_lumps, read_wad  # noqa: E402
from doom_env.wrapper import IWAD  # noqa: E402
from agent import DoomAgent  # noqa: E402

EXIT_KIND = {11: "switch", 51: "switch(secret)", 52: "walk", 124: "walk(secret)", 197: "shoot", 198: "shoot(secret)"}
KEY_TYPES = (5, 6, 13, 38, 39, 40)
CELL = 64.0


def map_facts(m):
    L = map_lumps(read_wad(IWAD), m)
    vx = [struct.unpack_from("<hh", L["VERTEXES"], i) for i in range(0, len(L["VERTEXES"]), 4)]
    exits = []
    for i in range(0, len(L["LINEDEFS"]), 14):
        v1, v2, _fl, sp, _tag, _f, _b = struct.unpack_from("<HHHHhHH", L["LINEDEFS"], i)
        if sp in EXIT_KIND:
            exits.append(((vx[v1][0] + vx[v2][0]) / 2, (vx[v1][1] + vx[v2][1]) / 2, EXIT_KIND[sp]))
    things = [struct.unpack_from("<hhhhh", L["THINGS"], i) for i in range(0, len(L["THINGS"]), 10)]
    start = next(((x, y) for x, y, _a, t, _f in things if t == 1), None)
    keys = sum(1 for *_, t, _f in things if t in KEY_TYPES)
    locks = sum(1 for v in load_mapinfo(IWAD, m).door_sectors.values() if v > 0)
    normal = [e for e in exits if "secret" not in e[2]] or exits
    dist = min(math.hypot(start[0] - x, start[1] - y) for x, y, _k in normal) if (start and normal) else float("nan")
    kinds = "/".join(sorted({k for *_, k in normal})) or "none"
    return normal, dict(map=m, exit=kinds, start_to_exit=round(dist), keys=keys, locks=locks)


def run_episode(agent, m, seed, exits):
    env = make_env(IWAD, m, episode_ticks=4200, render_buffers=False)
    obs, _ = env.reset(seed=seed); agent.reset()
    cells, closest, last_new_tic, term = set(), float("inf"), 0, False
    ps = env.privileged_state()
    while True:
        c = (int(ps["x"] // CELL), int(ps["y"] // CELL))
        if c not in cells:
            cells.add(c); last_new_tic = int(ps["tic"])
        closest = min(closest, min(math.hypot(ps["x"] - x, ps["y"] - y) for x, y, _k in exits))
        obs, _, term, trunc, _ = env.step(agent.act(obs))
        ps = env.privileged_state()
        if term or trunc:
            break
    outcome = "exit" if ps["exit_reached"] else ("death" if ps["dead"] else "timeout")
    env.close()
    return dict(seed=seed, outcome=outcome, closest=round(closest), last_new_cell_tic=last_new_tic,
                end_tic=int(ps["tic"]), cells=len(cells), kills=int(ps["kills"]), keys=int(np.sum(ps["keys"])))


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1]
    facts = {m: map_facts(m) for m in [f"MAP{i:02d}" for i in range(1, 33) if i != 30]}
    print("map    exit         start->exit  keys locks")
    for m, (_e, f) in facts.items():
        print(f"{m}  {f['exit']:12s} {f['start_to_exit']:10d}  {f['keys']:4d} {f['locks']:5d}")
    probe = ["MAP25", "MAP26", "MAP27", "MAP28", "MAP29", "MAP31", "MAP32", "MAP02", "MAP05", "MAP08"]
    dummy = make_env(str(ROOT / "maps" / "two_rooms.wad"), "MAP01", episode_ticks=2100, render_buffers=False)
    agent = DoomAgent(dummy.observation_space, dummy.action_space, "cpu"); dummy.close()
    rows = []
    for m in probe:
        exits, f = facts[m]
        for s in seeds:
            r = {**f, **run_episode(agent, m, s, exits)}
            r["stalled_tics"] = r["end_tic"] - r["last_new_cell_tic"]
            rows.append(r); print(r, flush=True)
    out = ROOT / "training" / "results" / "t3_checks" / "exit_blockers.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    assert len(rows) == len(probe) * len(seeds)
    n = len(rows)
    print(f"\nSUMMARY n={n}: exits {sum(r['outcome']=='exit' for r in rows)}, deaths {sum(r['outcome']=='death' for r in rows)}, "
          f"timeouts {sum(r['outcome']=='timeout' for r in rows)}; stalled>1000 tics {sum(r['stalled_tics']>1000 for r in rows)}; "
          f"closest<256 without exit {sum(r['closest']<256 and r['outcome']!='exit' for r in rows)}; "
          f"median closest {int(np.median([r['closest'] for r in rows]))}")
