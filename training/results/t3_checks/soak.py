"""Robustness soak of the shipped agent: every map in tests/fixtures + giant/real maps, seeded, CPU.
Flags exceptions, act() safety-net triggers, invalid actions and slow steps. usage: python training/results/t3_checks/soak.py"""
import contextlib, io, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "env")]
from doom_env import make_env  # noqa: E402
from doom_env.wad import read_wad  # noqa: E402
from doom_env.wrapper import IWAD  # noqa: E402
from agent import DoomAgent  # noqa: E402

def maps_in(wad):
    names = [lump.name for lump in read_wad(wad)]
    return [n for n in names if (n.startswith("MAP") and n[3:].isdigit()) or (len(n) == 4 and n[0] == "E" and n[2] == "M")]

jobs = [(str(w), m) for w in sorted((ROOT / "tests" / "fixtures").glob("*.wad")) for m in maps_in(w)[:1]]
jobs += [(IWAD, m) for m in ("MAP11", "MAP12", "MAP15", "MAP23")]  # giant maps excluded from training
f1 = str(Path(IWAD).with_name("freedoom1.wad"))
if Path(f1).exists():
    jobs += [(f1, "E2M2"), (f1, "E3M3")]
agent = None; bad = 0; worst = 0.0; rows = []
for wad, m in jobs:
    for seed in (0, 1):
        env = make_env(wad, m, episode_ticks=4200, render_buffers=False)
        if agent is None:
            agent = DoomAgent(env.observation_space, env.action_space, "cpu")
        obs, _ = env.reset(seed=seed); agent.reset(); torch.manual_seed(seed)
        buf = io.StringIO(); ts = []; err = None; term = trunc = False
        try:
            with contextlib.redirect_stdout(buf):
                while not (term or trunc):
                    t = time.perf_counter(); a = agent.act(obs); ts.append((time.perf_counter() - t) * 1000)
                    assert isinstance(a, np.ndarray) and a.shape == (9,) and set(np.unique(a)) <= {0, 1}, "invalid action"
                    obs, _, term, trunc, _ = env.step(a)
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        ps = env.privileged_state(); env.close()
        net = buf.getvalue().count("DoomAgent.act error")
        p99 = float(np.percentile(ts, 99)) if ts else 0.0; worst = max(worst, max(ts) if ts else 0.0)
        flag = err or net
        bad += bool(flag)
        rows.append(f"{Path(wad).name}:{m} s{seed} steps={len(ts)} exit={int(ps['exit_reached'])} p99={p99:.1f}ms "
                     f"safety_net={net}{' ERROR ' + err if err else ''}")
        print(rows[-1], flush=True)
print(f"SOAK: {len(rows)} episodes, problems={bad}, worst single step={worst:.1f} ms")
assert bad == 0, "soak found problems"
