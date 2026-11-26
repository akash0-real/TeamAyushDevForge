"""Agent-only act() latency on one pinned core, with no engine contention: replays recorded real-map observations.
usage: python training/results/t3_checks/agent_only_timing.py MAP32 [steps]   (Windows: pins via SetProcessAffinityMask)"""
import ctypes, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "env")]
from doom_env import make_env  # noqa: E402
from doom_env.wrapper import IWAD  # noqa: E402
from agent import DoomAgent  # noqa: E402
m, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300
env = make_env(IWAD, m, episode_ticks=4200, render_buffers=False)
agent = DoomAgent(env.observation_space, env.action_space, "cpu")
obs, _ = env.reset(seed=0); agent.reset(); rec = []
while len(rec) < n:  # record observations with the engine running (unpinned)
    rec.append(obs); obs, _, term, trunc, _ = env.step(agent.act(obs))
    if term or trunc: obs, _ = env.reset(seed=len(rec)); agent.reset()
env.close()
ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(), 1)  # 1 core, agent only
import torch; torch.set_num_threads(1)
agent.reset(); ts = []
for o in rec:
    t = time.perf_counter(); agent.act(o); ts.append((time.perf_counter() - t) * 1000)
ts = np.array(ts[1:])
print(f"{m}: agent-only on 1 core, n={len(ts)}: p50 {np.percentile(ts,50):.1f} ms, p99 {np.percentile(ts,99):.1f} ms, max {ts.max():.1f} ms, mean {ts.mean():.1f} ms")
