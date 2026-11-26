# Self-check for round 6 geodesic shaping (training/geodesic.py): python training/check_geodesic.py
import sys, time, math
sys.path[:0] = ["training", "env", "submission"]
import numpy as np, torch
from rewards import ShapedDoomEnv, IWAD
from model import load_policy

# 1) every training level: walkable grid builds, and the start can reach the first goal
reach = 0
maps = [f"MAP{i:02d}" for i in range(1, 25) if i not in (11, 12, 15, 23)]
for m in maps:
    env = ShapedDoomEnv([(IWAD, m, 2100)], seed=0)
    env.rng.random = lambda: 1.0  # no warps: measure from the real start
    t = time.perf_counter(); env.reset(); geo = env._geo
    ps = env._env.privileged_state()
    phi = geo.phi(ps["x"], ps["y"], env._geo_src) if geo else None
    straight = min(math.hypot(ps["x"] - x, ps["y"] - y) for x, y in env._geo_src) if geo and env._geo_src else float("nan")
    reach += phi is not None
    walk = "unreachable" if phi is None else f"{-phi * 1000:6.0f}"
    print(f"{m}: grid {geo.w}x{geo.h}  keys {len(geo.keys)}  walk {walk}  straight {straight:6.0f}  {time.perf_counter() - t:.1f}s")
    if phi is not None:
        assert -phi * 1000 >= straight - 64, "walking distance can't be shorter than the straight line"
    env.close()
print(f"start reaches its first goal on {reach}/{len(maps)} training maps")
assert reach >= 15, reach  # 17/20 measured; the rest need teleporters or switch-raised floors

# 2) the shipped agent with shaping: episodes that exit collect positive geodesic progress
pol = load_policy("submission/weights/best_agent.pt", torch.device("cpu")).eval()
env = ShapedDoomEnv([("gen3", "MAP01", 2100), (IWAD, "MAP08", 2100)], swap_every=1, seed=1, use_map=True)
pos = []
for ep in range(8):
    obs, _ = env.reset(); h = pol.initial_state(1, torch.device("cpu")); done = False
    phi0 = env._phi if env._geo is not None else None
    if phi0 is not None:  # upper bound: start -> first goal, plus key -> next goal for every key (goal switches)
        g = env._geo
        legs = [g.phi(x, y, tuple((a, b) for a, b, k2 in g.keys if k2 != k) or tuple(g.exits)) for x, y, k in g.keys]
        bound = -phi0 + sum(-p for p in legs if p is not None)
    while not done:
        with torch.no_grad():
            lg, _, h = pol.forward_step({k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}, h)
            a, _, _ = pol.sample_and_logprob(lg)
        obs, r, term, trunc, info = env.step(a[0].numpy()); done = term or trunc
    st = info["episode_stats"]
    print(f"{env._cur_spec[1]}/{env._cur_spec[0][-8:]} ep{ep}: exit {st['exit_reached']:.0f} keys {st['keys']:.0f} geo {st['geo']:+.2f}")
    if st["exit_reached"]:
        pos.append(st["geo"])
    if phi0 is not None:  # no farming: total progress can't exceed the start distance (teammate 3 review 23:40)
        assert st["geo"] <= bound + 0.5, (st["geo"], bound)
assert all(g > 0 for g in pos), pos
print("OK")
