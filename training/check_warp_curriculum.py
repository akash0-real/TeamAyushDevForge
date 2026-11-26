# Self-check for the start-near-exit curriculum in training/rewards.py: python training/check_warp_curriculum.py
import sys, math, numpy as np
sys.path.insert(0, "training"); sys.path.insert(0, "env")
from rewards import ShapedDoomEnv, exit_spots, warp_spots, IWAD
for m in [f"MAP{i:02d}" for i in range(1, 33)]:
    ex, sp = exit_spots(m)
    assert (ex and sp) or m == "MAP30", (m, len(ex), len(sp))  # MAP30: boss map, no exit line
print("all 32 maps have exits + spawn spots")
env = ShapedDoomEnv([(IWAD, "MAP03", 2100), (IWAD, "MAP07", 2100)], swap_every=1, seed=3, use_map=True)
warped = 0
kinds = set()
for ep in range(30):
    obs, info = env.reset()
    ps = env._env.privileged_state()
    ex, sp = warp_spots(env._cur_spec[1], env._warped or "exit")
    kinds.add(env._warped)
    d = min(math.hypot(ps["x"] - a, ps["y"] - b) for a, b in ex)
    warped += bool(env._warped)
    assert set(obs) >= {"rays", "map"} and env.ego_map.tic == (4 if env._warped else 0)
    if env._warped: assert d <= max(256, sp[min(2, len(sp) - 1)][0]) + 200, d  # within the radius, or the 3 nearest spots
    # run a short random episode to make sure stepping works after a warp
    for t in range(30):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc: break
    print(env._cur_spec[1], env._warped or "normal", f"dist to target {d:6.0f}", "tic", info["tic"])
assert 6 <= warped <= 24 and {"exit", "key", False} <= kinds, kinds
print("OK", warped, "of 30 warped (exit + key); radius", env._warp_radius)
