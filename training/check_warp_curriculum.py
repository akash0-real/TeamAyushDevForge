# Self-check for the start-near-exit curriculum in training/rewards.py: python training/check_warp_curriculum.py
import sys, math, numpy as np
sys.path.insert(0, "training"); sys.path.insert(0, "env")
from rewards import ShapedDoomEnv, exit_spots, IWAD
for m in [f"MAP{i:02d}" for i in range(1, 33)]:
    ex, sp = exit_spots(m)
    assert (ex and sp) or m == "MAP30", (m, len(ex), len(sp))  # MAP30: boss map, no exit line
print("all 32 maps have exits + spawn spots")
env = ShapedDoomEnv([(IWAD, "MAP01", 2100), (IWAD, "MAP07", 2100)], swap_every=1, seed=3, use_map=True)
warped = 0
for ep in range(12):
    obs, info = env.reset()
    ps = env._env.privileged_state()
    ex, sp = exit_spots(env._cur_spec[1])
    d = min(math.hypot(ps["x"] - a, ps["y"] - b) for a, b in ex)
    warped += env._warped
    assert set(obs) >= {"rays", "map"} and env.ego_map.tic == (4 if env._warped else 0)
    if env._warped: assert d <= 768 + 200, d  # within the starting radius
    # run a short random episode to make sure stepping works after a warp
    for t in range(30):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc: break
    print(env._cur_spec[1], "warped" if env._warped else "normal", f"dist to exit {d:6.0f}", "tic", info["tic"])
assert 2 <= warped <= 10
print("OK", warped, "of 12 warped; radius", env._warp_radius)
