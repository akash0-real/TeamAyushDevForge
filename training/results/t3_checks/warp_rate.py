"""Measure how often ShapedDoomEnv.reset() warps on real maps (WARP_PROB in training/rewards.py).
usage: python training/results/t3_checks/warp_rate.py <seed> <n_resets>"""
import sys
sys.path[:0] = ["training", "env", "submission"]
from rewards import ShapedDoomEnv, IWAD, WARP_PROB
seed, n = int(sys.argv[1]), int(sys.argv[2])
env = ShapedDoomEnv([(IWAD, "MAP05", 2100), (IWAD, "MAP08", 2100)], swap_every=1, seed=seed, use_map=False)
w = 0
for _ in range(n):
    env.reset(); w += env._warped
print(f"seed {seed}: {w}/{n} warped ({100*w/n:.0f}%), WARP_PROB={WARP_PROB}")
