# Candidate checkpoints (never merged into main)

Core and teammate 2 put checkpoints here for scoring. Teammate 3 scores each with 5 seeds (Task R2).
The core owner ships a winner by copying it into submission/weights/ on main after validate.py passes.

| file | from | notes |
|---|---|---|
| core_r1_2025.pt | core run 1 | round 3, 20:25 |
| core_r2_2025.pt | core run 2 | round 3, 20:25 |
| core_r2_2155.pt | core run 2 | round 4 (exit bonus for hp/time), 21:55; training T3 exits 35–46%. Compare vs shipped core_r2_2025 (bar 34.85) |
- `t2h_0037_score44_22.pt` (t2, 00:54 IST, ~11.7M steps, Round 6 + 256/192 warp): 5-seed CUDA 44.22 vs v3 43.11 (SHIP RULE PASS), T1 10/10, T2 9/10 (99.21), T3 8/10 (120.35), T4 0/10, exit_pts 52 vs 45, MAP25-32 -3.49, T5 19% in log.
| core_vA_0035.pt | core Vast run A | round 6 + warp fix, from v3; Vast 5-seed holdout +3.3 ± 2.6 vs v3, exit_points 49 vs 45, MAP25-32 +1.8; needs teammate 2's independent CUDA check |

- `t2h_0122_score49_23.pt` (2026-11-27 01:51 IST, teammate 2 RTX 3050, Round 6 Geodesic + 256/192 warp, ~12.4M steps): **deterministic seeded 5-seed CUDA SHIP RULE PASS** vs shipped-v6 — **MEAN 49.23 vs 40.95 (+8.28)**, T1 10/10 (101.21), T2 9/10 (103.78), **T3 9/10 (134.77 vs 5/10 87.37)**, T4 0/10 (15.57 vs 12.96), **exit_points 55 vs 45 (+10)**, **MAP25-32 +2.18 vs -5.10 (+7.28 ± 6.08)**; throwaway Linux 1-CPU CUDA validate.py PASS (two_rooms 55 steps, 130 MB VRAM). CSVs on main: `t2h_0122_seeded_s04.csv` vs `cuda_v6_seeded_s04.csv`.
| core_soupA_0200.pt | core weight average of v6 (t2h_0037) + Vast run A | with v8 settings (turn 0.5, strafe 0.5): seeds 10-19 +3.5 ± 1.4 vs v7, exit_points 106 vs 101, unseen-real +5.9; needs a second-machine check vs v8 |
