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
