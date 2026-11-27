# training/results

Raw per-episode eval CSVs from `training/eval_zeroshot.py`, one row per (map, seed). The docs cite them by file name:
[RESULTS.md](../../doom-rl-docs/RESULTS.md), [EVAL_I3.md](../../doom-rl-docs/EVAL_I3.md), [CHECKPOINTS.md](../../doom-rl-docs/CHECKPOINTS.md).
The name prefix is the run (checkpoint/sweep tag); `i3_*` are teammate-3 CPU re-runs. `t3_checks/` holds the check scripts
(ship rule, soak, timing). Final v11 regression: `i3_final_regress.csv` (holdout) and `i3_t4x_v11_20seeds.csv` (tier 4, 20 seeds).
