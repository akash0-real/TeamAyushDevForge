"""Direction + overfitting check (core, 00:45): is each training run getting better than the shipped v3, and is it
generalising or memorising? Reads training/results/{chk,trn}_<ckpt>_s<seed>.csv written by eval_zeroshot.py:
  chk_* = 17 holdout maps (unseen generated + unseen MAP25-32 + two_rooms), seeds 0-4
  trn_* = 8 real Freedoom maps the agent trains on (MAP01 03 05 07 09 13 17 21), seeds 0-2
usage: python training/direction_check.py v3 vA vB vC vD vE   (first name = baseline)"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"


def load(prefix, ck, seeds):
    rows = []
    for s in seeds:
        f = R / f"{prefix}_{ck}_s{s}.csv"
        if f.exists():
            rows += list(csv.DictReader(open(f)))
    return rows


def per_seed(rows, sel):
    out = {}
    for r in rows:
        if sel(r):
            out.setdefault(r["seed"], []).append(float(r["score"]))
    return {s: np.mean(v) for s, v in out.items()}


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = np.array([b[s] - a[s] for s in seeds])
    return (d.mean(), d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan"), len(d))


SLICES = {
    "overall": lambda r: True,
    "unseen-real": lambda r: r["map"].startswith(("freedoom2.wad:", "freedoom1.wad:")),  # MAP25-32 / Freedoom 1
    "T2": lambda r: r["tier"] == "2",
    "T3": lambda r: r["tier"] == "3",
    "T4": lambda r: r["tier"] == "4",
}
def tier_exits(rows):
    return {t: int(sum(float(r["exit"]) > 0 for r in rows if r["tier"] == str(t))) for t in (1, 2, 3, 4)}


def exit_points(rows):  # teammate 3 / human, 00:55: harder tiers weigh more
    return sum(t * n for t, n in tier_exits(rows).items())


PREFIX, NSEED = os.environ.get("DC_PREFIX", "chk"), int(os.environ.get("DC_SEEDS", "5"))  # final pick: DC_PREFIX=fin DC_SEEDS=10
base, cks = sys.argv[1], sys.argv[2:]
B = load(PREFIX, base, range(NSEED))
Btr = load("trn", base, range(3))
print(f"baseline {base}: holdout {np.mean([float(r['score']) for r in B]):.2f} (n={len(B)})")
for ck in cks:
    C = load(PREFIX, ck, range(NSEED))
    if not C:
        print(f"{ck}: no results yet")
        continue
    parts = []
    for name, sel in SLICES.items():
        m, se, n = paired(per_seed(B, sel), per_seed(C, sel))
        flag = " *BETTER*" if m > 2 * se else (" *WORSE*" if m < -2 * se else "")
        parts.append(f"{name} {m:+6.1f}±{se:4.1f}{flag}")
    ex = lambda rows: int(sum(float(r["exit"]) > 0 for r in rows if r["tier"] in ("2", "3", "4")))
    print(f"{ck}: mean {np.mean([float(r['score']) for r in C]):6.2f} | " + " | ".join(parts) + f" | T2-T4 exits {ex(C)} vs {ex(B)} (n seeds {len(per_seed(C, SLICES['overall']))})")
    print(f"    exits by tier {tier_exits(C)} vs {base} {tier_exits(B)} | exit_points {exit_points(C)} vs {exit_points(B)}")
# overfitting: training real maps vs unseen real maps (MAP25-32), per checkpoint
print("\noverfitting check: score on real maps it TRAINS on vs UNSEEN real maps (MAP25-32)")
for ck in [base] + cks:
    tr = [float(r["score"]) for r in load("trn", ck, range(3))]
    un = [float(r["score"]) for r in load(PREFIX, ck, range(NSEED)) if r["map"].startswith("freedoom2.wad:")]
    if tr and un:
        print(f"{ck}: train-maps {np.mean(tr):6.2f}  unseen {np.mean(un):6.2f}  gap {np.mean(tr) - np.mean(un):+6.2f}")
