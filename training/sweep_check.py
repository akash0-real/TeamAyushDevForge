"""Knob sweep vs v7 (core, 02:00): reads training/results/sw_<name>_s<seed>.csv (seeded eval, 17 holdout maps).
usage: python training/sweep_check.py base name1 name2 ...   [SW_SEEDS=0-9 default]"""
import csv, os, sys
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEEDS = range(int(os.environ.get("SW_LO", "0")), int(os.environ.get("SW_HI", "10")))
def load(n): return [r for s in SEEDS if os.path.exists(f"{R}/sw_{n}_s{s}.csv") for r in csv.DictReader(open(f"{R}/sw_{n}_s{s}.csv"))]
def by_seed(rows, sel=lambda r: True):
    d = {}
    for r in rows:
        if sel(r): d.setdefault(r["seed"], []).append(float(r["score"]))
    return {k: np.mean(v) for k, v in d.items()}
def ep_seed(rows):
    d = {}
    for r in rows: d[r["seed"]] = d.get(r["seed"], 0) + (int(r["tier"]) if float(r["exit"]) > 0 and r["tier"] != "0" else 0)
    return d
def tex(rows): return [int(sum(float(r["exit"]) > 0 for r in rows if r["tier"] == str(t))) for t in (1, 2, 3, 4)]
def pd(a, b):
    k = sorted(set(a) & set(b)); d = np.array([b[s] - a[s] for s in k]); return d.mean(), d.std(ddof=1) / np.sqrt(len(d))
base = sys.argv[1]; B = load(base)
print(f"{base}: mean {np.mean([float(r['score']) for r in B]):.2f} exits {tex(B)} exit_points {sum(ep_seed(B).values())} (n={len(B)})")
for n in sys.argv[2:]:
    C = load(n)
    if not C: print(n, "no results"); continue
    m, se = pd(by_seed(B), by_seed(C)); e, ee = pd(ep_seed(B), ep_seed(C))
    u, ue = pd(by_seed(B, lambda r: r["map"].startswith("freedoom2.wad:")), by_seed(C, lambda r: r["map"].startswith("freedoom2.wad:")))
    print(f"{n:10s} mean {m:+5.1f}±{se:3.1f} | exit_points/seed {e:+4.1f}±{ee:3.1f} (total {sum(ep_seed(C).values())}) | exits {tex(C)} | unseen-real {u:+5.1f}±{ue:3.1f}")
