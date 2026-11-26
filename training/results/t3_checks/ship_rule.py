"""Apply the COORDINATION ship rule (doom-rl-docs/CHECKPOINTS.md, 22:15) to i3 5-seed CSVs.
usage: python training/results/t3_checks/ship_rule.py <candidate_csv_stem> <shipped_csv_stem>
Rule: (1) paired 5-seed MEAN higher than shipped; (2) T2+T3+T4 exits (count over seeds) not fewer;
(3) MAP25-32 mean not worse by more than 2 SE (paired per seed)."""
import csv, re, statistics as st, sys
R = "training/results/"
def load(n): return list(csv.DictReader(open(R + n + ".csv", encoding="utf-8")))
def tier(m):
    g = re.search(r"gen_t([1-4])_", m); return int(g.group(1)) if g else 0
def per_seed(rows, keep):
    out = {}
    for sd in sorted({x["seed"] for x in rows}, key=int):
        m = {}
        for x in rows:
            if x["seed"] == sd and keep(x["map"]): m.setdefault(x["map"], []).append(float(x["score"]))
        out[sd] = st.mean(st.mean(v) for v in m.values())
    return out
c, b = load(sys.argv[1]), load(sys.argv[2])
assert {x["seed"] for x in c} == {x["seed"] for x in b}, "seed sets differ"
pc, pb = per_seed(c, lambda m: True), per_seed(b, lambda m: True)
mc, mb = st.mean(pc.values()), st.mean(pb.values())
ex = lambda rows: int(sum(float(x["exit"]) for x in rows if tier(x["map"]) in (2, 3, 4)))
ec, eb = ex(c), ex(b)
fc, fb = per_seed(c, lambda m: "freedoom2" in m), per_seed(b, lambda m: "freedoom2" in m)
d = [fc[s] - fb[s] for s in fb]; md, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
r1, r2, r3 = mc > mb, ec >= eb, md >= -2 * se
print(f"(1) MEAN {mc:.2f} vs {mb:.2f}: {'PASS' if r1 else 'FAIL'}")
print(f"(2) T2+T3+T4 exits {ec} vs {eb} (of {sum(1 for x in c if tier(x['map']) in (2,3,4))} episodes): {'PASS' if r2 else 'FAIL'}")
print(f"(3) MAP25-32 delta {md:+.2f} +/- {se:.2f} (limit -2 SE = {-2*se:.2f}): {'PASS' if r3 else 'FAIL'}")
print("SHIP RULE:", "PASS" if (r1 and r2 and r3) else "FAIL")
