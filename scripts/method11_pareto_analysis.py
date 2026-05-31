"""Method 11 — Pareto-front analysis across all Likert-judged populations.

Combines Methods 3, 7, 8 outputs into a unified dataset (~80 Likert-judged
hyps across 6 populations) and asks:

  1. Which Likert axes are independent vs correlated?
  2. Is there a 2D Pareto front of (named_mechanism, atlas_overlap) that
     identifies the actually-good hyps?
  3. Are atlas_overlap=3 hyps systematically better than 2 or 4 on the
     OTHER axes? (Validates 3 = sweet spot from Method 3.)
  4. What rank-order rule (axis weighting) best separates manually-
     identifiable strong from weak hyps?

No new LLM calls — analyzes existing judge outputs.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

_REPO = Path(__file__).resolve().parent.parent
SRCS = [
    ("artifacts/atlas_test/m3_out.jsonl", "M3"),
    ("artifacts/atlas_test/m7_out.jsonl", "M7"),
    ("artifacts/atlas_test/m8_out.jsonl", "M8"),
]
AXES = ("atlas_overlap", "forward_looking", "named_mechanism",
        "single_variable_test", "specific_scope")


def load_all():
    out = []
    for path, m in SRCS:
        for ln in open(_REPO / path):
            r = json.loads(ln)
            if "error" in r or not r.get("_judge_ok"):
                continue
            r["_src"] = m
            out.append(r)
    return out


def main():
    rows = load_all()
    print(f"Loaded {len(rows)} Likert-judged hyps across populations:")
    print(f"  {Counter((r['_src'], r['pop']) for r in rows)}")
    print()

    # --- 1. Per-axis distribution + per-source means -----------------------
    print("=== axis distributions ===")
    for ax in AXES:
        vals = [r[ax] for r in rows]
        d = dict(Counter(vals))
        print(f"  {ax:>22}  mean={mean(vals):.2f}±{stdev(vals):.2f}  "
              f"median={median(vals)}  dist={d}")

    # --- 2. Correlations among axes ---------------------------------------
    def corr(xs, ys):
        n = len(xs)
        if n < 3:
            return 0.0
        mx, my = sum(xs)/n, sum(ys)/n
        sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        sxx = sum((x-mx)**2 for x in xs)
        syy = sum((y-my)**2 for y in ys)
        d = (sxx*syy)**0.5
        return round(sxy/d, 3) if d else 0.0

    print()
    print("=== pairwise pearson correlations ===")
    print(f"  {'':>22} " + " ".join(f"{a[:6]:>8}" for a in AXES))
    for a in AXES:
        row = [corr([r[a] for r in rows], [r[b] for r in rows]) for b in AXES]
        print(f"  {a:>22} " + " ".join(f"{c:>8.3f}" for c in row))

    # --- 3. Are atlas_overlap=3 hyps better on other axes? ----------------
    print()
    print("=== other-axis means split by atlas_overlap value ===")
    print(f"  {'overlap':>10} {'n':>4} "
          + " ".join(f"{a[:9]:>10}" for a in AXES if a != "atlas_overlap"))
    for ovr in sorted(set(r["atlas_overlap"] for r in rows)):
        sub = [r for r in rows if r["atlas_overlap"] == ovr]
        if not sub:
            continue
        row = [round(mean([r[a] for r in sub]), 2)
               for a in AXES if a != "atlas_overlap"]
        print(f"  {ovr:>10} {len(sub):>4} " + " ".join(f"{v:>10.2f}" for v in row))

    # --- 4. Pareto front (named_mechanism, atlas_overlap) ------------------
    pairs = [(r, r["named_mechanism"], r["atlas_overlap"]) for r in rows]
    front = []
    for r, nm, ovr in pairs:
        dominated = any(nm2 >= nm and ovr2 >= ovr and (nm2 > nm or ovr2 > ovr)
                        for _, nm2, ovr2 in pairs)
        if not dominated:
            front.append((r, nm, ovr))
    print()
    print(f"=== Pareto front (named_mechanism, atlas_overlap), n={len(front)} ===")
    print(f"  {'hyp_id':>16} {'src':>4} {'pop':>10} {'nm':>4} {'ovr':>4} "
          f"{'fl':>4} {'svt':>4} {'sc':>4}")
    front.sort(key=lambda x: (-x[1], -x[2]))
    for r, nm, ovr in front:
        print(f"  {r['hyp_id'][:16]:>16} {r['_src']:>4} {r['pop'][:10]:>10} "
              f"{nm:>4} {ovr:>4} {r['forward_looking']:>4} "
              f"{r['single_variable_test']:>4} {r['specific_scope']:>4}")

    # --- 5. Try composite rank rules ---------------------------------------
    print()
    print("=== composite rank rules — sum of axis weights ===")
    rules = {
        "uniform 5-sum":         lambda r: sum(r[a] for a in AXES),
        "no-overlap 4-sum":      lambda r: sum(r[a] for a in AXES if a != "atlas_overlap"),
        "boost_nm + sweet_ovr":  lambda r: (1.5*r["named_mechanism"]
                                            + (3 if r["atlas_overlap"]==3
                                               else 2 if r["atlas_overlap"]==4
                                               else 1)
                                            + r["forward_looking"]
                                            + r["specific_scope"]),
        "anchored-novelty":      lambda r: (r["named_mechanism"]
                                            + {1:0.4, 2:0.6, 3:1.0, 4:0.8, 5:0.2}[r["atlas_overlap"]] * 4
                                            + r["forward_looking"]
                                            + r["specific_scope"]
                                            + 0.5*r["single_variable_test"]),
    }
    for name, fn in rules.items():
        scored = sorted(rows, key=fn, reverse=True)
        top10 = scored[:10]
        bot10 = scored[-10:]
        top_pops = Counter(r["pop"] for r in top10)
        bot_pops = Counter(r["pop"] for r in bot10)
        print(f"\n  Rule: {name}")
        print(f"    top-10 pops: {dict(top_pops)}")
        print(f"    bot-10 pops: {dict(bot_pops)}")


if __name__ == "__main__":
    main()
