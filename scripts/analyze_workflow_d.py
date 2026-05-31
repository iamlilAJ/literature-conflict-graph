"""Local analyzer for the Workflow D vs C experiment.

Reads wfd_out.jsonl (pulled from the server) and the prior genexp16_out.jsonl,
produces a clean report:
  - rubric mean for each arm (A, B, C, D) on the same 16 anomalies
  - per-criterion deltas (D - C)
  - blind pairwise C-vs-D winner counts
  - per-anomaly side-by-side flags (so we can read the cases where D wins/loses)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parent.parent
PRIOR = _REPO / "artifacts/atlas_test/genexp16_out.jsonl"


def main() -> None:
    wfd_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _REPO / "artifacts/atlas_test/wfd_out.jsonl"
    prior = [json.loads(l) for l in open(PRIOR) if l.strip()]
    wfd = [json.loads(l) for l in open(wfd_path) if l.strip()]
    prior_by_id = {r["anomaly_id"]: r for r in prior}

    paired = [(prior_by_id.get(r["anomaly_id"]), r) for r in wfd
              if "error" not in r and r.get("anomaly_id") in prior_by_id]
    if not paired:
        print("no overlapping anomalies", file=sys.stderr)
        return

    keys = ("forward_looking", "named_mechanism", "single_variable_test",
            "specific_scope", "composite")
    arms = ("A", "B", "C", "D")

    def m(arm: str, key: str) -> float:
        vs = []
        for p, w in paired:
            if arm == "D":
                vs.append(w["rubric_D"][key])
            else:
                vs.append(p["rubric"][arm][key])
        return round(mean(vs), 2)

    print(f"\n=== rubric means ({len(paired)} anomalies, all 4 arms on same set) ===")
    print(f"  {'criterion':>22} {'A':>6} {'B':>6} {'C':>6} {'D':>6}   ΔD-C   ΔD-B")
    for k in keys:
        row = {arm: m(arm, k) for arm in arms}
        dc = round(row["D"] - row["C"], 2)
        db = round(row["D"] - row["B"], 2)
        print(f"  {k:>22} {row['A']:>6.2f} {row['B']:>6.2f} {row['C']:>6.2f} "
              f"{row['D']:>6.2f}  {dc:>+5.2f}  {db:>+5.2f}")

    pair_counts = Counter(w["pair_CvD"] for _, w in paired)
    print(f"\n=== blind pairwise C vs D ({len(paired)} anomalies) ===")
    print(f"  D wins: {pair_counts.get('D', 0)}")
    print(f"  C wins: {pair_counts.get('C', 0)}")
    print(f"  tie:    {pair_counts.get('tie', 0)}")

    # Per-anomaly side-by-side (composite scores + winner)
    print(f"\n=== per-anomaly composite scores + pairwise winner ===")
    print(f"  {'anomaly':>10} {'type':>22}  A  B  C  D  CvD")
    for p, w in paired:
        ra, rb, rc = p["rubric"]["A"]["composite"], p["rubric"]["B"]["composite"], p["rubric"]["C"]["composite"]
        rd = w["rubric_D"]["composite"]
        cvd = w["pair_CvD"]
        atype = p["type"][:22]
        print(f"  {p['anomaly_id']:>10} {atype:>22}  {ra}  {rb}  {rc}  {rd}  {cvd}")

    # 3 best D-vs-C wins and 3 worst D losses (by composite delta)
    deltas = sorted(paired, key=lambda pw: pw[1]["rubric_D"]["composite"] - pw[0]["rubric"]["C"]["composite"], reverse=True)
    print("\n=== top-3 D advantages (D composite − C composite) ===")
    for p, w in deltas[:3]:
        d = w["rubric_D"]["composite"] - p["rubric"]["C"]["composite"]
        print(f"  {p['anomaly_id']} Δ={d:+d}")
        print(f"    hC: {p['hC'][:240]}...")
        print(f"    hD: {w['hD'][:240]}...")
        print(f"    pair_why: {w.get('pair_why','')[:200]}")
    print("\n=== bottom-3 (D regressions) ===")
    for p, w in deltas[-3:]:
        d = w["rubric_D"]["composite"] - p["rubric"]["C"]["composite"]
        print(f"  {p['anomaly_id']} Δ={d:+d}")
        print(f"    hC: {p['hC'][:240]}...")
        print(f"    hD: {w['hD'][:240]}...")
        print(f"    pair_why: {w.get('pair_why','')[:200]}")


if __name__ == "__main__":
    main()
