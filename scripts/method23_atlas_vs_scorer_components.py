"""Method 23 — what frozen-scorer dimension does atlas_overlap capture?

Spearman ρ of atlas_overlap against each ScoreBreakdown component, on the
259 production hyps where both are known. Locates the orthogonality:
  - If ρ(overlap, novelty) >> 0 → atlas is "just a better novelty score"
  - If ρ(overlap, all-low) → atlas is orthogonal to the entire frozen scorer
  - If ρ(overlap, explain) >> 0 → atlas overlaps the existing explain signal
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from aigraph.io import read_jsonl
from aigraph.models import Anomaly, Claim, Hypothesis
from aigraph.scoring import score_all
from aigraph_query import _load_atlas_overlap_sidecar


def spearman(rows, k1, k2):
    n = len(rows)
    if n < 2:
        return 0.0

    def ranks(vals):
        srt = sorted(range(n), key=lambda i: vals[i])
        rk = [0] * n
        for r, i in enumerate(srt):
            rk[i] = r + 1
        return rk
    v1 = [r[k1] for r in rows]
    v2 = [r[k2] for r in rows]
    r1, r2 = ranks(v1), ranks(v2)
    m1, m2 = mean(r1), mean(r2)
    num = sum((r1[i] - m1) * (r2[i] - m2) for i in range(n))
    d1 = sum((x - m1) ** 2 for x in r1) ** 0.5
    d2 = sum((x - m2) ** 2 for x in r2) ** 0.5
    return round(num / (d1 * d2), 3) if d1 and d2 else 0.0


def main():
    run = _REPO / "artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1"
    hyps = read_jsonl(run / "hypotheses_scored.jsonl", Hypothesis)
    anoms = read_jsonl(run / "anomalies.jsonl", Anomaly)
    claims = read_jsonl(run / "claims.jsonl", Claim)
    sidecar = _load_atlas_overlap_sidecar(run)
    sb = score_all(hyps, anoms, claims)

    components = ("explain", "grounding", "testability", "novelty",
                  "discriminability", "impact", "topology", "cost", "utility")
    rows = []
    for h in hyps:
        ov = sidecar.get(h.hypothesis_id)
        if ov is None:
            continue
        s = sb[h.hypothesis_id]
        row = {"atlas_overlap": ov}
        for c in components:
            row[c] = getattr(s, c)
        rows.append(row)
    print(f"joined {len(rows)} hyps (atlas_overlap × frozen scorer components)\n")

    print(f"  {'component':>17}  ρ(atlas_overlap, x)")
    for c in components:
        r = spearman(rows, "atlas_overlap", c)
        flag = "  ←" if abs(r) > 0.20 else ""
        print(f"  {c:>17}  {r:>+8.3f}{flag}")

    # Also: per-overlap-bucket means of each component
    print()
    print("=== mean of each scorer component, split by atlas_overlap ===")
    print(f"  {'overlap':>8} {'n':>4}  " + "  ".join(f"{c[:11]:>11}" for c in components))
    for ov in sorted(set(r["atlas_overlap"] for r in rows)):
        sub = [r for r in rows if r["atlas_overlap"] == ov]
        means = [round(sum(r[c] for r in sub) / len(sub), 3) for c in components]
        print(f"  {ov:>8} {len(sub):>4}  " + "  ".join(f"{v:>11.3f}" for v in means))


if __name__ == "__main__":
    main()
