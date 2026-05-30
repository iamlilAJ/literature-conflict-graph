"""Step 2 end-to-end test: does the FROZEN scorer reward forward hypotheses?

Population-level comparison since the random forward-gen sample doesn't
overlap with the run's 62 scored anomalies. Compares the FULL distribution
of forward composites vs the FULL distribution of frozen composites.
Also probes MMR survival on a merged pool. 0 LLM.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean, stdev

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
from aigraph.io import read_jsonl  # noqa: E402
from aigraph.models import Anomaly, Claim, Hypothesis  # noqa: E402
from aigraph.scoring import score_all, select_mmr  # noqa: E402

RUN = _REPO / "artifacts/runs/validation-v1-primary"
FWD = _REPO / "artifacts/atlas_test/forward_hyps.jsonl"


def _to_hyp(d: dict, idx: int) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=d.get("hypothesis_id") or f"fwd_{idx:03d}",
        anomaly_id=d["anomaly_id"],
        hypothesis=d.get("hypothesis", ""),
        mechanism=d.get("mechanism", ""),
        explains_claims=list(d.get("explains_claims", []) or []),
        predictions=list(d.get("predictions", []) or []),
        minimal_test=d.get("minimal_test", ""),
        scope_conditions=dict(d.get("scope_conditions", {}) or {}),
        evidence_gap=d.get("evidence_gap", ""),
    )


def main():
    anoms = read_jsonl(RUN / "anomalies.jsonl", Anomaly)
    claims = read_jsonl(RUN / "claims.jsonl", Claim)
    frozen = read_jsonl(RUN / "hypotheses_scored.jsonl", Hypothesis)
    fwd_raw = [json.loads(l) for l in open(FWD) if l.strip()]
    # Some forward hyps may have hypothesis_id collisions w/ frozen — re-prefix.
    forward = []
    for i, d in enumerate(fwd_raw):
        h = _to_hyp(d, i)
        h.hypothesis_id = f"fwd_{i:03d}"
        forward.append(h)

    print(f"forward hyps: {len(forward)} (covering {len({h.anomaly_id for h in forward})} anomalies)")
    print(f"frozen hyps : {len(frozen)} (covering {len({h.anomaly_id for h in frozen})} anomalies)")

    sb_f = score_all(forward, anoms, claims)
    sb_z = score_all(frozen, anoms, claims)

    keys = ("explain", "grounding", "testability", "novelty",
            "discriminability", "impact", "topology", "cost", "utility")
    def m(scores, key):
        vs = [getattr(s, key) for s in scores.values()]
        return round(mean(vs), 3), round(stdev(vs) if len(vs) > 1 else 0.0, 3)

    print("\n=== mean (±sd) per-component: FORWARD vs FROZEN (population) ===")
    print(f"  {'component':16} {'forward':>16} {'frozen':>16} {'Δ mean':>9}")
    for k in keys:
        fm, fs = m(sb_f, k); zm, zs = m(sb_z, k)
        print(f"  {k:16} {fm:>8.3f}±{fs:>5.3f} {zm:>8.3f}±{zs:>5.3f} {fm-zm:>+9.3f}")

    # MMR probe: merged pool, what fraction of top-K is forward?
    merged = forward + frozen
    sb_m = score_all(merged, anoms, claims)
    selected = select_mmr(merged, sb_m, k=8, lambda_=0.7, min_anomalies=1)
    fwd_sel = sum(1 for h in selected if h.hypothesis_id.startswith("fwd_"))
    print(f"\n=== MMR top-8 from merged pool ({len(merged)} candidates): "
          f"{fwd_sel} forward, {len(selected)-fwd_sel} frozen")
    # If forward hyps were random in their (anomaly_id, scoring-context) niche
    # they would represent ~24/(24+300) = ~7% of the pool, so picking >7% is a win.
    base = round(100 * len(forward) / len(merged), 1)
    pct_picked = round(100 * fwd_sel / len(selected), 1)
    print(f"  base rate (forward share of pool): {base}%  |  picked share of top-8: {pct_picked}%")
    for h in selected:
        kind = "FWD" if h.hypothesis_id.startswith("fwd_") else "frz"
        print(f"  [{kind}] {h.hypothesis_id[:18]:18} util={sb_m[h.hypothesis_id].utility:.3f}  anom={h.anomaly_id}")


if __name__ == "__main__":
    main()
