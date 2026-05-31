"""Method 22 — Soft weighted re-rank vs binary atlas-overlap filter.

Same data as Method 15 (12 diverse topics × top-8 on 540p-thaw1). Three arms:

  OFF:  baseline (no atlas signal)
  HARD: --min-atlas-overlap 3 (Method 13, drop overlap=1,2 entirely)
  SOFT: utility × overlap_weight, no drop; weight curve:
        {1: 0.4, 2: 0.6, 3: 1.0, 4: 0.9, 5: 0.5, 0/unscored: 1.0}

Pure analytical (no new LLM calls). Reuses the sidecar + scoring.score_all.

Question: does the soft re-rank surface DIFFERENT hyps than the hard filter?
If they match → hard filter is sufficient (simpler is better).
If soft surfaces novel-but-marginal hyps that hard correctly drops → keep
the hard filter for safety.
If soft surfaces overlap=4 hyps that hard ranks tied with overlap=3 → soft
wins for fine-grained quality distinctions.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from aigraph.io import read_jsonl  # noqa: E402
from aigraph.models import Anomaly, Claim, Hypothesis, Paper  # noqa: E402
from aigraph.scoring import score_all, select_mmr  # noqa: E402
from aigraph_query import (  # noqa: E402
    _load_atlas_overlap_sidecar, _load_run_dir, _tokenize, _topic_relevance,
    _is_degenerate_anomaly,
)

RUN = _REPO / "artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1"
TOPICS = [
    "chain of thought reasoning",
    "language model",
    "neural network",
    "evaluation",
    "fine-tuning",
    "retrieval augmented generation",
    "reinforcement learning",
    "in-context learning",
    "multi-agent",
    "alignment",
    "tool use",
    "code generation",
]
SOFT_WEIGHT = {1: 0.4, 2: 0.6, 3: 1.0, 4: 0.9, 5: 0.5, 0: 1.0}


def select_arm(hyps, anoms, claims, topic, sidecar, mode: str, k=8):
    """mode ∈ {'off', 'hard', 'soft'}. Returns the selected hyp_ids."""
    query_tokens = _tokenize(topic)
    anom_lookup = {a.anomaly_id: a for a in anoms}
    claim_lookup = {c.claim_id: c for c in claims}

    # Baseline filters (same as production today)
    scored = [(h, _topic_relevance(h, anom_lookup, claim_lookup, query_tokens))
              for h in hyps]
    matched = [(h, r) for (h, r) in scored if r >= 1]
    non_degen = [(h, r) for (h, r) in matched
                 if not _is_degenerate_anomaly(anom_lookup.get(h.anomaly_id))]
    if non_degen:
        matched = non_degen

    # Mode-specific atlas treatment
    if mode == "hard":
        # Drop overlap < 3, keep unscored (==0)
        anchored = [(h, r) for (h, r) in matched
                    if sidecar.get(h.hypothesis_id, 0) == 0
                    or sidecar[h.hypothesis_id] >= 3]
        if anchored:
            matched = anchored

    # Sort by topic relevance (same as production)
    matched.sort(key=lambda hr: -hr[1])

    # Cap per anomaly (same as production)
    per_anom = {}; capped = []
    for h, r in matched:
        n = per_anom.get(h.anomaly_id, 0) + 1
        per_anom[h.anomaly_id] = n
        if n <= 2:
            capped.append((h, r))
    matched = capped[:30]

    candidates = [h for h, _ in matched]
    breakdowns = score_all(candidates, anoms, claims)

    if mode == "soft":
        # Mutate ScoreBreakdown.utility in place to apply soft weight
        for h in candidates:
            ov = sidecar.get(h.hypothesis_id, 0)
            w = SOFT_WEIGHT.get(ov, 1.0)
            sb = breakdowns[h.hypothesis_id]
            # ScoreBreakdown is a pydantic model; use object.__setattr__ if frozen
            try:
                sb.utility = sb.utility * w
            except Exception:
                object.__setattr__(sb, "utility", sb.utility * w)

    selected = select_mmr(candidates, breakdowns, k=k, lambda_=0.7,
                          min_anomalies=2)
    return [h.hypothesis_id for h in selected], breakdowns


def main():
    hyps = read_jsonl(RUN / "hypotheses_scored.jsonl", Hypothesis)
    anoms = read_jsonl(RUN / "anomalies.jsonl", Anomaly)
    claims = read_jsonl(RUN / "claims.jsonl", Claim)
    sidecar = _load_atlas_overlap_sidecar(RUN)
    print(f"loaded {len(hyps)} hyps, sidecar entries: {len(sidecar)}\n")

    print(f"{'topic':>34} {'OFF':>40} {'HARD':>40} {'SOFT':>40}  off→hard  off→soft  hard→soft")
    rows = []
    for topic in TOPICS:
        off_ids, _ = select_arm(hyps, anoms, claims, topic, sidecar, "off")
        hard_ids, _ = select_arm(hyps, anoms, claims, topic, sidecar, "hard")
        soft_ids, _ = select_arm(hyps, anoms, claims, topic, sidecar, "soft")
        oh = len(set(off_ids) ^ set(hard_ids))
        os_ = len(set(off_ids) ^ set(soft_ids))
        hs = len(set(hard_ids) ^ set(soft_ids))
        rows.append((topic, off_ids, hard_ids, soft_ids, oh, os_, hs))
        print(f"  {topic:>32} "
              f"{str([(i, sidecar.get(i,0)) for i in off_ids[:4]]):>40} "
              f"{str([(i, sidecar.get(i,0)) for i in hard_ids[:4]]):>40} "
              f"{str([(i, sidecar.get(i,0)) for i in soft_ids[:4]]):>40} "
              f"{oh:>8} {os_:>8} {hs:>8}")

    # Aggregate
    print()
    print("=== aggregate ===")
    print(f"  total slot churn off→hard: {sum(r[4] for r in rows)}/{2*len(TOPICS)*8}")
    print(f"  total slot churn off→soft: {sum(r[5] for r in rows)}/{2*len(TOPICS)*8}")
    print(f"  total slot churn hard→soft: {sum(r[6] for r in rows)}/{2*len(TOPICS)*8}")

    # Leak counts in each arm
    from collections import Counter
    def leak_count(ids):
        return sum(1 for i in ids if 0 < sidecar.get(i, 0) < 3)
    off_leaks = sum(leak_count(r[1]) for r in rows)
    hard_leaks = sum(leak_count(r[2]) for r in rows)
    soft_leaks = sum(leak_count(r[3]) for r in rows)
    print(f"  leaks in OFF top-8 (overlap=1,2): {off_leaks}/{len(TOPICS)*8}")
    print(f"  leaks in HARD top-8:             {hard_leaks}/{len(TOPICS)*8}")
    print(f"  leaks in SOFT top-8:             {soft_leaks}/{len(TOPICS)*8}")

    # Overlap=5 (exact restatement) handling
    def restate_count(ids):
        return sum(1 for i in ids if sidecar.get(i) == 5)
    print()
    print(f"  overlap=5 (exact restatement) in OFF:  {sum(restate_count(r[1]) for r in rows)}")
    print(f"  overlap=5 in HARD: {sum(restate_count(r[2]) for r in rows)}")
    print(f"  overlap=5 in SOFT: {sum(restate_count(r[3]) for r in rows)}")

    out_path = _REPO / "artifacts/atlas_test/method22_results.json"
    out_data = [{"topic": r[0], "off": r[1], "hard": r[2], "soft": r[3]}
                for r in rows]
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
