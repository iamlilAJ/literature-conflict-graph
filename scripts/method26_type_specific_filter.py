"""Method 26 — Type-specific Atlas-overlap filter.

Method 24 showed:
  community_disconnect: 30% leak (overlap < 3)
  setting_mismatch: 27% leak
  bridge_opportunity: 6% leak
  metric_mismatch: 4% leak
  impact_conflict: 8% leak

Question: does a filter that only drops overlap<3 for "leaky" types
(community_disconnect + setting_mismatch) match the universal filter's
leak prevention while reducing slot churn?

Three arms, same 12 topics × top-8:
  OFF           (no filter)
  UNIVERSAL     (drop overlap<3 regardless of type — Method 13)
  TYPE_SPECIFIC (drop overlap<3 only when anomaly.type in {community_disconnect,
                 setting_mismatch})

Pure analytical, no LLM calls.
"""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from aigraph.io import read_jsonl  # noqa: E402
from aigraph.models import Anomaly, Claim, Hypothesis  # noqa: E402
from aigraph.scoring import score_all, select_mmr  # noqa: E402
from aigraph_query import (  # noqa: E402
    _load_atlas_overlap_sidecar, _tokenize, _topic_relevance,
    _is_degenerate_anomaly,
)

RUN = _REPO / "artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1"
TOPICS = [
    "chain of thought reasoning", "language model", "neural network",
    "evaluation", "fine-tuning", "retrieval augmented generation",
    "reinforcement learning", "in-context learning", "multi-agent",
    "alignment", "tool use", "code generation",
]
LEAKY_TYPES = {"community_disconnect", "setting_mismatch"}


def run_arm(hyps, anoms, claims, topic, sidecar, anom_by_id, mode: str, k=8):
    query_tokens = _tokenize(topic)
    anom_lookup = {a.anomaly_id: a for a in anoms}
    claim_lookup = {c.claim_id: c for c in claims}

    scored = [(h, _topic_relevance(h, anom_lookup, claim_lookup, query_tokens))
              for h in hyps]
    matched = [(h, r) for (h, r) in scored if r >= 1]
    non_degen = [(h, r) for (h, r) in matched
                 if not _is_degenerate_anomaly(anom_lookup.get(h.anomaly_id))]
    if non_degen:
        matched = non_degen

    def keep(h):
        ov = sidecar.get(h.hypothesis_id, 0)
        if ov == 0:
            return True  # unscored: defensive keep
        if mode == "off":
            return True
        if mode == "universal":
            return ov >= 3
        if mode == "type_specific":
            a = anom_by_id.get(h.anomaly_id)
            t = a.type if a else "unknown"
            if t in LEAKY_TYPES:
                return ov >= 3
            return True  # non-leaky type: no filter
        return True

    filtered = [(h, r) for (h, r) in matched if keep(h)]
    if filtered:
        matched = filtered

    matched.sort(key=lambda hr: -hr[1])

    per_anom = {}; capped = []
    for h, r in matched:
        n = per_anom.get(h.anomaly_id, 0) + 1
        per_anom[h.anomaly_id] = n
        if n <= 2:
            capped.append((h, r))
    matched = capped[:30]
    candidates = [h for h, _ in matched]
    breakdowns = score_all(candidates, anoms, claims)
    selected = select_mmr(candidates, breakdowns, k=k, lambda_=0.7, min_anomalies=2)
    return [h.hypothesis_id for h in selected]


def main():
    hyps = read_jsonl(RUN / "hypotheses_scored.jsonl", Hypothesis)
    anoms = read_jsonl(RUN / "anomalies.jsonl", Anomaly)
    claims = read_jsonl(RUN / "claims.jsonl", Claim)
    sidecar = _load_atlas_overlap_sidecar(RUN)
    anom_by_id = {a.anomaly_id: a for a in anoms}

    print(f"{'topic':>34}  off_leaks  uni_leaks  type_leaks  off-uni  off-type  uni-type")
    total_off_leaks = total_uni_leaks = total_type_leaks = 0
    total_off_uni = total_off_type = total_uni_type = 0

    for topic in TOPICS:
        off = run_arm(hyps, anoms, claims, topic, sidecar, anom_by_id, "off")
        uni = run_arm(hyps, anoms, claims, topic, sidecar, anom_by_id, "universal")
        ts = run_arm(hyps, anoms, claims, topic, sidecar, anom_by_id, "type_specific")

        def leaks(ids):
            return sum(1 for i in ids if 0 < sidecar.get(i, 0) < 3)

        ol = leaks(off); ul = leaks(uni); tl = leaks(ts)
        ou = len(set(off) ^ set(uni)); ot = len(set(off) ^ set(ts))
        ut = len(set(uni) ^ set(ts))
        total_off_leaks += ol; total_uni_leaks += ul; total_type_leaks += tl
        total_off_uni += ou; total_off_type += ot; total_uni_type += ut

        print(f"  {topic:>32}  {ol:>9}  {ul:>9}  {tl:>10}  {ou:>7}  {ot:>8}  {ut:>8}")

    n_slots = 8 * len(TOPICS)
    print()
    print(f"=== aggregate (12 topics × top-8 = {n_slots} slots) ===")
    print(f"  leaks OFF:           {total_off_leaks}/{n_slots} "
          f"({100*total_off_leaks/n_slots:.1f}%)")
    print(f"  leaks UNIVERSAL:     {total_uni_leaks}/{n_slots} "
          f"({100*total_uni_leaks/n_slots:.1f}%)")
    print(f"  leaks TYPE_SPECIFIC: {total_type_leaks}/{n_slots} "
          f"({100*total_type_leaks/n_slots:.1f}%)")
    print()
    print(f"  slot churn off→uni:  {total_off_uni}/{2*n_slots}")
    print(f"  slot churn off→type: {total_off_type}/{2*n_slots}")
    print(f"  slot churn uni→type: {total_uni_type}/{2*n_slots}  (lower = simpler+universal already sufficient)")


if __name__ == "__main__":
    main()
