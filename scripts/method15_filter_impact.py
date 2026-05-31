"""Method 15 — Atlas overlap filter deployment impact across diverse topics.

For each of N curated topics, run query_records(min_atlas_overlap=0) and
query_records(min_atlas_overlap=3) and characterize the difference:

  - n_matched before/after
  - top-K membership churn
  - distribution of dropped vs surfaced overlaps
  - which anomaly types tend to lose hyps

Tells us what end users actually experience when the filter is enabled.
No new LLM calls.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from aigraph_query import query_records, _load_atlas_overlap_sidecar  # noqa: E402

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


def main():
    sidecar = _load_atlas_overlap_sidecar(RUN)
    print(f"sidecar entries: {len(sidecar)}")
    print(f"global overlap distribution: {dict(Counter(sidecar.values()))}")
    print()

    summary = []
    for topic in TOPICS:
        off, off_stats = query_records(RUN, topic, k=8, min_atlas_overlap=0)
        on, on_stats = query_records(RUN, topic, k=8, min_atlas_overlap=3)
        if not off and not on:
            continue
        off_ids = [r["hypothesis_id"] for r in off]
        on_ids = [r["hypothesis_id"] for r in on]
        off_ovr = [sidecar.get(h, 0) for h in off_ids]
        on_ovr = [sidecar.get(h, 0) for h in on_ids]
        dropped = sorted(set(off_ids) - set(on_ids))
        surfaced = sorted(set(on_ids) - set(off_ids))
        # Were any of the off-top-8 hyps "leaks" (overlap 1 or 2)?
        leaks_off = sum(1 for o in off_ovr if 0 < o < 3)
        leaks_on = sum(1 for o in on_ovr if 0 < o < 3)
        summary.append({
            "topic": topic,
            "n_matched_off": off_stats["n_matched"],
            "n_matched_on": on_stats["n_matched"],
            "off_top8_overlaps": off_ovr,
            "on_top8_overlaps": on_ovr,
            "leaks_in_off_top8": leaks_off,
            "leaks_in_on_top8": leaks_on,
            "dropped": dropped,
            "surfaced": surfaced,
            "top8_churn": len(set(off_ids) ^ set(on_ids)),
        })

    print(f"{'topic':>34} {'matched':>14} {'off-ovr':>22} {'on-ovr':>22} "
          f"{'leaks→':>8} {'churn':>6}")
    for s in summary:
        ms = f"{s['n_matched_off']}→{s['n_matched_on']}"
        leakd = f"{s['leaks_in_off_top8']}→{s['leaks_in_on_top8']}"
        print(f"  {s['topic']:>32} {ms:>14} {str(s['off_top8_overlaps']):>22} "
              f"{str(s['on_top8_overlaps']):>22} {leakd:>8} {s['top8_churn']:>6}")

    # Aggregate
    total_leaks_off = sum(s["leaks_in_off_top8"] for s in summary)
    total_leaks_on = sum(s["leaks_in_on_top8"] for s in summary)
    n_topics = len(summary)
    n_total_slots = 8 * n_topics
    print()
    print(f"=== aggregate over {n_topics} topics × top-8 = {n_total_slots} slots ===")
    print(f"  leaks before (overlap=1 or 2): {total_leaks_off}/{n_total_slots} "
          f"({100*total_leaks_off/n_total_slots:.1f}%)")
    print(f"  leaks after  (overlap=1 or 2): {total_leaks_on}/{n_total_slots} "
          f"({100*total_leaks_on/n_total_slots:.1f}%)")
    print(f"  total top-8 churn (different ids): "
          f"{sum(s['top8_churn'] for s in summary)} / {2*n_total_slots}")

    # Write CSV for further analysis
    out = _REPO / "artifacts/atlas_test/method15_impact.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
