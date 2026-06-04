"""Method 10 — Atlas conflict graph as a research-activity signal.

The 9.6M-edge Atlas conflicts parquet records improver→paper_b directed
relations on dimensions (generalization, accuracy, training_stability, …).
Each edge is "improver beats paper_b on dim".

This script probes whether the conflict graph adds an orthogonal signal
beyond the bottleneck_signals quotes (used in Methods 3-29). Two tests:

  1. Conflict-edge count per dimension as "research activity" proxy.
     A dim with millions of improver→paper_b edges is hotter than one with
     thousands. Compare to the joint_anomalies bottleneck_dim distribution.

  2. Per-dimension atlas_overlap distribution. Do hyps in high-conflict
     dims (generalization, computational_complexity) anchor differently
     than hyps in low-conflict dims?

If conflict-edge count adds signal beyond what bottleneck quotes already
capture, the conflict graph could supplement the filter or seed novel
anomaly selection. If they're redundant, the bottleneck quotes are enough.
"""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from aigraph.io import read_jsonl
from aigraph.models import Anomaly, Hypothesis
from aigraph_query import _load_atlas_overlap_sidecar


def main():
    run = _REPO / "artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1"
    parquet = _REPO / "artifacts/atlas_conflicts/atlas_conflicts.parquet"
    joint_anoms_path = _REPO / "artifacts/atlas_test/joint_anomalies.jsonl"

    # 1. Edge count per dim
    print("Loading 9.6M conflict edges...", file=sys.stderr)
    table = pq.read_table(parquet, columns=["dim", "joint_conf"])
    dims = table.column("dim").to_pylist()
    confs = table.column("joint_conf").to_pylist()
    dim_count = Counter(dims)
    print(f"\n=== Atlas conflict edge counts per dim (total {len(dims):,}) ===")
    for d, n in dim_count.most_common(15):
        print(f"  {d:>26}  {n:>10,}")
    # High-confidence subset
    hi_conf_dim_count = Counter(d for d, c in zip(dims, confs) if c and c >= 0.7)
    print(f"\n=== Same, restricted to joint_conf >= 0.7 (n={sum(hi_conf_dim_count.values()):,}) ===")
    for d, n in hi_conf_dim_count.most_common(10):
        print(f"  {d:>26}  {n:>10,}  ({100*n/dim_count[d]:.0f}% of dim's total)")

    # 2. Compare to joint_anomalies bottleneck_dim distribution
    joint = [json.loads(l) for l in open(joint_anoms_path)]
    anom_dim = Counter(j["shared_entities"].get("bottleneck_dimension") for j in joint)
    print(f"\n=== joint_anomalies.jsonl bottleneck_dim distribution (n={len(joint)}) ===")
    for d, n in anom_dim.most_common():
        ratio = anom_dim[d] / dim_count.get(d, 1)
        print(f"  {d:>26}  {n:>4}  ({1000*ratio:.2f} anomalies per 1000 conflicts)")

    # 3. atlas_overlap by dim — do high-conflict dims have different overlap
    # distributions?
    sidecar = _load_atlas_overlap_sidecar(run)
    anoms = read_jsonl(run / "anomalies.jsonl", Anomaly)
    hyps = read_jsonl(run / "hypotheses_scored.jsonl", Hypothesis)
    anom_by_id = {a.anomaly_id: a for a in anoms}
    # The 540p-thaw1 hyps' anomalies are aigraph-detected, not joint-Atlas;
    # they don't have bottleneck_dim. So Method 10's overlap-by-dim test only
    # makes sense on the creator hyps that derived from joint anomalies (the
    # jb… ids). The non-joint hyps just have type info.
    by_dim_overlap = defaultdict(list)
    for h in hyps:
        ov = sidecar.get(h.hypothesis_id)
        if ov is None: continue
        a = anom_by_id.get(h.anomaly_id)
        # See if this hyp's anomaly is a joint anomaly with a bottleneck dim
        # OR if the closest_atlas_quote dim is recorded in the sidecar
        # (we stored 'closest_dim' there).
        # Read sidecar directly for closest_dim:
        pass
    # Reload sidecar with closest_dim
    sc_full = []
    for line in open(run / "atlas_overlap.jsonl"):
        sc_full.append(json.loads(line))
    by_closest = defaultdict(list)
    for r in sc_full:
        if r.get("closest_dim"):
            by_closest[r["closest_dim"]].append(r["atlas_overlap"])
    print(f"\n=== atlas_overlap distribution by hyp's closest_atlas_dim ===")
    print(f"  {'dim':>26}  {'n':>4} {'mean':>6}  {'%≥3':>5}  {'%1+2':>5}  ({'atlas conflict count':>22})")
    for d in sorted(by_closest, key=lambda k: -len(by_closest[k])):
        ovs = by_closest[d]
        if len(ovs) < 5: continue
        anchored = sum(1 for o in ovs if o >= 3)
        leaks = sum(1 for o in ovs if o < 3)
        cc = dim_count.get(d, 0)
        print(f"  {d:>26}  {len(ovs):>4}  {mean(ovs):.2f}  {100*anchored/len(ovs):>4.0f}%  "
              f"{100*leaks/len(ovs):>4.0f}%  ({cc:>20,})")


if __name__ == "__main__":
    main()
