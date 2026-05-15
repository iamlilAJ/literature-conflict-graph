"""Q4 redo on validation_v1_primary cohort (NeurIPS/ICML/ICLR 2018-2020).

Same protocol as Q4 but uses 1790-paper validation cohort where
`replaces` edge density is expected to be much higher.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import polars as pl

SRC_CLAIMS = Path(
    "/Users/liuanjie/Documents/New project/hypothesis_generation/artifacts/runs/validation-v1-primary/claims.jsonl"
)
SRC_PAPERS = Path(
    "/Users/liuanjie/Documents/New project/hypothesis_generation/artifacts/runs/validation-v1-primary/papers.jsonl"
)
ATLAS_EDGES = "/Users/liuanjie/Documents/New project/hypothesis_generation/data/intern_atlas/data/paper_evolution_edges/*.parquet"
OUT_CSV = Path(__file__).parent / "Q4_redo_val1_primary_replaces_stance_xval.csv"

SEED = 17


def main():
    random.seed(SEED)

    aigraph_arxiv_ids = set()
    aigraph_pid_to_arxiv = {}
    for line in SRC_PAPERS.open():
        p = json.loads(line)
        aid = p.get("arxiv_id_base") or (p.get("arxiv_id_full", "").split("v")[0] if p.get("arxiv_id_full") else "")
        if aid:
            aigraph_arxiv_ids.add(aid)
            aigraph_pid_to_arxiv[p["paper_id"]] = aid
    print(f"val1_primary arxiv_ids: {len(aigraph_arxiv_ids)}")

    edges = (
        pl.scan_parquet(ATLAS_EDGES)
        .filter(pl.col("evolution_relation") == "replaces")
        .select(
            "paper_a_id", "paper_a_title", "paper_a_arxiv_id",
            "paper_b_id", "paper_b_title", "paper_b_arxiv_id",
            "bottleneck_json", "raw_reason",
        )
        .filter(pl.col("paper_a_arxiv_id").is_in(list(aigraph_arxiv_ids)))
        .filter(pl.col("paper_b_arxiv_id").is_in(list(aigraph_arxiv_ids)))
        .collect()
    )
    print(f"Atlas replaces edges with BOTH endpoints in val1_primary: {len(edges)}")

    if len(edges) == 0:
        print("ZERO intersection.")
        return

    n_sample = min(50, len(edges))
    sample = edges.sample(n_sample, seed=SEED).to_dicts()
    print(f"Sampled {len(sample)}")

    claims_by_paper = {}
    for line in SRC_CLAIMS.open():
        c = json.loads(line)
        pid = c.get("paper_id")
        claims_by_paper.setdefault(pid, []).append(c)
    arxiv_to_pid = {aid: pid for pid, aid in aigraph_pid_to_arxiv.items()}

    rows = []
    for e in sample:
        a_arxiv = e["paper_a_arxiv_id"]
        b_arxiv = e["paper_b_arxiv_id"]
        a_pid = arxiv_to_pid.get(a_arxiv)
        b_pid = arxiv_to_pid.get(b_arxiv)
        a_claims = claims_by_paper.get(a_pid, []) if a_pid else []
        b_claims = claims_by_paper.get(b_pid, []) if b_pid else []

        a_methods = {(c.get("canonical_method") or c.get("method") or "").lower() for c in a_claims}
        a_methods.discard("")
        a_tasks = {(c.get("canonical_task") or c.get("task") or "").lower() for c in a_claims}
        a_tasks.discard("")

        b_neg = [c for c in b_claims if c.get("direction") == "negative"]
        b_neg_methods = {(c.get("canonical_method") or c.get("method") or "").lower() for c in b_neg}
        b_neg_methods.discard("")
        b_neg_tasks = {(c.get("canonical_task") or c.get("task") or "").lower() for c in b_neg}
        b_neg_tasks.discard("")

        method_overlap = b_neg_methods & a_methods
        task_overlap = b_neg_tasks & a_tasks
        xval_signal = bool(method_overlap or task_overlap)

        rows.append({
            "paper_a_arxiv": a_arxiv,
            "paper_b_arxiv": b_arxiv,
            "paper_a_title": (e["paper_a_title"] or "")[:80],
            "paper_b_title": (e["paper_b_title"] or "")[:80],
            "a_claim_count": len(a_claims),
            "b_claim_count": len(b_claims),
            "b_negative_claim_count": len(b_neg),
            "method_overlap": ";".join(sorted(method_overlap)),
            "task_overlap": ";".join(sorted(task_overlap)),
            "xval_signal": xval_signal,
            "atlas_bottleneck": (e.get("bottleneck_json") or "")[:200],
        })

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_yes = sum(1 for r in rows if r["xval_signal"])
    print(f"\n=== Q4 redo (val1_primary 1790p) cross-validation rate ===")
    print(f"  rows with B's negative claim touching A's method/task: {n_yes}/{len(rows)} = {100*n_yes/max(1,len(rows)):.1f}%")
    print(f"  brief threshold: >30% supports J2, <10% suggests different reading")
    print(f"  vs Q4 original on 540p: 4 edges, 0/4 = 0%")


if __name__ == "__main__":
    main()
