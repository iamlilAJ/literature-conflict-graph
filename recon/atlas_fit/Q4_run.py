"""Q4: Atlas `replaces` ↔ aigraph stance cross-validation.

Sample 50 Atlas `replaces` edges where BOTH endpoints are in aigraph 540p corpus.
For each: does aigraph independently extract any negative-stance claim B→A?

Output: 50-row CSV + aggregate rate.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import polars as pl

SRC_CLAIMS = Path(
    "/Users/liuanjie/Documents/New project/hypothesis_generation/artifacts/runs/arxiv-reasoning-v0.7-540p/claims.jsonl"
)
SRC_PAPERS = Path(
    "/Users/liuanjie/Documents/New project/hypothesis_generation/artifacts/runs/arxiv-reasoning-v0.7-540p/papers.jsonl"
)
ATLAS_EDGES_GLOB = "/Users/liuanjie/Documents/New project/hypothesis_generation/data/intern_atlas/data/paper_evolution_edges/*.parquet"
OUT_CSV = Path(__file__).parent / "Q4_replaces_stance_xval.csv"

SEED = 17


def main():
    random.seed(SEED)

    # 1. Build the set of paper_ids in 540p run (matched via arxiv_id_base)
    aigraph_arxiv_ids = set()
    aigraph_pid_to_arxiv = {}
    for line in SRC_PAPERS.open():
        p = json.loads(line)
        aid = p.get("arxiv_id_base") or p.get("arxiv_id_full")
        if aid:
            # Normalize to just the base
            aid_clean = aid.replace("v1", "").replace("v2", "").replace("v3", "").replace("v4", "").replace("v5", "")
            # Actually keep both forms
            for s in {aid, aid_clean}:
                aigraph_arxiv_ids.add(s)
                aigraph_pid_to_arxiv[p["paper_id"]] = aid_clean
    print(f"540p arxiv_ids: {len(aigraph_arxiv_ids)}")

    # 2. Filter Atlas replaces edges to both-endpoints-in-540p
    edges = (
        pl.scan_parquet(ATLAS_EDGES_GLOB)
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
    print(f"Atlas replaces edges with BOTH endpoints in 540p: {len(edges)}")

    if len(edges) == 0:
        print("ZERO intersection — Atlas/540p have no replaces edges in common. J2/J3 mooted by data.")
        # write empty CSV
        with OUT_CSV.open("w") as f:
            f.write("# Atlas replaces × 540p intersection: 0\n")
        return

    # 3. Sample up to 50
    n_sample = min(50, len(edges))
    sample = edges.sample(n_sample, seed=SEED).to_dicts()
    print(f"Sampled {len(sample)} edges to cross-check")

    # 4. Load aigraph claims indexed by paper_id
    claims_by_paper = {}
    for line in SRC_CLAIMS.open():
        c = json.loads(line)
        pid = c.get("paper_id")
        claims_by_paper.setdefault(pid, []).append(c)
    # We need arxiv_id → paper_id reverse map
    arxiv_to_pid = {}
    for pid, aid in aigraph_pid_to_arxiv.items():
        if aid: arxiv_to_pid[aid] = pid

    # 5. For each replaces edge (a→b means b replaces a per Atlas convention?),
    #    look for negative-stance claims about paper A IN paper B's claim set.
    #    Atlas convention: paper_a is the OLDER one, paper_b CITES/IMPROVES it.
    #    "replaces" means paper_b's method replaces paper_a's. So negative claims
    #    about paper_a should appear in paper_b's text — but aigraph extracts claims
    #    OF a paper, not ABOUT external papers. So we check: does paper_b have any
    #    negative-direction claim, AND does paper_b's claim's method/task overlap
    #    with paper_a's claim method/task?
    rows = []
    for e in sample:
        a_arxiv = e["paper_a_arxiv_id"]
        b_arxiv = e["paper_b_arxiv_id"]
        a_pid = arxiv_to_pid.get(a_arxiv)
        b_pid = arxiv_to_pid.get(b_arxiv)
        a_claims = claims_by_paper.get(a_pid, []) if a_pid else []
        b_claims = claims_by_paper.get(b_pid, []) if b_pid else []

        # paper_a's methods + tasks
        a_methods = {(c.get("canonical_method") or c.get("method") or "").lower() for c in a_claims}
        a_methods.discard("")
        a_tasks = {(c.get("canonical_task") or c.get("task") or "").lower() for c in a_claims}
        a_tasks.discard("")

        # paper_b's negative-direction claims
        b_neg = [c for c in b_claims if c.get("direction") == "negative"]
        b_neg_methods = {(c.get("canonical_method") or c.get("method") or "").lower() for c in b_neg}
        b_neg_methods.discard("")
        b_neg_tasks = {(c.get("canonical_task") or c.get("task") or "").lower() for c in b_neg}
        b_neg_tasks.discard("")

        # Overlap signal: do B's negative claims touch A's methods OR tasks?
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

    # 6. Write CSV
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 7. Stats
    n_yes = sum(1 for r in rows if r["xval_signal"])
    print(f"\n=== Q4 cross-validation rate ===")
    print(f"  rows with B's negative claim touching A's method/task: {n_yes}/{len(rows)} = {100*n_yes/max(1,len(rows)):.1f}%")
    print(f"  brief threshold: > 30% supports J2, < 10% suggests very different reading")
    print(f"  verdict: {'GO (supports J2)' if n_yes/max(1,len(rows)) > 0.30 else ('INCONCLUSIVE' if n_yes/max(1,len(rows)) > 0.10 else 'KILL (different reading)')}")
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
