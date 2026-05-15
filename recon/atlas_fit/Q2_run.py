"""Q2 — Atlas weakness signal × aigraph weakness signal overlap.

For 30 papers in (aigraph 540p ∩ Atlas) where both sides have weakness
signal:
  - Atlas: bottleneck_json from inbound edges (paper_b = P)
  - aigraph: P's negative-direction claims + limitation-type claims
            (proxy for "first-party weakness" since open_questions
             were not extracted on the 540p run)

LLM judge each PAPER (1 row per paper) on dominant relation:
{same_signal, complementary, unrelated, contradictory}.

Model: gpt-5.4 (Sonnet-tier substitution; endpoint blocks Anthropic).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

load_dotenv("/Users/liuanjie/Documents/New project/hypothesis_generation/.env")
from openai import OpenAI

SRC_CLAIMS = Path(
    "/Users/liuanjie/Documents/New project/hypothesis_generation/artifacts/runs/arxiv-reasoning-v0.7-540p/claims.jsonl"
)
SRC_PAPERS = Path(
    "/Users/liuanjie/Documents/New project/hypothesis_generation/artifacts/runs/arxiv-reasoning-v0.7-540p/papers.jsonl"
)
ATLAS_EDGES = "/Users/liuanjie/Documents/New project/hypothesis_generation/data/intern_atlas/data/paper_evolution_edges/*.parquet"
OUT_CSV = Path(__file__).parent / "Q2_weakness_overlap.csv"
JUDGE_MODEL = "gpt-5.4"
SEED = 17
N_TARGET = 30


def main():
    random.seed(SEED)

    # Build 540p arxiv_id → paper_id reverse map
    aigraph_arxiv_to_pid = {}
    aigraph_pid_to_arxiv = {}
    for line in SRC_PAPERS.open():
        p = json.loads(line)
        aid = p.get("arxiv_id_base") or (p.get("arxiv_id_full", "").split("v")[0] if p.get("arxiv_id_full") else "")
        if aid:
            aigraph_arxiv_to_pid[aid] = p["paper_id"]
            aigraph_pid_to_arxiv[p["paper_id"]] = aid

    # Load aigraph claims by paper_id, filter to weakness signals
    weakness_claims_by_pid = {}
    for line in SRC_CLAIMS.open():
        c = json.loads(line)
        pid = c.get("paper_id")
        if c.get("direction") == "negative" or c.get("claim_type") == "limitation":
            weakness_claims_by_pid.setdefault(pid, []).append(c)

    print(f"aigraph papers with weakness signal: {len(weakness_claims_by_pid)}", file=sys.stderr)

    # Atlas bottlenecks by P (where paper_b = P)
    print("scanning Atlas edges for inbound bottlenecks...", file=sys.stderr)
    atlas_540p = list(aigraph_arxiv_to_pid.keys())
    atlas_edges = (
        pl.scan_parquet(ATLAS_EDGES)
        .filter(pl.col("paper_b_arxiv_id").is_in(atlas_540p))
        .filter(pl.col("bottleneck_json").is_not_null() & (pl.col("bottleneck_json") != ""))
        .select(
            "paper_b_arxiv_id", "paper_a_arxiv_id", "paper_a_title",
            "evolution_relation", "bottleneck_json",
        )
        .collect()
    )
    print(f"Atlas inbound edges with bottleneck on 540p papers: {atlas_edges.height}", file=sys.stderr)

    # Group bottlenecks by target paper
    bottlenecks_by_p = {}
    for row in atlas_edges.to_dicts():
        p_arxiv = row["paper_b_arxiv_id"]
        bottlenecks_by_p.setdefault(p_arxiv, []).append(row)

    # Candidate papers: have BOTH sides
    candidates = []
    for p_arxiv, btls in bottlenecks_by_p.items():
        pid = aigraph_arxiv_to_pid.get(p_arxiv)
        if pid and pid in weakness_claims_by_pid:
            candidates.append({
                "arxiv_id": p_arxiv,
                "paper_id": pid,
                "n_bottlenecks": len(btls),
                "n_weakness_claims": len(weakness_claims_by_pid[pid]),
            })

    print(f"papers with BOTH atlas bottlenecks AND aigraph weakness claims: {len(candidates)}", file=sys.stderr)
    if len(candidates) < N_TARGET:
        print(f"warning: only {len(candidates)} candidates, less than target {N_TARGET}", file=sys.stderr)
    sample = random.sample(candidates, min(N_TARGET, len(candidates)))
    print(f"sampled {len(sample)} for Q2", file=sys.stderr)

    # Build prompts and call LLM
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["AIGRAPH_BASE_URL"],
    )
    SYSTEM_PROMPT = (
        "You are an analyst comparing two independent observations about an AI paper's weaknesses.\n"
        "OBSERVATION_A: bottlenecks asserted by THIRD-PARTY papers about paper P (from Atlas).\n"
        "OBSERVATION_B: paper P's own first-party limitations and negative-direction claims (from aigraph).\n\n"
        "Compare the two SETS as a whole. Choose ONE relation label:\n"
        "- same_signal: A and B identify substantively the same weakness(es). Redundant.\n"
        "- complementary: A and B identify DIFFERENT but compatible weaknesses; together richer than either.\n"
        "- unrelated: A and B are about different topics; no overlap in subject matter.\n"
        "- contradictory: A asserts weakness W, but B explicitly DENIES W (or vice versa); they conflict.\n\n"
        "Output STRICT JSON:\n"
        "{\"label\": \"<one_of_4>\", \"rationale\": \"<one sentence>\"}"
    )

    def parse_bottleneck(raw: str) -> str:
        """Pull description text from bottleneck_json (which is a JSON string)."""
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj.get("description") or obj.get("quote") or raw[:200]
            return str(obj)[:200]
        except Exception:
            return raw[:200]

    rows = []
    for i, cand in enumerate(sample):
        if i % 5 == 0:
            print(f"  judging {i}/{len(sample)}", file=sys.stderr)
        pid = cand["paper_id"]
        p_arxiv = cand["arxiv_id"]

        # Atlas side: top 3 bottlenecks
        btls = bottlenecks_by_p[p_arxiv][:3]
        a_text = "\n".join(
            f"  - (from {b.get('paper_a_arxiv_id') or 'unknown'}, rel={b['evolution_relation']}): "
            f"{parse_bottleneck(b['bottleneck_json'])[:300]}"
            for b in btls
        )

        # aigraph side: top 3 weakness claims
        claims = weakness_claims_by_pid[pid][:3]
        b_text = "\n".join(
            f"  - [{c.get('claim_type','?')} / {c.get('direction','?')}] {c.get('claim_text', '')[:250]}"
            for c in claims
        )

        user = (
            f"Paper P: arxiv:{p_arxiv}\n\n"
            f"OBSERVATION_A (Atlas — third-party bottlenecks about P):\n{a_text}\n\n"
            f"OBSERVATION_B (aigraph — P's own limitations / negative claims):\n{b_text}\n\n"
            "Compare the two SETS and choose ONE relation label."
        )

        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_tokens=300,
                temperature=0,
            )
            raw = resp.choices[0].message.content
            s, e = raw.find("{"), raw.rfind("}")
            obj = json.loads(raw[s:e+1]) if s != -1 else {}
            label = obj.get("label")
            rationale = obj.get("rationale")
        except Exception as exc:
            label = "ERROR"
            rationale = f"call failed: {str(exc)[:100]}"

        rows.append({
            "paper_id": pid,
            "arxiv_id": p_arxiv,
            "n_bottlenecks": cand["n_bottlenecks"],
            "n_weakness_claims": cand["n_weakness_claims"],
            "atlas_bottlenecks_top3": a_text,
            "aigraph_weakness_top3": b_text,
            "judge_label": label,
            "judge_rationale": rationale,
        })

    # Write CSV (multi-line cells — use json.dumps escape)
    import csv
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Stats
    from collections import Counter
    counter = Counter(r["judge_label"] for r in rows)
    print(f"\n=== Q2 4-class distribution ===", file=sys.stderr)
    for label, n in counter.most_common():
        print(f"  {label}: {n}/{len(rows)} = {100*n/len(rows):.1f}%", file=sys.stderr)
    print(f"\nwrote {OUT_CSV}", file=sys.stderr)

    # Save also json with full pairs for memo reference
    out_json = OUT_CSV.with_suffix(".rows.json")
    out_json.write_text(json.dumps({
        "model": JUDGE_MODEL,
        "seed": SEED,
        "n_candidates_total": len(candidates),
        "n_sampled": len(sample),
        "distribution": dict(counter),
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    print(f"wrote {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
