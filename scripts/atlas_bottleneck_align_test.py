"""Empirical test of the J2 joint anomaly `bottleneck_open_q_alignment`.

Aligns Atlas's THIRD-PARTY bottleneck_json (paper B asserts paper P's
weakness) with aigraph's FIRST-PARTY weakness signal (P's own
negative-direction + limitation claims). Run on the val1-primary
cohort (1790 mature NeurIPS/ICML/ICLR 2018-2020 papers — the cohort the
recon recommended over the thin 540p reasoning slice).

Phase 1 — candidates + LLM 4-class judging (same protocol as recon Q2,
          scaled up since cost is not a concern here).
Phase 2 — for `complementary` papers, generate a JOINT hypothesis
          (aigraph anomaly + Atlas bottleneck) vs an aigraph-ONLY
          baseline, and have gpt-5.4 judge which is better grounded.
          This is the actual "does Atlas improve the deliverable" test.

Outputs under artifacts/atlas_test/.
gpt-5.4 via AIGRAPH_BASE_URL. Concurrency for throughput.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")
from openai import OpenAI  # noqa: E402

RUN = REPO / "artifacts/runs/validation-v1-primary"
ATLAS_EDGES = str(REPO / "data/intern_atlas/data/paper_evolution_edges/*.parquet")
OUT = REPO / "artifacts/atlas_test"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "gpt-5.4"
SEED = 17

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["AIGRAPH_BASE_URL"])


def _chat(system: str, user: str, max_tokens: int = 400) -> str:
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=0,
    )
    return resp.choices[0].message.content or ""


def _parse_json(raw: str) -> dict:
    s, e = raw.find("{"), raw.rfind("}")
    try:
        return json.loads(raw[s:e + 1]) if s != -1 else {}
    except Exception:
        return {}


def _arxiv(p: dict) -> str:
    return (p.get("arxiv_id_base")
            or (p.get("arxiv_id_full", "").split("v")[0] if p.get("arxiv_id_full") else "")
            or (p["paper_id"].split(":", 1)[1] if str(p.get("paper_id", "")).startswith("arxiv:") else ""))


def _bottleneck_fields(raw: str) -> dict:
    try:
        o = json.loads(raw)
        if isinstance(o, dict):
            return {"description": o.get("description") or o.get("quote") or "",
                    "dimension": o.get("dimension") or "", "severity": o.get("severity") or ""}
    except Exception:
        pass
    return {"description": (raw or "")[:300], "dimension": "", "severity": ""}


def build_candidates():
    papers = [json.loads(l) for l in (RUN / "papers.jsonl").open()]
    claims = [json.loads(l) for l in (RUN / "claims.jsonl").open()]
    arxiv_to_pid, pid_to_paper = {}, {}
    for p in papers:
        a = _arxiv(p)
        if a:
            arxiv_to_pid[a] = p["paper_id"]
        pid_to_paper[p["paper_id"]] = p

    weakness_by_pid: dict[str, list] = {}
    claims_by_pid: dict[str, list] = {}
    for c in claims:
        claims_by_pid.setdefault(c["paper_id"], []).append(c)
        if c.get("direction") == "negative" or c.get("claim_type") == "limitation":
            weakness_by_pid.setdefault(c["paper_id"], []).append(c)

    run_arxiv = list(arxiv_to_pid.keys())
    print(f"scanning Atlas inbound bottlenecks for {len(run_arxiv)} papers...", file=sys.stderr)
    edges = (pl.scan_parquet(ATLAS_EDGES)
             .filter(pl.col("paper_b_arxiv_id").is_in(run_arxiv))
             .filter(pl.col("bottleneck_json").is_not_null() & (pl.col("bottleneck_json") != ""))
             .select("paper_b_arxiv_id", "paper_a_arxiv_id", "paper_a_title",
                     "evolution_relation", "relation_confidence", "bottleneck_json")
             .collect())
    print(f"Atlas inbound bottleneck edges on cohort: {edges.height}", file=sys.stderr)

    btl_by_arxiv: dict[str, list] = {}
    for r in edges.to_dicts():
        btl_by_arxiv.setdefault(r["paper_b_arxiv_id"], []).append(r)

    cands = []
    for a, btls in btl_by_arxiv.items():
        pid = arxiv_to_pid.get(a)
        if pid and pid in weakness_by_pid:
            cands.append({"arxiv_id": a, "paper_id": pid, "title": pid_to_paper[pid].get("title", ""),
                          "n_bottlenecks": len(btls), "n_weakness": len(weakness_by_pid[pid])})
    return cands, btl_by_arxiv, weakness_by_pid, claims_by_pid, pid_to_paper


JUDGE_SYS = (
    "You compare two independent observations about an AI paper P's weaknesses.\n"
    "OBSERVATION_A: bottlenecks asserted by THIRD-PARTY papers about P (Atlas).\n"
    "OBSERVATION_B: P's OWN first-party limitations / negative-direction claims (aigraph).\n\n"
    "Compare the two SETS. Choose ONE label:\n"
    "- same_signal: same weakness(es); redundant.\n"
    "- complementary: DIFFERENT but compatible weaknesses; together richer.\n"
    "- unrelated: different subject matter; no overlap.\n"
    "- contradictory: A asserts weakness W but B denies W (or vice versa).\n\n"
    'STRICT JSON: {"label":"<one_of_4>","rationale":"<one sentence>"}'
)


def judge_one(cand, btl_by_arxiv, weakness_by_pid):
    a = cand["arxiv_id"]
    btls = btl_by_arxiv[a][:3]
    a_text = "\n".join(
        f"  - (from {b.get('paper_a_arxiv_id') or '?'}, rel={b['evolution_relation']}, dim={_bottleneck_fields(b['bottleneck_json'])['dimension']}): "
        f"{_bottleneck_fields(b['bottleneck_json'])['description'][:300]}" for b in btls)
    wk = weakness_by_pid[cand["paper_id"]][:3]
    b_text = "\n".join(f"  - [{c.get('claim_type','?')}/{c.get('direction','?')}] {c.get('claim_text','')[:250]}" for c in wk)
    user = (f"Paper P: arxiv:{a}\n\nOBSERVATION_A (Atlas third-party bottlenecks):\n{a_text}\n\n"
            f"OBSERVATION_B (aigraph first-party weakness):\n{b_text}\n\nChoose ONE relation label.")
    obj = _parse_json(_chat(JUDGE_SYS, user, 300))
    return {**cand, "atlas_bottlenecks": a_text, "aigraph_weakness": b_text,
            "label": obj.get("label", "ERROR"), "rationale": obj.get("rationale", "")}


def dump_candidates(out_path: str, n: int):
    """Build candidates + pre-render the A/B text blocks, write JSONL.
    No LLM — judging happens on the remote (the LLM endpoint is only
    reachable there). Each row carries everything the judge needs."""
    random.seed(SEED)
    cands, btl_by_arxiv, weakness_by_pid, claims_by_pid, pid_to_paper = build_candidates()
    print(f"CANDIDATES (Atlas bottleneck ∧ aigraph weakness): {len(cands)}", file=sys.stderr)
    sample = cands if n <= 0 or n >= len(cands) else random.sample(cands, n)
    rows = []
    for c in sample:
        a = c["arxiv_id"]
        btls = btl_by_arxiv[a][:3]
        a_text = "\n".join(
            f"  - (from {b.get('paper_a_arxiv_id') or '?'}, rel={b['evolution_relation']}, "
            f"dim={_bottleneck_fields(b['bottleneck_json'])['dimension']}, "
            f"sev={_bottleneck_fields(b['bottleneck_json'])['severity']}): "
            f"{_bottleneck_fields(b['bottleneck_json'])['description'][:300]}" for b in btls)
        wk = weakness_by_pid[c["paper_id"]][:3]
        b_text = "\n".join(
            f"  - [{w.get('claim_type','?')}/{w.get('direction','?')}] {w.get('claim_text','')[:250]}" for w in wk)
        rows.append({"paper_id": c["paper_id"], "arxiv_id": a, "title": c["title"],
                     "n_bottlenecks": c["n_bottlenecks"], "n_weakness": c["n_weakness"],
                     "atlas_text": a_text, "aigraph_text": b_text})
    Path(out_path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    meta = {"cohort": "val1-primary", "n_candidates": len(cands), "n_dumped": len(rows)}
    Path(out_path + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(rows)} candidate prompts -> {out_path}", file=sys.stderr)
    print(json.dumps(meta))


def phase1(n: int, workers: int):
    random.seed(SEED)
    cands, btl_by_arxiv, weakness_by_pid, _, _ = build_candidates()
    print(f"CANDIDATES (papers with BOTH Atlas bottleneck AND aigraph weakness): {len(cands)}", file=sys.stderr)
    sample = cands if n <= 0 or n >= len(cands) else random.sample(cands, n)
    print(f"judging {len(sample)} with {MODEL} ({workers} workers)...", file=sys.stderr)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge_one, c, btl_by_arxiv, weakness_by_pid): c for c in sample}
        for i, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if i % 20 == 0:
                print(f"  {i}/{len(sample)}", file=sys.stderr)
    dist = Counter(r["label"] for r in rows)
    total = len(rows)
    summary = {"cohort": "val1-primary", "model": MODEL, "n_candidates": len(cands),
               "n_judged": total, "distribution": dict(dist),
               "pct": {k: round(100 * v / total, 1) for k, v in dist.items()}}
    (OUT / "phase1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (OUT / "phase1_rows.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (OUT / "phase1_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print("\n=== PHASE 1: bottleneck_open_q_alignment 4-class on val1-primary ===")
    print(f"candidates (joint-anomaly firing count): {len(cands)}")
    for k, v in dist.most_common():
        print(f"  {k}: {v}/{total} = {100*v/total:.1f}%")
    print(f"wrote {OUT/'phase1_summary.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--n", type=int, default=150, help="sample size; <=0 = all candidates")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dump", type=str, default="", help="phase 0: write candidate prompts JSONL here (no LLM)")
    args = ap.parse_args()
    if args.dump:
        dump_candidates(args.dump, args.n)
    elif args.phase == 1:
        phase1(args.n, args.workers)
