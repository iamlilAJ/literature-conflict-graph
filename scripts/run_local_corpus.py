"""End-to-end pipeline run on a pre-built local corpus.

Loads papers + artifacts from a corpus directory (e.g.
``data/corpus/arxiv_reasoning/``), filters to a topic-relevant subset,
and runs the v0.7-frozen aigraph pipeline:

    papers -> claim extract -> graph build -> anomaly detect
    -> hierarchy -> hypothesis generate -> influence score -> MMR select

Outputs go to ``<out>/``. Composes the existing ``aigraph`` CLI subcommands
via subprocess; no frozen module is touched.

Example::

    python3 scripts/run_local_corpus.py \
        --corpus data/corpus/arxiv_reasoning \
        --max-papers 100 \
        --year-min 2023 \
        --keywords "reasoning,chain-of-thought,planning,tool use,agent" \
        --out artifacts/runs/arxiv-reasoning-v0.7-100p \
        --generator llm
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_artifact_id(paper_id: str) -> str:
    """Mirror aigraph.corpus.artifact_dir naming."""
    return paper_id.replace(":", "__").replace("/", "_")


def filter_papers(
    corpus_dir: Path,
    year_min: int,
    keywords: list[str],
    max_papers: int,
) -> list[dict]:
    """Read corpus papers.jsonl, keep rows that:
    - have year >= year_min
    - title or abstract contains any keyword (case-insensitive)
    - have a corresponding artifacts/<safe_id>/sections.json on disk
    Sort by (year desc, paper_id asc), take first ``max_papers``.
    """
    papers_path = corpus_dir / "papers.jsonl"
    if not papers_path.exists():
        raise FileNotFoundError(f"corpus papers.jsonl missing: {papers_path}")
    artifacts_root = corpus_dir / "artifacts"

    needles = [k.strip().lower() for k in keywords if k.strip()]
    candidates: list[dict] = []
    seen_no_sections = 0
    with papers_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            year = p.get("year") or 0
            if year < year_min:
                continue
            haystack = ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()
            if not any(n in haystack for n in needles):
                continue
            sections = artifacts_root / _safe_artifact_id(p.get("paper_id", "")) / "sections.json"
            if not sections.exists():
                seen_no_sections += 1
                continue
            candidates.append(p)
    candidates.sort(key=lambda r: (-(r.get("year") or 0), r.get("paper_id", "")))
    print(
        f"[filter] {len(candidates)} candidates pass year+keyword+sections "
        f"(skipped {seen_no_sections} matches that lacked parsed sections)",
        flush=True,
    )
    return candidates[:max_papers]


def _run_stage(name: str, cmd: list[str]) -> None:
    """Run a CLI stage; raise CalledProcessError on non-zero exit."""
    print(f"\n=== [{name}] {' '.join(cmd)} ===", flush=True)
    t0 = time.monotonic()
    subprocess.run(cmd, check=True)
    print(f"=== [{name}] done in {time.monotonic() - t0:.1f}s ===", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--max-papers", type=int, default=100)
    ap.add_argument("--year-min", type=int, default=2023)
    ap.add_argument("--keywords", required=True, type=str)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--generator", default="llm", choices=["llm", "template"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mmr-lambda", type=float, default=0.7)
    args = ap.parse_args()

    corpus_dir = args.corpus
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    started_at = _now_iso()
    t0 = time.monotonic()

    # Stage 1: filter corpus -> papers.jsonl
    selected = filter_papers(corpus_dir, args.year_min, keywords, args.max_papers)
    if not selected:
        print("[filter] no papers passed filter; aborting.", file=sys.stderr)
        sys.exit(2)
    papers_path = out_dir / "papers.jsonl"
    with papers_path.open("w") as f:
        for p in selected:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[filter] wrote {len(selected)} papers -> {papers_path}", flush=True)

    # AIGRAPH_CORPUS_ROOT lets the extractor find artifacts/<id>/sections.json
    env_root = str(corpus_dir.resolve())
    os.environ["AIGRAPH_CORPUS_ROOT"] = env_root
    print(f"[env ] AIGRAPH_CORPUS_ROOT={env_root}", flush=True)

    claims_path = out_dir / "claims.jsonl"
    graph_path = out_dir / "graph.json"
    anomalies_path = out_dir / "anomalies.jsonl"
    hypotheses_path = out_dir / "hypotheses.jsonl"
    hierarchy_path = out_dir / "hierarchy.json"
    scored_path = out_dir / "hypotheses_scored.jsonl"
    report_path = out_dir / "selected_hypotheses.md"

    aigraph = [sys.executable, "-m", "aigraph.cli"]

    extract_cmd = aigraph + [
        "extract",
        "--input", str(papers_path),
        "--output", str(claims_path),
        "--extractor", "llm",
        "--reader", "heuristic",
        "--workers", str(args.workers),
    ]
    if args.model:
        extract_cmd += ["--model", args.model]
    _run_stage("extract", extract_cmd)

    _run_stage("build-graph", aigraph + [
        "build-graph",
        "--claims", str(claims_path),
        "--papers", str(papers_path),
        "--output", str(graph_path),
    ])

    _run_stage("detect-anomalies", aigraph + [
        "detect-anomalies",
        "--graph", str(graph_path),
        "--claims", str(claims_path),
        "--output", str(anomalies_path),
    ])

    gen_cmd = aigraph + [
        "generate-hypotheses",
        "--anomalies", str(anomalies_path),
        "--claims", str(claims_path),
        "--output", str(hypotheses_path),
        "--generator", args.generator,
    ]
    if args.model:
        gen_cmd += ["--model", args.model]
    _run_stage("generate-hypotheses", gen_cmd)

    _run_stage("build-hierarchy", aigraph + [
        "build-hierarchy",
        "--claims", str(claims_path),
        "--papers", str(papers_path),
        "--anomalies", str(anomalies_path),
        "--graph", str(graph_path),
        "--output", str(hierarchy_path),
    ])

    _run_stage("predict-influence", aigraph + [
        "predict-influence",
        "--hypotheses", str(hypotheses_path),
        "--claims", str(claims_path),
        "--hierarchy", str(hierarchy_path),
        "--output", str(scored_path),
    ])

    _run_stage("select", aigraph + [
        "select",
        "--hypotheses", str(scored_path),
        "--claims", str(claims_path),
        "--anomalies", str(anomalies_path),
        "--papers", str(papers_path),
        "--k", str(args.k),
        "--lambda", str(args.mmr_lambda),
        "--output", str(report_path),
    ])

    finished_at = _now_iso()
    wall = int(time.monotonic() - t0)

    # Anomaly type breakdown for the report-back / thaw-flag check
    anomaly_type_counts: dict[str, int] = {}
    if anomalies_path.exists():
        with anomalies_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                t = rec.get("type") or "unknown"
                anomaly_type_counts[t] = anomaly_type_counts.get(t, 0) + 1

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    git_tag = subprocess.run(
        ["git", "tag", "--points-at", git_sha], capture_output=True, text=True
    ).stdout.strip().split("\n")[0] or "v0.7-frozen"

    metadata = {
        "topic": "LLM reasoning (arxiv 2023+)",
        "corpus_dir": str(corpus_dir),
        "n_papers_in_corpus": sum(1 for _ in (corpus_dir / "papers.jsonl").open()),
        "n_papers_filtered_in": len(selected),
        "n_papers_selected": len(selected),
        "year_min": args.year_min,
        "keywords": keywords,
        "git_sha": git_sha,
        "git_tag": git_tag,
        "model": args.model or os.environ.get("AIGRAPH_MODEL", "(env-default)"),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": wall,
        "anomaly_type_counts": anomaly_type_counts,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print(f"\n[done] wall={wall}s  out={out_dir}/", flush=True)
    print(f"       anomaly types: {anomaly_type_counts}", flush=True)


if __name__ == "__main__":
    main()
