"""Continuation runner for run_local_corpus.py outputs.

Use this after a run has produced papers/claims/graph/anomalies but
before/instead of generate-hypotheses, when the anomaly count blew up
beyond the spec (~50) and the serial generate-hypotheses pipeline
would take hours.

Pipeline:
    1. Read <out>/anomalies.jsonl, filter top-N by topology_score,
       write anomalies_top.jsonl
    2. Run generate-hypotheses on anomalies_top.jsonl
    3. Run build-hierarchy on full anomalies set
    4. Run predict-influence
    5. Run select
    6. Update run_metadata.json with the cap + per-type counts

Composes existing aigraph CLI subcommands; no frozen module touched.

Example::

    python3 scripts/finish_local_run.py \
        --out artifacts/runs/arxiv-reasoning-v0.7-100p \
        --max-anomalies 50 \
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_stage(name: str, cmd: list[str]) -> None:
    print(f"\n=== [{name}] {' '.join(cmd)} ===", flush=True)
    t0 = time.monotonic()
    subprocess.run(cmd, check=True)
    print(f"=== [{name}] done in {time.monotonic() - t0:.1f}s ===", flush=True)


def cap_anomalies_top_n(anomalies_path: Path, n: int) -> tuple[Path, dict[str, int]]:
    """Read full anomalies, sort by topology_score desc, take top N.
    Returns (filtered_path, type_counter_for_full_set)."""
    rows: list[dict] = []
    with anomalies_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    full_count_by_type: dict[str, int] = {}
    for r in rows:
        t = r.get("type") or "unknown"
        full_count_by_type[t] = full_count_by_type.get(t, 0) + 1
    rows.sort(key=lambda r: -float(r.get("topology_score") or 0.0))
    top = rows[:n]
    out = anomalies_path.parent / "anomalies_top.jsonl"
    with out.open("w") as f:
        for r in top:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"[cap] {len(rows)} anomalies -> top {len(top)} by topology_score "
        f"(types in full set: {full_count_by_type})",
        flush=True,
    )
    return out, full_count_by_type


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-anomalies", type=int, default=50)
    ap.add_argument("--generator", default="llm", choices=["llm", "template"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mmr-lambda", type=float, default=0.7)
    args = ap.parse_args()

    out_dir = Path(args.out)
    papers_path = out_dir / "papers.jsonl"
    claims_path = out_dir / "claims.jsonl"
    graph_path = out_dir / "graph.json"
    anomalies_path = out_dir / "anomalies.jsonl"

    for p in (papers_path, claims_path, graph_path, anomalies_path):
        if not p.exists():
            print(f"missing prerequisite: {p}", file=sys.stderr)
            sys.exit(2)

    t0 = time.monotonic()
    started_at = _now_iso()

    # Step 1: cap anomalies
    top_path, type_counts_full = cap_anomalies_top_n(anomalies_path, args.max_anomalies)

    hypotheses_path = out_dir / "hypotheses.jsonl"
    hierarchy_path = out_dir / "hierarchy.json"
    scored_path = out_dir / "hypotheses_scored.jsonl"
    report_path = out_dir / "selected_hypotheses.md"

    aigraph = [sys.executable, "-m", "aigraph.cli"]

    gen_cmd = aigraph + [
        "generate-hypotheses",
        "--anomalies", str(top_path),
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
        "--anomalies", str(top_path),
        "--papers", str(papers_path),
        "--k", str(args.k),
        "--lambda", str(args.mmr_lambda),
        "--output", str(report_path),
    ])

    finished_at = _now_iso()
    wall = int(time.monotonic() - t0)

    # Update run_metadata.json (or create) with the cap info.
    metadata_path = out_dir / "run_metadata.json"
    metadata: dict = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            metadata = {}
    metadata.setdefault("topic", "LLM reasoning (arxiv 2023+)")
    metadata["finish_run_started_at"] = started_at
    metadata["finish_run_finished_at"] = finished_at
    metadata["finish_run_wall_seconds"] = wall
    metadata["max_anomalies_cap"] = args.max_anomalies
    metadata["anomaly_type_counts_full"] = type_counts_full

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    git_tag = subprocess.run(
        ["git", "tag", "--points-at", git_sha], capture_output=True, text=True
    ).stdout.strip().split("\n")[0] or "v0.7-frozen"
    metadata.setdefault("git_sha", git_sha)
    metadata.setdefault("git_tag", git_tag)
    metadata.setdefault("model", args.model or os.environ.get("AIGRAPH_MODEL", "(env-default)"))

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\n[done] finish_run wall={wall}s  out={out_dir}/", flush=True)


if __name__ == "__main__":
    main()
