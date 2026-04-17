"""Persistent community corpus built from completed runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .anomalies import detect_anomalies
from .graph import build_graph, save_graph
from .io import read_jsonl, write_jsonl
from .models import Anomaly, Claim, Insight, Paper
from .visualize import render_visualization


def community_dir(runs_dir: str | Path) -> Path:
    return Path(runs_dir).resolve() / "_community"


def ingest_run(run_dir: str | Path, runs_dir: str | Path, run_id: str | None = None) -> dict:
    """Merge one completed run into the living community corpus."""
    run_path = Path(run_dir)
    target = community_dir(runs_dir)
    target.mkdir(parents=True, exist_ok=True)

    existing_papers = _read_optional(target / "papers.jsonl", Paper)
    existing_claims = _read_optional(target / "claims.jsonl", Claim)
    existing_insights = _read_optional(target / "insights.jsonl", Insight)

    run_papers = _read_optional(run_path / "papers.jsonl", Paper)
    run_claims = _read_optional(run_path / "claims.jsonl", Claim)
    run_insights = _read_optional(run_path / "insights.jsonl", Insight)
    run_id = run_id or run_path.name

    papers = _merge_papers(existing_papers, run_papers)
    claims = _merge_claims(existing_claims, run_claims, run_id)
    insights = _merge_insights(existing_insights, run_insights, run_id)
    graph = build_graph(claims, papers=papers)
    anomalies = detect_anomalies(graph, claims)

    write_jsonl(target / "papers.jsonl", papers)
    write_jsonl(target / "claims.jsonl", claims)
    write_jsonl(target / "anomalies.jsonl", anomalies)
    write_jsonl(target / "insights.jsonl", insights)
    write_jsonl(target / "hypotheses.jsonl", [])
    save_graph(graph, target / "graph.json")
    render_visualization(target, target / "index.html")

    status = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": _count_ingested_runs(claims),
        "papers": len(papers),
        "claims": len(claims),
        "anomalies": len(anomalies),
        "insights": len(insights),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "graph_url": "/community/index.html",
    }
    (target / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def rebuild_community(runs_dir: str | Path) -> dict:
    target = community_dir(runs_dir)
    if target.exists():
        for path in target.iterdir():
            if path.is_file():
                path.unlink()
    summary: dict = {}
    root = Path(runs_dir)
    for status_path in sorted(root.glob("*/status.json")):
        if status_path.parent.name.startswith("_"):
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if status.get("status") != "done":
            continue
        summary = ingest_run(status_path.parent, root, run_id=status_path.parent.name)
    if not summary:
        target.mkdir(parents=True, exist_ok=True)
        summary = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "runs": 0,
            "papers": 0,
            "claims": 0,
            "anomalies": 0,
            "insights": 0,
            "nodes": 0,
            "edges": 0,
            "graph_url": "/community/index.html",
        }
        (target / "status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def read_community_status(runs_dir: str | Path) -> dict:
    path = community_dir(runs_dir) / "status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def community_digest(runs_dir: str | Path) -> dict:
    root = Path(runs_dir).resolve()
    status = read_community_status(root)
    newest_runs: list[dict] = []
    for path in sorted(root.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.parent.name.startswith("_"):
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if row.get("status") != "done":
            continue
        newest_runs.append(
            {
                "run_id": path.parent.name,
                "topic": row.get("topic", "Untitled run"),
                "claims": int(row.get("claims") or 0),
                "anomalies": int(row.get("anomalies") or 0),
                "insights": int(row.get("insights") or 0),
            }
        )
        if len(newest_runs) >= 4:
            break

    hottest_topics = _hot_topics(root)
    biggest_conflicts = _top_conflicts(community_dir(root) / "anomalies.jsonl")
    newest_bridges = _top_bridges(community_dir(root) / "insights.jsonl")
    return {
        "status": status,
        "newest_runs": newest_runs,
        "hottest_topics": hottest_topics,
        "biggest_conflicts": biggest_conflicts,
        "newest_bridges": newest_bridges,
    }


def _read_optional(path: Path, model):
    if not path.exists():
        return []
    return read_jsonl(path, model)


def _merge_papers(existing: list[Paper], new: list[Paper]) -> list[Paper]:
    by_id = {p.paper_id: p for p in existing}
    for paper in new:
        current = by_id.get(paper.paper_id)
        if current is None or _paper_quality(paper) > _paper_quality(current):
            by_id[paper.paper_id] = paper
    return sorted(by_id.values(), key=lambda p: (p.year or 0, p.paper_id), reverse=True)


def _paper_quality(paper: Paper) -> tuple[int, float, int]:
    return (int(paper.cited_by_count or 0), float(paper.selection_score or 0.0), len(paper.abstract or ""))


def _merge_claims(existing: list[Claim], new: list[Claim], run_id: str) -> list[Claim]:
    by_id = {c.claim_id: c for c in existing}
    for claim in new:
        gid = claim.claim_id if claim.claim_id.startswith(f"{run_id}:") else f"{run_id}:{claim.claim_id}"
        if gid not in by_id:
            by_id[gid] = claim.model_copy(update={"claim_id": gid})
    return sorted(by_id.values(), key=lambda c: c.claim_id)


def _merge_insights(existing: list[Insight], new: list[Insight], run_id: str) -> list[Insight]:
    by_id = {i.insight_id: i for i in existing}
    for insight in new:
        gid = insight.insight_id if insight.insight_id.startswith(f"{run_id}:") else f"{run_id}:{insight.insight_id}"
        if gid not in by_id:
            by_id[gid] = insight.model_copy(update={"insight_id": gid})
    return sorted(by_id.values(), key=lambda i: i.insight_id)


def _count_ingested_runs(claims: list[Claim]) -> int:
    return len({claim.claim_id.split(":", 1)[0] for claim in claims if ":" in claim.claim_id})


def _hot_topics(root: Path) -> list[dict]:
    analytics = root / "_analytics" / "requests.jsonl"
    if not analytics.exists():
        return []
    counts: dict[str, int] = {}
    for line in analytics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        topic = str(row.get("topic") or "").strip()
        if not topic:
            continue
        counts[topic] = counts.get(topic, 0) + 1
    return [
        {"topic": topic, "count": count}
        for topic, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
    ]


def _top_conflicts(path: Path) -> list[dict]:
    rows = _read_optional(path, Anomaly)
    rows.sort(key=lambda a: (float(a.topology_score or 0.0), len(a.claim_ids)), reverse=True)
    return [
        {
            "anomaly_id": row.anomaly_id,
            "type": row.type,
            "question": row.central_question,
            "claim_count": len(row.claim_ids),
        }
        for row in rows[:3]
    ]


def _top_bridges(path: Path) -> list[dict]:
    rows = _read_optional(path, Insight)
    rows.sort(key=lambda i: (float(i.confidence_score or 0.0), float(i.topology_score or 0.0)), reverse=True)
    return [
        {
            "insight_id": row.insight_id,
            "title": row.title,
            "communities": row.communities,
        }
        for row in rows[:3]
    ]
