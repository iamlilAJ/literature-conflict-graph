"""Tests for src/aigraph/hierarchy.py — domain/community/cluster aggregation."""

from __future__ import annotations

from pathlib import Path

from aigraph.graph import build_graph
from aigraph.hierarchy import build_hierarchy, load_hierarchy, save_hierarchy
from aigraph.models import Anomaly, Claim, Paper


def _claim(
    cid: str,
    paper_id: str,
    *,
    domain: str | None = None,
    canonical_method: str | None = None,
    canonical_task: str | None = None,
    method: str = "RAG",
    task: str = "QA",
) -> Claim:
    return Claim(
        claim_id=cid,
        paper_id=paper_id,
        claim_text="x",
        method=method,
        task=task,
        canonical_method=canonical_method,
        canonical_task=canonical_task,
        domain=domain,
        direction="positive",
    )


def _paper(paper_id: str) -> Paper:
    return Paper(paper_id=paper_id, title=paper_id, year=2024, venue="ACL")


def test_build_hierarchy_aggregates_domains():
    """Three domains, distinct method/task profiles — verify rollup."""
    claims = [
        _claim("c1", "p1", domain="medical", canonical_method="RAG", canonical_task="medical-QA"),
        _claim("c2", "p2", domain="medical", canonical_method="RAG", canonical_task="medical-QA"),
        _claim("c3", "p3", domain="legal", canonical_method="DPR", canonical_task="legal-QA"),
        _claim("c4", "p4", domain="finance", canonical_method="LSTM", canonical_task="forecasting"),
    ]
    papers = [_paper(p) for p in ("p1", "p2", "p3", "p4")]
    g = build_graph(claims, papers=papers)
    h = build_hierarchy(claims, papers, [], g)

    domains = h["domains"]
    assert set(domains) == {"medical", "legal", "finance"}
    assert domains["medical"]["paper_count"] == 2
    assert domains["medical"]["claim_count"] == 2
    assert "RAG" in domains["medical"]["top_methods"]
    assert "medical-QA" in domains["medical"]["top_tasks"]
    assert domains["legal"]["paper_count"] == 1
    assert domains["finance"]["top_methods"] == ["LSTM"]


def test_build_hierarchy_handles_missing_domain():
    """Claims with domain=None/placeholder land in the "uncategorized" bucket."""
    claims = [
        _claim("c1", "p1", domain=None, canonical_method="RAG", canonical_task="QA"),
        _claim("c2", "p2", domain="other", canonical_method="RAG", canonical_task="QA"),
        _claim("c3", "p3", domain="medical", canonical_method="RAG", canonical_task="QA"),
    ]
    papers = [_paper(p) for p in ("p1", "p2", "p3")]
    g = build_graph(claims, papers=papers)
    h = build_hierarchy(claims, papers, [], g)

    assert "uncategorized" in h["domains"]
    assert h["domains"]["uncategorized"]["claim_count"] == 2
    assert "medical" in h["domains"]


def test_build_hierarchy_clusters_by_canonical_fields():
    """Reuses _cluster_key — claims sharing (canonical_method, canonical_task)
    fall into one cluster keyed as f"{method}__{task}" (lowercased)."""
    claims = [
        _claim("c1", "p1", canonical_method="RAG", canonical_task="QA"),
        _claim("c2", "p2", canonical_method="RAG", canonical_task="QA"),
        _claim("c3", "p3", canonical_method="DPR", canonical_task="retrieval"),
    ]
    papers = [_paper(p) for p in ("p1", "p2", "p3")]
    g = build_graph(claims, papers=papers)
    h = build_hierarchy(claims, papers, [], g)

    clusters = h["clusters"]
    # _cluster_key lowercases, so RAG+QA -> "rag__qa"
    assert "rag__qa" in clusters
    assert "dpr__retrieval" in clusters
    rag_cluster = clusters["rag__qa"]
    assert sorted(rag_cluster["claim_ids"]) == ["c1", "c2"]
    assert rag_cluster["paper_count"] == 2


def test_save_load_hierarchy_roundtrip(tmp_path):
    """Atomic save then load returns identical content (sets become sorted lists)."""
    claims = [
        _claim("c1", "p1", domain="medical", canonical_method="RAG", canonical_task="QA"),
        _claim("c2", "p2", domain="medical", canonical_method="RAG", canonical_task="QA"),
    ]
    papers = [_paper(p) for p in ("p1", "p2")]
    g = build_graph(claims, papers=papers)
    h = build_hierarchy(claims, papers, [], g)

    out = tmp_path / "h.json"
    save_hierarchy(h, out)
    assert out.exists()
    loaded = load_hierarchy(out)
    assert loaded == h


def test_load_hierarchy_returns_empty_on_missing_file(tmp_path):
    """Graceful degradation when the file doesn't exist or is corrupt."""
    missing = tmp_path / "nope.json"
    h = load_hierarchy(missing)
    assert h == {
        "domains": {},
        "communities": {},
        "clusters": {},
        "cluster_to_community": {},
        "anomaly_to_cluster": {},
    }

    # Also handles malformed JSON.
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    h2 = load_hierarchy(corrupt)
    assert h2 == h
