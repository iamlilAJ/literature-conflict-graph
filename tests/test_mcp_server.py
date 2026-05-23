"""Smoke tests for the aigraph MCP server.

Builds the FastMCP app against a tmp runs_root and asserts tools are
registered + return the expected structured shapes. No network, no LLM.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

mcp_sdk = pytest.importorskip("mcp", reason="mcp SDK not installed")


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    """Minimal complete run dir the query layer can read."""
    root = tmp_path / "runs"
    run = root / "demo_run"
    run.mkdir(parents=True)
    # claims
    (run / "claims.jsonl").write_text(
        "\n".join(
            json.dumps(c) for c in [
                {"claim_id": "demo#c1", "paper_id": "arxiv:1", "claim_text": "RAG helps QA",
                 "method": "RAG", "task": "factual-QA", "direction": "positive",
                 "canonical_method": "RAG", "canonical_task": "factual-QA"},
                {"claim_id": "demo#c2", "paper_id": "arxiv:2", "claim_text": "RAG hurts QA",
                 "method": "RAG", "task": "factual-QA", "direction": "negative",
                 "canonical_method": "RAG", "canonical_task": "factual-QA"},
            ]
        ) + "\n"
    )
    # papers
    (run / "papers.jsonl").write_text(
        "\n".join(
            json.dumps(p) for p in [
                {"paper_id": "arxiv:1", "title": "Paper One", "year": 2024, "venue": "X"},
                {"paper_id": "arxiv:2", "title": "Paper Two", "year": 2024, "venue": "Y"},
            ]
        ) + "\n"
    )
    # one anomaly + one hypothesis pointing at it
    (run / "anomalies.jsonl").write_text(
        json.dumps({
            "anomaly_id": "a001", "type": "benchmark_inconsistency",
            "central_question": "When does RAG help on factual-QA?",
            "claim_ids": ["demo#c1", "demo#c2"],
            "shared_entities": {"method": "RAG", "task": "factual-QA"},
        }) + "\n"
    )
    (run / "hypotheses_scored.jsonl").write_text(
        json.dumps({
            "hypothesis_id": "h001", "anomaly_id": "a001",
            "hypothesis": "Corpus cleanliness flips RAG's sign on factual-QA.",
            "mechanism": "noisy corpora inject distractors",
            "explains_claims": ["demo#c1", "demo#c2"],
            "predictions": ["RAG gains shrink on noisy corpora"],
            "minimal_test": "run RAG on clean vs noisy corpus",
            "scope_conditions": {"method": "RAG"},
        }) + "\n"
    )
    return root


def _tool_names(mcp) -> set[str]:
    tools = asyncio.run(mcp.list_tools())
    return {t.name for t in tools}


def _call(mcp, name, args):
    """Call a registered tool and return its parsed result.

    FastMCP.call_tool returns a list of content blocks (TextContent with
    a JSON string for structured returns), or a (content, structured)
    tuple in some SDK versions. Normalize to the parsed Python object.
    """
    result = asyncio.run(mcp.call_tool(name, args))
    if isinstance(result, tuple):
        # (content_list, structured) — prefer structured
        return result[1]
    # list of content blocks — parse the first TextContent's JSON
    first = result[0]
    text = getattr(first, "text", None)
    if text is not None:
        return json.loads(text)
    return result


def test_tools_registered(runs_root):
    from aigraph.mcp_server import build_mcp

    mcp = build_mcp(runs_root)
    names = _tool_names(mcp)
    assert {"list_runs", "get_run_summary", "query_hypotheses",
            "get_conflict_graph", "start_run", "get_run_status"} <= names


def test_list_runs(runs_root):
    from aigraph.mcp_server import build_mcp

    mcp = build_mcp(runs_root)
    out = _call(mcp, "list_runs", {})
    # structured result wraps the list under "result" in some SDK versions
    runs = out.get("result", out) if isinstance(out, dict) else out
    ids = [r["id"] for r in runs]
    assert "demo_run" in ids


def test_query_hypotheses(runs_root):
    from aigraph.mcp_server import build_mcp

    mcp = build_mcp(runs_root)
    out = _call(mcp, "query_hypotheses", {"topic": "RAG factual QA", "run": "demo_run", "k": 3})
    assert out.get("run") == "demo_run"
    assert out["stats"]["llm_calls"] == 0
    assert len(out["hypotheses"]) >= 1
    h = out["hypotheses"][0]
    assert h["hypothesis_id"] == "h001"
    assert h["anomaly_type"] == "benchmark_inconsistency"
    assert len(h["evidence_claims"]) == 2
    assert h["evidence_claims"][0]["title"] in {"Paper One", "Paper Two"}


def test_get_run_summary(runs_root):
    from aigraph.mcp_server import build_mcp

    mcp = build_mcp(runs_root)
    out = _call(mcp, "get_run_summary", {"run": "demo_run"})
    assert out["n_papers"] == 2
    assert out["n_hypotheses"] == 1
    assert out["anomaly_types"].get("benchmark_inconsistency") == 1


def test_start_run_disabled_without_service(runs_root):
    from aigraph.mcp_server import build_mcp

    mcp = build_mcp(runs_root, search_service=None)
    out = _call(mcp, "start_run", {"topic": "x"})
    assert "disabled" in out.get("error", "")


def test_path_traversal_rejected(runs_root):
    from aigraph.mcp_server import build_mcp

    mcp = build_mcp(runs_root)
    out = _call(mcp, "get_run_summary", {"run": "../../etc"})
    assert "error" in out


def test_readonly_drops_paid_tools(runs_root):
    from aigraph.mcp_server import build_mcp

    mcp = build_mcp(runs_root, readonly=True)
    names = _tool_names(mcp)
    # the 4 read tools stay; the paid run-trigger tools are gone
    assert {"list_runs", "get_run_summary", "query_hypotheses",
            "get_conflict_graph"} <= names
    assert "start_run" not in names
    assert "get_run_status" not in names
