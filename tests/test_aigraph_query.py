"""Unit tests for the per-query layer (service-mode POC entry point)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from aigraph_query import _tokenize, _topic_relevance, query  # noqa: E402

from aigraph.models import Anomaly, Claim, Hypothesis  # noqa: E402


# --- token helpers ---


def test_tokenize_drops_stopwords_and_lowercases():
    out = _tokenize("The Reasoning of an Agent on a Task")
    # 'the', 'of', 'an', 'on', 'a' all dropped; rest lowercased
    assert out == {"reasoning", "agent", "task"}


def test_tokenize_drops_short_tokens():
    # length <= 1 always drops, even non-stopwords
    out = _tokenize("a b c reasoning agent")
    assert out == {"reasoning", "agent"}


def test_tokenize_handles_punctuation():
    out = _tokenize("retrieval-augmented generation, evaluated on QA-style tasks")
    assert "retrieval" in out
    assert "augmented" in out
    assert "qa" in out
    assert "style" in out


def test_tokenize_empty_input():
    assert _tokenize("") == set()
    assert _tokenize(None) == set()  # type: ignore[arg-type]


# --- _topic_relevance ---


def _make_hyp(hyp_id: str, **kwargs) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hyp_id,
        anomaly_id=kwargs.get("anomaly_id", "a1"),
        hypothesis=kwargs.get("hypothesis", ""),
        mechanism=kwargs.get("mechanism", ""),
        explains_claims=kwargs.get("explains_claims", []),
        predictions=kwargs.get("predictions", []),
        evidence_gap=kwargs.get("evidence_gap", ""),
    )


def _make_anom(anom_id: str, **kwargs) -> Anomaly:
    return Anomaly(
        anomaly_id=anom_id,
        type=kwargs.get("type", "evidence_gap"),
        central_question=kwargs.get("central_question", ""),
        shared_entities=kwargs.get("shared_entities", {}),
        topology_score=kwargs.get("topology_score", 0.5),
    )


def _make_claim(cid: str, **kwargs) -> Claim:
    return Claim(
        claim_id=cid,
        paper_id=kwargs.get("paper_id", "p"),
        claim_text=kwargs.get("claim_text", ""),
        method=kwargs.get("method"),
        task=kwargs.get("task"),
        dataset=kwargs.get("dataset"),
        metric=kwargs.get("metric"),
    )


def test_topic_relevance_counts_overlap_in_hypothesis_text():
    """Tokens are exact-match (no stemming): 'agent' won't match 'agents'.
    The runner relies on the LLM-generated hypothesis text containing the
    user's query terms verbatim."""
    hyp = _make_hyp("h1", hypothesis="Reasoning over agent in long context")
    score = _topic_relevance(hyp, {}, {}, {"reasoning", "agent"})
    assert score == 2


def test_topic_relevance_includes_anomaly_central_question():
    hyp = _make_hyp("h1", hypothesis="x")
    anom = _make_anom("a1", central_question="why does multimodal differ on video reasoning")
    rel = _topic_relevance(hyp, {"a1": anom}, {}, {"video", "multimodal"})
    assert rel == 2


def test_topic_relevance_includes_cited_claims():
    hyp = _make_hyp("h1", explains_claims=["c1"])
    claim = _make_claim("c1", claim_text="RAG improves QA accuracy on HotpotQA")
    rel = _topic_relevance(hyp, {}, {"c1": claim}, {"rag", "hotpotqa"})
    assert rel == 2


def test_topic_relevance_zero_on_no_overlap():
    hyp = _make_hyp("h1", hypothesis="reasoning agents")
    rel = _topic_relevance(hyp, {}, {}, {"diffusion", "stable"})
    assert rel == 0


# --- query() integration ---


def _write_run_dir(tmp_path: Path, hyps, anoms, claims, papers=None) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    with (run / "hypotheses_scored.jsonl").open("w") as f:
        for h in hyps:
            f.write(h.model_dump_json() + "\n")
    with (run / "anomalies.jsonl").open("w") as f:
        for a in anoms:
            f.write(a.model_dump_json() + "\n")
    with (run / "claims.jsonl").open("w") as f:
        for c in claims:
            f.write(c.model_dump_json() + "\n")
    if papers:
        from aigraph.models import Paper as _Paper
        with (run / "papers.jsonl").open("w") as f:
            for p in papers:
                f.write(p.model_dump_json() + "\n")
    else:
        (run / "papers.jsonl").write_text("")
    return run


def test_query_returns_no_match_message_on_off_topic(tmp_path):
    hyps = [_make_hyp("h1", hypothesis="Reasoning over agents")]
    anoms = [_make_anom("a1", central_question="why does method X differ")]
    claims = [_make_claim("c1", claim_text="agents help reasoning")]
    run_dir = _write_run_dir(tmp_path, hyps, anoms, claims)

    md, stats = query(run_dir, topic="quantum nanotube", k=3, min_anomalies=1)
    assert "No matches for topic" in md
    assert stats["n_matched"] == 0
    assert stats["n_selected"] == 0
    assert stats["llm_calls"] == 0


def test_query_filters_then_selects(tmp_path):
    """Two hypotheses; one matches the topic, the other doesn't.
    Only the matching one is returned."""
    from aigraph.models import Paper
    hyps = [
        _make_hyp("h_match", anomaly_id="a1", hypothesis="Reasoning over agent",
                  predictions=["agent prediction"], evidence_gap="missing data"),
        _make_hyp("h_miss", anomaly_id="a2", hypothesis="Stable diffusion image quality",
                  predictions=["image gen prediction"]),
    ]
    anoms = [
        _make_anom("a1", central_question="why does agent differ on reasoning",
                   shared_entities={"method": "agent", "task": "reasoning"}),
        _make_anom("a2", central_question="why does diffusion differ on image",
                   shared_entities={"method": "diffusion"}),
    ]
    claims: list[Claim] = []
    papers = [Paper(paper_id="p1", title="dummy", year=2024, venue="x")]
    run_dir = _write_run_dir(tmp_path, hyps, anoms, claims, papers=papers)

    md, stats = query(run_dir, topic="agent reasoning", k=2, min_anomalies=1)
    assert stats["n_matched"] == 1
    assert stats["n_selected"] == 1
    assert stats["llm_calls"] == 0
    # The matched hypothesis text or its mechanism token shows in markdown.
    assert "agent" in md.lower() or "Reasoning" in md


def test_query_raises_on_empty_topic(tmp_path):
    run_dir = _write_run_dir(tmp_path, [], [], [])
    with pytest.raises(ValueError, match="no usable tokens"):
        query(run_dir, topic="the of an a", k=3)
