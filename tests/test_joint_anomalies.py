"""Tests for the Atlas-grounded joint anomaly detector.

Deterministic — injects inbound_bottlenecks so no parquet/polars needed.
"""
from __future__ import annotations

from aigraph.models import Anomaly, Claim, Paper
from aigraph.joint_anomalies import detect_bottleneck_open_q_alignment, merge_joint_anomalies


def _claim(cid, pid, **kw):
    return Claim(claim_id=cid, paper_id=pid, claim_text=kw.pop("text", "t"), **kw)


def _papers():
    return [Paper(paper_id="arxiv:2401.00001", title="P1", year=2024, venue="X",
                  arxiv_id_base="2401.00001"),
            Paper(paper_id="arxiv:2401.00002", title="P2", year=2024, venue="Y",
                  arxiv_id_base="2401.00002")]


def test_fires_when_weakness_and_bottleneck_align():
    claims = [
        _claim("c1", "arxiv:2401.00001", direction="negative", method="RAG", task="QA",
               canonical_method="RAG", canonical_task="QA", text="RAG degrades on noisy corpora"),
        _claim("c2", "arxiv:2401.00001", claim_type="limitation", method="RAG", task="QA",
               text="evaluation limited to English"),
        # P2 has weakness but NO atlas bottleneck -> must not fire
        _claim("c3", "arxiv:2401.00002", direction="negative", method="CoT", task="math",
               text="CoT brittle on long chains"),
    ]
    inbound = {"2401.00001": [
        {"source_paper": "2406.12345", "source_title": "Later work", "relation": "improves",
         "confidence": 0.9, "dimension": "accuracy", "severity": "fundamental",
         "description": "RAG retrieval injects distractors", "quote": "...distractors..."},
    ]}
    anoms = detect_bottleneck_open_q_alignment(claims, _papers(), inbound_bottlenecks=inbound)
    assert len(anoms) == 1
    a = anoms[0]
    assert a.type == "bottleneck_open_q_alignment"
    assert set(a.claim_ids) == {"c1", "c2"}          # both weakness claims on P1
    assert a.shared_entities["method"] == "RAG"
    assert a.shared_entities["bottleneck_dimension"] == "accuracy"
    assert a.bottleneck_signals and a.bottleneck_signals[0]["source_paper"] == "2406.12345"
    assert "accuracy" in a.central_question


def test_no_fire_without_weakness_or_without_bottleneck():
    # positive-only claim + a bottleneck -> no first-party weakness -> no fire
    claims = [_claim("c1", "arxiv:2401.00001", direction="positive", method="RAG", task="QA")]
    inbound = {"2401.00001": [{"source_paper": "x", "dimension": "accuracy",
                               "severity": "minor", "confidence": 0.5, "quote": "q"}]}
    assert detect_bottleneck_open_q_alignment(claims, _papers(), inbound_bottlenecks=inbound) == []
    # weakness but empty atlas
    claims2 = [_claim("c1", "arxiv:2401.00001", direction="negative", method="RAG", task="QA")]
    assert detect_bottleneck_open_q_alignment(claims2, _papers(), inbound_bottlenecks={}) == []


def test_anomaly_validates_and_is_serializable():
    claims = [_claim("c1", "arxiv:2401.00001", direction="negative", method="RAG", task="QA")]
    inbound = {"2401.00001": [{"source_paper": "x", "dimension": "speed", "severity": "significant",
                               "confidence": 0.8, "quote": "slow"}]}
    a = detect_bottleneck_open_q_alignment(claims, _papers(), inbound_bottlenecks=inbound)[0]
    # round-trips through the pydantic model (new type accepted)
    again = Anomaly.model_validate(a.model_dump())
    assert again.type == "bottleneck_open_q_alignment"


def test_merge_keeps_ids_unique():
    frozen = [Anomaly(anomaly_id="a001", type="evidence_gap", central_question="q")]
    joint = [Anomaly(anomaly_id="a001", type="bottleneck_open_q_alignment", central_question="q2")]
    merged = merge_joint_anomalies(frozen, joint)
    assert len(merged) == 2
    assert len({a.anomaly_id for a in merged}) == 2
