"""Tests for src/aigraph/creator.py multi-grain pipeline.

The single-grain ``generate_creator_hypotheses`` is exercised end-to-end on
the real corpus; these tests cover the new ``generate_creator_hypotheses_multi_grain``
which orchestrates three LLM passes per anomaly (fine → coarse → synthesize).
"""

from __future__ import annotations

import json

from aigraph.creator import (
    CREATOR_SYSTEM_PROMPT,
    SYSTEM_PROMPT_COARSE,
    SYSTEM_PROMPT_SYNTHESIZE,
    generate_creator_hypotheses_multi_grain,
)
from aigraph.models import Anomaly, Claim, OpenQuestion


# --- Fake LLM client (chat fallback path) ----------------------------------


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletionsQueue:
    """Returns each queued response in order; raises if drained.

    Mirrors the fake-LLM template at tests/test_llm_hypotheses.py:8-40 but
    queues multiple responses so we can exercise the 3 sequential calls
    (fine -> coarse -> synthesize) per anomaly.
    """

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._contents:
            raise RuntimeError("ran out of fake LLM responses")
        return _FakeCompletion(self._contents.pop(0))


class _FakeChat:
    def __init__(self, contents: list[str]):
        self.completions = _FakeCompletionsQueue(contents)


class _FakeClient:
    def __init__(self, contents: list[str]):
        self.chat = _FakeChat(contents)


# --- Fixtures ---------------------------------------------------------------


def _anomaly() -> Anomaly:
    return Anomaly(
        anomaly_id="a001",
        type="benchmark_inconsistency",
        central_question="When does RAG help domain QA?",
        claim_ids=["c001", "c002"],
        positive_claims=["c001"],
        negative_claims=["c002"],
        shared_entities={"method": "RAG", "task": "domain-QA"},
        topology_score=2.5,
    )


def _claims() -> list[Claim]:
    return [
        Claim(
            claim_id="c001",
            paper_id="p1",
            claim_text="RAG improves medical QA.",
            method="RAG",
            task="medical QA",
            canonical_method="RAG",
            canonical_task="QA",
            domain="medical",
            direction="positive",
            evidence_span="On MedQA we observe a +5.2 EM gain.",
        ),
        Claim(
            claim_id="c002",
            paper_id="p2",
            claim_text="RAG hurts multilingual medical QA.",
            method="RAG",
            task="medical QA",
            canonical_method="RAG",
            canonical_task="QA",
            domain="medical",
            direction="negative",
            evidence_span="On non-English MedQA EM drops by 4.1.",
        ),
    ]


def _open_questions() -> list[OpenQuestion]:
    return [
        OpenQuestion(
            open_question_id="p1#oq01",
            paper_id="p1",
            text="Does the corpus cover non-English medical facts?",
            kind="acknowledged_limitation",
            evidence_span="We did not test non-English medical sources.",
        ),
        OpenQuestion(
            open_question_id="p2#oq01",
            paper_id="p2",
            text="A multilingual retrieval index might recover gains.",
            kind="future_work_suggestion",
            evidence_span="Future work could explore multilingual indices.",
        ),
    ]


def _hierarchy() -> dict:
    return {
        "domains": {
            "medical": {
                "paper_count": 2,
                "claim_count": 2,
                "top_methods": ["RAG", "DPR"],
                "top_tasks": ["medical QA", "QA"],
                "anomaly_type_counts": {"benchmark_inconsistency": 1},
                "sample_claim_ids": ["c001", "c002"],
            }
        },
        "communities": {
            "c000": {
                "paper_ids": ["Paper:p1", "Paper:p2"],
                "paper_count": 2,
                "top_concepts": ["rag__qa"],
                "anomaly_count": 1,
            }
        },
        "clusters": {
            "rag__qa": {
                "claim_ids": ["c001", "c002"],
                "anomaly_ids": ["a001"],
                "paper_count": 2,
                "sample_claim_ids": ["c001", "c002"],
            },
            "dpr__retrieval": {
                "claim_ids": ["c003"],
                "anomaly_ids": [],
                "paper_count": 1,
                "sample_claim_ids": ["c003"],
            },
        },
        "cluster_to_community": {"rag__qa": "c000", "dpr__retrieval": "c000"},
        "anomaly_to_cluster": {"a001": "rag__qa"},
    }


def _payload_fine() -> str:
    return json.dumps({
        "creator_hypotheses": [{
            "proposed_method": "Multilingual Medical RAG",
            "mechanism": "Use a multilingual retriever atop a translation-aligned medical corpus.",
            "predictions": ["Non-English MedQA EM rises by ≥3.0.", "English EM unchanged."],
            "minimal_test": "Run RAG-vs-multilingual-RAG on bilingual MedQA splits.",
            "inspired_by": ["c001", "c002", "p1#oq01"],
            "distinguishes_from": "Differs from English-only RAG by adding cross-lingual retrieval.",
            "anomaly_resolution": "Explains the gap as English bias in the corpus.",
        }]
    })


def _payload_coarse() -> str:
    return json.dumps({
        "creator_hypotheses": [{
            "proposed_method": "Cross-Domain Retrieval Atlas",
            "mechanism": "A unified retrieval index spanning medical, legal, and finance domains.",
            "predictions": ["Cross-domain transfer improves on all three benchmarks."],
            "minimal_test": "Train on medical+legal, evaluate on finance MedQA-style tasks.",
            "inspired_by": ["claim_xyz_NOTREAL", "c001"],
            "distinguishes_from": "Existing single-domain retrievers do not share semantics.",
            "anomaly_resolution": "Cross-domain coverage closes evidence gaps.",
        }]
    })


def _payload_synth() -> str:
    return json.dumps({
        "creator_hypotheses": [{
            "proposed_method": "Multilingual Cross-Domain Medical RAG",
            "mechanism": "Multilingual retrieval over a translation-aligned, cross-domain medical+legal+finance corpus.",
            "predictions": [
                "Non-English MedQA EM rises by ≥3.0.",
                "Transfer to legal-QA improves over single-domain baseline by ≥2.0.",
            ],
            "minimal_test": "Bilingual + cross-domain ablation on MedQA + LegalQA.",
            "inspired_by": ["c001", "c002", "p1#oq01", "p2#oq01", "claim_xyz_NOTREAL"],
            "distinguishes_from": "Combines cross-lingual retrieval with cross-domain transfer; neither single-domain RAG nor monolingual atlases address both.",
            "anomaly_resolution": "Explains corpus-coverage gap and unifies cross-domain transfer in one mechanism.",
        }]
    })


# --- Tests ------------------------------------------------------------------


def test_multi_grain_calls_three_llm_passes(monkeypatch):
    """One anomaly with claims + OQs triggers exactly 3 LLM calls in the
    order fine -> coarse -> synthesize, with the system prompts mapping
    one-to-one."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    fake = _FakeClient([_payload_fine(), _payload_coarse(), _payload_synth()])

    out = generate_creator_hypotheses_multi_grain(
        [_anomaly()],
        _claims(),
        _open_questions(),
        _hierarchy(),
        model="stub",
        api_key="test-key",
        client=fake,
    )

    calls = fake.chat.completions.calls
    assert len(calls) == 3
    systems = [c["messages"][0]["content"] for c in calls]
    assert systems[0] == CREATOR_SYSTEM_PROMPT
    assert systems[1] == SYSTEM_PROMPT_COARSE
    assert systems[2] == SYSTEM_PROMPT_SYNTHESIZE
    assert len(out) == 1
    rec = out[0]
    assert rec["hypothesis_id"] == "a001#mg01"
    assert rec["anomaly_id"] == "a001"
    assert "multi_grain" in rec
    assert "fine" in rec["multi_grain"]
    assert "coarse" in rec["multi_grain"]


def test_multi_grain_synthesized_carries_grounding_from_fine(monkeypatch):
    """The synthesizer's inspired_by includes a hallucinated id ("claim_xyz_NOTREAL");
    the final record's allowed-grounding filter must drop it. Both fine and coarse
    raw items must round-trip into multi_grain unchanged."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    fake = _FakeClient([_payload_fine(), _payload_coarse(), _payload_synth()])

    out = generate_creator_hypotheses_multi_grain(
        [_anomaly()],
        _claims(),
        _open_questions(),
        _hierarchy(),
        model="stub",
        api_key="test-key",
        client=fake,
    )

    rec = out[0]
    inspired = rec["scope_conditions"].get("inspired_by", "")
    assert "claim_xyz_NOTREAL" not in inspired
    # Real grounding ids — claim_ids and open_question_ids — must survive.
    assert "c001" in inspired or "p1#oq01" in inspired
    # explains_claims is sourced from anomaly.claim_ids, never from synth payload.
    assert set(rec["explains_claims"]) <= {"c001", "c002"}
    # Both intermediates round-trip as raw dicts.
    assert rec["multi_grain"]["fine"]["proposed_method"] == "Multilingual Medical RAG"
    assert rec["multi_grain"]["coarse"]["proposed_method"] == "Cross-Domain Retrieval Atlas"


def test_multi_grain_falls_back_when_hierarchy_empty(monkeypatch):
    """An empty hierarchy must NOT crash the coarse pass; the LLM still gets a
    payload (with all-zero stats) and the pipeline still produces a synthesized
    record."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    fake = _FakeClient([_payload_fine(), _payload_coarse(), _payload_synth()])
    empty_hierarchy = {
        "domains": {},
        "communities": {},
        "clusters": {},
        "cluster_to_community": {},
        "anomaly_to_cluster": {},
    }

    out = generate_creator_hypotheses_multi_grain(
        [_anomaly()],
        _claims(),
        _open_questions(),
        empty_hierarchy,
        model="stub",
        api_key="test-key",
        client=fake,
    )

    assert len(out) == 1
    # Coarse pass user prompt was built from the empty hierarchy and is still
    # a valid JSON envelope with the expected top-level keys.
    coarse_user_msg = fake.chat.completions.calls[1]["messages"][1]["content"]
    coarse_payload = json.loads(coarse_user_msg)
    for key in ("anomaly", "domain", "community", "sibling_clusters", "anchor_claims"):
        assert key in coarse_payload
    assert coarse_payload["domain"]["paper_count"] == 0
    assert coarse_payload["sibling_clusters"] == []
