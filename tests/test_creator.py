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
    _hypothesis_to_creator_dict,
    _prompt_payload_coarse,
    _prompt_payload_coarse_community_disconnect,
    generate_creator_hypotheses_multi_grain,
)
from aigraph.models import Anomaly, Claim, Hypothesis, OpenQuestion


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


def _existing_hypothesis(anomaly_id: str) -> Hypothesis:
    """Build a Hypothesis matching the shape that generate_creator_hypotheses
    emits — includes scope_conditions with comma-joined inspired_by + a
    distinguishes_from line, and evidence_gap as the anomaly_resolution."""
    return Hypothesis(
        hypothesis_id=f"{anomaly_id}#cr01",
        anomaly_id=anomaly_id,
        hypothesis="Existing-Anchor RAG Method",
        mechanism="A baseline retrieval-augmented method that the existing creator already produced.",
        explains_claims=["c001", "c002"],
        predictions=["existing prediction A", "existing prediction B"],
        minimal_test="Compare against vanilla RAG on MedQA.",
        scope_conditions={
            "distinguishes_from": "Differs from naive RAG by a calibrated retriever.",
            "inspired_by": "c001, c002, p1#oq01",
        },
        evidence_gap="Existing single-grain anomaly resolution sentence.",
        graph_bridge={"from": "open_questions", "to": "creator_hypothesis"},
    )


def test_multi_grain_ablate_fine_when_existing_provided(monkeypatch):
    """When an existing Hypothesis covers the anomaly, fine LLM call is
    skipped — only 2 calls fire (coarse + synthesize) — and the synthesis
    user prompt carries the existing hypothesis as the fine anchor.
    multi_grain.fine_source must be 'existing'."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    fake = _FakeClient([_payload_coarse(), _payload_synth()])

    out = generate_creator_hypotheses_multi_grain(
        [_anomaly()],
        _claims(),
        _open_questions(),
        _hierarchy(),
        model="stub",
        api_key="test-key",
        client=fake,
        existing_hypotheses=[_existing_hypothesis("a001")],
    )

    calls = fake.chat.completions.calls
    assert len(calls) == 2, f"expected 2 LLM calls, got {len(calls)}"
    systems = [c["messages"][0]["content"] for c in calls]
    assert systems[0] == SYSTEM_PROMPT_COARSE
    assert systems[1] == SYSTEM_PROMPT_SYNTHESIZE
    # Synthesize user prompt must carry the existing hypothesis content as fine.
    synth_user_msg = json.loads(calls[1]["messages"][1]["content"])
    assert synth_user_msg["fine"]["proposed_method"] == "Existing-Anchor RAG Method"
    assert synth_user_msg["fine"]["mechanism"].startswith("A baseline retrieval-augmented")
    assert "c001" in synth_user_msg["fine"]["inspired_by"]
    rec = out[0]
    assert rec["multi_grain"]["fine_source"] == "existing"
    assert rec["multi_grain"]["fine"]["proposed_method"] == "Existing-Anchor RAG Method"


def test_multi_grain_falls_back_to_3_calls_when_existing_missing(monkeypatch):
    """When existing_hypotheses is None or doesn't cover the anomaly, all 3
    LLM calls fire and multi_grain.fine_source == 'generated'."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    fake = _FakeClient([_payload_fine(), _payload_coarse(), _payload_synth()])

    # existing_hypotheses provided but for a DIFFERENT anomaly_id
    out = generate_creator_hypotheses_multi_grain(
        [_anomaly()],
        _claims(),
        _open_questions(),
        _hierarchy(),
        model="stub",
        api_key="test-key",
        client=fake,
        existing_hypotheses=[_existing_hypothesis("a999")],
    )

    assert len(fake.chat.completions.calls) == 3
    rec = out[0]
    assert rec["multi_grain"]["fine_source"] == "generated"


def test_hypothesis_to_creator_dict_field_mapping():
    """Spot-check the exact field mapping between Hypothesis (single-grain
    output) and the creator-payload dict that the synthesize prompt expects.
    Important because the existing creator stores `inspired_by` as a
    comma-joined STRING in scope_conditions, not as a list."""
    h = _existing_hypothesis("a042")
    d = _hypothesis_to_creator_dict(h)
    assert d["proposed_method"] == "Existing-Anchor RAG Method"
    assert d["mechanism"].startswith("A baseline")
    assert d["inspired_by"] == ["c001", "c002", "p1#oq01"]  # split from string
    assert d["distinguishes_from"].startswith("Differs from naive RAG")
    assert d["anomaly_resolution"] == "Existing single-grain anomaly resolution sentence."


def _community_disconnect_anomaly() -> Anomaly:
    """A community_disconnect anomaly matching the real schema observed on
    the 1000-paper run: shared_entities carries community_from/_to plus a
    comma-joined shared_concepts string, NOT method/task."""
    return Anomaly(
        anomaly_id="a423",
        type="community_disconnect",
        central_question=(
            "Why do AGI and LLM-serving communities both reason about "
            "forecasting + text but rarely cite each other?"
        ),
        claim_ids=["c001", "c002"],
        positive_claims=["c001"],
        negative_claims=["c002"],
        shared_entities={
            "community_from": "artificial general intelligence",
            "community_to": "generative large language model serving",
            "shared_concepts": "forecasting, text",
        },
        topology_score=0.86,
    )


def test_multi_grain_community_disconnect_uses_coarse_directly(monkeypatch):
    """For community_disconnect anomalies, only the coarse LLM call fires;
    fine and synthesize are skipped. The coarse output is the final
    hypothesis. multi_grain.fine_source must be 'skipped_community_disconnect'
    and synthesize_source 'coarse_direct'. Critically, this works WITHOUT
    paper_oqs — the existing skip-rule that filtered these anomalies out
    in the v0.3 smoke run is bypassed."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    fake = _FakeClient([_payload_coarse()])

    out = generate_creator_hypotheses_multi_grain(
        [_community_disconnect_anomaly()],
        _claims(),
        # NO open questions — community_disconnect path must not require them.
        [],
        _hierarchy(),
        model="stub",
        api_key="test-key",
        client=fake,
    )

    assert len(out) == 1
    calls = fake.chat.completions.calls
    assert len(calls) == 1, f"expected 1 LLM call (coarse only), got {len(calls)}"
    assert calls[0]["messages"][0]["content"] == SYSTEM_PROMPT_COARSE
    rec = out[0]
    assert rec["multi_grain"]["fine_source"] == "skipped_community_disconnect"
    assert rec["multi_grain"]["synthesize_source"] == "coarse_direct"
    assert rec["multi_grain"]["fine"] is None
    # Coarse output became the final hypothesis.
    assert rec["hypothesis"].startswith("Cross-Domain Retrieval Atlas")


def test_multi_grain_payload_community_disconnect_includes_shared_concepts():
    """The coarse payload for a community_disconnect anomaly must contain
    shared_concepts (parsed from comma string), community_from/_to, and a
    framing_note that asks for a unifying mechanism. Other anomaly types
    must NOT include the framing_note."""
    cdis = _community_disconnect_anomaly()
    payload_cdis = json.loads(
        _prompt_payload_coarse_community_disconnect(cdis, {}, {})
    )

    assert payload_cdis["anomaly"]["community_from"] == "artificial general intelligence"
    assert payload_cdis["anomaly"]["community_to"] == "generative large language model serving"
    assert payload_cdis["shared_concepts"] == ["forecasting", "text"]
    assert "framing_note" in payload_cdis
    assert "unifying" in payload_cdis["framing_note"].lower()

    # Sanity check: the regular path does NOT carry community_from / framing_note.
    payload_regular = json.loads(
        _prompt_payload_coarse(_anomaly(), {c.claim_id: c for c in _claims()}, _hierarchy())
    )
    assert "framing_note" not in payload_regular
    assert "community_from" not in payload_regular["anomaly"]


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
