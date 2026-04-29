import json
from typing import get_args

from aigraph.llm_hypotheses import LLMHypothesisGenerator, SYSTEM_PROMPTS
from aigraph.models import Anomaly, AnomalyType, Claim


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeCompletion(self._content)


class _FakeChat:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str):
        self.chat = _FakeChat(content)


def test_llm_hypothesis_generator_parses_strict_json():
    anomaly = Anomaly(
        anomaly_id="a001",
        type="benchmark_inconsistency",
        central_question="When does RAG help?",
        claim_ids=["c001", "c002"],
        positive_claims=["c001"],
        negative_claims=["c002"],
        shared_entities={"method": "RAG", "task": "domain-QA"},
    )
    claims = {
        "c001": Claim(
            claim_id="c001",
            paper_id="p1",
            claim_text="RAG improves medical QA.",
            method="RAG",
            task="medical QA",
            direction="positive",
        ),
        "c002": Claim(
            claim_id="c002",
            paper_id="p2",
            claim_text="RAG struggles on multilingual medical QA.",
            method="RAG",
            task="medical QA",
            direction="negative",
        ),
    }
    payload = {
        "hypotheses": [
            {
                "hypothesis": "Language coverage moderates RAG gains.",
                "mechanism": "Retrieval corpora cover English medical facts better than non-English facts.",
                "explains_claims": ["c001", "c002", "not-real"],
                "predictions": ["English gains exceed non-English gains.", "Adding multilingual corpora narrows the gap."],
                "minimal_test": "Evaluate matched English and non-English medical QA with the same retriever.",
                "scope_conditions": {"language": "multilingual"},
                "evidence_gap": "The claims do not report corpus language coverage.",
                "graph_bridge": {"from": "RAG", "to": "domain-QA"},
            }
        ]
    }
    client = _FakeClient(json.dumps(payload))
    generator = LLMHypothesisGenerator(model="stub", client=client, api_key="test-key")

    out = generator.generate(anomaly, claims)

    assert len(out) == 1
    assert out[0].hypothesis_id == "h001"
    assert out[0].explains_claims == ["c001", "c002"]
    assert out[0].minimal_test.startswith("Evaluate matched")
    assert client.chat.completions.calls[0]["model"] == "stub"


def _make_anomaly(anomaly_id: str, anomaly_type: str) -> Anomaly:
    """Construct a minimal Anomaly with the given type. Uses model_construct so
    callers can pass non-Literal values for the unknown-type fallback test."""
    return Anomaly.model_construct(
        anomaly_id=anomaly_id,
        type=anomaly_type,
        central_question="?",
        claim_ids=["c001"],
        positive_claims=["c001"],
        negative_claims=[],
        shared_entities={"method": "RAG", "task": "QA"},
        varying_settings=[],
        local_graph_nodes=[],
        local_graph_edges=[],
        evidence_impact=0.0,
        recent_activity=0.0,
        impact_balance=0.0,
        citation_bridge_score=0.0,
        replication_score=0.0,
        topology_score=0.0,
    )


def _claim(claim_id: str) -> Claim:
    return Claim(
        claim_id=claim_id,
        paper_id="p1",
        claim_text="x",
        method="RAG",
        task="QA",
        direction="positive",
    )


def test_system_prompts_covers_every_anomaly_type():
    """Every value in the AnomalyType Literal must have a specialized prompt.
    Adding a new anomaly type without a prompt entry should fail CI rather
    than silently fall through to _DEFAULT_PROMPT."""
    expected = set(get_args(AnomalyType))
    actual = set(SYSTEM_PROMPTS.keys())
    missing = expected - actual
    assert not missing, f"SYSTEM_PROMPTS missing entries for: {sorted(missing)}"


def test_distinct_anomaly_types_route_to_distinct_prompts(monkeypatch):
    """Two different anomaly types must produce two different system prompts
    in the LLM call. Forcing AIGRAPH_LLM_ENDPOINT=chat keeps the existing
    _FakeClient mock pattern (which only stubs .chat.completions)."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    fake = _FakeClient(json.dumps({"hypotheses": []}))
    gen = LLMHypothesisGenerator(model="stub", client=fake, api_key="test-key")

    a_rep = _make_anomaly("a1", "replication_conflict")
    a_bri = _make_anomaly("a2", "bridge_opportunity")
    claims = {"c001": _claim("c001")}
    gen.generate(a_rep, claims)
    gen.generate(a_bri, claims)

    calls = fake.chat.completions.calls
    assert len(calls) == 2
    systems = [c["messages"][0]["content"] for c in calls]
    assert systems[0] != systems[1]
    # Replication framing should mention reproduction/replication; bridge
    # framing should mention transfer/forward-looking experiments.
    assert "replication" in systems[0].lower() or "reproduce" in systems[0].lower()
    assert "transfer" in systems[1].lower() or "forward-looking" in systems[1].lower()


def test_unknown_anomaly_type_falls_back_to_default_and_logs(monkeypatch, caplog):
    """An anomaly with a type not in SYSTEM_PROMPTS must fall back to
    _DEFAULT_PROMPT and emit one INFO log line per unknown type per process."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    # Reset the dedup set so the test sees its own log entry even if a prior
    # test in the same process already warned about this same type.
    import aigraph.llm_hypotheses as mod
    mod._warned_unknown_types.clear()

    fake = _FakeClient(json.dumps({"hypotheses": []}))
    gen = LLMHypothesisGenerator(model="stub", client=fake, api_key="test-key")
    a = _make_anomaly("aX", "totally_fake_type")

    with caplog.at_level("INFO", logger="aigraph.llm_hypotheses"):
        gen.generate(a, {"c001": _claim("c001")})

    system_used = fake.chat.completions.calls[0]["messages"][0]["content"]
    assert system_used == mod._DEFAULT_PROMPT
    assert any("totally_fake_type" in r.message for r in caplog.records), (
        "Expected an INFO log mentioning the unknown anomaly type"
    )
    # Second call with the same unknown type must NOT log again — dedup test.
    caplog.clear()
    with caplog.at_level("INFO", logger="aigraph.llm_hypotheses"):
        gen.generate(a, {"c001": _claim("c001")})
    assert not any("totally_fake_type" in r.message for r in caplog.records), (
        "Second call with same unknown type should be deduped"
    )

