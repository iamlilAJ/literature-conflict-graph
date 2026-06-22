import json

from aigraph.llm_hypotheses import LLMHypothesisGenerator, _SYSTEM
from aigraph.models import Anomaly, Claim


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


# --------------------------------------------------------------------------- #
# §7 Thaw #4: the forward-design contract. The generator no longer routes a
# distinct prompt per anomaly type — one type-agnostic FORWARD prompt is used
# for every type (the type + central_question travel in the user payload, see
# test_payload_includes_signals_block). These tests re-pin the thawed behavior.
# --------------------------------------------------------------------------- #

def test_forward_prompt_is_type_agnostic_and_bans_boilerplate(monkeypatch):
    """Every anomaly type uses the SAME forward system prompt, and that prompt
    explicitly bans the pre-thaw retrospective boilerplate."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    fake = _FakeClient(json.dumps({"hypotheses": []}))
    gen = LLMHypothesisGenerator(model="stub", client=fake, api_key="test-key")

    for i, atype in enumerate(("replication_conflict", "bridge_opportunity",
                               "metric_mismatch", "totally_fake_type")):
        gen.generate(_make_anomaly(f"a{i}", atype), {"c001": _claim("c001")})

    systems = [c["messages"][0]["content"] for c in fake.chat.completions.calls]
    assert len(systems) == 4
    assert all(s == _SYSTEM for s in systems)          # one prompt for all types
    low = _SYSTEM.lower()
    assert "banned" in low and "forward" in low
    assert "unreported moderator" in low               # the boilerplate it forbids
    assert "interior_optimum" in low and "mechanism" in low


def _multi_payload(items: list[dict]) -> str:
    return json.dumps({"hypotheses": items})


def test_parse_stamps_shape_into_scope_conditions(monkeypatch):
    """The generator's own shape label is recorded in scope_conditions so
    downstream/eval can read it without re-judging."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    payload = _multi_payload([
        {"shape": "interior_optimum",
         "hypothesis": "RAG top-k helps multi-hop QA up to ~10 then hurts.",
         "mechanism": "Beyond the optimum, distractor passages dilute support.",
         "explains_claims": ["c001"],
         "predictions": ["Accuracy is unimodal in k.", "Peak near k=10."],
         "minimal_test": "Sweep top-k 1..30 on HotpotQA; F1; falsifier: no decline."},
    ])
    fake = _FakeClient(payload)
    gen = LLMHypothesisGenerator(model="stub", client=fake, api_key="test-key")
    out = gen.generate(_make_anomaly("a1", "benchmark_inconsistency"), {"c001": _claim("c001")})
    assert len(out) == 1
    assert out[0].scope_conditions.get("shape") == "interior_optimum"


def test_parse_caps_retrospective_conflict_attribution(monkeypatch):
    """At most ONE retrospective conflict-attribution item survives per anomaly,
    so the thawed generator cannot regress to the pre-thaw monoculture. Forward
    items are unaffected, and the count is NOT padded to a fixed 3."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    retro = {
        "shape": "conflict_attribution",
        "hypothesis": "An unreported moderator variable drives the conflicting results.",
        "mechanism": "A confound in preprocessing correlates with outcome direction.",
        "explains_claims": ["c001"],
        "predictions": ["Holding it fixed shrinks variance.", "Covariate analysis explains the flip."],
        "minimal_test": "Replay all claims in a common harness with identical prompts and decoding.",
    }
    fwd = {
        "shape": "mechanism",
        "hypothesis": "RAG helps only when the answer needs multi-document synthesis.",
        "mechanism": "Single-doc questions are already answerable without retrieval.",
        "explains_claims": ["c001"],
        "predictions": ["Gains concentrate on multi-hop items.", "Single-hop gains ~0."],
        "minimal_test": "Split HotpotQA into single/multi-hop; compare RAG F1; falsifier: equal gains.",
    }
    fake = _FakeClient(_multi_payload([retro, retro, retro, fwd]))
    gen = LLMHypothesisGenerator(model="stub", client=fake, api_key="test-key")
    out = gen.generate(_make_anomaly("a1", "impact_conflict"), {"c001": _claim("c001")})

    shapes = [h.scope_conditions.get("shape") for h in out]
    assert shapes.count("conflict_attribution") == 1   # 3 retro collapse to 1
    assert "mechanism" in shapes                        # forward item kept
    assert len(out) == 2                                # not padded to 3


def test_variable_count_is_not_forced_to_three(monkeypatch):
    """Two forward hypotheses in → two out (the EXACTLY-3 rule is gone)."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    items = [
        {"shape": "interior_optimum", "hypothesis": f"H{i} sweet spot exists.",
         "mechanism": "m", "explains_claims": ["c001"],
         "predictions": ["p1", "p2"], "minimal_test": "sweep; metric; falsifier."}
        for i in range(2)
    ]
    fake = _FakeClient(_multi_payload(items))
    gen = LLMHypothesisGenerator(model="stub", client=fake, api_key="test-key")
    out = gen.generate(_make_anomaly("a1", "evidence_gap"), {"c001": _claim("c001")})
    assert len(out) == 2


def test_payload_includes_signals_block(monkeypatch):
    """The user-side JSON payload must carry the anomaly's numeric signals
    (evidence_impact / recent_activity / replication_score / etc.) so the LLM
    can calibrate hypothesis emphasis based on impact and activity levels."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    fake = _FakeClient(json.dumps({"hypotheses": []}))
    gen = LLMHypothesisGenerator(model="stub", client=fake, api_key="test-key")

    anomaly = Anomaly(
        anomaly_id="a042",
        type="replication_conflict",
        central_question="Why does B fail to reproduce A?",
        claim_ids=["c001"],
        positive_claims=["c001"],
        negative_claims=[],
        shared_entities={"method": "RAG", "task": "QA"},
        evidence_impact=4.5,
        recent_activity=2.1,
        impact_balance=0.3,
        citation_bridge_score=0.0,
        replication_score=1.0,
        topology_score=3.7,
    )
    gen.generate(anomaly, {"c001": _claim("c001")})

    user_msg = fake.chat.completions.calls[0]["messages"][1]["content"]
    payload = json.loads(user_msg)
    signals = payload["anomaly"]["signals"]

    assert signals["evidence_impact"] == 4.5
    assert signals["recent_activity"] == 2.1
    assert signals["impact_balance"] == 0.3
    assert signals["replication_score"] == 1.0
    assert signals["topology_score"] == 3.7
    # Output schema is unchanged — anomaly block still has all the legacy keys.
    for k in ("anomaly_id", "type", "central_question", "positive_claims",
              "negative_or_mixed_claims", "shared_entities", "varying_settings"):
        assert k in payload["anomaly"], f"missing legacy key: {k}"


def test_signals_block_is_robust_to_legacy_anomaly(monkeypatch):
    """A legacy Anomaly missing the numeric signal fields entirely (built via
    model_construct without them, or constructed before those fields existed)
    must still serialize cleanly and produce 0.0 in every signal slot — the
    getattr safety net in _anomaly_signals does the work."""
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "chat")
    fake = _FakeClient(json.dumps({"hypotheses": []}))
    gen = LLMHypothesisGenerator(model="stub", client=fake, api_key="test-key")

    a = _make_anomaly("aL", "benchmark_inconsistency")
    # Strip the signal attrs from the model_construct'd instance so the
    # getattr fallback path is exercised even on pydantic v2 versions that
    # populate defaults during model_construct.
    for field in (
        "evidence_impact", "recent_activity", "impact_balance",
        "citation_bridge_score", "replication_score", "topology_score",
    ):
        a.__dict__.pop(field, None)

    gen.generate(a, {"c001": _claim("c001")})

    user_msg = fake.chat.completions.calls[0]["messages"][1]["content"]
    payload = json.loads(user_msg)
    signals = payload["anomaly"]["signals"]
    assert signals["evidence_impact"] == 0.0
    assert signals["recent_activity"] == 0.0
    assert signals["impact_balance"] == 0.0
    assert signals["citation_bridge_score"] == 0.0
    assert signals["replication_score"] == 0.0
    assert signals["topology_score"] == 0.0

