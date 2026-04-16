import json

from aigraph.llm_hypotheses import LLMHypothesisGenerator
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

