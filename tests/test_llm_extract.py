import json

from aigraph.llm_extract import LLMClaimExtractor
from aigraph.models import Paper


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


class _FakeResponseObject:
    def __init__(self, content: str):
        self.output_text = content


class _FakeResponses:
    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponseObject(self._content)


class _FakeResponsesClient(_FakeClient):
    def __init__(self, content: str):
        super().__init__(content)
        self.responses = _FakeResponses(content)


def _paper_with_abstract(text: str) -> Paper:
    return Paper(
        paper_id="openalex:W1",
        title="Study of RAG",
        year=2024,
        venue="ACL",
        abstract=text,
        text=f"Study of RAG\n\n{text}",
    )


def test_parses_strict_json_response_into_claim():
    abstract = "RAG improves factual QA on NaturalQuestions by +8.2 EM over a closed-book LLM."
    payload = {
        "claims": [
            {
                "claim_text": "RAG improves factual QA by +8.2 EM.",
                "claim_type": "performance_improvement",
                "method": "RAG",
                "model": "GPT-3.5",
                "task": "factual QA",
                "dataset": "NaturalQuestions",
                "metric": "Exact Match",
                "baseline": "closed-book LLM",
                "result": "+8.2",
                "direction": "positive",
                "setting": {
                    "retriever": "DPR",
                    "top_k": "5",
                    "context_length": "4k",
                    "task_type": "factual",
                },
                "evidence_span": abstract,
                "domain": "finance",
                "data_modality": "text + time series",
                "mechanism": "event grounding",
                "failure_mode": "temporal leakage",
                "evaluation_protocol": "backtesting",
                "assumption": "market regimes shift",
                "risk_type": "financial risk",
                "temporal_property": "non-stationarity",
            }
        ]
    }
    client = _FakeClient(json.dumps(payload))
    extractor = LLMClaimExtractor(model="stub-model", client=client, api_key="test-key")
    claims = extractor.extract(_paper_with_abstract(abstract))
    assert len(claims) == 1
    assert claims[0].claim_id == "c001"
    assert claims[0].method == "RAG"
    assert claims[0].setting.task_type == "factual"
    assert claims[0].evidence_span == abstract
    assert claims[0].domain == "finance"
    assert claims[0].data_modality == "text + time series"
    assert claims[0].mechanism == "event grounding"
    assert claims[0].failure_mode == "temporal leakage"
    assert claims[0].evaluation_protocol == "backtesting"
    assert claims[0].temporal_property == "non-stationarity"


def test_uses_responses_endpoint_when_available(monkeypatch):
    abstract = "RAG improves factual QA on NaturalQuestions."
    payload = {"claims": [{"claim_text": "RAG helps factual QA.", "method": "RAG", "direction": "positive", "evidence_span": abstract}]}
    client = _FakeResponsesClient(json.dumps(payload))
    monkeypatch.setenv("AIGRAPH_LLM_ENDPOINT", "responses")
    monkeypatch.setenv("AIGRAPH_REASONING_EFFORT", "high")
    extractor = LLMClaimExtractor(model="stub", client=client, api_key="test-key")
    claims = extractor.extract(_paper_with_abstract(abstract))
    assert len(claims) == 1
    call = client.responses.calls[0]
    assert call["model"] == "stub"
    assert call["reasoning"] == {"effort": "high"}
    assert "max_output_tokens" in call


def test_strips_markdown_fences():
    abstract = "RAG may hurt HotpotQA when top_k is large."
    wrapped = "```json\n" + json.dumps({
        "claims": [
            {
                "claim_text": "RAG hurts HotpotQA at high top_k.",
                "claim_type": "limitation",
                "method": "RAG",
                "task": "multi-hop QA",
                "direction": "negative",
                "setting": {"top_k": "20", "task_type": "multi-hop"},
                "evidence_span": abstract,
            }
        ]
    }) + "\n```"
    client = _FakeClient(wrapped)
    extractor = LLMClaimExtractor(model="stub", client=client, api_key="test-key")
    claims = extractor.extract(_paper_with_abstract(abstract))
    assert len(claims) == 1
    assert claims[0].direction == "negative"


def test_rejects_hallucinated_evidence_span():
    abstract = "RAG improves factual QA on NaturalQuestions."
    payload = {
        "claims": [
            {
                "claim_text": "RAG helps.",
                "claim_type": "performance_improvement",
                "method": "RAG",
                "direction": "positive",
                "evidence_span": "Completely fabricated quote about transformers.",
            }
        ]
    }
    client = _FakeClient(json.dumps(payload))
    extractor = LLMClaimExtractor(model="stub", client=client, api_key="test-key")
    claims = extractor.extract(_paper_with_abstract(abstract))
    # Claim is still produced, but the bogus span is dropped.
    assert len(claims) == 1
    assert claims[0].evidence_span == ""


def test_malformed_json_returns_empty_list():
    client = _FakeClient("not json at all")
    extractor = LLMClaimExtractor(model="stub", client=client, api_key="test-key")
    claims = extractor.extract(_paper_with_abstract("something"))
    assert claims == []


def test_invalid_enum_values_are_normalized():
    abstract = "RAG is fine."
    payload = {
        "claims": [
            {
                "claim_text": "x",
                "claim_type": "speculation",  # not in allowed set
                "direction": "up",  # not in allowed set
                "setting": {"task_type": "unknown-type"},  # not in allowed set
                "evidence_span": abstract,
            }
        ]
    }
    client = _FakeClient(json.dumps(payload))
    extractor = LLMClaimExtractor(model="stub", client=client, api_key="test-key")
    claims = extractor.extract(_paper_with_abstract(abstract))
    assert len(claims) == 1
    assert claims[0].claim_type == "performance_improvement"
    assert claims[0].direction == "positive"
    assert claims[0].setting.task_type is None


def test_canonical_method_and_task_are_validated():
    abstract = "RAG improves medical question answering."
    payload = {
        "claims": [
            {
                "claim_text": "RAG helps medical QA.",
                "method": "RAG",
                "canonical_method": "rag",  # case-insensitive match
                "task": "medical question answering",
                "canonical_task": "domain-QA",
                "direction": "positive",
                "evidence_span": abstract,
            },
            {
                "claim_text": "y",
                "method": "RAG",
                "canonical_method": "made-up-category",  # invalid -> None
                "canonical_task": "factual-qa",  # case-insensitive
                "direction": "positive",
                "evidence_span": abstract,
            },
        ]
    }
    client = _FakeClient(json.dumps(payload))
    extractor = LLMClaimExtractor(model="stub", client=client, api_key="test-key")
    claims = extractor.extract(_paper_with_abstract(abstract))
    assert len(claims) == 2
    assert claims[0].canonical_method == "RAG"
    assert claims[0].canonical_task == "domain-QA"
    assert claims[1].canonical_method is None
    assert claims[1].canonical_task == "factual-QA"
