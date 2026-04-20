import json

from aigraph.models import Paper
from aigraph.paper_reader import HeuristicPaperReader, LLMPaperReaderMini, read_paper_candidates


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


def test_heuristic_reader_returns_grounded_candidates_from_text():
    paper = Paper(
        paper_id="p001",
        title="RAG study",
        year=2024,
        venue="ACL",
        text=(
            "RAG improves factual QA on NaturalQuestions by +8.2 EM over a closed-book LLM. "
            "We discuss general background. "
            "Long-context prompting hurts performance on HotpotQA by -3 F1."
        ),
    )
    result = HeuristicPaperReader(max_prefilter_sentences=8).read(paper)
    assert len(result.candidates) == 2
    assert result.candidates[0].evidence_source_field == "text"
    assert result.candidates[0].evidence_char_start is not None
    assert result.candidates[0].metric_raw in {"Exact Match", "F1"}


def test_reader_candidate_caps_are_enforced():
    paper = Paper(
        paper_id="p002",
        title="Many claims",
        year=2024,
        venue="ACL",
        abstract=" ".join(
            f"RAG improves factual QA on NaturalQuestions by +{idx}.0 EM."
            for idx in range(1, 8)
        ),
    )
    result = read_paper_candidates(
        paper,
        mode="heuristic",
        max_candidates=2,
        prefilter_sentences=3,
    )
    assert len(result.candidates) == 2
    assert result.prefilter_count == 3


def test_mini_reader_accepts_valid_json_and_maps_back_to_grounded_candidates():
    paper = Paper(
        paper_id="p003",
        title="Mini reader",
        year=2024,
        venue="ACL",
        text="RAG improves factual QA on NaturalQuestions by +8.2 EM over a closed-book LLM.",
    )
    pool = HeuristicPaperReader(max_prefilter_sentences=4).read(paper).candidates
    payload = {
        "candidates": [
            {
                "candidate_index": 0,
                "evidence_span": "RAG improves factual QA on NaturalQuestions by +8.2 EM over a closed-book LLM.",
                "subject_raw": "RAG",
                "predicate": "improves",
                "object_raw": "factual QA",
                "dataset_raw": "NaturalQuestions",
                "metric_raw": "Exact Match",
                "baseline_raw": "closed-book LLM",
                "direction": "positive",
                "magnitude_text": "+8.2 EM",
                "conditions": ["top_k = 5"],
                "scope": ["factual QA"],
                "candidate_score": 0.95,
                "selection_reason": "strong empirical result",
            }
        ]
    }
    client = _FakeClient(json.dumps(payload))
    result = LLMPaperReaderMini(
        model="stub-mini",
        max_candidates=2,
        candidates=pool,
        client=client,
        api_key="test-key",
    ).read(paper)
    assert len(result.candidates) == 1
    assert result.candidates[0].evidence_char_start == 0
    assert result.candidates[0].dataset_raw == "NaturalQuestions"


def test_mini_reader_falls_back_to_heuristic_on_invalid_json():
    paper = Paper(
        paper_id="p004",
        title="Fallback",
        year=2024,
        venue="ACL",
        abstract="RAG improves factual QA on NaturalQuestions by +4.0 EM.",
    )
    client = _FakeClient("not-json")
    result = read_paper_candidates(
        paper,
        mode="mini",
        max_candidates=2,
        prefilter_sentences=4,
        client=client,
        api_key="test-key",
    )
    assert result.mode_used == "heuristic"
    assert result.fallback_used is True
    assert len(result.candidates) == 1
