import pytest

from aigraph.cli import _build_extractor, _extract_claims_incremental
from aigraph.extract import RuleBasedExtractor, extract_claims
from aigraph.models import Claim, Paper


def test_structured_hint_extraction():
    paper = Paper(
        paper_id="p001",
        title="Test",
        year=2024,
        venue="X",
        abstract="",
        structured_hint=[
            {
                "claim_text": "RAG improves NQ by +8.",
                "method": "RAG",
                "task": "factual QA",
                "direction": "positive",
                "setting": {"top_k": "5", "task_type": "factual"},
            },
            {
                "claim_text": "RAG hurts HotpotQA.",
                "method": "RAG",
                "task": "multi-hop QA",
                "direction": "negative",
                "setting": {"top_k": "20", "task_type": "multi-hop"},
            },
        ],
    )

    claims = RuleBasedExtractor().extract(paper)

    assert len(claims) == 2
    assert claims[0].claim_id == "c001"
    assert claims[0].method == "RAG"
    assert claims[0].setting.top_k == "5"
    assert claims[0].subject_raw == "RAG"
    assert claims[0].predicate == "improves"
    assert claims[1].direction == "negative"


def test_structured_hint_tuple_shape_is_supported():
    paper = Paper(
        paper_id="p003",
        title="Tuple",
        year=2024,
        venue="X",
        abstract="",
        structured_hint=[
            {
                "subject_raw": "RLHF on 7B models",
                "predicate": "improves helpfulness",
                "object_raw": "assistant behavior",
                "dataset_raw": "HH-RLHF",
                "metric_raw": "win rate",
                "baseline_raw": "SFT only",
                "magnitude_text": "+14pp",
                "conditions": ["batch size >= 64"],
                "scope": ["7B models"],
                "claim_text": "RLHF on 7B models improves helpfulness by +14pp.",
            }
        ],
    )
    claims = RuleBasedExtractor().extract(paper)
    assert len(claims) == 1
    assert claims[0].subject_raw == "RLHF on 7B models"
    assert claims[0].dataset_raw == "HH-RLHF"
    assert claims[0].dataset_canonical == "hh-rlhf"
    assert claims[0].metric_canonical == "win-rate"
    assert claims[0].magnitude_value == 14.0
    assert claims[0].conditions == ["batch size >= 64"]
    assert claims[0].scope == ["7B models"]


def test_heuristic_fallback_detects_direction():
    paper = Paper(
        paper_id="p002",
        title="Heuristic",
        year=2024,
        venue="X",
        abstract="RAG improves factual QA on NaturalQuestions by +4.0 EM. Long-context models degrades the RAG advantage.",
    )

    claims = extract_claims([paper])

    assert len(claims) == 2
    assert claims[0].direction == "positive"
    assert claims[0].method == "RAG"
    assert claims[0].subject_raw == "RAG"
    assert claims[0].predicate == "improves"
    assert claims[0].magnitude_value == 4.0
    assert claims[0].evidence_sentence_index == 0
    assert claims[0].evidence_char_start is not None
    assert claims[0].evidence_char_end is not None
    assert claims[1].direction == "negative"
    assert claims[1].magnitude_value is None
    assert claims[1].conditions == []
    assert claims[1].scope == []


def test_structured_hint_negative_magnitude_is_parsed():
    paper = Paper(
        paper_id="p005",
        title="Negative",
        year=2024,
        venue="X",
        structured_hint=[
            {
                "claim_text": "Long-context hurts factual QA by -3 EM.",
                "method": "long-context",
                "task": "factual QA",
                "direction": "negative",
                "magnitude_text": "-3 EM",
            }
        ],
    )
    claims = RuleBasedExtractor().extract(paper)
    assert len(claims) == 1
    assert claims[0].magnitude_value == -3.0
    assert claims[0].magnitude_unit == "em"


def test_grounding_keeps_span_but_can_leave_offsets_null_when_match_is_normalized():
    paper = Paper(
        paper_id="p004",
        title="Grounding",
        year=2024,
        venue="X",
        text="RAG improves factual QA on\nNaturalQuestions by +8.2 EM over a closed-book LLM.",
        structured_hint=[
            {
                "claim_text": "RAG improves factual QA by +8.2 EM.",
                "method": "RAG",
                "task": "factual QA",
                "direction": "positive",
                "evidence_span": "RAG improves factual QA on NaturalQuestions by +8.2 EM over a closed-book LLM.",
            }
        ],
    )
    claims = RuleBasedExtractor().extract(paper)
    assert len(claims) == 1
    assert claims[0].evidence_span
    assert claims[0].evidence_char_start is None
    assert claims[0].evidence_char_end is None


def test_claim_ids_are_globally_unique_across_papers():
    papers = [
        Paper(
            paper_id="p001",
            title="A",
            year=2024,
            venue="X",
            structured_hint=[{"claim_text": "x", "method": "RAG", "task": "factual QA", "direction": "positive"}],
        ),
        Paper(
            paper_id="p002",
            title="B",
            year=2024,
            venue="X",
            structured_hint=[
                {"claim_text": "y", "method": "RAG", "task": "factual QA", "direction": "negative"},
                {"claim_text": "z", "method": "RAG", "task": "multi-hop QA", "direction": "negative"},
            ],
        ),
    ]
    claims = extract_claims(papers)
    ids = [c.claim_id for c in claims]
    assert ids == ["c001", "c002", "c003"]


def test_cli_extract_defaults_to_rule_based():
    assert isinstance(_build_extractor("rule", None), RuleBasedExtractor)
    # Default value in CLI is "rule"; passing None should behave like "rule".
    assert isinstance(_build_extractor(None, None), RuleBasedExtractor)


def test_cli_extract_rejects_unknown_extractor():
    import typer
    with pytest.raises(typer.BadParameter):
        _build_extractor("neural", None)


class _FakeExtractor:
    def __init__(self):
        self.calls: list[str] = []

    def extract(self, paper: Paper, start_index: int = 0, *, candidates=None) -> list[Claim]:
        self.calls.append(paper.paper_id)
        return [
            Claim(
                claim_id=f"c{start_index + 1:03d}",
                paper_id=paper.paper_id,
                claim_text=f"{paper.title} claim",
                method="RAG",
                task="factual QA",
                direction="positive",
            )
        ]


def test_incremental_extraction_writes_and_resumes(tmp_path):
    papers = [
        Paper(paper_id="p001", title="A", year=2024, venue="X"),
        Paper(paper_id="p002", title="B", year=2024, venue="X"),
    ]
    output = tmp_path / "claims.jsonl"

    first = _FakeExtractor()
    claims = _extract_claims_incremental(papers, first, output, resume=False)
    assert [c.claim_id for c in claims] == ["c001", "c002"]
    assert output.read_text(encoding="utf-8").count("\n") == 2

    second = _FakeExtractor()
    resumed = _extract_claims_incremental(papers, second, output, resume=True)
    assert [c.claim_id for c in resumed] == ["c001", "c002"]
    assert second.calls == []


def test_extract_claims_reader_off_preserves_rule_heuristic_behavior():
    paper = Paper(
        paper_id="p006",
        title="Reader Off",
        year=2024,
        venue="X",
        abstract="RAG improves factual QA on NaturalQuestions by +4.0 EM. Long-context models degrades the RAG advantage.",
    )
    claims = extract_claims([paper], extractor=RuleBasedExtractor(), reader_mode="off")
    assert len(claims) == 2


def test_extract_claims_reader_heuristic_limits_candidates():
    paper = Paper(
        paper_id="p007",
        title="Reader On",
        year=2024,
        venue="X",
        abstract=(
            "RAG improves factual QA on NaturalQuestions by +4.0 EM. "
            "RAG improves multi-hop QA on HotpotQA by +3.0 F1. "
            "RAG hurts TruthfulQA by -2.0 accuracy."
        ),
    )
    claims = extract_claims(
        [paper],
        extractor=RuleBasedExtractor(),
        reader_mode="heuristic",
        reader_max_candidates=2,
    )
    assert len(claims) == 2
