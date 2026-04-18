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
    assert claims[1].direction == "negative"


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
    assert claims[1].direction == "negative"


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

    def extract(self, paper: Paper, start_index: int = 0) -> list[Claim]:
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
