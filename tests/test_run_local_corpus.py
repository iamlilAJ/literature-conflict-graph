"""Unit tests for the runner script's filter logic.

The full e2e flow is exercised by the bundled
``artifacts/runs/arxiv-reasoning-v0.7-100p/`` integration run; these
tests pin the standalone filter helper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# scripts/ is not a package; add it to sys.path then import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_local_corpus import filter_papers, _safe_artifact_id  # noqa: E402


def _write_corpus(
    tmp_path: Path,
    papers: list[dict],
    *,
    sections_for_ids: list[str] | None = None,
) -> Path:
    """Write a small corpus directory + optional sections.json artifacts.
    Returns the corpus root path."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    papers_path = corpus / "papers.jsonl"
    with papers_path.open("w") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    artifacts = corpus / "artifacts"
    artifacts.mkdir()
    if sections_for_ids:
        for pid in sections_for_ids:
            d = artifacts / _safe_artifact_id(pid)
            d.mkdir(parents=True)
            (d / "sections.json").write_text("[]")
    return corpus


def test_filter_papers_keeps_only_year_keyword_and_sections_match(tmp_path):
    """Three filter dimensions must all pass: year_min, keyword (any),
    and the presence of artifacts/<safe_id>/sections.json."""
    papers = [
        # passes all
        {"paper_id": "arxiv:2024.01", "title": "Chain-of-Thought Reasoning", "abstract": "...", "year": 2024},
        # year too old
        {"paper_id": "arxiv:2018.01", "title": "Reasoning over graphs", "abstract": "x", "year": 2018},
        # no keyword in title or abstract
        {"paper_id": "arxiv:2024.02", "title": "An unrelated topic", "abstract": "boring", "year": 2024},
        # passes year + keyword but has no sections.json on disk
        {"paper_id": "arxiv:2024.03", "title": "Reasoning soup", "abstract": "more reasoning", "year": 2024},
    ]
    corpus = _write_corpus(
        tmp_path,
        papers,
        sections_for_ids=["arxiv:2024.01"],  # only one paper has sections
    )

    selected = filter_papers(
        corpus_dir=corpus,
        year_min=2023,
        keywords=["reasoning"],
        max_papers=10,
    )

    assert len(selected) == 1
    assert selected[0]["paper_id"] == "arxiv:2024.01"


def test_filter_papers_sorts_by_year_desc_then_paper_id(tmp_path):
    """Tie-breaker: same-year papers ordered by paper_id ascending."""
    papers = [
        {"paper_id": "arxiv:2023.99", "title": "agent reasoning", "abstract": "x" * 100, "year": 2023},
        {"paper_id": "arxiv:2024.01", "title": "agent thing", "abstract": "x" * 100, "year": 2024},
        {"paper_id": "arxiv:2024.05", "title": "another agent", "abstract": "x" * 100, "year": 2024},
        {"paper_id": "arxiv:2025.07", "title": "Agent System 2", "abstract": "x" * 100, "year": 2025},
    ]
    corpus = _write_corpus(
        tmp_path,
        papers,
        sections_for_ids=[p["paper_id"] for p in papers],
    )

    selected = filter_papers(
        corpus_dir=corpus,
        year_min=2023,
        keywords=["agent"],
        max_papers=10,
    )
    ids = [p["paper_id"] for p in selected]
    assert ids == [
        "arxiv:2025.07",  # year=2025 first
        "arxiv:2024.01",  # year=2024, lower paper_id wins tie
        "arxiv:2024.05",
        "arxiv:2023.99",
    ]


def test_filter_papers_caps_at_max(tmp_path):
    """max_papers truncates after sorting."""
    papers = [
        {"paper_id": f"arxiv:2024.{i:02d}", "title": "reasoning", "abstract": "x" * 100, "year": 2024}
        for i in range(10)
    ]
    corpus = _write_corpus(tmp_path, papers, sections_for_ids=[p["paper_id"] for p in papers])
    selected = filter_papers(corpus, year_min=2023, keywords=["reasoning"], max_papers=3)
    assert len(selected) == 3


def test_filter_papers_keyword_matching_is_case_insensitive(tmp_path):
    """Mixed-case keywords + abstracts still match."""
    papers = [
        {"paper_id": "arxiv:2024.01", "title": "REASONING in LLMs", "abstract": "x" * 100, "year": 2024},
        {"paper_id": "arxiv:2024.02", "title": "Vision survey", "abstract": "We discuss Reasoning at length.", "year": 2024},
        {"paper_id": "arxiv:2024.03", "title": "Vision-only", "abstract": "no token of interest", "year": 2024},
    ]
    corpus = _write_corpus(tmp_path, papers, sections_for_ids=[p["paper_id"] for p in papers])
    selected = filter_papers(corpus, year_min=2023, keywords=["reasoning"], max_papers=10)
    ids = sorted(p["paper_id"] for p in selected)
    assert ids == ["arxiv:2024.01", "arxiv:2024.02"]


def test_filter_papers_missing_corpus_raises(tmp_path):
    """Pointing to a non-existent corpus path must surface a clear error,
    not silently return an empty list."""
    with pytest.raises(FileNotFoundError):
        filter_papers(
            corpus_dir=tmp_path / "does-not-exist",
            year_min=2023,
            keywords=["x"],
            max_papers=10,
        )
