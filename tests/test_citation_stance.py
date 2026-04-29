"""Tests for src/aigraph/citation_stance.py — LLM stance classification on
cites edges."""

from __future__ import annotations

import json

import pytest

from aigraph.citation_stance import (
    STANCE_LABELS,
    _extract_citation_context,
    _strip_bibliography,
    classify_cites_edges,
)
from aigraph.graph import build_graph
from aigraph.models import Claim, Paper


# --- Fake LLM client (mirrors tests/test_llm_hypotheses.py:8-40) -----------


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


# --- Fixture helpers --------------------------------------------------------


def _papers_b_cites_a(*, b_text: str = "", a_title: str = "Foo Method") -> list[Paper]:
    """Construct a 2-paper corpus where B cites A."""
    return [
        Paper(
            paper_id="A",
            title=a_title,
            year=2023,
            venue="ACL",
            cited_by_count=10,
        ),
        Paper(
            paper_id="B",
            title="A study that cites Foo Method",
            year=2024,
            venue="EMNLP",
            cited_by_count=5,
            referenced_works=["A"],
            text=b_text,
        ),
    ]


def _claims_for(papers: list[Paper]) -> list[Claim]:
    """One trivial claim per paper so build_graph has something to anchor on."""
    return [
        Claim(
            claim_id=f"c{p.paper_id}",
            paper_id=p.paper_id,
            claim_text="x",
            method="m",
            task="t",
            direction="positive",
        )
        for p in papers
    ]


def _stance_payload(stance: str = "extends", confidence: float = 0.85) -> str:
    return json.dumps(
        {
            "stance": stance,
            "confidence": confidence,
            "rationale": "B explicitly builds on A's framework.",
        }
    )


def _cites_data(g, u: str, v: str) -> dict:
    """Return the data dict of the (single) cites edge between u and v."""
    edge_data = g.get_edge_data(u, v) or {}
    for nd in edge_data.values():
        if nd.get("edge_type") == "cites":
            return nd
    raise AssertionError(f"no cites edge between {u} and {v}")


# --- _extract_citation_context (pure unit tests) ----------------------------


def test_extract_citation_context_returns_surrounding_sentences():
    text = "First sentence. We compare to Foo Method on this benchmark. The result was X."
    contexts = _extract_citation_context(text, "Foo Method")
    assert len(contexts) == 1
    assert "First sentence" in contexts[0]
    assert "Foo Method" in contexts[0]
    assert "result was X" in contexts[0]


def test_extract_citation_context_returns_empty_when_no_match():
    text = "We compare to a totally different baseline. The result was X."
    contexts = _extract_citation_context(text, "Foo Method")
    assert contexts == []


def test_extract_citation_context_strips_bibliography_section():
    """A title appearing only in the References section should not produce a
    context — _strip_bibliography removes that section before matching."""
    text = (
        "We propose a new approach. Section body that does not mention the cited paper.\n"
        "References\n"
        "[1] Foo Method by Author et al, 2023.\n"
    )
    stripped = _strip_bibliography(text)
    contexts = _extract_citation_context(stripped, "Foo Method")
    assert contexts == []


def test_extract_citation_context_handles_long_titles_via_prefix():
    """Titles >10 words match on the first 5 words, so a truncated citation
    in the body still hits."""
    long_title = "Self-Consistency Improves Chain Of Thought Reasoning In Large Language Models By A Lot"
    text = "Following Self-Consistency Improves Chain of Thought, we apply majority voting."
    contexts = _extract_citation_context(text, long_title)
    assert len(contexts) >= 1


# --- classify_cites_edges --------------------------------------------------


def test_classify_attaches_stance_to_cites_edges():
    """Happy path: B's text mentions A's title, LLM returns 'extends', edge
    picks up stance + confidence + rationale."""
    papers = _papers_b_cites_a(
        b_text="We extend Foo Method with a new sampling strategy.",
        a_title="Foo Method",
    )
    g = build_graph(_claims_for(papers), papers=papers)
    fake = _FakeClient(_stance_payload(stance="extends", confidence=0.9))

    counters = classify_cites_edges(g, papers, client=fake, model="stub")

    assert counters["classified"] == 1
    assert counters["skipped_no_text"] == 0
    edge = _cites_data(g, "Paper:B", "Paper:A")
    assert edge["stance"] == "extends"
    assert edge["stance_confidence"] == 0.9
    assert edge["stance_rationale"]
    assert len(fake.chat.completions.calls) == 1


def test_classify_skips_edges_with_existing_stance():
    """Idempotency: edges already carrying stance are skipped (no LLM call)
    when skip_classified=True (default)."""
    papers = _papers_b_cites_a(
        b_text="We extend Foo Method.",
        a_title="Foo Method",
    )
    g = build_graph(_claims_for(papers), papers=papers)
    # Pre-stamp the cites edge.
    edge_data = g.get_edge_data("Paper:B", "Paper:A") or {}
    for nd in edge_data.values():
        if nd.get("edge_type") == "cites":
            nd["stance"] = "mentions"
            break
    fake = _FakeClient(_stance_payload())

    counters = classify_cites_edges(g, papers, client=fake, model="stub")

    assert counters["classified"] == 0
    assert counters["skipped_already"] == 1
    assert fake.chat.completions.calls == []


def test_classify_skips_when_paper_b_has_no_text():
    """Paper B has empty text — edge stays unchanged, LLM never called."""
    papers = _papers_b_cites_a(b_text="", a_title="Foo Method")
    g = build_graph(_claims_for(papers), papers=papers)
    fake = _FakeClient(_stance_payload())

    counters = classify_cites_edges(g, papers, client=fake, model="stub")

    assert counters["classified"] == 0
    assert counters["skipped_no_text"] == 1
    assert fake.chat.completions.calls == []
    assert "stance" not in _cites_data(g, "Paper:B", "Paper:A")


def test_classify_skips_when_a_title_not_in_b_text():
    """B has real text but doesn't mention A's title — no LLM call."""
    papers = _papers_b_cites_a(
        b_text="This paper studies entirely unrelated topics like cooking.",
        a_title="Foo Method",
    )
    g = build_graph(_claims_for(papers), papers=papers)
    fake = _FakeClient(_stance_payload())

    counters = classify_cites_edges(g, papers, client=fake, model="stub")

    assert counters["classified"] == 0
    assert counters["skipped_no_match"] == 1
    assert fake.chat.completions.calls == []


def test_classify_handles_invalid_stance_gracefully():
    """LLM returns nonsense stance — no edge mutation, llm_failed counter
    incremented. Do NOT fall back to a default label silently."""
    papers = _papers_b_cites_a(
        b_text="We extend Foo Method to a new domain.",
        a_title="Foo Method",
    )
    g = build_graph(_claims_for(papers), papers=papers)
    fake = _FakeClient(json.dumps({"stance": "lolwhat", "confidence": 0.5}))

    counters = classify_cites_edges(g, papers, client=fake, model="stub")

    assert counters["classified"] == 0
    assert counters["llm_failed"] == 1
    assert "stance" not in _cites_data(g, "Paper:B", "Paper:A")


def test_classify_skips_self_citation():
    """An edge where u == v (paper cites itself, possible from a data quirk)
    is dropped before any LLM call."""
    papers = [
        Paper(
            paper_id="A",
            title="Foo Method",
            year=2023,
            venue="ACL",
            referenced_works=["A"],  # self-reference
            text="Foo Method describes itself.",
        ),
    ]
    g = build_graph(_claims_for(papers), papers=papers)
    fake = _FakeClient(_stance_payload())

    counters = classify_cites_edges(g, papers, client=fake, model="stub")

    assert counters["skipped_self_citation"] == 1
    assert counters["classified"] == 0
    assert fake.chat.completions.calls == []


def test_dry_run_returns_count_without_llm_calls():
    """dry_run=True reports would_classify but makes zero LLM calls."""
    papers = _papers_b_cites_a(
        b_text="We extend Foo Method.",
        a_title="Foo Method",
    )
    g = build_graph(_claims_for(papers), papers=papers)
    fake = _FakeClient(_stance_payload())

    counters = classify_cites_edges(
        g, papers, client=fake, model="stub", dry_run=True
    )

    assert counters["would_classify"] == 1
    assert "classified" not in counters or counters.get("classified", 0) == 0
    assert fake.chat.completions.calls == []
    assert "stance" not in _cites_data(g, "Paper:B", "Paper:A")


def test_max_edges_caps_classification():
    """max_edges=1 with 2 candidate edges processes only the first."""
    papers = [
        Paper(paper_id="A", title="Method A", year=2023, venue="ACL"),
        Paper(paper_id="X", title="Method X", year=2023, venue="EMNLP"),
        Paper(
            paper_id="B",
            title="A study",
            year=2024,
            venue="ICLR",
            referenced_works=["A", "X"],
            text="We extend Method A and also Method X here in the body.",
        ),
    ]
    g = build_graph(_claims_for(papers), papers=papers)
    fake = _FakeClient(_stance_payload())

    counters = classify_cites_edges(
        g, papers, client=fake, model="stub", max_edges=1
    )

    assert counters["classified"] == 1
    assert len(fake.chat.completions.calls) == 1


def test_returned_stance_is_in_stance_labels():
    """Structural assertion — whatever stance comes back must be one of the
    five canonical labels. Locks the validator boundary."""
    papers = _papers_b_cites_a(b_text="We extend Foo Method.", a_title="Foo Method")
    g = build_graph(_claims_for(papers), papers=papers)
    fake = _FakeClient(_stance_payload(stance="extends"))

    classify_cites_edges(g, papers, client=fake, model="stub")

    edge = _cites_data(g, "Paper:B", "Paper:A")
    assert edge["stance"] in STANCE_LABELS
