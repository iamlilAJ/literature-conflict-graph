"""Tests for the post-hypothesis arxiv novelty check module."""

from __future__ import annotations

import json
from typing import Any

import pytest

from aigraph.models import Hypothesis
from aigraph.novelty_check import (
    ARXIV_QUERY_URL,
    check_hypothesis_novelty,
    query_arxiv,
)


def test_arxiv_query_url_uses_https():
    """Regression: arxiv 301-redirects http -> https. httpx defaults
    follow_redirects=False, so an http URL silently 0s out the
    candidate list and downstream is_novel becomes null. Pin to https."""
    assert ARXIV_QUERY_URL.startswith("https://"), (
        f"ARXIV_QUERY_URL must be https to avoid 301-redirect data loss "
        f"(got {ARXIV_QUERY_URL!r})"
    )


# --- LLM mock template (mirrors tests/test_llm_hypotheses.py:8-40) ---


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

    def create(self, **kwargs: Any) -> _FakeCompletion:
        self.calls.append(kwargs)
        return _FakeCompletion(self._content)


class _FakeChat:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)


class _FakeLLMClient:
    def __init__(self, content: str):
        self.chat = _FakeChat(content)


# --- arXiv mock template ---


_ATOM_TWO_ENTRIES = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.11111v1</id>
    <title>Foo Method for Reasoning</title>
    <summary>An older approach to reasoning over text using Foo Method.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.22222v2</id>
    <title>Old Method Survey</title>
    <summary>A survey of old methods for retrieval-augmented reasoning.</summary>
  </entry>
</feed>
"""


class _FakeArxivResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeArxivClient:
    def __init__(self, text: str):
        self._text = text
        self.calls: list[dict] = []

    def get(self, url: str, params: dict) -> _FakeArxivResponse:
        self.calls.append({"url": url, "params": params})
        return _FakeArxivResponse(self._text)


class _BrokenArxivClient:
    """Client that raises a transport-style error on every request, used
    to simulate arxiv being unavailable. Mirrors what httpx.ConnectError
    would surface up to ``check_hypothesis_novelty``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get(self, url: str, params: dict) -> _FakeArxivResponse:
        self.calls.append({"url": url, "params": params})
        try:
            import httpx

            raise httpx.ConnectError("simulated arxiv unavailable")
        except ImportError:
            import urllib.error

            raise urllib.error.URLError("simulated arxiv unavailable")


# --- helpers ---


def _make_hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis_id="h001",
        anomaly_id="a001",
        hypothesis=(
            "Adaptive Retrieval Reranker improves retrieval-augmented "
            "reasoning under domain shift on Adaptive Retrieval Reranker tasks."
        ),
        mechanism=(
            "Adaptive Retrieval Reranker reweights candidate passages using "
            "domain-specific Adaptive Retrieval Reranker signals before "
            "reasoning, so the reasoning step sees better-matched evidence."
        ),
    )


# --- tests ---


def test_query_arxiv_returns_candidate_titles():
    fake = _FakeArxivClient(_ATOM_TWO_ENTRIES)
    candidates = query_arxiv("ti:foo OR abs:foo", max_results=5, http_client=fake)
    assert len(candidates) == 2
    assert candidates[0]["arxiv_id"] == "2401.11111v1"
    assert candidates[0]["title"] == "Foo Method for Reasoning"
    assert "older approach" in candidates[0]["abstract"]
    assert candidates[1]["arxiv_id"] == "2402.22222v2"
    assert candidates[1]["title"] == "Old Method Survey"
    # And the query went through to the right endpoint with the expected
    # params (search_query passed through verbatim, max_results respected).
    assert fake.calls[0]["params"]["search_query"] == "ti:foo OR abs:foo"
    assert fake.calls[0]["params"]["max_results"] == 5


def test_check_hypothesis_classifies_as_novel():
    arxiv_client = _FakeArxivClient(_ATOM_TWO_ENTRIES)
    llm_payload = {
        "is_novel": True,
        "similar": [{"arxiv_id": "1234.5", "title": "Old Method"}],
        "rationale": "core mechanism is new",
    }
    llm_client = _FakeLLMClient(json.dumps(llm_payload))

    result = check_hypothesis_novelty(
        _make_hypothesis(),
        http_client=arxiv_client,
        llm_client=llm_client,
        model="stub",
    )
    assert result["is_novel"] is True
    assert result["rationale"] == "core mechanism is new"
    assert result["similar_papers"] == [{"arxiv_id": "1234.5", "title": "Old Method"}]
    # The LLM was actually invoked (we exercised the chat completions path).
    assert llm_client.chat.completions.calls, "expected at least one LLM call"
    assert llm_client.chat.completions.calls[0]["model"] == "stub"


def test_check_hypothesis_handles_arxiv_unavailable(monkeypatch):
    """Arxiv transport failure must not propagate; the function returns a
    result dict with ``is_novel=None``, no similar papers, and a rationale
    string mentioning the failure."""
    # Elide retry sleeps — the connection error is treated as transient
    # by the new retry path, so the client is hit 4 times before giving up.
    monkeypatch.setattr("aigraph.novelty_check.time.sleep", lambda *a, **k: None)

    arxiv_client = _BrokenArxivClient()
    # The LLM client is a sentinel that would explode if the function
    # mistakenly tried to call it after an arxiv failure — none of its
    # methods are invoked in this path.
    sentinel_llm = object()

    result = check_hypothesis_novelty(
        _make_hypothesis(),
        http_client=arxiv_client,
        llm_client=sentinel_llm,
        model="stub",
    )

    assert result["is_novel"] is None
    assert result["similar_papers"] == []
    assert isinstance(result["rationale"], str)
    rationale_lower = result["rationale"].lower()
    assert (
        "arxiv" in rationale_lower
        or "unavailable" in rationale_lower
        or "failed" in rationale_lower
        or "no candidates" in rationale_lower
    ), f"rationale should mention the failure: {result['rationale']!r}"
    # All retries exhausted — initial + 3 retries = 4 attempts.
    assert len(arxiv_client.calls) == 4, (
        f"expected 4 attempts (initial + 3 retries), got {len(arxiv_client.calls)}"
    )


class _FlakyArxivClient:
    """Fails with httpx.TimeoutException for the first ``fail_first``
    calls, then returns the canned Atom response. Used to verify the
    retry-then-success path."""

    def __init__(self, atom_xml: str, fail_first: int = 1) -> None:
        self._atom_xml = atom_xml
        self._fail_first = fail_first
        self.calls: list[dict] = []

    def get(self, url: str, params: dict) -> _FakeArxivResponse:
        self.calls.append({"url": url, "params": params})
        if len(self.calls) <= self._fail_first:
            import httpx
            raise httpx.TimeoutException("simulated read timeout")
        return _FakeArxivResponse(self._atom_xml)


def test_query_arxiv_retries_on_timeout(monkeypatch):
    """Transient timeout on attempt 1 should trigger retry; attempt 2
    succeeds and the parsed candidates flow through."""
    monkeypatch.setattr("aigraph.novelty_check.time.sleep", lambda *a, **k: None)

    flaky = _FlakyArxivClient(_ATOM_TWO_ENTRIES, fail_first=1)
    candidates = query_arxiv("ti:foo OR abs:foo", max_results=5, http_client=flaky)

    assert len(candidates) == 2
    assert candidates[0]["arxiv_id"] == "2401.11111v1"
    assert len(flaky.calls) == 2, "expected exactly 1 retry then success"
