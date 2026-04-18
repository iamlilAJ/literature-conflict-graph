"""Claim extraction. Deterministic rule-based for the MVP; LLM extractor is a future swap-in."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Iterable

from .models import Claim, Paper, Setting


class ClaimExtractor(ABC):
    @abstractmethod
    def extract(self, paper: Paper, start_index: int = 0) -> list[Claim]:
        """Return claims for a single paper. `start_index` lets the caller allocate global IDs."""


_DIRECTION_CUES = {
    "positive": ("improve", "improves", "boost", "boosts", "gains", "helps", "outperform"),
    "negative": ("hurts", "degrades", "reduces", "drops", "fails", "worse"),
    "mixed": ("mixed", "uneven", "noisy", "inconsistent", "introduces irrelevant"),
}

_DELTA_RE = re.compile(r"([+\-−])\s*(\d+(?:\.\d+)?)")


class RuleBasedExtractor(ClaimExtractor):
    """Prefer `paper.structured_hint` if present; fall back to regex heuristics."""

    def extract(self, paper: Paper, start_index: int = 0) -> list[Claim]:
        if paper.structured_hint:
            claims: list[Claim] = []
            for i, hint in enumerate(paper.structured_hint):
                claim_id = f"c{start_index + i + 1:03d}"
                data = {
                    "claim_id": claim_id,
                    "paper_id": paper.paper_id,
                    "claim_text": hint.get("claim_text", ""),
                    "claim_type": hint.get("claim_type", "performance_improvement"),
                    "method": hint.get("method"),
                    "model": hint.get("model"),
                    "task": hint.get("task"),
                    "dataset": hint.get("dataset"),
                    "metric": hint.get("metric"),
                    "baseline": hint.get("baseline"),
                    "result": hint.get("result"),
                    "direction": hint.get("direction", "positive"),
                    "setting": Setting.model_validate(hint.get("setting", {})),
                    "evidence_span": hint.get("evidence_span", ""),
                }
                claims.append(Claim.model_validate(data))
            return claims

        return self._heuristic_extract(paper, start_index)

    def _heuristic_extract(self, paper: Paper, start_index: int) -> list[Claim]:
        text = f"{paper.abstract}\n{paper.text}".strip()
        if not text:
            return []

        claims: list[Claim] = []
        for i, sentence in enumerate(_split_sentences(text)):
            direction = _guess_direction(sentence)
            delta = _DELTA_RE.search(sentence)
            if not (direction or delta):
                continue
            claim_id = f"c{start_index + len(claims) + 1:03d}"
            claims.append(
                Claim(
                    claim_id=claim_id,
                    paper_id=paper.paper_id,
                    claim_text=sentence.strip(),
                    claim_type="performance_improvement" if direction == "positive" else "limitation",
                    method=_guess_method(sentence),
                    task=_guess_task(sentence),
                    direction=direction or "positive",
                    result=(delta.group(0) if delta else None),
                    evidence_span=sentence.strip(),
                )
            )
        return claims


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _guess_direction(sentence: str) -> str | None:
    low = sentence.lower()
    for direction, cues in _DIRECTION_CUES.items():
        if any(cue in low for cue in cues):
            return direction
    return None


_METHOD_KEYWORDS = [
    "RAG",
    "DPR",
    "BM25",
    "retrieval-augmented",
    "chain-of-thought",
    "reranking",
]


def _guess_method(sentence: str) -> str | None:
    low = sentence.lower()
    for kw in _METHOD_KEYWORDS:
        if kw.lower() in low:
            return kw
    return None


_TASK_KEYWORDS = [
    ("multi-hop", "multi-hop QA"),
    ("hotpotqa", "multi-hop QA"),
    ("naturalquestions", "factual QA"),
    ("factual", "factual QA"),
    ("long-context", "long-context QA"),
    ("agentic", "agentic QA"),
]


def _guess_task(sentence: str) -> str | None:
    low = sentence.lower()
    for needle, canonical in _TASK_KEYWORDS:
        if needle in low:
            return canonical
    return None


def extract_claims(
    papers: Iterable[Paper],
    extractor: ClaimExtractor | None = None,
) -> list[Claim]:
    extractor = extractor or RuleBasedExtractor()
    all_claims: list[Claim] = []
    for paper in papers:
        new_claims = extractor.extract(paper, start_index=len(all_claims))
        all_claims.extend(new_claims)
    return all_claims
