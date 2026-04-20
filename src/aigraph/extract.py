"""Claim extraction. Deterministic rule-based for the MVP; LLM extractor is a future swap-in."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .claim_schema import (
    normalize_structured_claim_payload,
)
from .models import Claim, Paper, PaperReadCandidate, Setting
from .paper_reader import (
    HeuristicPaperReader,
    read_paper_candidates,
)


class ClaimExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        paper: Paper,
        start_index: int = 0,
        *,
        candidates: list[PaperReadCandidate] | None = None,
    ) -> list[Claim]:
        """Return claims for a single paper. `start_index` lets the caller allocate global IDs."""


class RuleBasedExtractor(ClaimExtractor):
    """Prefer `paper.structured_hint` if present; fall back to regex heuristics."""

    def extract(
        self,
        paper: Paper,
        start_index: int = 0,
        *,
        candidates: list[PaperReadCandidate] | None = None,
    ) -> list[Claim]:
        if paper.structured_hint:
            claims: list[Claim] = []
            for i, hint in enumerate(paper.structured_hint):
                claim_id = f"c{start_index + i + 1:03d}"
                normalized = normalize_structured_claim_payload(
                    hint,
                    paper,
                    trust_evidence_if_no_body=True,
                )
                if normalized is None:
                    continue
                data = {
                    "claim_id": claim_id,
                    "paper_id": paper.paper_id,
                    "claim_type": hint.get("claim_type", "performance_improvement"),
                    "direction": normalized.get("direction") or hint.get("direction", "positive"),
                    "setting": Setting.model_validate(hint.get("setting", {})),
                    **normalized,
                }
                claims.append(Claim.model_validate(data))
            return claims

        if candidates:
            return self._claims_from_candidates(paper, candidates, start_index)

        return self._heuristic_extract(paper, start_index)

    def _heuristic_extract(self, paper: Paper, start_index: int) -> list[Claim]:
        candidates = HeuristicPaperReader(max_prefilter_sentences=1000).read(paper).candidates
        return self._claims_from_candidates(paper, candidates, start_index)

    def _claims_from_candidates(
        self,
        paper: Paper,
        candidates: list[PaperReadCandidate],
        start_index: int,
    ) -> list[Claim]:
        claims: list[Claim] = []
        for candidate in candidates:
            claim_id = f"c{start_index + len(claims) + 1:03d}"
            normalized = normalize_structured_claim_payload(
                {
                    "claim_text": candidate.sentence.strip(),
                    "method": candidate.subject_raw,
                    "task": candidate.object_raw,
                    "dataset": candidate.dataset_raw,
                    "metric": candidate.metric_raw,
                    "baseline": candidate.baseline_raw,
                    "direction": candidate.direction or "positive",
                    "subject_raw": candidate.subject_raw,
                    "predicate": candidate.predicate,
                    "object_raw": candidate.object_raw,
                    "magnitude_text": candidate.magnitude_text,
                    "conditions": candidate.conditions,
                    "scope": candidate.scope,
                    "evidence_span": candidate.evidence_span,
                },
                paper,
            )
            if normalized is None:
                continue
            normalized["evidence_source_field"] = candidate.evidence_source_field
            normalized["evidence_sentence_index"] = candidate.evidence_sentence_index
            normalized["evidence_char_start"] = candidate.evidence_char_start
            normalized["evidence_char_end"] = candidate.evidence_char_end
            claims.append(
                Claim.model_validate(
                    {
                        "claim_id": claim_id,
                        "paper_id": paper.paper_id,
                        "claim_type": "performance_improvement" if candidate.direction == "positive" else "limitation",
                        "direction": candidate.direction or "positive",
                        **normalized,
                    }
                )
            )
        return claims


def extract_claims(
    papers: Iterable[Paper],
    extractor: ClaimExtractor | None = None,
    *,
    reader_mode: str | None = None,
    reader_model: str | None = None,
    reader_max_candidates: int | None = None,
) -> list[Claim]:
    extractor = extractor or RuleBasedExtractor()
    all_claims: list[Claim] = []
    for paper in papers:
        candidates = read_paper_candidates(
            paper,
            mode=reader_mode,
            model=reader_model,
            max_candidates=reader_max_candidates,
        ).candidates
        new_claims = extractor.extract(paper, start_index=len(all_claims), candidates=candidates)
        all_claims.extend(new_claims)
    return all_claims
