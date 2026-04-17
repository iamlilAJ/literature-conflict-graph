"""Representative paper selection for retrieval candidate pools."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Literal

from .models import Paper


SelectionStrategy = Literal["balanced", "high-impact", "recent"]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
PHRASE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("large language models", ("large language models", "language models", "llm", "llms")),
    ("retrieval augmented generation", ("retrieval augmented generation", "retrieval-augmented generation", "rag")),
    ("time series", ("time series", "time-series")),
    ("scientific question answering", ("scientific qa", "scientific question answering")),
    ("medical question answering", ("medical qa", "medical question answering")),
    ("reinforcement learning from human feedback", ("rlhf", "reinforcement learning from human feedback")),
    ("direct preference optimization", ("dpo", "direct preference optimization")),
    ("proximal policy optimization", ("ppo", "proximal policy optimization")),
)
TOKEN_ALIASES = {
    "forecasting": "forecasting",
    "forecast": "forecasting",
    "financial": "finance",
    "temporal": "temporal",
    "retrieval": "retrieval",
    "alignment": "alignment",
    "safety": "safety",
}
QUERY_FILLER_PHRASES: tuple[str, ...] = (
    "best design for",
    "design for",
    "how to",
    "what is the best",
    "help me find",
    "papers about",
    "research about",
    "research on",
    "studies on",
    "using",
    "with",
    "for",
)
COMPLEX_QUERY_HINTS = {
    "best",
    "design",
    "mechanism",
    "architecture",
    "memory",
    "agent",
    "agents",
    "signal",
    "post",
    "training",
    "fine-tuning",
    "finetuning",
    "collaboration",
}
GENERIC_RESEARCH_TERMS = {
    "ai",
    "artificial",
    "intelligence",
    "large",
    "language",
    "languages",
    "model",
    "models",
    "llm",
    "llms",
    "learning",
    "machine",
    "deep",
    "neural",
    "study",
    "paper",
    "research",
    "analysis",
    "method",
    "methods",
}


def normalized_title(title: str) -> str:
    """Normalize a title for duplicate detection."""
    return " ".join(re.findall(r"[a-z0-9]+", (title or "").lower()))


def normalize_topic_query(query: str) -> str:
    """Normalize broad user topics into a cleaner retrieval string."""
    lower = " ".join(str(query or "").strip().lower().split())
    if not lower:
        return ""
    parts: list[str] = []
    consumed: set[str] = set()
    for canonical, variants in PHRASE_ALIASES:
        if any(variant in lower for variant in variants):
            parts.append(canonical)
            consumed.update(re.findall(r"[a-z0-9]+", canonical))
    for token in re.findall(r"[a-z0-9]+", lower):
        token = TOKEN_ALIASES.get(token, token)
        if len(token) <= 1 or token in STOPWORDS or token in consumed:
            continue
        parts.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part not in seen:
            seen.add(part)
            deduped.append(part)
    return " ".join(deduped)


def decompose_topic_query(query: str) -> dict[str, Any]:
    """Split a natural-language topic into retrieval-friendly pieces."""
    original = " ".join(str(query or "").strip().split())
    lower = original.lower()
    if not original:
        return {
            "original": "",
            "normalized_topic": "",
            "core_terms": [],
            "modifiers": [],
            "retrieval_variants": [],
            "needs_llm_fallback": False,
        }

    cleaned = lower
    for phrase in QUERY_FILLER_PHRASES:
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned)
    cleaned = re.sub(r"[,:;/()+]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    normalized = normalize_topic_query(cleaned or original)
    normalized_tokens = normalized.split()
    core_terms: list[str] = []
    modifiers: list[str] = []
    for term in normalized_tokens:
        if term in GENERIC_RESEARCH_TERMS or term in STOPWORDS:
            continue
        if term in {"alignment", "safety", "finance", "forecasting", "retrieval", "medical"}:
            core_terms.append(term)
        elif term in {"time", "series", "temporal", "reward", "policy", "memory", "agents", "agent"}:
            modifiers.append(term)
        else:
            core_terms.append(term)

    retrieval_variants: list[str] = []
    if normalized:
        retrieval_variants.append(normalized)
    if core_terms:
        retrieval_variants.append(" ".join(dict.fromkeys(core_terms)))
    if core_terms and modifiers:
        retrieval_variants.append(" ".join(dict.fromkeys(core_terms[:4] + modifiers[:3])))

    unique_variants: list[str] = []
    seen: set[str] = set()
    for variant in retrieval_variants:
        compact = " ".join(variant.split())
        if compact and compact not in seen:
            seen.add(compact)
            unique_variants.append(compact)

    raw_tokens = re.findall(r"[a-z0-9+.-]+", lower)
    meaningful_raw = [token for token in raw_tokens if len(token) > 2 and token not in STOPWORDS]
    needs_llm_fallback = (
        len(meaningful_raw) >= 6
        and (
            len(set(core_terms)) <= 2
            or sum(1 for token in meaningful_raw if token in COMPLEX_QUERY_HINTS) >= 2
            or "?" in original
        )
    )
    return {
        "original": original,
        "normalized_topic": normalized or original,
        "core_terms": list(dict.fromkeys(core_terms))[:8],
        "modifiers": list(dict.fromkeys(modifiers))[:8],
        "retrieval_variants": unique_variants[:3],
        "needs_llm_fallback": needs_llm_fallback,
    }


def dedupe_papers(papers: list[Paper]) -> list[Paper]:
    """Dedupe by paper id, DOI, then normalized title while keeping stronger metadata."""
    by_key: dict[tuple[str, str], Paper] = {}
    alias_to_key: dict[tuple[str, str], tuple[str, str]] = {}
    order: list[tuple[str, str]] = []
    for paper in papers:
        aliases = _dedupe_aliases(paper)
        key = next((alias_to_key[alias] for alias in aliases if alias in alias_to_key), None)
        if key is None:
            key = aliases[0] if aliases else ("object", str(id(paper)))
            order.append(key)
        for alias in aliases:
            alias_to_key[alias] = key
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = paper
            continue
        if _paper_quality_tuple(paper) > _paper_quality_tuple(existing):
            by_key[key] = paper
    return [by_key[key] for key in order]


def score_paper(
    paper: Paper,
    query: str,
    strategy: str = "balanced",
    current_year: int | None = None,
    citation_weight: float | None = None,
) -> dict[str, float | str]:
    """Compute retrieval features and a strategy-specific selection score."""
    current_year = current_year or datetime.now().year
    query = normalize_topic_query(query)
    age = _paper_age(paper, current_year)
    age_normalized_impact = math.log1p(max(0, paper.cited_by_count)) / math.sqrt(age)
    impact_score = min(1.0, age_normalized_impact / 6.0)
    recency_score = 1.0 / math.sqrt(age)
    relevance_score = title_relevance_score(paper, query)
    signal_score = _topic_signal_score(paper)

    if strategy == "high-impact":
        impact_w, relevance_w, recency_w, signal_w, channel_w = _score_weights(
            strategy,
            citation_weight,
        )
        selection_score = (
            impact_w * impact_score
            + relevance_w * relevance_score
            + recency_w * recency_score
            + signal_w * signal_score
            + channel_w * _channel_score(paper)
        )
    elif strategy == "recent":
        selection_score = 0.70 * recency_score + 0.20 * relevance_score + 0.10 * signal_score
    else:
        impact_w, relevance_w, recency_w, signal_w, channel_w = _score_weights(
            strategy,
            citation_weight,
        )
        selection_score = (
            relevance_w * relevance_score
            + impact_w * impact_score
            + recency_w * recency_score
            + signal_w * signal_score
            + channel_w * _channel_score(paper)
        )

    return {
        "age_normalized_impact": age_normalized_impact,
        "impact_score": impact_score,
        "recency_score": recency_score,
        "title_relevance_score": relevance_score,
        "topic_signal_score": signal_score,
        "selection_score": round(selection_score, 4),
        "selection_reason": _selection_reason(
            paper,
            strategy,
            relevance_score=relevance_score,
            impact_score=impact_score,
            recency_score=recency_score,
            signal_score=signal_score,
        ),
    }


def title_relevance_score(paper: Paper, query: str) -> float:
    query_tokens = _tokenize(normalize_topic_query(query))
    if not query_tokens:
        return 0.0
    paper_tokens = _tokenize(f"{paper.title} {paper.abstract[:800]}")
    if not paper_tokens:
        return 0.0
    overlap = len(query_tokens & paper_tokens)
    return min(1.0, overlap / max(1, len(query_tokens)))


def select_representative_papers(
    papers: list[Paper],
    query: str,
    limit: int,
    strategy: str = "balanced",
    current_year: int | None = None,
    citation_weight: float | None = None,
    min_relevance: float | None = None,
    require_core_match: bool = True,
) -> list[Paper]:
    """Select a representative, annotated paper subset from a candidate pool."""
    if limit <= 0:
        return []
    strategy = strategy if strategy in {"balanced", "high-impact", "recent"} else "balanced"
    current_year = current_year or datetime.now().year
    query = normalize_topic_query(query)
    candidates = dedupe_papers(papers)
    scored: list[tuple[Paper, dict[str, float | str]]] = [
        (
            paper,
            score_paper(
                paper,
                query,
                strategy=strategy,
                current_year=current_year,
                citation_weight=citation_weight,
            ),
        )
        for paper in candidates
    ]
    scored = _filter_relevant_candidates(
        scored,
        query=query,
        strategy=strategy,
        min_relevance=min_relevance,
        require_core_match=require_core_match,
    )
    scored = _second_stage_topic_cleanup(scored, query=query, strategy=strategy)
    if strategy in {"high-impact", "recent"}:
        return [
            _annotate_paper(paper, metrics, strategy)
            for paper, metrics in sorted(
                scored,
                key=lambda item: (
                    float(item[1]["selection_score"]),
                    int(item[0].cited_by_count),
                    int(item[0].year or 0),
                ),
                reverse=True,
            )[:limit]
        ]

    selected: list[tuple[Paper, dict[str, float | str]]] = []
    remaining = scored[:]
    while remaining and len(selected) < limit:
        best_idx = 0
        best_score = -1.0
        for idx, (paper, metrics) in enumerate(remaining):
            diversity = _diversity_score(paper, [p for p, _ in selected])
            mmr_score = 0.82 * float(metrics["selection_score"]) + 0.18 * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        selected.append(remaining.pop(best_idx))
    return [_annotate_paper(paper, metrics, strategy) for paper, metrics in selected]


def _dedupe_aliases(paper: Paper) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    if paper.paper_id:
        aliases.append(("paper_id", paper.paper_id.lower()))
    if paper.doi:
        doi = paper.doi.lower().removeprefix("https://doi.org/").removeprefix("doi:")
        aliases.append(("doi", doi))
    title = normalized_title(paper.title)
    if title:
        aliases.append(("title", title))
    return aliases


def _paper_quality_tuple(paper: Paper) -> tuple[int, int, int]:
    return (int(paper.cited_by_count or 0), int(paper.year or 0), len(paper.abstract or ""))


def _paper_age(paper: Paper, current_year: int) -> int:
    year = int(paper.year or current_year)
    return max(1, current_year - year + 1)


def _filter_relevant_candidates(
    scored: list[tuple[Paper, dict[str, float | str]]],
    *,
    query: str,
    strategy: str,
    min_relevance: float | None,
    require_core_match: bool,
) -> list[tuple[Paper, dict[str, float | str]]]:
    if not scored:
        return []
    threshold = min_relevance
    if threshold is None:
        threshold = 0.30 if strategy != "recent" else 0.20
    query_core = _core_query_tokens(query)
    filtered = [
        (paper, metrics)
        for paper, metrics in scored
        if float(metrics["title_relevance_score"]) >= threshold
        and (not require_core_match or _has_core_overlap(paper, query_core))
    ]
    if len(filtered) >= 1:
        return filtered
    # If the query is extremely narrow or abstracts are unavailable, fall back to
    # the best relevance-ranked candidates rather than returning an empty pool.
    return sorted(scored, key=lambda item: float(item[1]["title_relevance_score"]), reverse=True)[: max(1, min(5, len(scored)))]


def _core_query_tokens(query: str) -> set[str]:
    return {token for token in _tokenize(normalize_topic_query(query)) if token not in GENERIC_RESEARCH_TERMS}


def _has_core_overlap(paper: Paper, query_core: set[str]) -> bool:
    if not query_core:
        return True
    text_tokens = _tokenize(f"{paper.title} {paper.abstract[:1200]}")
    overlap = query_core & text_tokens
    if len(query_core) <= 2:
        return bool(overlap)
    return len(overlap) >= 2


def _second_stage_topic_cleanup(
    scored: list[tuple[Paper, dict[str, float | str]]],
    *,
    query: str,
    strategy: str,
) -> list[tuple[Paper, dict[str, float | str]]]:
    if len(scored) <= 2:
        return scored
    query_core = _core_query_tokens(query)
    if len(query_core) < 2:
        return scored

    cleaned: list[tuple[Paper, dict[str, float | str]]] = []
    for paper, metrics in scored:
        title_tokens = _tokenize(paper.title)
        abstract_tokens = _tokenize((paper.abstract or "")[:1200])
        title_overlap = len(query_core & title_tokens)
        abstract_overlap = len(query_core & abstract_tokens)
        strict_match = 0.72 * (title_overlap / max(1, len(query_core))) + 0.28 * (abstract_overlap / max(1, len(query_core)))
        if len(query_core) <= 3:
            passes = title_overlap >= 1 and strict_match >= 0.20
        else:
            passes = title_overlap >= 2 or strict_match >= (0.22 if strategy == "recent" else 0.26)
        if passes:
            cleaned.append((paper, metrics))

    # Keep the stricter cleanup only when it still leaves a usable pool.
    if len(cleaned) >= max(2, min(5, len(scored) // 2)):
        return cleaned
    return scored


def _score_weights(strategy: str, citation_weight: float | None) -> tuple[float, float, float, float, float]:
    """Return impact, relevance, recency, signal, channel weights."""
    if strategy == "high-impact":
        default_impact = 0.75
        other_defaults = (0.15, 0.10, 0.0, 0.0)
    else:
        default_impact = 0.24
        other_defaults = (0.38, 0.20, 0.12, 0.06)
    impact = default_impact if citation_weight is None else _clamp(citation_weight, 0.0, 0.85)
    remaining = max(0.0, 1.0 - impact)
    other_total = sum(other_defaults)
    if other_total <= 0:
        return impact, 0.0, 0.0, 0.0, 0.0
    relevance, recency, signal, channel = [remaining * (w / other_total) for w in other_defaults]
    return impact, relevance, recency, signal, channel


def _clamp(value: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, parsed))


def _topic_signal_score(paper: Paper) -> float:
    text = f"{paper.title} {paper.abstract}".lower()
    groups = [
        ("survey", "review", "overview"),
        ("benchmark", "evaluation", "dataset"),
        ("limitation", "challenge", "failure", "robustness", "bias", "risk"),
    ]
    hits = sum(1 for group in groups if any(term in text for term in group))
    return min(1.0, hits / 2.0)


def _channel_score(paper: Paper) -> float:
    channel = (paper.retrieval_channel or "").lower()
    if any(key in channel for key in ("survey", "critical", "impact", "recent")):
        return 1.0
    return 0.35 if channel else 0.0


def _selection_reason(
    paper: Paper,
    strategy: str,
    *,
    relevance_score: float,
    impact_score: float,
    recency_score: float,
    signal_score: float,
) -> str:
    reasons: list[str] = []
    if relevance_score >= 0.35:
        reasons.append("high title/abstract relevance")
    if impact_score >= 0.20:
        reasons.append("strong age-normalized citation impact")
    if recency_score >= 0.45:
        reasons.append("recent publication")
    if signal_score >= 0.50:
        reasons.append("survey/benchmark/critical topic signal")
    if paper.retrieval_channel:
        reasons.append(f"retrieved via {paper.retrieval_channel} channel")
    if not reasons:
        reasons.append(f"best available candidate under {strategy} strategy")
    return "; ".join(reasons)


def _annotate_paper(paper: Paper, metrics: dict[str, float | str], strategy: str) -> Paper:
    reason = str(metrics["selection_reason"])
    if paper.paper_id.startswith("arxiv:"):
        reason = f"{reason}; arXiv has no citation metadata; selected by relevance/recency/topic signal"
    return paper.model_copy(
        update={
            "selection_score": float(metrics["selection_score"]),
            "selection_reason": reason,
            "retrieval_channel": paper.retrieval_channel or strategy,
        }
    )


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def _diversity_score(paper: Paper, selected: list[Paper]) -> float:
    if not selected:
        return 1.0
    tokens = _tokenize(f"{paper.title} {paper.abstract[:400]}")
    if not tokens:
        return 0.0
    max_sim = 0.0
    for other in selected:
        other_tokens = _tokenize(f"{other.title} {other.abstract[:400]}")
        if not other_tokens:
            continue
        max_sim = max(max_sim, len(tokens & other_tokens) / max(1, len(tokens | other_tokens)))
    return 1.0 - max_sim
