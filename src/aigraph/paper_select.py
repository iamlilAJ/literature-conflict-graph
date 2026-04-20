"""Representative paper selection for retrieval candidate pools."""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from typing import Any, Literal

from .llm_client import build_openai_client, call_llm_text, configured_api_key, configured_model
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
PAPER_ROLES: tuple[str, ...] = ("survey", "benchmark", "dataset", "method", "industry", "failure", "other")
ROLE_LABELS = {
    "survey": "Survey paper",
    "benchmark": "Benchmark paper",
    "dataset": "Dataset/resource paper",
    "method": "Method paper",
    "industry": "Industry/system paper",
    "failure": "Failure-analysis paper",
    "other": "Other paper",
}
ROLE_EXPLANATIONS = {
    "survey": "Maps the field, taxonomy, or prior landscape.",
    "benchmark": "Defines an evaluation protocol, benchmark, or systematic comparison.",
    "dataset": "Introduces a dataset, corpus, benchmark resource, or data card.",
    "method": "Primarily proposes or improves a model, method, or framework.",
    "industry": "Focuses on deployment, systems, serving, or real-world operation.",
    "failure": "Revisits claims, highlights limitations, or analyzes failure modes.",
    "other": "Included as supporting context when no clear role signal wins.",
}
ROLE_KEYWORDS: dict[str, tuple[tuple[str, float], ...]] = {
    "survey": (
        ("survey", 3.4),
        ("review", 3.0),
        ("overview", 2.8),
        ("taxonomy", 3.0),
        ("tutorial", 2.6),
        ("systematic review", 3.4),
    ),
    "benchmark": (
        ("benchmark", 3.4),
        ("benchmarking", 3.2),
        ("evaluation", 2.8),
        ("leaderboard", 2.8),
        ("challenge", 2.4),
        ("empirical study", 2.2),
    ),
    "dataset": (
        ("dataset", 3.2),
        ("corpus", 2.8),
        ("resource", 2.4),
        ("data card", 2.8),
        ("benchmark dataset", 3.2),
    ),
    "failure": (
        ("revisiting", 2.8),
        ("revisit", 2.4),
        ("rethinking", 2.8),
        ("re-evaluating", 2.8),
        ("reassessing", 2.8),
        ("understanding", 2.0),
        ("demystifying", 2.6),
        ("limitations", 2.8),
        ("limitation", 2.6),
        ("fails", 2.6),
        ("failure", 2.4),
        ("robustness", 2.4),
        ("bias", 2.2),
        ("risk", 2.0),
    ),
    "industry": (
        ("system", 2.0),
        ("deployment", 3.0),
        ("production", 3.0),
        ("practical", 2.4),
        ("serving", 2.6),
        ("real-world", 2.8),
    ),
    "method": (
        ("framework", 1.8),
        ("method", 1.6),
        ("model", 1.4),
        ("architecture", 1.8),
        ("approach", 1.6),
        ("improving", 1.6),
        ("scalable", 1.4),
        ("efficient", 1.4),
    ),
}
REVISION_CUES = ("revisiting", "revisit", "rethinking", "re-evaluating", "reassessing", "understanding", "demystifying")
DATA_CENTRIC_TERMS = {"dataset", "datasets", "corpus", "resource", "data", "benchmark"}
INDUSTRY_TERMS = {"production", "deployment", "serving", "system", "systems", "practical"}


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
    if len(meaningful_raw := [token for token in re.findall(r"[a-z0-9+.-]+", lower) if len(token) > 2 and token not in STOPWORDS]) >= 4:
        base = " ".join(dict.fromkeys(core_terms[:4] + modifiers[:3])) or normalized or original
        retrieval_variants.extend(
            [
                f"{base} survey review".strip(),
                f"{base} benchmark evaluation".strip(),
                f"{base} limitation failure robustness".strip(),
                f"{base} dataset corpus resource".strip(),
            ]
        )

    unique_variants: list[str] = []
    seen: set[str] = set()
    for variant in retrieval_variants:
        compact = " ".join(variant.split())
        if compact and compact not in seen:
            seen.add(compact)
            unique_variants.append(compact)

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
        "retrieval_variants": unique_variants[:6],
        "needs_llm_fallback": needs_llm_fallback,
    }


def paper_role_label(role: str | None) -> str:
    return ROLE_LABELS.get(str(role or "other"), ROLE_LABELS["other"])


def paper_role_explanation(role: str | None) -> str:
    return ROLE_EXPLANATIONS.get(str(role or "other"), ROLE_EXPLANATIONS["other"])


def infer_paper_role(
    title: str,
    abstract: str,
    venue: str | None = None,
    retrieval_channel: str | None = None,
    *,
    allow_llm_fallback: bool = False,
) -> dict[str, Any]:
    deterministic = _infer_paper_role_deterministic(title, abstract, venue=venue, retrieval_channel=retrieval_channel)
    if not allow_llm_fallback or not _paper_role_is_ambiguous(deterministic) or not configured_api_key():
        return deterministic
    try:
        client = build_openai_client()
        raw = call_llm_text(
            client,
            model=configured_model(),
            system=(
                "Classify an academic paper into exactly one role from "
                "[survey, benchmark, dataset, method, industry, failure, other]. "
                "Return strict JSON with keys paper_role, paper_role_score, paper_role_rationale."
            ),
            user=json.dumps(
                {
                    "title": title,
                    "abstract": abstract[:2500],
                    "venue": venue or "",
                    "retrieval_channel": retrieval_channel or "",
                    "deterministic_guess": deterministic,
                },
                ensure_ascii=False,
            ),
            temperature=0.0,
            max_tokens=220,
        )
        parsed = json.loads(raw)
    except Exception:
        return deterministic
    role = str(parsed.get("paper_role") or deterministic["role"]).strip().lower()
    if role not in PAPER_ROLES:
        return deterministic
    score = _clamp(float(parsed.get("paper_role_score") or deterministic["score"]), 0.0, 1.0)
    rationale = str(parsed.get("paper_role_rationale") or "").strip()
    signals = list(deterministic.get("signals") or [])
    if rationale:
        signals.append(f"llm:{rationale}")
    return {
        "role": role,
        "score": max(score, float(deterministic["score"])),
        "signals": signals[:6],
        "scores": deterministic.get("scores", {}),
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
    allow_role_llm_fallback: bool | None = None,
) -> list[Paper]:
    """Select a representative, annotated paper subset from a candidate pool."""
    if limit <= 0:
        return []
    strategy = strategy if strategy in {"balanced", "high-impact", "recent"} else "balanced"
    current_year = current_year or datetime.now().year
    query = normalize_topic_query(query)
    if allow_role_llm_fallback is None:
        allow_role_llm_fallback = os.environ.get("AIGRAPH_ROLE_LLM_FALLBACK", "0") == "1"
    candidates = [
        annotate_paper_role(paper, allow_llm_fallback=allow_role_llm_fallback)
        for paper in dedupe_papers(papers)
    ]
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

    selected, remaining = _role_seed_selection(scored, query=query, limit=limit)
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


def annotate_paper_role(paper: Paper, *, allow_llm_fallback: bool = False) -> Paper:
    if paper.paper_role and paper.paper_role in PAPER_ROLES and paper.paper_role_score > 0:
        return paper
    info = infer_paper_role(
        paper.title,
        paper.abstract,
        venue=paper.venue,
        retrieval_channel=paper.retrieval_channel,
        allow_llm_fallback=allow_llm_fallback,
    )
    return paper.model_copy(
        update={
            "paper_role": info["role"],
            "paper_role_score": float(info["score"]),
            "paper_role_signals": list(info.get("signals") or []),
        }
    )


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
        if float(metrics["title_relevance_score"]) >= _role_adjusted_threshold(paper, threshold)
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
    if (paper.paper_role or "") in {"survey", "benchmark", "dataset", "method", "failure", "industry"} and overlap:
        return True
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


def _role_adjusted_threshold(paper: Paper, threshold: float) -> float:
    if (paper.paper_role or "") in {"survey", "benchmark", "dataset", "failure", "industry", "method"}:
        return max(0.18, threshold - 0.10)
    return threshold


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
    if paper.paper_role in {"survey", "benchmark", "dataset", "failure", "industry"}:
        hits += 1
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
    if paper.paper_role and paper.paper_role != "other":
        reasons.append(f"{paper_role_label(paper.paper_role).lower()} anchor")
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
            "paper_role": paper.paper_role,
            "paper_role_score": paper.paper_role_score,
            "paper_role_signals": list(paper.paper_role_signals or []),
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


def _infer_paper_role_deterministic(
    title: str,
    abstract: str,
    *,
    venue: str | None = None,
    retrieval_channel: str | None = None,
) -> dict[str, Any]:
    title_lower = " ".join(str(title or "").lower().split())
    abstract_lower = " ".join(str(abstract or "").lower().split())
    venue_lower = str(venue or "").lower()
    scores = {role: 0.0 for role in PAPER_ROLES}
    signals: list[str] = []
    for role, phrases in ROLE_KEYWORDS.items():
        for phrase, weight in phrases:
            if phrase in title_lower:
                scores[role] += weight
                signals.append(f"title:{phrase}")
            elif phrase in abstract_lower:
                scores[role] += weight * 0.55
                signals.append(f"abstract:{phrase}")
    if retrieval_channel:
        channel = retrieval_channel.lower()
        if "survey" in channel:
            scores["survey"] += 1.8
            signals.append("channel:survey")
        if any(key in channel for key in ("critical", "failure")):
            scores["failure"] += 1.6
            signals.append("channel:critical")
        if "impact" in channel:
            scores["benchmark"] += 0.25
        if "recent" in channel:
            scores["failure"] += 0.2
    if "system" in venue_lower:
        scores["industry"] += 0.6
        signals.append("venue:system")

    has_revision = any(term in title_lower or term in abstract_lower for term in REVISION_CUES)
    has_benchmark = any(term in title_lower or term in abstract_lower for term in ("benchmark", "benchmarking", "evaluation", "leaderboard", "challenge"))
    has_dataset = any(term in title_lower or term in abstract_lower for term in ("dataset", "datasets", "corpus", "resource", "data card"))
    if has_revision and has_benchmark:
        scores["benchmark"] += 2.2
        scores["failure"] += 0.6
        signals.append("composite:revision+benchmark")
    elif has_revision and has_dataset:
        scores["dataset"] += 1.6
        scores["failure"] += 0.6
        signals.append("composite:revision+dataset")
    elif has_revision:
        scores["failure"] += 1.8
        signals.append("composite:revision")

    if any(term in title_lower or term in abstract_lower for term in INDUSTRY_TERMS):
        scores["industry"] += 0.8
        signals.append("composite:industry")
    if any(term in title_lower or term in abstract_lower for term in DATA_CENTRIC_TERMS):
        scores["dataset"] += 0.2

    top_role, top_score = max(scores.items(), key=lambda item: item[1])
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if top_score < 1.15:
        method_boost = _method_fallback_score(title_lower, abstract_lower)
        if method_boost >= 1.2:
            scores["method"] += method_boost
            top_role, top_score = "method", scores["method"]
            signals.append("fallback:method")
        else:
            top_role = "other"
            top_score = 0.2
    elif top_role == "other":
        method_boost = _method_fallback_score(title_lower, abstract_lower)
        if method_boost >= 1.2:
            scores["method"] += method_boost
            top_role, top_score = "method", scores["method"]
            signals.append("fallback:method")
    score = min(1.0, top_score / 5.0)
    if top_role == "other":
        score = min(score, 0.35)
    return {
        "role": top_role,
        "score": round(score, 3),
        "signals": list(dict.fromkeys(signals))[:6],
        "scores": {role: round(value, 3) for role, value in scores.items() if value > 0},
        "margin": round(max(0.0, top_score - second_score), 3),
    }


def _method_fallback_score(title_lower: str, abstract_lower: str) -> float:
    score = 0.0
    method_like = ("framework", "architecture", "approach", "model", "method", "improving", "efficient", "scalable", "forecasting", "prediction")
    for phrase in method_like:
        if phrase in title_lower:
            score += 0.7
        elif phrase in abstract_lower:
            score += 0.25
    return score


def _paper_role_is_ambiguous(info: dict[str, Any]) -> bool:
    role = str(info.get("role") or "other")
    margin = float(info.get("margin") or 0.0)
    score = float(info.get("score") or 0.0)
    return role == "other" or score < 0.45 or margin < 0.7


def _role_seed_selection(
    scored: list[tuple[Paper, dict[str, float | str]]],
    *,
    query: str,
    limit: int,
) -> tuple[list[tuple[Paper, dict[str, float | str]]], list[tuple[Paper, dict[str, float | str]]]]:
    if not scored or limit <= 0:
        return [], []
    roleful = [item for item in scored if (item[0].paper_role or "other") != "other"]
    if limit <= 1 or len(scored) <= 2 or len({item[0].paper_role for item in roleful}) <= 1:
        return [], list(scored)
    selected: list[tuple[Paper, dict[str, float | str]]] = []
    remaining = list(scored)

    def pick_best(role_group: set[str]) -> None:
        nonlocal remaining
        matches = [item for item in remaining if (item[0].paper_role or "other") in role_group]
        if not matches:
            return
        best = max(
            matches,
            key=lambda item: (
                float(item[1]["selection_score"]),
                float(item[0].paper_role_score or 0.0),
                int(item[0].cited_by_count or 0),
                int(item[0].year or 0),
            ),
        )
        selected.append(best)
        remaining = [item for item in remaining if item[0].paper_id != best[0].paper_id]

    pick_best({"survey", "benchmark"})
    if len(selected) < limit:
        pick_best({"failure"})
    if len(selected) < limit and _query_prefers_dataset(query):
        pick_best({"dataset"})
    if len(selected) < limit and _query_prefers_industry(query):
        pick_best({"industry"})
    if len(selected) < limit:
        pick_best({"method"})
    if len(selected) < limit and limit >= 8:
        pick_best({"survey", "benchmark"})
    if len(selected) < limit and limit >= 6:
        pick_best({"failure"})
    return selected, remaining


def _query_prefers_dataset(query: str) -> bool:
    tokens = _tokenize(query)
    return bool(tokens & {"dataset", "datasets", "corpus", "resource", "resources", "benchmark", "evaluation", "medical", "scientific"})


def _query_prefers_industry(query: str) -> bool:
    tokens = _tokenize(query)
    return bool(tokens & {"deployment", "production", "system", "systems", "serving", "practical", "real", "industry"})
