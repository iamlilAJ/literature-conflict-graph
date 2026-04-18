"""Community-level insights from graph topology and citation gaps."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any, Optional

import networkx as nx
from pydantic import ValidationError

from .graph import build_citation_graph
from .llm_client import (
    DEFAULT_MAX_TOKENS,
    build_openai_client,
    call_llm_text,
    configured_api_key,
    configured_base_url,
    configured_model,
)
from .llm_extract import _load_json
from .models import Anomaly, Claim, Insight, Paper


logger = logging.getLogger(__name__)

SEMANTIC_FIELDS: tuple[str, ...] = (
    "mechanism",
    "failure_mode",
    "evaluation_protocol",
    "assumption",
    "risk_type",
    "temporal_property",
    "data_modality",
)
TRANSFER_FIELDS: tuple[str, ...] = ("evaluation_protocol", "failure_mode", "temporal_property")
GENERIC_COMMUNITIES = {
    "other",
    "research",
    "literature",
    "scientific research",
    "scientific literature",
    "community",
    "communities",
}
GENERIC_SHARED_CONCEPTS = {
    "forecasting",
    "evaluation protocol",
    "temporal reasoning",
    "shared concepts",
}


class InsightGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        g: nx.MultiDiGraph,
        claims: list[Claim],
        papers: list[Paper],
        anomalies: list[Anomaly],
    ) -> list[Insight]:
        """Return graph-grounded community insights."""


class TemplateInsightGenerator(InsightGenerator):
    """Deterministic topology/citation insight generator."""

    def __init__(self, min_claims_per_community: int = 2, max_insights: int = 20):
        self.min_claims_per_community = min_claims_per_community
        self.max_insights = max_insights

    def generate(
        self,
        g: nx.MultiDiGraph,
        claims: list[Claim],
        papers: list[Paper],
        anomalies: list[Anomaly],
    ) -> list[Insight]:
        communities = _community_groups(claims, min_size=self.min_claims_per_community)
        concepts_by_community = {name: _concepts_by_field(group) for name, group in communities.items()}
        flat_concepts_by_community = {name: _flatten_concepts(concepts) for name, concepts in concepts_by_community.items()}
        transfer_concepts_by_community = {
            name: _flatten_selected(concepts, TRANSFER_FIELDS)
            for name, concepts in concepts_by_community.items()
        }
        citation_graph = build_citation_graph(g)
        insights: list[Insight] = []
        for (name_a, claims_a), (name_b, claims_b) in combinations(communities.items(), 2):
            concepts_a = concepts_by_community[name_a]
            concepts_b = concepts_by_community[name_b]
            flat_a = flat_concepts_by_community[name_a]
            flat_b = flat_concepts_by_community[name_b]
            shared = flat_a & flat_b
            if len(shared) < 2:
                continue
            concept_jaccard = len(shared) / max(1, len(flat_a | flat_b))
            if concept_jaccard < 0.25 and not (len(shared) >= 4 and concept_jaccard >= 0.12):
                continue

            distance = _citation_distance_between(citation_graph, claims_a, claims_b)
            if distance is not None and distance <= 2:
                continue

            pair_claims = claims_a + claims_b
            evidence_claims = [c.claim_id for c in pair_claims]
            evidence_papers = sorted({c.paper_id for c in pair_claims})
            impact = _impact_score(g, evidence_papers)
            topology = min(1.0, concept_jaccard + (0.25 if distance is None else 0.1))
            confidence = min(1.0, 0.45 * concept_jaccard + 0.25 * topology + 0.30 * min(1.0, len(shared) / 4))

            insight_id = f"i{len(insights) + 1:03d}"
            insights.append(
                Insight(
                    insight_id=insight_id,
                    type="unifying_theory",
                    title=f"{_title(name_a)} and {_title(name_b)} may share a unifying mechanism",
                    insight=(
                        f"The {name_a} and {name_b} communities share concepts such as "
                        f"{_join(sorted(shared), max_items=4)}, but their citation connection is weak. "
                        "This suggests they may be studying different instances of the same underlying problem."
                    ),
                    communities=[name_a, name_b],
                    shared_concepts=sorted(shared),
                    evidence_claims=evidence_claims,
                    evidence_papers=evidence_papers,
                    citation_gap=_citation_gap_text(distance),
                    unifying_frame=_unifying_frame(name_a, name_b, shared),
                    transfer_suggestions=_transfer_suggestions(name_a, name_b, concepts_a, concepts_b),
                    impact_score=round(impact, 4),
                    topology_score=round(topology, 4),
                    confidence_score=round(confidence, 4),
                )
            )

            transfer_shared = transfer_concepts_by_community[name_a] & transfer_concepts_by_community[name_b]
            if transfer_shared and _evaluation_asymmetry(concepts_a, concepts_b):
                insight_id = f"i{len(insights) + 1:03d}"
                insights.append(
                    Insight(
                        insight_id=insight_id,
                        type="transfer_opportunity",
                        title=f"Evaluation practice may transfer between {_title(name_a)} and {_title(name_b)}",
                        insight=(
                            f"{name_a} and {name_b} share evaluation or failure concepts "
                            f"({_join(sorted(transfer_shared), max_items=3)}), but one side has a clearer "
                            "evaluation protocol. This marks a concrete method-transfer opportunity."
                        ),
                        communities=[name_a, name_b],
                        shared_concepts=sorted(transfer_shared),
                        evidence_claims=evidence_claims,
                        evidence_papers=evidence_papers,
                        citation_gap=_citation_gap_text(distance),
                        unifying_frame=_unifying_frame(name_a, name_b, transfer_shared),
                        transfer_suggestions=_transfer_suggestions(name_a, name_b, concepts_a, concepts_b),
                        impact_score=round(impact, 4),
                        topology_score=round(topology, 4),
                        confidence_score=round(min(1.0, confidence + 0.05), 4),
                    )
                )

            if len(insights) >= self.max_insights:
                break
        return insights[: self.max_insights]


class LLMInsightGenerator(InsightGenerator):
    """Rewrite deterministic insight patterns with an OpenAI-compatible LLM."""

    def __init__(
        self,
        model: Optional[str] = None,
        client: Any | None = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fallback: InsightGenerator | None = None,
    ):
        self.model = configured_model(model)
        self._client = client
        self._api_key = configured_api_key(api_key)
        self._base_url = configured_base_url(base_url)
        self.fallback = fallback or TemplateInsightGenerator()

    def generate(
        self,
        g: nx.MultiDiGraph,
        claims: list[Claim],
        papers: list[Paper],
        anomalies: list[Anomaly],
    ) -> list[Insight]:
        base = self.fallback.generate(g, claims, papers, anomalies)
        out: list[Insight] = []
        for insight in base:
            try:
                rewritten = self._rewrite(insight, claims, papers)
                out.append(rewritten or insight)
            except Exception as e:  # pragma: no cover - network errors
                logger.warning("LLM insight rewrite failed for %s: %s", insight.insight_id, e)
                out.append(insight)
        return out

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = build_openai_client(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _rewrite(self, insight: Insight, claims: list[Claim], papers: list[Paper]) -> Insight | None:
        client = self._get_client()
        prompt = _insight_prompt(insight, claims, papers)
        raw = call_llm_text(
            client,
            model=self.model,
            system=(
                "You convert graph topology patterns into concise scientific community insights. "
                "Use only the provided fields. Do not invent paper facts. Return strict JSON."
            ),
            user=prompt,
            temperature=0.0,
            max_tokens=int(os.environ.get("AIGRAPH_INSIGHT_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        )
        payload = _load_json(raw)
        if not isinstance(payload, dict):
            return None
        merged = insight.model_dump()
        for key in ("title", "insight", "unifying_frame", "transfer_suggestions", "citation_gap"):
            if key in payload:
                merged[key] = payload[key]
        try:
            return Insight.model_validate(merged)
        except ValidationError:
            return None


def generate_insights(
    g: nx.MultiDiGraph,
    claims: list[Claim],
    papers: list[Paper],
    anomalies: list[Anomaly],
    generator: InsightGenerator | None = None,
) -> list[Insight]:
    generator = generator or TemplateInsightGenerator()
    return generator.generate(g, claims, papers, anomalies)


def prune_insights(insights: list[Insight], claims: list[Claim], max_keep: int = 10) -> list[Insight]:
    claims_by_id = {claim.claim_id: claim for claim in claims}
    cleaned: list[Insight] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for insight in insights:
        candidate = _clean_insight(insight, claims_by_id)
        quality = _insight_quality(candidate, claims_by_id)
        if quality < 0.38:
            continue
        candidate = candidate.model_copy(update={"quality_score": round(quality, 4)})
        key = (candidate.type, tuple(candidate.communities[:2]), candidate.title.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(candidate)
    cleaned.sort(
        key=lambda insight: (
            float(insight.quality_score or 0.0),
            float(insight.confidence_score or 0.0),
            float(insight.topology_score or 0.0),
            len(insight.shared_concepts),
        ),
        reverse=True,
    )
    return cleaned[:max_keep]


def _community_groups(claims: list[Claim], min_size: int) -> dict[str, list[Claim]]:
    groups: dict[str, list[Claim]] = {}
    for claim in claims:
        key = _clean(claim.domain) or _clean(claim.canonical_task) or _clean(claim.task)
        if not key:
            continue
        groups.setdefault(key, []).append(claim)
    return {key: value for key, value in groups.items() if len(value) >= min_size}


def _concepts_by_field(claims: list[Claim]) -> dict[str, set[str]]:
    out = {field: set() for field in SEMANTIC_FIELDS}
    for claim in claims:
        for field in SEMANTIC_FIELDS:
            value = _clean(getattr(claim, field))
            if value:
                out[field].add(value)
        for concept in _inferred_concepts(claim):
            out.setdefault("mechanism", set()).add(concept)
    return out


def _flatten_concepts(concepts: dict[str, set[str]]) -> set[str]:
    out: set[str] = set()
    for values in concepts.values():
        out.update(values)
    return out


def _flatten_selected(concepts: dict[str, set[str]], fields: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for field in fields:
        out.update(concepts.get(field, set()))
    return out


def _evaluation_asymmetry(a: dict[str, set[str]], b: dict[str, set[str]]) -> bool:
    return bool(a.get("evaluation_protocol")) != bool(b.get("evaluation_protocol"))


def _citation_distance_between(citation_graph: nx.Graph, a: list[Claim], b: list[Claim]) -> int | None:
    sources = {f"Paper:{c.paper_id}" for c in a if f"Paper:{c.paper_id}" in citation_graph}
    targets = {f"Paper:{c.paper_id}" for c in b if f"Paper:{c.paper_id}" in citation_graph}
    if not sources or not targets:
        return None
    if sources & targets:
        return 0
    return _shortest_distance_between_sets(citation_graph, sources, targets)


def _impact_score(g: nx.MultiDiGraph, paper_ids: list[str]) -> float:
    total = 0.0
    for paper_id in set(paper_ids):
        node = f"Paper:{paper_id}"
        if node in g:
            total += float(g.nodes[node].get("age_normalized_impact") or g.nodes[node].get("impact_score") or 0.0)
    return min(1.0, total / 8.0)


def _citation_gap_text(distance: int | None) -> str:
    if distance is None:
        return "No internal citation path was found between these communities in the fetched corpus."
    return f"The shortest internal citation path between these communities is {distance} edges."


def _unifying_frame(a: str, b: str, shared: set[str]) -> str:
    return (
        f"Frame {a} and {b} as domain-specific instances of a shared problem around "
        f"{_join(sorted(shared), max_items=4)}."
    )


def _transfer_suggestions(
    a: str,
    b: str,
    concepts_a: dict[str, set[str]],
    concepts_b: dict[str, set[str]],
) -> list[str]:
    suggestions = [
        f"Compare {a} and {b} papers on the shared concepts rather than only by domain labels.",
    ]
    eval_a = concepts_a.get("evaluation_protocol", set())
    eval_b = concepts_b.get("evaluation_protocol", set())
    if eval_a and not eval_b:
        suggestions.append(f"Test whether {_join(sorted(eval_a), max_items=2)} from {a} improves evaluation in {b}.")
    elif eval_b and not eval_a:
        suggestions.append(f"Test whether {_join(sorted(eval_b), max_items=2)} from {b} improves evaluation in {a}.")
    else:
        suggestions.append("Run matched ablations on the shared mechanism/failure nodes across both communities.")
    return suggestions


def _insight_prompt(insight: Insight, claims: list[Claim], papers: list[Paper]) -> str:
    claims_by_id = {c.claim_id: c for c in claims}
    papers_by_id = {p.paper_id: p for p in papers}
    evidence = []
    for cid in insight.evidence_claims[:10]:
        claim = claims_by_id.get(cid)
        if not claim:
            continue
        paper = papers_by_id.get(claim.paper_id)
        evidence.append(
            {
                "claim_id": cid,
                "paper": paper.title if paper else claim.paper_id,
                "claim_text": claim.claim_text,
                "domain": claim.domain,
                "mechanism": claim.mechanism,
                "failure_mode": claim.failure_mode,
                "evaluation_protocol": claim.evaluation_protocol,
                "temporal_property": claim.temporal_property,
            }
        )
    payload = {
        "pattern": insight.model_dump(),
        "evidence": evidence,
        "required_output": {
            "title": "short string",
            "insight": "2-3 sentence graph-grounded community insight",
            "unifying_frame": "one concise unifying theoretical frame",
            "citation_gap": "short description of the citation/community disconnect",
            "transfer_suggestions": ["concrete cross-community comparison or evaluation transfer"],
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    clean = value.strip().lower()
    return clean or None


def _clean_insight(insight: Insight, claims_by_id: dict[str, Claim]) -> Insight:
    evidence_claims = [claims_by_id[cid] for cid in insight.evidence_claims if cid in claims_by_id]
    communities = [_refine_community_label(name, evidence_claims) for name in insight.communities]
    communities = [community for community in communities if community]
    shared_concepts = _clean_shared_concepts(insight.shared_concepts)
    title = _clean_insight_title(insight, communities, shared_concepts)
    summary = _clean_insight_summary(insight, communities, shared_concepts)
    return insight.model_copy(
        update={
            "communities": communities,
            "shared_concepts": shared_concepts,
            "title": title,
            "insight": summary,
            "unifying_frame": _clean_sentence(insight.unifying_frame),
            "citation_gap": _clean_sentence(insight.citation_gap),
        }
    )


def _refine_community_label(name: str, claims: list[Claim]) -> str:
    raw = _clean(name) or ""
    if raw and raw not in GENERIC_COMMUNITIES:
        return raw
    if not claims:
        return ""
    domain = _mode_specific(claims, ("domain",))
    task = _mode_specific(claims, ("task", "canonical_task"))
    method = _mode_specific(claims, ("method", "model"))
    mechanism = _mode_specific(claims, ("mechanism", "temporal_property", "failure_mode"))
    parts = [part for part in (domain, task, method or mechanism) if part]
    if not parts:
        return ""
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}".strip()
    return parts[0]


def _mode_specific(claims: list[Claim], fields: tuple[str, ...]) -> str:
    counts: dict[str, int] = {}
    for claim in claims:
        for field in fields:
            value = _clean(getattr(claim, field))
            if value and value not in GENERIC_COMMUNITIES and value != "other":
                counts[value] = counts.get(value, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], len(item[0])))[0]


def _clean_shared_concepts(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        norm = _clean_sentence(value)
        if not norm:
            continue
        lowered = norm.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(norm)
    return cleaned


def _clean_insight_title(insight: Insight, communities: list[str], shared_concepts: list[str]) -> str:
    if len(communities) >= 2:
        subject = shared_concepts[0] if shared_concepts else "shared mechanism"
        if insight.type == "transfer_opportunity":
            return f"{_title(communities[0])} and {_title(communities[1])} may share evaluation practice"
        return f"{_title(communities[0])} and {_title(communities[1])} may share a {subject} mechanism"
    title = _clean_sentence(insight.title)
    return title or "Communities may share a hidden mechanism"


def _clean_insight_summary(insight: Insight, communities: list[str], shared_concepts: list[str]) -> str:
    if len(communities) >= 2 and shared_concepts:
        concepts = _join(shared_concepts, max_items=3)
        return (
            f"{communities[0]} and {communities[1]} converge on {concepts}, "
            "but the citation graph still treats them as separate conversations."
        )
    return _clean_sentence(insight.insight)


def _insight_quality(insight: Insight, claims_by_id: dict[str, Claim]) -> float:
    label_quality = _community_label_quality(insight.communities)
    concept_specificity = _concept_specificity(insight.shared_concepts)
    readability = _readability_score(insight.title, insight.insight)
    evidence_density = min(1.0, len([cid for cid in insight.evidence_claims if cid in claims_by_id]) / 4.0)
    score = (
        0.26 * float(insight.confidence_score or 0.0)
        + 0.22 * float(insight.topology_score or 0.0)
        + 0.16 * concept_specificity
        + 0.16 * label_quality
        + 0.12 * evidence_density
        + 0.08 * readability
    )
    if len(insight.shared_concepts) < 2 and concept_specificity < 0.45:
        score -= 0.20
    if any((_clean(c) or "") in GENERIC_COMMUNITIES for c in insight.communities):
        score -= 0.22
    if "unifying mechanism" in (insight.title or "").lower() and concept_specificity < 0.55:
        score -= 0.08
    return max(0.0, min(1.0, score))


def _community_label_quality(communities: list[str]) -> float:
    if not communities:
        return 0.0
    scores = []
    for community in communities:
        norm = _clean(community) or ""
        if not norm or norm in GENERIC_COMMUNITIES:
            scores.append(0.0)
            continue
        score = 0.45
        if " " in norm:
            score += 0.25
        if len(norm) >= 12:
            score += 0.15
        scores.append(min(1.0, score))
    return sum(scores) / len(scores)


def _concept_specificity(shared_concepts: list[str]) -> float:
    if not shared_concepts:
        return 0.0
    specific = 0
    for concept in shared_concepts:
        lowered = (concept or "").lower()
        if lowered not in GENERIC_SHARED_CONCEPTS and len(lowered) >= 8:
            specific += 1
    return min(1.0, 0.35 + 0.25 * specific)


def _readability_score(title: str, summary: str) -> float:
    text = " ".join(part for part in (title, summary) if part)
    if not text:
        return 0.0
    score = 0.35
    if 40 <= len(title or "") <= 95:
        score += 0.25
    if len(summary or "") <= 180:
        score += 0.2
    if "  " in text or ";" in text:
        score -= 0.1
    return max(0.0, min(1.0, score))


def _clean_sentence(value: str | None) -> str:
    text = " ".join(str(value or "").split()).strip(" ,;")
    text = text.replace("  ", " ")
    return text


def _inferred_concepts(claim: Claim) -> set[str]:
    """Map sparse/free-text claim fields into coarse reusable topology concepts."""
    text = " ".join(
        str(v or "")
        for v in (
            claim.claim_text,
            claim.domain,
            claim.data_modality,
            claim.mechanism,
            claim.failure_mode,
            claim.evaluation_protocol,
            claim.assumption,
            claim.risk_type,
            claim.temporal_property,
            claim.task,
            claim.dataset,
            claim.metric,
        )
    ).lower()
    concepts: set[str] = set()
    if any(tok in text for tok in ("forecast", "prediction", "predict", "future")):
        concepts.add("forecasting")
    if any(tok in text for tok in ("time series", "temporal", "historical", "trend", "horizon", "short-term", "long-term")):
        concepts.add("temporal reasoning")
    if any(tok in text for tok in ("non-station", "regime shift", "market downturn", "distribution shift", "changing")):
        concepts.add("non-stationarity")
    if any(tok in text for tok in ("numeric", "numerical", "quantitative", "number", "high dimensional")):
        concepts.add("numeric reasoning")
    if any(tok in text for tok in ("benchmark", "evaluation", "empirical", "backtest", "tested datasets")):
        concepts.add("evaluation protocol")
    if any(tok in text for tok in ("modality", "multimodal", "text + time series", "visual", "plots")):
        concepts.add("modality alignment")
    if any(tok in text for tok in ("hallucination", "false alarm", "bias", "privacy", "risk")):
        concepts.add("reliability risk")
    if any(tok in text for tok in ("scaling", "scale", "power-law", "parameter count", "compute")):
        concepts.add("scaling behavior")
    return concepts


def _title(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split())


def _join(values: list[str], max_items: int) -> str:
    clipped = values[:max_items]
    if not clipped:
        return "shared concepts"
    if len(values) > max_items:
        return ", ".join(clipped) + ", and related concepts"
    if len(clipped) == 1:
        return clipped[0]
    return ", ".join(clipped[:-1]) + f", and {clipped[-1]}"


def _shortest_distance_between_sets(
    citation_graph: nx.Graph,
    sources: set[str],
    targets: set[str],
) -> int | None:
    frontier = set(sources)
    seen = set(sources)
    distance = 0
    while frontier:
        if frontier & targets:
            return distance
        distance += 1
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in citation_graph.adj[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
    return None
