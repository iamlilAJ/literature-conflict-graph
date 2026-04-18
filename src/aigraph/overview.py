"""Build user-facing search overviews from generated graph artifacts."""

from __future__ import annotations

import re

from .models import Anomaly, Claim, Hypothesis, Insight, Paper, ScoreBreakdown
from .paper_select import select_representative_papers

MIN_OVERVIEW_COMMUNITY_DISCONNECT = 0.25
MIN_OVERVIEW_INSIGHT_CONFIDENCE = 0.25
MIN_CURATED_LINE_QUALITY = 0.55
GENERIC_VALUES = {
    "",
    "other",
    "the method",
    "the task",
    "the model",
    "method",
    "task",
    "model",
    "approach",
    "technique",
}
MECHANISM_WORDS = {
    "baseline",
    "retrieval",
    "contamination",
    "leakage",
    "distractor",
    "noise",
    "scale",
    "bridge",
    "shift",
    "mismatch",
    "inflates",
    "shrinks",
    "non-stationarity",
    "temporal",
}


def build_search_overview(
    topic: str,
    papers: list[Paper],
    claims: list[Claim],
    anomalies: list[Anomaly],
    insights: list[Insight],
    selected: list[Hypothesis],
    scores: dict[str, ScoreBreakdown] | None = None,
) -> dict:
    """Create a compact, screenshot-friendly overview for the web result page."""
    papers = _hydrate_selection_scores(papers, topic)
    supported_insights = _supported_insights(insights)
    top_conflicts = _top_conflicts(anomalies)
    hidden_bridges = _hidden_bridges(insights)
    top_papers = _top_papers(papers)
    reading_path = _reading_path(papers, insights)
    best_lines = _curated_best_lines(
        anomalies,
        insights,
        selected,
        scores or {},
        allow_hypothesis_hero=bool(supported_insights),
    )
    why_this_matters = _why_this_matters(
        top_conflicts,
        hidden_bridges,
        selected,
        reading_path,
        papers,
        supported_insight_count=len(supported_insights),
    )
    return {
        "headline": _headline(topic, top_conflicts, hidden_bridges, papers, claims),
        "hero_line": best_lines["hero_line"],
        "why_this_matters": why_this_matters,
        "best_conflict_lines": best_lines["best_conflict_lines"],
        "best_bridge_lines": best_lines["best_bridge_lines"],
        "best_explanation_lines": best_lines["best_explanation_lines"],
        "top_conflicts": top_conflicts,
        "hidden_bridges": hidden_bridges,
        "top_papers": top_papers,
        "reading_path": reading_path,
        "selected_hypotheses": [
            {
                "hypothesis_id": h.hypothesis_id,
                "anomaly_id": h.anomaly_id,
                "hypothesis": h.hypothesis,
                "utility": round((scores or {}).get(h.hypothesis_id, ScoreBreakdown(hypothesis_id=h.hypothesis_id)).utility, 3),
            }
            for h in selected[:5]
        ],
    }


def _no_papers_next_step() -> str:
    return "Try a shorter or broader topic query, then rerun the search."


def _headline(topic: str, conflicts: list[dict], bridges: list[dict], papers: list[Paper], claims: list[Claim]) -> str:
    if not papers:
        return f"For '{topic}', no papers were retrieved, so the run could not build a literature map."
    if conflicts and bridges:
        return (
            f"For '{topic}', we found {len(conflicts)} visible conflict/gap region(s) "
            f"and {len(bridges)} hidden community bridge(s) across {len(papers)} papers and {len(claims)} claims."
        )
    if conflicts:
        return f"For '{topic}', the strongest signal is disagreement or missing evidence across {len(conflicts)} region(s)."
    if bridges:
        return f"For '{topic}', the strongest signal is hidden overlap between weakly connected communities."
    return f"For '{topic}', the run produced a starter map. Add more papers or full text to expose sharper conflicts."


def _curated_best_lines(
    anomalies: list[Anomaly],
    insights: list[Insight],
    selected: list[Hypothesis],
    scores: dict[str, ScoreBreakdown],
    *,
    allow_hypothesis_hero: bool,
) -> dict:
    anomaly_by_id = {a.anomaly_id: a for a in anomalies}
    conflict_lines = _rank_line_candidates(
        _conflict_line_candidates(anomalies),
        limit=3,
    )
    bridge_lines = _rank_line_candidates(
        _bridge_line_candidates(insights),
        limit=3,
    )
    explanation_lines = _rank_line_candidates(
        _explanation_line_candidates(selected, anomaly_by_id, scores),
        limit=4,
    )
    hero_line = conflict_lines[0] if conflict_lines else None
    if hero_line is None and bridge_lines:
        hero_line = bridge_lines[0]
    if hero_line is None and explanation_lines and allow_hypothesis_hero:
        hero_line = explanation_lines[0]
    return {
        "hero_line": hero_line,
        "best_conflict_lines": conflict_lines,
        "best_bridge_lines": bridge_lines,
        "best_explanation_lines": explanation_lines,
    }


def _conflict_line_candidates(anomalies: list[Anomaly]) -> list[dict]:
    candidates: list[dict] = []
    for anomaly in anomalies:
        if anomaly.type == "bridge_opportunity":
            continue
        if anomaly.type == "community_disconnect" and anomaly.topology_score < MIN_OVERVIEW_COMMUNITY_DISCONNECT:
            continue
        line = _rewrite_conflict_line(anomaly)
        quality = _quality_score(
            line,
            source_type="anomaly",
            boost=float(anomaly.topology_score or 0.0) * 0.4 + float(anomaly.evidence_impact or 0.0) * 0.08,
        )
        if quality < MIN_CURATED_LINE_QUALITY:
            continue
        candidates.append(
            {
                "line": line,
                "source_type": "anomaly",
                "source_id": anomaly.anomaly_id,
                "quality": round(quality, 3),
                "supporting_text": f"{len(anomaly.claim_ids)} claims · +{len(anomaly.positive_claims)} / -{len(anomaly.negative_claims)}",
                "anomaly_type": anomaly.type,
            }
        )
    return candidates


def _bridge_line_candidates(insights: list[Insight]) -> list[dict]:
    candidates: list[dict] = []
    for insight in insights:
        if (
            float(insight.confidence_score or 0.0) < MIN_OVERVIEW_INSIGHT_CONFIDENCE
            and float(insight.topology_score or 0.0) < MIN_OVERVIEW_INSIGHT_CONFIDENCE
        ):
            continue
        line = _rewrite_bridge_line(insight)
        quality = _quality_score(
            line,
            source_type="insight",
            boost=float(insight.confidence_score or 0.0) * 0.45 + float(insight.topology_score or 0.0) * 0.35,
        )
        if quality < MIN_CURATED_LINE_QUALITY:
            continue
        candidates.append(
            {
                "line": line,
                "source_type": "insight",
                "source_id": insight.insight_id,
                "quality": round(quality, 3),
                "supporting_text": " ↔ ".join(insight.communities[:2]) if insight.communities else "",
                "insight_type": insight.type,
            }
        )
    return candidates


def _explanation_line_candidates(
    selected: list[Hypothesis],
    anomaly_by_id: dict[str, Anomaly],
    scores: dict[str, ScoreBreakdown],
) -> list[dict]:
    candidates: list[dict] = []
    for hypothesis in selected:
        anomaly = anomaly_by_id.get(hypothesis.anomaly_id)
        line = _rewrite_explanation_line(hypothesis, anomaly)
        utility = float(scores.get(hypothesis.hypothesis_id, ScoreBreakdown(hypothesis_id=hypothesis.hypothesis_id)).utility)
        quality = _quality_score(line, source_type="hypothesis", boost=max(0.0, utility) * 0.45)
        if quality < MIN_CURATED_LINE_QUALITY:
            continue
        candidates.append(
            {
                "line": line,
                "source_type": "hypothesis",
                "source_id": hypothesis.hypothesis_id,
                "quality": round(quality, 3),
                "supporting_text": _hypothesis_supporting_text(anomaly, utility),
                "anomaly_id": hypothesis.anomaly_id,
            }
        )
    return candidates


def _hypothesis_supporting_text(anomaly: Anomaly | None, utility: float) -> str:
    bits: list[str] = []
    if anomaly is not None and anomaly.claim_ids:
        bits.append(f"{len(anomaly.claim_ids)} claims")
        if anomaly.positive_claims or anomaly.negative_claims:
            bits.append(f"+{len(anomaly.positive_claims)} / -{len(anomaly.negative_claims)}")
    bits.append(f"utility {round(utility, 3)}")
    return " · ".join(bits)


def _rank_line_candidates(candidates: list[dict], limit: int) -> list[dict]:
    ranked = sorted(
        candidates,
        key=lambda item: (float(item.get("quality") or 0.0), len(item.get("line") or "")),
        reverse=True,
    )
    return ranked[:limit]


def _rewrite_conflict_line(anomaly: Anomaly) -> str:
    method = _entity_value(anomaly.shared_entities, "method")
    task = _entity_value(anomaly.shared_entities, "task")
    question = _normalize_candidate_text(anomaly.central_question, anomaly=anomaly)
    lower = question.lower()
    if anomaly.type == "evidence_gap":
        subject = method or "The evidence"
        if method and task:
            return _limit_line(f"The evidence for {method} on {task} is thinner than it looks.")
        if method:
            return _limit_line(f"The evidence around {method} is thinner than it looks.")
        if task:
            return _limit_line(f"The evidence on {task} is thinner than it looks.")
    if question.startswith("When does ") and " and when does it fail" in lower:
        subject = method or "The method"
        if task:
            return _limit_line(f"{subject} can flip from win to failure on {task} depending on the setup.")
        return _limit_line(f"{subject} flips from win to failure depending on the setup.")
    if "mismatch" in anomaly.type and task:
        subject = method or "Results"
        return _limit_line(f"{subject} changes on {task} when the evaluation setup shifts.")
    if question:
        return _limit_line(question.rstrip("?") + ".")
    subject = method or "The method"
    if task:
        return _limit_line(f"{subject} shows conflicting results on {task}.")
    return _limit_line(f"{subject} shows conflicting results across the map.")


def _rewrite_bridge_line(insight: Insight) -> str:
    text = _normalize_candidate_text(insight.title or insight.insight or "", insight=insight)
    communities = [c for c in insight.communities if c]
    if communities and insight.shared_concepts:
        concept = insight.shared_concepts[0]
        if len(communities) >= 2:
            return _limit_line(f"{communities[0]} and {communities[1]} may share a {concept} story without talking to each other.")
    if text:
        return _limit_line(text.rstrip(".") + ".")
    if communities:
        return _limit_line(f"{' and '.join(communities[:2])} may be closer than the citation graph suggests.")
    return "Two weakly connected communities may be studying the same idea."


def _rewrite_explanation_line(hypothesis: Hypothesis, anomaly: Anomaly | None) -> str:
    text = _normalize_candidate_text(hypothesis.hypothesis, anomaly=anomaly)
    lower = text.lower()
    method = _best_subject(text, anomaly)
    task = _entity_value(getattr(anomaly, "shared_entities", {}) or {}, "task")
    if "inflated when compared against weak baselines" in lower:
        return _limit_line(f"Weak baselines can make {method} look stronger than it is.")
    if "partial test-set contamination in pretraining inflates closed-book baselines" in lower:
        if method.lower() != "the method":
            return _limit_line(f"Pretraining leakage can erase the visible gain from {method}.")
        return _limit_line("Pretraining leakage can make closed-book baselines look deceptively strong.")
    if "high top-k retrieval injects distractor passages" in lower:
        if task:
            return _limit_line(f"Too much retrieval can drown {task} in distractors.")
        return _limit_line("Too much retrieval can drown the model in distractors.")
    if "decreases with generator scale" in lower:
        return _limit_line(f"As the base model gets stronger, {method} matters less.")
    if "unreported moderator variable drives the conflicting results" in lower:
        if task:
            return _limit_line(f"A hidden setup variable may be driving the {task} disagreement.")
        return _limit_line("A hidden setup variable may be driving the disagreement.")
    return _limit_line(text.rstrip(".") + ".")


def _normalize_candidate_text(text: str, anomaly: Anomaly | None = None, insight: Insight | None = None) -> str:
    text = " ".join(str(text or "").strip().split())
    if not text:
        return ""
    method = _entity_value(getattr(anomaly, "shared_entities", {}) or {}, "method")
    task = _entity_value(getattr(anomaly, "shared_entities", {}) or {}, "task")
    replacements = {
        r"\bthe method\b": method,
        r"\bthe task\b": task,
    }
    for pattern, value in replacements.items():
        if value:
            text = re.sub(pattern, value, text, flags=re.IGNORECASE)
    if re.search(r"\bother\b", text, flags=re.IGNORECASE):
        if method:
            text = re.sub(r"\bother\b", method, text, flags=re.IGNORECASE)
        else:
            return ""
    text = re.sub(r"\bthe model\b", method or "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ;,")
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\band related benchmarks\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthe reported evaluation metric\b", "the reported metric", text, flags=re.IGNORECASE)
    text = re.sub(r"\breported effects\b", "reported results", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ;,")
    return text


def _best_subject(text: str, anomaly: Anomaly | None) -> str:
    method = _entity_value(getattr(anomaly, "shared_entities", {}) or {}, "method")
    if method:
        return method
    match = re.search(r"Gains attributed to (.+?) are inflated", text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        if _is_specific_value(candidate):
            return candidate
    return "the method"


def _quality_score(text: str, source_type: str, boost: float = 0.0) -> float:
    if not text:
        return 0.0
    lower = text.lower()
    if any(bad in lower for bad in (" other ", "the method", "the task", "various factors", "complex interactions")):
        return 0.0
    score = 0.25
    length = len(text)
    if 45 <= length <= 110:
        score += 0.28
    elif length <= 140:
        score += 0.16
    else:
        score -= 0.12
    score += min(0.22, sum(1 for word in MECHANISM_WORDS if word in lower) * 0.06)
    if any(ch in text for ch in (";", ":")):
        score -= 0.05
    if source_type == "hypothesis":
        score += 0.1
    if source_type == "insight":
        score += 0.06
    if re.search(r"\b(can|may|might)\b", lower):
        score += 0.04
    if re.search(r"\b(help|fail|flip|erase|drown|inflate|share|shift)\b", lower):
        score += 0.08
    return max(0.0, min(1.0, score + boost))


def _limit_line(text: str, max_len: int = 108) -> str:
    text = " ".join((text or "").split()).strip()
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    truncated = text[: max_len - 1].rstrip(" ,;:")
    words = truncated.split()
    if len(words) > 4:
        truncated = " ".join(words[:-1])
    return truncated.rstrip(" ,;:") + "..."


def _entity_value(shared_entities: dict[str, str], key: str) -> str:
    value = str((shared_entities or {}).get(key) or "").strip()
    return value if _is_specific_value(value) else ""


def _is_specific_value(value: str | None) -> bool:
    norm = str(value or "").strip().lower()
    return bool(norm and norm not in GENERIC_VALUES)


def _top_conflicts(anomalies: list[Anomaly]) -> list[dict]:
    primary = [
        a
        for a in anomalies
        if a.type != "bridge_opportunity"
        and (a.type != "community_disconnect" or a.topology_score >= MIN_OVERVIEW_COMMUNITY_DISCONNECT)
        and a.positive_claims
        and a.negative_claims
    ]
    ranked = sorted(
        primary,
        key=lambda a: (
            float(a.topology_score or 0.0),
            float(a.evidence_impact or 0.0),
            len(a.claim_ids),
        ),
        reverse=True,
    )
    return [
        {
            "anomaly_id": a.anomaly_id,
            "type": a.type,
            "question": a.central_question,
            "claim_count": len(a.claim_ids),
            "positive": len(a.positive_claims),
            "negative": len(a.negative_claims),
            "impact": round(float(a.evidence_impact or 0.0), 3),
            "impact_available": bool(a.evidence_impact),
            "topology": round(float(a.topology_score or 0.0), 3),
        }
        for a in ranked[:4]
    ]


def _hidden_bridges(insights: list[Insight]) -> list[dict]:
    insights = [
        i
        for i in insights
        if i.confidence_score >= MIN_OVERVIEW_INSIGHT_CONFIDENCE
        or i.topology_score >= MIN_OVERVIEW_INSIGHT_CONFIDENCE
    ]
    ranked = sorted(
        insights,
        key=lambda i: (
            float(i.quality_score or 0.0),
            float(i.confidence_score or 0.0),
            float(i.topology_score or 0.0),
            float(i.impact_score or 0.0),
        ),
        reverse=True,
    )
    return [
        {
            "insight_id": i.insight_id,
            "type": i.type,
            "title": i.title,
            "communities": i.communities,
            "shared_concepts": i.shared_concepts[:6],
            "unifying_frame": i.unifying_frame,
            "citation_gap": i.citation_gap,
            "confidence": round(float(i.confidence_score or 0.0), 3),
        }
        for i in ranked[:3]
    ]


def _why_this_matters(
    top_conflicts: list[dict],
    hidden_bridges: list[dict],
    selected: list[Hypothesis],
    reading_path: list[dict],
    papers: list[Paper] | None = None,
    *,
    supported_insight_count: int = 0,
) -> dict:
    if not (papers or []):
        return {
            "line": "No papers were retrieved for this topic, so there is no evidence to summarize yet.",
            "next_step": _no_papers_next_step(),
        }
    if supported_insight_count <= 0:
        if top_conflicts:
            line = (
                "This run is exploratory: it surfaced claim-level disagreement, but no synthesized insight is "
                "supported strongly enough yet to treat the takeaway as settled."
            )
        elif selected:
            line = (
                "This run is exploratory: it found candidate explanations, but no synthesized insight is supported "
                "yet, so treat the output as hypotheses to test rather than a conclusion."
            )
        else:
            line = "This run is still a starter map, so the next gain will come from adding sharper evidence rather than reading the graph literally."
    elif top_conflicts:
        conflict = top_conflicts[0]
        conflict_type = str(conflict.get("type") or "")
        if conflict_type in {"benchmark_inconsistency", "impact_conflict"}:
            line = "This disagreement changes which benchmarks and baselines you should trust when reading the literature."
        elif conflict_type in {"setting_mismatch", "metric_mismatch"}:
            line = "Small evaluation changes may be flipping the reported conclusion, so implementation decisions here are not plug-and-play."
        elif conflict_type == "evidence_gap":
            line = "The field looks more settled than the evidence actually is, so strong claims may be outrunning the data."
        else:
            line = "The main result in this area is conditional, not universal, and that changes how you should interpret new papers."
    elif hidden_bridges:
        line = "The strongest signal is a community disconnect: related ideas are close conceptually but still not informing each other."
    else:
        line = "This run is still a starter map, so the next gain will come from adding sharper evidence rather than reading the graph literally."

    next_step = ""
    if selected:
        minimal_test = _limit_line(str(selected[0].minimal_test or ""), max_len=126)
        if minimal_test:
            next_step = minimal_test
    if not next_step and reading_path:
        first = reading_path[0]
        next_step = f"Start with {first.get('title', 'the top paper')}."

    return {
        "line": line,
        "next_step": next_step,
    }


def _supported_insights(insights: list[Insight]) -> list[Insight]:
    return [
        insight
        for insight in insights
        if insight.evidence_claims
        and (
            insight.confidence_score >= MIN_OVERVIEW_INSIGHT_CONFIDENCE
            or insight.topology_score >= MIN_OVERVIEW_INSIGHT_CONFIDENCE
        )
    ]


def _top_papers(papers: list[Paper]) -> list[dict]:
    ranked = sorted(
        papers,
        key=lambda p: (
            float(p.selection_score or 0.0),
            int(p.cited_by_count or 0),
            int(p.year or 0),
        ),
        reverse=True,
    )
    return [_paper_card(p) for p in ranked[:6]]


def _reading_path(papers: list[Paper], insights: list[Insight]) -> list[dict]:
    picks: list[tuple[str, str, Paper]] = []
    survey = _find_paper(papers, ("survey", "review", "benchmark", "evaluation"))
    if survey:
        picks.append(("Start with the landscape", "Survey, benchmark, or evaluation signal.", survey))
    cited = max(papers, key=lambda p: int(p.cited_by_count or 0), default=None)
    if cited:
        picks.append(("Anchor on the most-cited evidence", "High-impact paper in the selected pool.", cited))
    recent_critical = _find_paper(papers, ("limitation", "challenge", "failure", "robustness", "bias"), newest=True)
    if recent_critical:
        picks.append(("Check the failure edge", "Recent critical or robustness signal.", recent_critical))
    bridge_paper = _bridge_paper(papers, insights)
    if bridge_paper:
        picks.append(("Read the bridge evidence", "Evidence used by a community insight.", bridge_paper))

    seen: set[str] = set()
    path: list[dict] = []
    for title, why, paper in picks:
        if paper.paper_id in seen:
            continue
        seen.add(paper.paper_id)
        card = _paper_card(paper)
        card["step"] = title
        card["why_read"] = why
        path.append(card)
        if len(path) >= 4:
            break
    return path


def _find_paper(papers: list[Paper], keywords: tuple[str, ...], newest: bool = False) -> Paper | None:
    matches = [
        p
        for p in papers
        if any(k in f"{p.title} {p.abstract} {p.selection_reason or ''}".lower() for k in keywords)
    ]
    if not matches:
        return None
    return max(
        matches,
        key=lambda p: (
            int(p.year or 0) if newest else float(p.selection_score or 0.0),
            int(p.cited_by_count or 0),
        ),
    )


def _bridge_paper(papers: list[Paper], insights: list[Insight]) -> Paper | None:
    by_id = {p.paper_id: p for p in papers}
    for insight in insights:
        for paper_id in insight.evidence_papers:
            if paper_id in by_id:
                return by_id[paper_id]
    return None


def _paper_card(paper: Paper) -> dict:
    citation_available = not paper.paper_id.startswith("arxiv:")
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "year": paper.year,
        "venue": paper.venue,
        "url": paper.url,
        "citations": int(paper.cited_by_count or 0),
        "citation_available": citation_available,
        "selection_score": round(float(paper.selection_score or 0.0), 3),
        "selection_reason": paper.selection_reason or "",
        "retrieval_channel": paper.retrieval_channel or "",
    }


def _hydrate_selection_scores(papers: list[Paper], topic: str) -> list[Paper]:
    if not papers:
        return []
    if all(p.selection_score or p.selection_reason for p in papers):
        return papers
    strategy = "recent" if all(p.paper_id.startswith("arxiv:") for p in papers) else "balanced"
    selected = select_representative_papers(
        papers,
        query=topic,
        limit=len(papers),
        strategy=strategy,
    )
    by_id = {p.paper_id: p for p in selected}
    return [by_id.get(p.paper_id, p) for p in papers]
