"""Markdown renderer for the selected-hypothesis report."""

from __future__ import annotations

from collections import defaultdict

from typing import Optional

from .models import Anomaly, Claim, Hypothesis, Insight, Paper, ScoreBreakdown


def render_report(
    selected: list[Hypothesis],
    anomalies: list[Anomaly],
    claims: list[Claim],
    scores: dict[str, ScoreBreakdown],
    paper_lookup: Optional[dict[str, Paper]] = None,
    insights: Optional[list[Insight]] = None,
    topic: Optional[str] = None,
    paper_count: Optional[int] = None,
) -> str:
    anom_by_id = {a.anomaly_id: a for a in anomalies}
    claims_by_id = {c.claim_id: c for c in claims}
    paper_lookup = paper_lookup or {}

    grouped: dict[str, list[Hypothesis]] = defaultdict(list)
    for h in selected:
        grouped[h.anomaly_id].append(h)
    ordered_anomaly_ids = list(dict.fromkeys(h.anomaly_id for h in selected))

    lines: list[str] = ["# Selected Hypotheses", ""]
    if paper_count == 0:
        topic_label = f" for '{topic}'" if topic else ""
        lines.append(f"No papers were retrieved{topic_label}, so there are no hypotheses or evidence claims to report yet.")
        lines.append("")
        lines.append("Next step: try a shorter or broader topic query, then rerun the search.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"Selected **{len(selected)}** hypotheses across **{len(grouped)}** anomalies.")
    lines.append("")
    if not insights:
        lines.append(
            "Exploratory report: no synthesized insights were supported yet, so the sections below show claim-level "
            "evidence and candidate explanations rather than a settled takeaway."
        )
        lines.append("")

    if insights:
        lines.extend(_render_insights(insights, paper_lookup))

    conflict_sections: list[tuple[Anomaly, list[Hypothesis]]] = []
    bridge_sections: list[tuple[Anomaly, list[Hypothesis]]] = []
    for anomaly_id in ordered_anomaly_ids:
        anomaly = anom_by_id.get(anomaly_id)
        hyps = grouped.get(anomaly_id) or []
        if anomaly is None or not hyps:
            continue
        if anomaly.type == "bridge_opportunity":
            bridge_sections.append((anomaly, hyps))
        else:
            conflict_sections.append((anomaly, hyps))

    if conflict_sections:
        lines.append("## Conflict Hypotheses")
        lines.append("")
        for anomaly, hyps in conflict_sections:
            lines.extend(_render_anomaly(anomaly, hyps, claims_by_id, scores, paper_lookup, heading_level=3))

    if bridge_sections:
        lines.append("## Bridge Opportunities")
        lines.append("")
        lines.append("These transfer questions are listed separately from conflicts because they do not imply opposing evidence.")
        lines.append("")
        for anomaly, hyps in bridge_sections:
            lines.extend(_render_anomaly(anomaly, hyps, claims_by_id, scores, paper_lookup, heading_level=3))

    lines.append("## Evidence claims")
    lines.append("")
    referenced: set[str] = set()
    for h in selected:
        referenced.update(h.explains_claims)
    for cid in sorted(referenced):
        c = claims_by_id.get(cid)
        if c is None:
            continue
        lines.append(f"- **{cid}** ({_paper_label(c, paper_lookup)}, {c.direction}): {c.claim_text}")
    lines.append("")

    return "\n".join(lines)


def _paper_label(claim: Claim, paper_lookup: dict[str, Paper]) -> str:
    paper = paper_lookup.get(claim.paper_id)
    if paper and paper.title:
        year = paper.year or ""
        year_str = f", {year}" if year else ""
        return f'{claim.paper_id} — "{paper.title}"{year_str}'
    return claim.paper_id


def _paper_title(paper_id: str, paper_lookup: dict[str, Paper]) -> str:
    paper = paper_lookup.get(paper_id)
    if paper and paper.title:
        year = f", {paper.year}" if paper.year else ""
        meta: list[str] = []
        if paper.cited_by_count:
            meta.append(f"citations={paper.cited_by_count}")
        if paper.selection_score:
            meta.append(f"selection={paper.selection_score:.2f}")
        if paper.retrieval_channel:
            meta.append(f"channel={paper.retrieval_channel}")
        meta_str = f" ({'; '.join(meta)})" if meta else ""
        return f'{paper_id} — "{paper.title}"{year}{meta_str}'
    return paper_id


def _render_insights(insights: list[Insight], paper_lookup: dict[str, Paper]) -> list[str]:
    lines: list[str] = ["# Community Insights", ""]
    for insight in insights:
        lines.append(f"## {insight.insight_id} — {insight.title}")
        lines.append("")
        lines.append(f"**Type:** {insight.type}")
        lines.append("")
        if insight.communities:
            lines.append(f"**Communities:** {', '.join(insight.communities)}")
            lines.append("")
        if insight.shared_concepts:
            lines.append(f"**Shared concepts:** {', '.join(insight.shared_concepts)}")
            lines.append("")
        lines.append(insight.insight)
        lines.append("")
        if insight.unifying_frame:
            lines.append(f"**Unifying frame.** {insight.unifying_frame}")
            lines.append("")
        if insight.citation_gap:
            lines.append(f"**Citation gap.** {insight.citation_gap}")
            lines.append("")
        if insight.transfer_suggestions:
            lines.append("**Transfer suggestions:**")
            for suggestion in insight.transfer_suggestions:
                lines.append(f"- {suggestion}")
            lines.append("")
        if insight.evidence_papers:
            lines.append("**Evidence papers:**")
            for paper_id in insight.evidence_papers[:8]:
                lines.append(f"- {_paper_title(paper_id, paper_lookup)}")
                paper = paper_lookup.get(paper_id)
                if paper and paper.selection_reason:
                    lines.append(f"  - selection: {paper.selection_reason}")
            lines.append("")
        lines.append(
            f"**Scores:** impact={insight.impact_score:.2f}, "
            f"topology={insight.topology_score:.2f}, confidence={insight.confidence_score:.2f}"
        )
        lines.append("")
    return lines


def _render_anomaly(
    anomaly: Anomaly,
    hyps: list[Hypothesis],
    claims_by_id: dict[str, Claim],
    scores: dict[str, ScoreBreakdown],
    paper_lookup: dict[str, Paper],
    *,
    heading_level: int = 2,
) -> list[str]:
    lines: list[str] = []
    heading = "#" * max(1, heading_level)
    if anomaly.type == "bridge_opportunity":
        section_label = "Bridge opportunity"
    elif anomaly.type == "replication_conflict":
        section_label = "Replication conflict"
    else:
        section_label = "Anomaly"
    lines.append(f"{heading} {section_label} {anomaly.anomaly_id} — {anomaly.type}")
    lines.append("")
    if anomaly.type == "bridge_opportunity":
        question_label = "Transfer question"
    elif anomaly.type == "replication_conflict":
        question_label = "Replication question"
    else:
        question_label = "Central question"
    lines.append(f"**{question_label}:** {anomaly.central_question}")
    lines.append("")
    if anomaly.shared_entities:
        items = ", ".join(f"{k}={v}" for k, v in anomaly.shared_entities.items())
        lines.append(f"**Shared entities:** {items}")
    if anomaly.varying_settings:
        lines.append(f"**Varying settings:** {', '.join(anomaly.varying_settings)}")
    lines.append("")
    lines.append("**Evidence claims:**")
    for cid in anomaly.claim_ids:
        c = claims_by_id.get(cid)
        if c is None:
            continue
        lines.append(f"- `{cid}` ({_paper_label(c, paper_lookup)}, {c.direction}): {c.claim_text}")
    lines.append("")

    for h in hyps:
        lines.extend(_render_hypothesis(h, scores))
    return lines


def _render_hypothesis(h: Hypothesis, scores: dict[str, ScoreBreakdown]) -> list[str]:
    s = scores.get(h.hypothesis_id)
    lines: list[str] = []
    lines.append(f"### {h.hypothesis_id} — {h.hypothesis}")
    lines.append("")
    if h.mechanism:
        lines.append(f"**Mechanism.** {h.mechanism}")
        lines.append("")
    if h.predictions:
        lines.append("**Predictions:**")
        for p in h.predictions:
            lines.append(f"- {p}")
        lines.append("")
    if h.minimal_test:
        lines.append(f"**Minimal test.** {h.minimal_test}")
        lines.append("")
    if h.scope_conditions:
        cond = ", ".join(f"{k}={v}" for k, v in h.scope_conditions.items())
        lines.append(f"**Scope.** {cond}")
        lines.append("")
    if h.evidence_gap:
        lines.append(f"**Evidence gap.** {h.evidence_gap}")
        lines.append("")
    if h.graph_bridge and (h.graph_bridge.from_ or h.graph_bridge.to):
        lines.append(f"**Graph bridge.** {h.graph_bridge.from_} → {h.graph_bridge.to}")
        lines.append("")
    if s is not None:
        lines.append("**Utility breakdown**")
        lines.append("")
        lines.append("| explain | grounding | testability | novelty | discrim | impact | topology | cost | utility |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        lines.append(
            f"| {s.explain:.2f} | {s.grounding:.2f} | {s.testability:.2f} | "
            f"{s.novelty:.2f} | {s.discriminability:.2f} | {s.impact:.2f} | "
            f"{s.topology:.2f} | {s.cost:.2f} | {s.utility:.2f} |"
        )
        lines.append("")
    return lines
