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
) -> str:
    anom_by_id = {a.anomaly_id: a for a in anomalies}
    claims_by_id = {c.claim_id: c for c in claims}
    paper_lookup = paper_lookup or {}

    grouped: dict[str, list[Hypothesis]] = defaultdict(list)
    for h in selected:
        grouped[h.anomaly_id].append(h)

    lines: list[str] = ["# Selected Hypotheses", ""]
    lines.append(f"Selected **{len(selected)}** hypotheses across **{len(grouped)}** anomalies.")
    lines.append("")

    if insights:
        lines.extend(_render_insights(insights, paper_lookup))

    for anomaly_id, hyps in grouped.items():
        anomaly = anom_by_id.get(anomaly_id)
        if anomaly is None:
            continue
        lines.extend(_render_anomaly(anomaly, hyps, claims_by_id, scores, paper_lookup))

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
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Anomaly {anomaly.anomaly_id} — {anomaly.type}")
    lines.append("")
    lines.append(f"**Central question:** {anomaly.central_question}")
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
