"""Typed claim graph built from extracted claims."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

import networkx as nx

from .models import Claim, Paper
from .paper_select import paper_role_explanation, paper_role_label

# Map entity fields on Claim -> (node type, edge type)
ENTITY_EDGES: dict[str, tuple[str, str]] = {
    "method": ("Method", "uses"),
    "model": ("Model", "evaluated_with"),
    "task": ("Task", "targets"),
    "dataset": ("Dataset", "evaluated_on"),
    "metric": ("Metric", "measured_by"),
    "baseline": ("Baseline", "compares_against"),
}

SEMANTIC_EDGES: dict[str, tuple[str, str]] = {
    "domain": ("Domain", "situated_in"),
    "data_modality": ("DataModality", "uses_modality"),
    "mechanism": ("Mechanism", "explains_by"),
    "failure_mode": ("FailureMode", "fails_by"),
    "evaluation_protocol": ("EvaluationProtocol", "evaluated_with_protocol"),
    "assumption": ("Assumption", "assumes"),
    "risk_type": ("RiskType", "has_risk"),
    "temporal_property": ("TemporalProperty", "has_temporal_property"),
}

SETTING_FIELDS: tuple[str, ...] = ("retriever", "top_k", "context_length", "task_type")


def _entity_node_id(node_type: str, value: str) -> str:
    return f"{node_type}:{value.strip().lower()}"


def _setting_node_id(field: str, value: str) -> str:
    return f"Setting:{field}={value.strip().lower()}"


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    return v or None


def build_graph(
    claims: list[Claim],
    papers: list[Paper] | None = None,
    current_year: int | None = None,
) -> nx.MultiDiGraph:
    g: nx.MultiDiGraph = nx.MultiDiGraph()

    current_year = current_year or date.today().year
    papers_by_id = {p.paper_id: p for p in (papers or [])}
    seen_papers: set[str] = set()
    for paper in papers_by_id.values():
        _add_paper_node(g, paper, current_year)
        seen_papers.add(paper.paper_id)

    for claim in claims:
        paper_node = f"Paper:{claim.paper_id}"
        claim_node = f"Claim:{claim.claim_id}"
        if claim.paper_id not in seen_papers:
            paper = papers_by_id.get(claim.paper_id)
            if paper is not None:
                _add_paper_node(g, paper, current_year)
            else:
                g.add_node(paper_node, node_type="Paper", paper_id=claim.paper_id)
            seen_papers.add(claim.paper_id)

        g.add_node(
            claim_node,
            node_type="Claim",
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            direction=claim.direction,
            method=claim.method,
            task=claim.task,
            dataset=claim.dataset,
            metric=claim.metric,
            domain=claim.domain,
            data_modality=claim.data_modality,
            mechanism=claim.mechanism,
            failure_mode=claim.failure_mode,
            evaluation_protocol=claim.evaluation_protocol,
            assumption=claim.assumption,
            risk_type=claim.risk_type,
            temporal_property=claim.temporal_property,
        )
        g.add_edge(paper_node, claim_node, edge_type="makes")

        for field, (node_type, edge_type) in {**ENTITY_EDGES, **SEMANTIC_EDGES}.items():
            value = getattr(claim, field)
            if not value:
                continue
            nid = _entity_node_id(node_type, value)
            if nid not in g:
                g.add_node(nid, node_type=node_type, name=value)
            g.add_edge(claim_node, nid, edge_type=edge_type)

        setting = claim.setting
        for field in SETTING_FIELDS:
            value = getattr(setting, field)
            if not value:
                continue
            nid = _setting_node_id(field, value)
            if nid not in g:
                g.add_node(nid, node_type="Setting", field=field, value=value)
            g.add_edge(claim_node, nid, edge_type="conditioned_on")

    _add_citation_edges(g, papers_by_id)
    _add_claim_claim_edges(g, claims)
    return g


def citation_metrics(paper: Paper, current_year: int | None = None) -> dict[str, float | int]:
    current_year = current_year or date.today().year
    age = max(1, current_year - int(paper.year or current_year) + 1)
    impact_score = math.log1p(max(0, int(paper.cited_by_count or 0)))
    recent_cutoff = current_year - 2
    recent = 0
    for item in paper.counts_by_year or []:
        try:
            year = int(item.get("year") or 0)
            count = int(item.get("cited_by_count") or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        if year >= recent_cutoff:
            recent += max(0, count)
    velocity = recent / age
    return {
        "cited_by_count": max(0, int(paper.cited_by_count or 0)),
        "recent_citations": recent,
        "citation_velocity": velocity,
        "impact_score": impact_score,
        "age_normalized_impact": impact_score / math.sqrt(age),
    }


def _add_paper_node(g: nx.MultiDiGraph, paper: Paper, current_year: int) -> None:
    metrics = citation_metrics(paper, current_year=current_year)
    paper_node = f"Paper:{paper.paper_id}"
    g.add_node(
        paper_node,
        node_type="Paper",
        paper_id=paper.paper_id,
        title=paper.title,
        year=paper.year,
        venue=paper.venue,
        url=paper.url,
        doi=paper.doi,
        referenced_works=list(paper.referenced_works or []),
        paper_role=paper.paper_role,
        paper_role_score=float(paper.paper_role_score or 0.0),
        paper_role_signals=list(paper.paper_role_signals or []),
        **metrics,
    )
    if paper.paper_role:
        role_node = f"Role:{paper.paper_role}"
        if role_node not in g:
            g.add_node(
                role_node,
                node_type="Role",
                role=paper.paper_role,
                name=paper_role_label(paper.paper_role),
                description=paper_role_explanation(paper.paper_role),
            )
        g.add_edge(paper_node, role_node, edge_type="has_role")


def _add_citation_edges(g: nx.MultiDiGraph, papers_by_id: dict[str, Paper]) -> None:
    paper_ids = set(papers_by_id)
    for paper in papers_by_id.values():
        source = f"Paper:{paper.paper_id}"
        for ref in paper.referenced_works or []:
            if ref not in paper_ids:
                continue
            target = f"Paper:{ref}"
            if source in g and target in g:
                g.add_edge(source, target, edge_type="cites")



def _add_claim_claim_edges(g: nx.MultiDiGraph, claims: list[Claim]) -> None:
    clusters: dict[
        tuple[str, str],
        dict[str, object],
    ] = defaultdict(
        lambda: {
            "positive": [],
            "non_positive": [],
            "overlap_groups": defaultdict(list),
        }
    )
    for index, claim in enumerate(claims):
        method = _norm(claim.method)
        task = _norm(claim.task)
        if not method or not task:
            continue
        cluster = clusters[(method, task)]
        node_id = f"Claim:{claim.claim_id}"
        entry = (index, claim, node_id)
        if claim.direction == "positive":
            cluster["positive"].append(entry)
        elif claim.direction in ("negative", "mixed"):
            cluster["non_positive"].append(entry)

        dataset = _norm(claim.dataset)
        metric = _norm(claim.metric)
        if dataset and metric:
            cluster["overlap_groups"][(dataset, metric)].append((index, node_id))

    for cluster in clusters.values():
        positives = cluster["positive"]
        non_positives = cluster["non_positive"]
        overlap_groups = cluster["overlap_groups"]

        for pos_index, pos_claim, pos_node in positives:
            for other_index, other_claim, other_node in non_positives:
                if pos_index < other_index:
                    source, target = pos_node, other_node
                else:
                    source, target = other_node, pos_node
                g.add_edge(source, target, edge_type="contradicts")
                if _settings_differ(pos_claim, other_claim):
                    g.add_edge(source, target, edge_type="setting_mismatch")

        for group in overlap_groups.values():
            for (_, node_a), (_, node_b) in combinations(group, 2):
                g.add_edge(node_a, node_b, edge_type="overlap")


def build_citation_graph(g: nx.MultiDiGraph) -> nx.Graph:
    cites = nx.Graph()
    for node, data in g.nodes(data=True):
        if data.get("node_type") == "Paper":
            cites.add_node(node)
    for u, v, data in g.edges(data=True):
        if data.get("edge_type") == "cites":
            cites.add_edge(u, v)
    return cites


def _opposite(a: str, b: str) -> bool:
    if {a, b} == {"positive", "negative"}:
        return True
    if {a, b} == {"positive", "mixed"}:
        return True
    return False


def _settings_differ(a: Claim, b: Claim) -> bool:
    for field in SETTING_FIELDS:
        va, vb = _norm(getattr(a.setting, field)), _norm(getattr(b.setting, field))
        if va and vb and va != vb:
            return True
    return False


def save_graph(g: nx.MultiDiGraph, path: str | Path) -> None:
    data = _node_link_data(g)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return _node_link_graph(data)


def _node_link_data(g: nx.MultiDiGraph) -> dict:
    # networkx>=3.4 switched default `edges` key to "edges"; be explicit for stability.
    try:
        return nx.node_link_data(g, edges="edges")
    except TypeError:  # older networkx
        return nx.node_link_data(g)


def _node_link_graph(data: dict) -> nx.MultiDiGraph:
    try:
        return nx.node_link_graph(data, edges="edges", directed=True, multigraph=True)
    except TypeError:
        return nx.node_link_graph(data, directed=True, multigraph=True)
