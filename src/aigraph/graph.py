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

# Core entity fields on Claim -> (node type, edge type). The graph keeps only
# the entities used by anomaly detection and hypothesis templates; semantic
# fields (domain, mechanism, failure_mode, ...) live as Claim attributes and
# are read directly without spawning per-value nodes.
ENTITY_EDGES: dict[str, tuple[str, str]] = {
    "method": ("Method", "uses"),
    "task": ("Task", "targets"),
    "dataset": ("Dataset", "evaluated_on"),
    "metric": ("Metric", "measured_by"),
}

CLAIM_ENTITY_FIELDS: dict[str, tuple[str, str]] = ENTITY_EDGES
SETTING_FIELDS: tuple[str, ...] = ("retriever", "top_k", "context_length", "task_type")

# Per-entity canonical-form attribute on the Claim model. When the canonical
# field is filled in (and not a placeholder like "other"), it is used as the
# entity node id and the raw field becomes an alias on that node. This stops
# "CoT" / "chain-of-thought" / "Chain Of Thought" from spawning three Method
# nodes that fragment downstream cluster degree.
_CANONICAL_FIELDS: dict[str, str] = {
    "method": "canonical_method",
    "task": "canonical_task",
    "dataset": "dataset_canonical",
    "metric": "metric_canonical",
}
_PLACEHOLDER_LABELS = frozenset(
    {"other", "unknown", "misc", "n/a", "na", "none", "null", ""}
)
# Bibliographic coupling: papers A and B that share at least this many cited
# works are linked by a co_cites edge with weight = number of shared refs.
# Threshold of 2 keeps single-ref coincidences from cluttering the graph.
_MIN_COUPLING_WEIGHT = 2


def _entity_node_id(node_type: str, value: str) -> str:
    return f"{node_type}:{value.strip().lower()}"


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    return v or None


def _resolve_entity_value(claim: Claim, field: str) -> tuple[str | None, str | None]:
    """Pick the canonical value for an entity field. Returns (canonical, alias)
    where alias is the raw value when it differs from canonical (so the node
    can record what surface forms reached it), or None when there is no
    canonical override."""
    raw = getattr(claim, field, None)
    raw_clean = raw.strip() if isinstance(raw, str) and raw.strip() else None
    canonical_attr = _CANONICAL_FIELDS.get(field)
    canonical = getattr(claim, canonical_attr, None) if canonical_attr else None
    canonical_clean = canonical.strip() if isinstance(canonical, str) and canonical.strip() else None
    if canonical_clean and canonical_clean.lower() not in _PLACEHOLDER_LABELS:
        if raw_clean and raw_clean.lower() != canonical_clean.lower():
            return canonical_clean, raw_clean
        return canonical_clean, None
    return raw_clean, None


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

        for field, (node_type, edge_type) in CLAIM_ENTITY_FIELDS.items():
            canonical_value, alias_value = _resolve_entity_value(claim, field)
            if not canonical_value:
                continue
            nid = _entity_node_id(node_type, canonical_value)
            if nid not in g:
                g.add_node(nid, node_type=node_type, name=canonical_value, aliases=[])
            if alias_value:
                aliases = list(g.nodes[nid].get("aliases") or [])
                if alias_value not in aliases:
                    aliases.append(alias_value)
                    g.nodes[nid]["aliases"] = aliases
            g.add_edge(claim_node, nid, edge_type=edge_type)

    _add_citation_edges(g, papers_by_id)
    _add_bibliographic_coupling_edges(g, papers_by_id)
    _add_claim_claim_edges(g, claims, papers_by_id)
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
        arxiv_id_base=paper.arxiv_id_base,
        arxiv_id_full=paper.arxiv_id_full,
        **metrics,
    )


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


def _add_bibliographic_coupling_edges(
    g: nx.MultiDiGraph, papers_by_id: dict[str, Paper]
) -> None:
    """For every pair of papers that cite at least _MIN_COUPLING_WEIGHT shared
    works, add a co_cites edge weighted by the number of shared references.
    This surfaces "implicit" relatedness — papers that rarely cite each other
    directly but draw from the same prior — and gives community-detection
    algorithms a structural signal to work with."""
    if not papers_by_id:
        return
    ref_to_citers: dict[str, set[str]] = defaultdict(set)
    for paper in papers_by_id.values():
        for ref in paper.referenced_works or []:
            ref_to_citers[ref].add(paper.paper_id)
    coupling: dict[tuple[str, str], int] = defaultdict(int)
    for citers in ref_to_citers.values():
        if len(citers) < 2:
            continue
        for a, b in combinations(sorted(citers), 2):
            coupling[(a, b)] += 1
    for (a, b), weight in coupling.items():
        if weight < _MIN_COUPLING_WEIGHT:
            continue
        node_a = f"Paper:{a}"
        node_b = f"Paper:{b}"
        if node_a in g and node_b in g:
            g.add_edge(node_a, node_b, edge_type="co_cites", weight=weight)


def _contradicts_weight(
    claim_a: Claim, claim_b: Claim, papers_by_id: dict[str, Paper]
) -> float:
    pa = papers_by_id.get(claim_a.paper_id)
    pb = papers_by_id.get(claim_b.paper_id)
    impact_a = math.log1p(int(pa.cited_by_count or 0)) if pa else 0.0
    impact_b = math.log1p(int(pb.cited_by_count or 0)) if pb else 0.0
    impact_factor = max(0.1, impact_a) * max(0.1, impact_b)
    mag_a = abs(claim_a.magnitude_value or 0.0)
    mag_b = abs(claim_b.magnitude_value or 0.0)
    mag_diff = max(0.1, abs(mag_a - mag_b))
    return round(impact_factor * mag_diff, 4)


def _add_claim_claim_edges(
    g: nx.MultiDiGraph,
    claims: list[Claim],
    papers_by_id: dict[str, Paper],
) -> None:
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
        method_canonical, _ = _resolve_entity_value(claim, "method")
        task_canonical, _ = _resolve_entity_value(claim, "task")
        method = _norm(method_canonical)
        task = _norm(task_canonical)
        if not method or not task:
            continue
        cluster = clusters[(method, task)]
        node_id = f"Claim:{claim.claim_id}"
        entry = (index, node_id, _setting_signature(claim), claim)
        if claim.direction == "positive":
            cluster["positive"].append(entry)
        elif claim.direction in ("negative", "mixed"):
            cluster["non_positive"].append(entry)

        dataset_canonical, _ = _resolve_entity_value(claim, "dataset")
        metric_canonical, _ = _resolve_entity_value(claim, "metric")
        dataset = _norm(dataset_canonical)
        metric = _norm(metric_canonical)
        if dataset and metric:
            cluster["overlap_groups"][(dataset, metric)].append(node_id)

    for cluster in clusters.values():
        positives = cluster["positive"]
        non_positives = cluster["non_positive"]
        overlap_groups = cluster["overlap_groups"]

        for pos_index, pos_node, pos_setting, pos_claim in positives:
            for other_index, other_node, other_setting, other_claim in non_positives:
                if pos_index < other_index:
                    source, target = pos_node, other_node
                    source_claim, target_claim = pos_claim, other_claim
                else:
                    source, target = other_node, pos_node
                    source_claim, target_claim = other_claim, pos_claim
                weight = _contradicts_weight(source_claim, target_claim, papers_by_id)
                g.add_edge(source, target, edge_type="contradicts", weight=weight)
                if _setting_signatures_differ(pos_setting, other_setting):
                    g.add_edge(source, target, edge_type="setting_mismatch")

        for group in overlap_groups.values():
            for node_a, node_b in combinations(group, 2):
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
    return _setting_signatures_differ(_setting_signature(a), _setting_signature(b))


def _setting_signature(claim: Claim) -> tuple[str | None, ...]:
    return tuple(_norm(getattr(claim.setting, field)) for field in SETTING_FIELDS)


def _setting_signatures_differ(
    a: tuple[str | None, ...],
    b: tuple[str | None, ...],
) -> bool:
    for va, vb in zip(a, b):
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
