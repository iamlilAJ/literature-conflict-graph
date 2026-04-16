"""Typed claim graph built from extracted claims."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import networkx as nx

from .models import Claim

# Map entity fields on Claim -> (node type, edge type)
ENTITY_EDGES: dict[str, tuple[str, str]] = {
    "method": ("Method", "uses"),
    "model": ("Model", "evaluated_with"),
    "task": ("Task", "targets"),
    "dataset": ("Dataset", "evaluated_on"),
    "metric": ("Metric", "measured_by"),
    "baseline": ("Baseline", "compares_against"),
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


def build_graph(claims: list[Claim]) -> nx.MultiDiGraph:
    g: nx.MultiDiGraph = nx.MultiDiGraph()

    seen_papers: set[str] = set()
    for claim in claims:
        paper_node = f"Paper:{claim.paper_id}"
        claim_node = f"Claim:{claim.claim_id}"
        if claim.paper_id not in seen_papers:
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
        )
        g.add_edge(paper_node, claim_node, edge_type="makes")

        for field, (node_type, edge_type) in ENTITY_EDGES.items():
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

    _add_claim_claim_edges(g, claims)
    return g


def _add_claim_claim_edges(g: nx.MultiDiGraph, claims: list[Claim]) -> None:
    for a, b in combinations(claims, 2):
        ma, mb = _norm(a.method), _norm(b.method)
        ta, tb = _norm(a.task), _norm(b.task)
        if ma and mb and ta and tb and ma == mb and ta == tb:
            da, db = a.direction, b.direction
            if _opposite(da, db):
                na, nb = f"Claim:{a.claim_id}", f"Claim:{b.claim_id}"
                g.add_edge(na, nb, edge_type="contradicts")
                if _settings_differ(a, b):
                    g.add_edge(na, nb, edge_type="setting_mismatch")
            if (
                _norm(a.dataset) and _norm(a.dataset) == _norm(b.dataset)
                and _norm(a.metric) and _norm(a.metric) == _norm(b.metric)
            ):
                g.add_edge(f"Claim:{a.claim_id}", f"Claim:{b.claim_id}", edge_type="overlap")


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
