"""Tests for src/aigraph/citation_stance_validation.py — using cites
edge stance to validate or downweight contradicts edges."""

from __future__ import annotations

import networkx as nx

from aigraph.citation_stance_validation import (
    CONFIRM_MULTIPLIER,
    SHADOW_MULTIPLIER,
    validate_contradicts_via_stance,
)


def _build_pair_with_contradicts(stance: str | None) -> nx.MultiDiGraph:
    """Helper: build a tiny graph with paper A, paper B, claim A1
    (positive), claim B1 (negative). cites edge B->A optionally carries
    stance. contradicts edge between the two claims has weight=2.0
    pre-validation."""
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_node("Paper:A", node_type="Paper", paper_id="A")
    g.add_node("Paper:B", node_type="Paper", paper_id="B")
    g.add_node("Claim:A1", node_type="Claim", claim_id="A1", direction="positive")
    g.add_node("Claim:B1", node_type="Claim", claim_id="B1", direction="negative")
    g.add_edge("Paper:A", "Claim:A1", edge_type="makes")
    g.add_edge("Paper:B", "Claim:B1", edge_type="makes")
    cites_attrs = {"edge_type": "cites"}
    if stance is not None:
        cites_attrs["stance"] = stance
        cites_attrs["stance_confidence"] = 0.85
    g.add_edge("Paper:B", "Paper:A", **cites_attrs)
    g.add_edge("Claim:A1", "Claim:B1", edge_type="contradicts", weight=2.0)
    return g


def test_downweight_when_cites_extends():
    """When the cites edge says B 'extends' A, the contradicts edge
    weight is multiplied by SHADOW_MULTIPLIER and annotated with
    stance_validation='downweighted'."""
    g = _build_pair_with_contradicts(stance="extends")
    counters = validate_contradicts_via_stance(g)

    assert counters["edges_seen"] == 1
    assert counters["downweighted"] == 1
    assert counters["confirmed"] == 0
    assert counters["no_stance"] == 0

    edge_data = g.get_edge_data("Claim:A1", "Claim:B1") or {}
    contradicts = [d for d in edge_data.values() if d.get("edge_type") == "contradicts"]
    assert len(contradicts) == 1
    nd = contradicts[0]
    assert nd["weight"] == round(2.0 * SHADOW_MULTIPLIER, 4)
    assert nd["stance_validation"] == "downweighted"
    assert nd["stance_evidence"] == ["extends"]


def test_downweight_when_cites_builds_on():
    """`builds_on` should also downweight."""
    g = _build_pair_with_contradicts(stance="builds_on")
    counters = validate_contradicts_via_stance(g)
    assert counters["downweighted"] == 1
    nd = list(g.get_edge_data("Claim:A1", "Claim:B1").values())[0]
    assert nd["stance_validation"] == "downweighted"


def test_confirm_when_cites_contradicts():
    """When the cites edge already says contradicts/contrasts, the
    contradicts edge is upweighted by CONFIRM_MULTIPLIER and annotated
    with stance_validation='confirmed'."""
    g = _build_pair_with_contradicts(stance="contradicts")
    counters = validate_contradicts_via_stance(g)

    assert counters["confirmed"] == 1
    assert counters["downweighted"] == 0

    edge_data = g.get_edge_data("Claim:A1", "Claim:B1") or {}
    contradicts = [d for d in edge_data.values() if d.get("edge_type") == "contradicts"]
    nd = contradicts[0]
    assert nd["weight"] == round(2.0 * CONFIRM_MULTIPLIER, 4)
    assert nd["stance_validation"] == "confirmed"
    assert nd["stance_evidence"] == ["contradicts"]


def test_mentions_stance_treated_as_no_signal():
    """`mentions` stance is too weak in either direction — leave the
    contradicts edge alone, count as no_stance."""
    g = _build_pair_with_contradicts(stance="mentions")
    counters = validate_contradicts_via_stance(g)

    assert counters["no_stance"] == 1
    assert counters["downweighted"] == 0
    assert counters["confirmed"] == 0
    nd = list(g.get_edge_data("Claim:A1", "Claim:B1").values())[0]
    assert nd["weight"] == 2.0
    assert "stance_validation" not in nd


def test_no_op_when_no_stance_data():
    """Cites edge has no stance attribute — function early-returns
    after the index pass. The contradicts edge is unchanged and
    stance_validation is not set."""
    g = _build_pair_with_contradicts(stance=None)
    counters = validate_contradicts_via_stance(g)

    # When ZERO cites edges have stance, the function bails before
    # touching any contradicts edges. Counters all start at zero.
    assert counters == {"edges_seen": 0, "downweighted": 0,
                         "confirmed": 0, "no_stance": 0}

    nd = list(g.get_edge_data("Claim:A1", "Claim:B1").values())[0]
    assert nd["weight"] == 2.0
    assert "stance_validation" not in nd


def test_intra_paper_contradicts_unaffected():
    """Two claims from the SAME paper that contradict each other have
    no cross-paper cites edge to consult. The validator falls into the
    no_stance bucket and leaves the contradicts edge unchanged."""
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    g.add_node("Paper:A", node_type="Paper", paper_id="A")
    g.add_node("Paper:B", node_type="Paper", paper_id="B")
    # Both intra-paper claims belong to A.
    g.add_node("Claim:A1", node_type="Claim", claim_id="A1", direction="positive")
    g.add_node("Claim:A2", node_type="Claim", claim_id="A2", direction="negative")
    g.add_edge("Paper:A", "Claim:A1", edge_type="makes")
    g.add_edge("Paper:A", "Claim:A2", edge_type="makes")
    # Force at least one cites edge with stance somewhere else in the
    # graph so the early-return guard doesn't skip the contradicts walk.
    g.add_node("Paper:C", node_type="Paper", paper_id="C")
    g.add_edge("Paper:B", "Paper:C", edge_type="cites", stance="extends")
    g.add_edge("Claim:A1", "Claim:A2", edge_type="contradicts", weight=2.0)

    counters = validate_contradicts_via_stance(g)
    assert counters["edges_seen"] == 1
    assert counters["no_stance"] == 1
    nd = list(g.get_edge_data("Claim:A1", "Claim:A2").values())[0]
    assert nd["weight"] == 2.0
    assert "stance_validation" not in nd
