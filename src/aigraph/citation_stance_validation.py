"""Validate (or downweight) ``contradicts`` claim-claim edges using the
``stance`` attribute that ``citation_stance.classify_cites_edges`` may
have written onto the corresponding ``cites`` paper-paper edge.

When paper B cites paper A with stance ``extends`` or ``builds_on``, a
claim-level ``contradicts`` edge between their claims is most likely a
false positive — B is building on A's framework, not disagreeing with
it. Conversely, ``contradicts`` / ``contrasts`` cite stances *confirm*
the conflict signal and warrant a small upweight.

This module is the first downstream consumer of the stance attribute
shipped in commit f560140. It runs unconditionally at the end of
``build_graph`` and is a no-op when no cites edges carry stance — so
graphs built without ``classify_cites_edges`` (or via the standalone
``aigraph classify-stance`` CLI) are untouched.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx


logger = logging.getLogger(__name__)


# Stance labels that suggest a NON-conflicting cite relationship between
# two papers. When B cites A with one of these, a claim-level contradicts
# edge between their claims is suspicious.
NON_CONFLICTING_STANCES = frozenset({"extends", "builds_on"})

# Stance labels that confirm a conflicting cite relationship.
CONFLICTING_STANCES = frozenset({"contradicts", "contrasts"})

# Multiplier applied to a contradicts edge's `weight` when the cite
# stance is non-conflicting. 0.3 = strongly de-prioritize without
# dropping; downstream MMR / scoring naturally surfaces real conflicts.
SHADOW_MULTIPLIER = 0.3

# Small bump when stance confirms the conflict.
CONFIRM_MULTIPLIER = 1.2


def validate_contradicts_via_stance(g: nx.MultiDiGraph) -> dict[str, int]:
    """Walk every contradicts edge in the graph, look up the parallel
    cites edge in paper space (via the ``makes`` edges from each Claim
    back to its parent Paper), and adjust the contradicts edge's
    ``weight`` plus annotate ``stance_validation`` and
    ``stance_evidence`` based on the cites stance.

    Returns counters: edges_seen, downweighted, confirmed, no_stance.

    No-op when no cites edges carry stance attributes (graphs built
    without classify_cites_edges). The first pass — building the
    paper_pair_stances index — short-circuits in that case before any
    contradicts edge is touched.
    """
    counters = {"edges_seen": 0, "downweighted": 0, "confirmed": 0, "no_stance": 0}

    # Build a lookup: sorted (paper_a, paper_b) tuple -> set of stance
    # strings observed on any cites edge between them. The sort makes the
    # lookup direction-agnostic (cites is directional but stance applies
    # to the relationship between the two papers).
    paper_pair_stances: dict[tuple[str, str], set[str]] = defaultdict(set)
    for u, v, data in g.edges(data=True):
        if data.get("edge_type") != "cites":
            continue
        stance = data.get("stance")
        if stance:
            paper_pair_stances[tuple(sorted([u, v]))].add(stance)

    if not paper_pair_stances:
        # No stance data on any cites edge — nothing to validate. Stay
        # silent (this path runs unconditionally on every build_graph).
        return counters

    # Walk contradicts edges. They live between Claim nodes; recover the
    # parent Paper node for each via the inbound `makes` edge.
    for u, v, _key, data in list(g.edges(keys=True, data=True)):
        if data.get("edge_type") != "contradicts":
            continue
        counters["edges_seen"] += 1
        paper_u = _claim_paper_node(g, u)
        paper_v = _claim_paper_node(g, v)
        if paper_u is None or paper_v is None or paper_u == paper_v:
            # Either we can't recover both parent papers, or both claims
            # came from the same paper (intra-paper contradicts — no
            # cross-paper stance to consult). Falls into the "no_stance"
            # bucket; the contradicts edge is left unchanged.
            counters["no_stance"] += 1
            continue
        stances = paper_pair_stances.get(tuple(sorted([paper_u, paper_v])))
        if not stances:
            counters["no_stance"] += 1
            continue
        if stances & NON_CONFLICTING_STANCES:
            old = float(data.get("weight", 1.0))
            data["weight"] = round(old * SHADOW_MULTIPLIER, 4)
            data["stance_validation"] = "downweighted"
            data["stance_evidence"] = sorted(stances)
            counters["downweighted"] += 1
        elif stances & CONFLICTING_STANCES:
            old = float(data.get("weight", 1.0))
            data["weight"] = round(old * CONFIRM_MULTIPLIER, 4)
            data["stance_validation"] = "confirmed"
            data["stance_evidence"] = sorted(stances)
            counters["confirmed"] += 1
        else:
            # `mentions` stance is too weak a signal in either direction.
            counters["no_stance"] += 1
    return counters


def _claim_paper_node(g: nx.MultiDiGraph, claim_node: str) -> str | None:
    """Find the Paper node whose ``makes`` edge points to this claim.

    A Claim node should have exactly one such inbound edge from a
    Paper. If the graph is corrupted (zero or multiple), pick the first
    deterministically — better to validate one stance than zero. Return
    None when no makes-edge predecessor exists at all.
    """
    for u, _, edge_data in g.in_edges(claim_node, data=True):
        if edge_data.get("edge_type") == "makes":
            return u
    return None
