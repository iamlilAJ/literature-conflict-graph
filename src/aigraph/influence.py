"""Phase 1 of hypothesis influence prediction — 4 cheap dimensions.

Implements 4 of the 5 dimensions from
``docs/influence-prediction-design.md``:

  I_2  Community Reach        — span across communities/clusters
  I_4  Novelty vs External    — consume existing novelty_check field
  I_5  Grounding Depth        — average evidence quality of cited claims
  Risk_scope  Overreach       — Jaccard mismatch between claimed scope
                                and observed scope of explains_claims

Phase 2 will add I_1 Structural Impact (requires graph_repair.py).
Phase 3 will add I_3 Test Efficiency (requires LLM-judge).

All Phase 1 dimensions are deterministic and non-LLM. Output is an
``InfluenceScore`` named tuple with a per-dimension breakdown so the
caller can render explainable rankings (see design doc §4 for the
intended UI).

The module is dependency-light on purpose: it does NOT import
``hierarchy.py`` or ``novelty_check.py`` — both live on still-in-flight
feature branches. Hierarchy is consumed as a plain dict (use
``_load_hierarchy_dict`` to load from disk), and novelty is read via
``getattr(h, "novelty_check", None)`` which returns None when the
field is absent (graceful "unknown" fallback).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple, Optional

from .models import Claim, Hypothesis


logger = logging.getLogger(__name__)


# Default per-dimension weights. Tunable; the validation script can
# grid-search alternatives. Sum is 1.0 by construction (overreach is
# applied as ``1 - overreach`` so its contribution is a bonus when
# scope matches grounding well).
WEIGHTS_PHASE1: dict[str, float] = {
    "community_reach": 0.25,    # I_2
    "novelty": 0.30,            # I_4
    "grounding_depth": 0.25,    # I_5
    "scope_overreach": 0.20,    # Risk_scope (subtractive)
}


_EMPTY_HIERARCHY: dict = {
    "domains": {},
    "communities": {},
    "clusters": {},
    "cluster_to_community": {},
    "anomaly_to_cluster": {},
}


class InfluenceScore(NamedTuple):
    """Phase 1 influence breakdown. All score fields are in [0, 1].

    Higher is better. ``scope_overreach_risk`` is the raw risk (also in
    [0, 1]); the contribution to ``total`` is ``(1 - risk)`` weighted
    by ``WEIGHTS_PHASE1["scope_overreach"]``.
    """

    community_reach: float       # I_2
    novelty: float               # I_4
    grounding_depth: float       # I_5
    scope_overreach_risk: float  # Risk_scope (subtractive — see docs)
    total: float                 # weighted combination, in [0, 1]
    # Diagnostics for explainable ranking output:
    n_communities_touched: int
    n_inspired_claims: int
    is_novel: Optional[bool]     # None = unknown (no novelty_check)
    n_similar_papers: int


def _load_hierarchy_dict(path: str | Path) -> dict:
    """Load a hierarchy JSON from disk; return the empty schema on
    missing or malformed file. Inlined here rather than importing
    ``hierarchy.load_hierarchy`` because that module lives on a
    different (still-in-flight) feature branch — this PR must run on
    a checkout from origin/main.
    """
    p = Path(path)
    if not p.exists():
        return dict(_EMPTY_HIERARCHY)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_EMPTY_HIERARCHY)


def _build_claim_to_cluster_index(hierarchy: dict) -> dict[str, str]:
    """Build a one-shot ``claim_id -> cluster_key`` lookup.

    The naive approach in the design (iterate ``hierarchy["clusters"]``
    per claim) is O(N_clusters) per claim. With ~1500 clusters and ~5
    claims per hypothesis on a 95-hypothesis run, that is ~700K dict
    operations. Building this index once amortizes the cost to O(N)
    overall.
    """
    index: dict[str, str] = {}
    for cluster_key, cinfo in (hierarchy.get("clusters") or {}).items():
        for c_id in (cinfo.get("claim_ids") or []):
            index.setdefault(c_id, cluster_key)
    return index


def community_reach(
    h: Hypothesis,
    hierarchy: dict,
    *,
    claim_to_cluster: Optional[dict[str, str]] = None,
) -> tuple[float, int]:
    """Fraction of corpus communities the hypothesis grounds in.

    Returns ``(reach, n_communities_touched)`` where ``reach`` is in
    ``[0, 1]``. A hypothesis whose ``explains_claims`` cover N out of K
    total communities scores ``N / K``.

    When ``claim_to_cluster`` is provided (e.g. by ``predict_influence_batch``
    which builds the index once), uses it for O(1) lookup. Otherwise
    builds it on the fly.
    """
    inspired_ids = list(h.explains_claims or [])
    if not inspired_ids:
        return 0.0, 0

    cluster_to_community = hierarchy.get("cluster_to_community") or {}
    total_communities = len(hierarchy.get("communities") or {})
    if total_communities == 0:
        return 0.0, 0

    index = claim_to_cluster if claim_to_cluster is not None else _build_claim_to_cluster_index(hierarchy)

    communities_touched: set[str] = set()
    for c_id in inspired_ids:
        cluster_key = index.get(c_id)
        if cluster_key is None:
            continue
        community = cluster_to_community.get(cluster_key)
        if community:
            communities_touched.add(community)

    return len(communities_touched) / total_communities, len(communities_touched)


def novelty_score(h: Hypothesis) -> tuple[float, Optional[bool], int]:
    """Read the ``novelty_check`` extras attribute (set by the
    arxiv-novelty-check pipeline) and convert to a score in [0, 1].

    Returns ``(score, is_novel, n_similar_papers)``.

    Score semantics:
      - novelty_check missing -> (0.5, None, 0) — neutral fallback.
      - is_novel=False -> (0.0, False, len(similar)).
      - is_novel=True with N similar papers -> (1 / (1 + N), True, N).

    The novelty_check field is an extras key on Hypothesis (LooseModel
    extra="ignore") and is only populated when the hypothesis was
    loaded from a JSONL emitted by the check-novelty CLI as raw dict.
    Hypotheses produced fresh by other pipelines have no such field;
    this function treats that case as "unknown" (score 0.5).
    """
    nc = getattr(h, "novelty_check", None)
    if nc is None or not isinstance(nc, dict):
        return 0.5, None, 0

    is_n = nc.get("is_novel")
    similar = nc.get("similar_papers") or []
    n_sim = len(similar) if isinstance(similar, list) else 0

    if is_n is False:
        return 0.0, False, n_sim
    if is_n is True:
        return 1.0 / (1.0 + n_sim), True, n_sim
    # is_novel is None / unknown / missing
    return 0.5, None, n_sim


def compute_evidence_quality(c: Claim) -> float:
    """Q(c) per design doc §I_5 — heuristic claim-quality score in [0, 1].

    Boosters above the 0.5 baseline:
      - substantive evidence_span (>50 chars):           +0.20
      - canonical_method or canonical_task is set:       +0.10
      - dataset_canonical is set:                        +0.05
      - magnitude_value is not None (concrete number):   +0.10

    Note: an earlier draft of this design referenced ``Claim.confidence``,
    but that field does not exist on the Claim model. ``magnitude_value``
    is used as the strongest available signal of concreteness.
    """
    score = 0.5
    if len(c.evidence_span or "") > 50:
        score += 0.20
    if c.canonical_method or c.canonical_task:
        score += 0.10
    if c.dataset_canonical:
        score += 0.05
    if c.magnitude_value is not None:
        score += 0.10
    return min(1.0, score)


def grounding_depth(
    h: Hypothesis,
    claims_by_id: dict[str, Claim],
) -> float:
    """Average ``compute_evidence_quality`` across explains_claims.

    Returns 0.0 when ``explains_claims`` is empty or none of its ids
    resolve in ``claims_by_id``.
    """
    explained = list(h.explains_claims or [])
    if not explained:
        return 0.0
    qualities = [
        compute_evidence_quality(claims_by_id[c_id])
        for c_id in explained
        if c_id in claims_by_id
    ]
    if not qualities:
        return 0.0
    return sum(qualities) / len(qualities)


def scope_overreach(
    h: Hypothesis,
    claims_by_id: dict[str, Claim],
) -> float:
    """Risk_scope: fraction of declared ``scope_conditions`` not
    supported by ``explains_claims``.

    Returns a value in ``[0, 1]``. 0.0 means every scope claim has at
    least one matching observation in the cited evidence; 1.0 means
    none do (full overreach).

    Edge cases:
      - ``scope_conditions`` is None or empty: 0.0 (no scope, no
        overreach).
      - ``explains_claims`` empty or none resolve: 1.0 (any declared
        scope is unsupported).
    """
    claimed_scope_dict = h.scope_conditions or {}
    if not claimed_scope_dict:
        return 0.0

    claimed_scope: set[tuple[str, str]] = {
        (str(k).strip().lower(), str(v).strip().lower())
        for k, v in claimed_scope_dict.items()
        if str(v).strip()
    }
    if not claimed_scope:
        return 0.0

    observed_scope: set[tuple[str, str]] = set()
    for c_id in (h.explains_claims or []):
        c = claims_by_id.get(c_id)
        if c is None:
            continue
        if c.canonical_task:
            observed_scope.add(("task", c.canonical_task.strip().lower()))
        if c.canonical_method:
            observed_scope.add(("method", c.canonical_method.strip().lower()))
        if c.dataset_canonical:
            observed_scope.add(("dataset", c.dataset_canonical.strip().lower()))
        if c.domain:
            observed_scope.add(("domain", c.domain.strip().lower()))

    if not observed_scope:
        return 1.0

    intersection = claimed_scope & observed_scope
    return 1.0 - (len(intersection) / len(claimed_scope))


def predict_influence_phase1(
    h: Hypothesis,
    hierarchy: dict,
    claims_by_id: dict[str, Claim],
    *,
    weights: Optional[dict[str, float]] = None,
    claim_to_cluster: Optional[dict[str, str]] = None,
) -> InfluenceScore:
    """Compute the Phase 1 (4-dim) influence score for one hypothesis.

    ``weights`` defaults to ``WEIGHTS_PHASE1``. ``claim_to_cluster`` is
    an optional pre-built index from
    ``_build_claim_to_cluster_index(hierarchy)``; pass it from
    ``predict_influence_batch`` to amortize the index build across many
    hypotheses.

    The combination formula: ``total = w_reach * reach + w_nov * nov +
    w_ground * grounding + w_scope * (1 - overreach)``. Note that the
    overreach contribution is ``(1 - risk)`` — low overreach raises the
    total, high overreach lowers it.
    """
    weights = weights or WEIGHTS_PHASE1
    reach, n_comms = community_reach(h, hierarchy, claim_to_cluster=claim_to_cluster)
    nov, is_nov, n_sim = novelty_score(h)
    grounding = grounding_depth(h, claims_by_id)
    overreach = scope_overreach(h, claims_by_id)

    total = (
        weights.get("community_reach", 0.0) * reach
        + weights.get("novelty", 0.0) * nov
        + weights.get("grounding_depth", 0.0) * grounding
        + weights.get("scope_overreach", 0.0) * (1.0 - overreach)
    )

    return InfluenceScore(
        community_reach=reach,
        novelty=nov,
        grounding_depth=grounding,
        scope_overreach_risk=overreach,
        total=total,
        n_communities_touched=n_comms,
        n_inspired_claims=len(h.explains_claims or []),
        is_novel=is_nov,
        n_similar_papers=n_sim,
    )


def predict_influence_batch(
    hypotheses: list[Hypothesis],
    hierarchy: dict,
    claims_by_id: dict[str, Claim],
    *,
    weights: Optional[dict[str, float]] = None,
) -> list[InfluenceScore]:
    """Batch wrapper. Builds the ``claim_to_cluster`` index once and
    reuses it across all hypotheses.

    Phase 1 has no expensive operations; this batching exists for
    consistency and to make Phase 3's LLM-judge cache straightforward
    to slot in (the cache will be keyed on hypothesis_id).
    """
    index = _build_claim_to_cluster_index(hierarchy)
    return [
        predict_influence_phase1(
            h, hierarchy, claims_by_id,
            weights=weights, claim_to_cluster=index,
        )
        for h in hypotheses
    ]
