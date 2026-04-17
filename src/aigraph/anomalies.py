"""Anomaly detection over the typed claim graph."""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

import networkx as nx

from .graph import SETTING_FIELDS, build_citation_graph
from .models import Anomaly, Claim


_STOPWORDS = {"the", "a", "an", "of", "on", "in", "for", "and", "to", "qa", "llm"}
MIN_COMMUNITY_DISCONNECT_TOPOLOGY = 0.25
_PLACEHOLDER_LABELS = {"other", "unknown", "misc", "n/a", "na", "none", "null"}


def detect_anomalies(g: nx.MultiDiGraph, claims: list[Claim]) -> list[Anomaly]:
    citation_graph = build_citation_graph(g)
    anomalies: list[Anomaly] = []
    anomalies.extend(_detect_benchmark_inconsistency(g, claims))
    anomalies.extend(_detect_setting_mismatch(claims))
    anomalies.extend(_detect_metric_mismatch(claims))
    anomalies.extend(_detect_evidence_gap(g, claims))
    anomalies.extend(_detect_community_disconnect(citation_graph, claims))
    anomalies.extend(_detect_bridge_opportunity(g, claims))

    claims_by_id = {c.claim_id: c for c in claims}
    for i, a in enumerate(anomalies, start=1):
        a.anomaly_id = f"a{i:03d}"
        _annotate_topology_scores(g, a, claims_by_id)
        nodes, edges = _local_subgraph(g, a.claim_ids)
        a.local_graph_nodes = nodes
        a.local_graph_edges = edges
    anomalies = [
        a
        for a in anomalies
        if a.type != "community_disconnect" or a.topology_score >= MIN_COMMUNITY_DISCONNECT_TOPOLOGY
    ]
    for i, a in enumerate(anomalies, start=1):
        a.anomaly_id = f"a{i:03d}"
    return anomalies


def _cluster_key(c: Claim) -> tuple[str, str] | None:
    """Prefer canonical fields unless they are generic placeholders like 'other'."""
    method = _norm(_preferred_label(c.canonical_method, c.method))
    task = _norm(_preferred_label(c.canonical_task, c.task))
    if not method or not task or method in _PLACEHOLDER_LABELS or task in _PLACEHOLDER_LABELS:
        return None
    return (method, task)


def _group_by_method_task(claims: list[Claim]) -> dict[tuple[str, str], list[Claim]]:
    groups: dict[tuple[str, str], list[Claim]] = defaultdict(list)
    for c in claims:
        key = _cluster_key(c)
        if key is None:
            continue
        groups[key].append(c)
    return groups


def _preferred_label(primary: str | None, fallback: str | None) -> str | None:
    preferred = _norm(primary)
    alternate = _norm(fallback)
    if preferred and preferred not in _PLACEHOLDER_LABELS:
        return primary.strip() if primary else None
    if alternate and alternate not in _PLACEHOLDER_LABELS:
        return fallback.strip() if fallback else None
    if preferred:
        return primary.strip() if primary else None
    if alternate:
        return fallback.strip() if fallback else None
    return None


def _display_method_label(claim: Claim) -> str:
    label = _preferred_label(claim.canonical_method, claim.method)
    if _norm(label) in _PLACEHOLDER_LABELS:
        return "this approach"
    return label or "this approach"


def _display_task_label(claim: Claim) -> str:
    label = _preferred_label(claim.canonical_task, claim.task)
    if _norm(label) in _PLACEHOLDER_LABELS:
        return "this task"
    return label or "this task"


def _detect_benchmark_inconsistency(g: nx.MultiDiGraph, claims: list[Claim]) -> list[Anomaly]:
    out: list[Anomaly] = []
    for (method, task), group in _group_by_method_task(claims).items():
        positives = [c for c in group if c.direction == "positive"]
        non_positives = [c for c in group if c.direction in ("negative", "mixed")]
        if not positives or not non_positives:
            continue
        method_label = _display_method_label(group[0])
        task_label = _display_task_label(group[0])
        pos_impact = _paper_impact_sum(g, positives)
        neg_impact = _paper_impact_sum(g, non_positives)
        anomaly_type = "impact_conflict" if pos_impact >= 2.0 and neg_impact >= 2.0 else "benchmark_inconsistency"
        prefix = "Why do high-impact papers disagree about" if anomaly_type == "impact_conflict" else "When does"
        question = (
            f"{prefix} {method_label} on {task_label}?"
            if anomaly_type == "impact_conflict"
            else f"When does {method_label} help on {task_label}, and when does it fail?"
        )
        out.append(
            Anomaly(
                anomaly_id="pending",
                type=anomaly_type,
                central_question=question,
                claim_ids=[c.claim_id for c in positives + non_positives],
                positive_claims=[c.claim_id for c in positives],
                negative_claims=[c.claim_id for c in non_positives],
                shared_entities={"method": method_label, "task": task_label},
            )
        )
    return out


def _detect_metric_mismatch(claims: list[Claim]) -> list[Anomaly]:
    out: list[Anomaly] = []
    for (_, _), group in _group_by_method_task(claims).items():
        by_metric: dict[str, list[Claim]] = defaultdict(list)
        for claim in group:
            metric = _norm(claim.metric)
            if metric:
                by_metric[metric].append(claim)
        if len(by_metric) < 2:
            continue
        directions_by_metric = {
            metric: {c.direction for c in metric_claims}
            for metric, metric_claims in by_metric.items()
        }
        all_dirs = set().union(*directions_by_metric.values())
        if not ("positive" in all_dirs and ({"negative", "mixed"} & all_dirs)):
            continue
        method_label = _display_method_label(group[0])
        task_label = _display_task_label(group[0])
        selected = [c for claims_for_metric in by_metric.values() for c in claims_for_metric]
        out.append(
            Anomaly(
                anomaly_id="pending",
                type="metric_mismatch",
                central_question=(
                    f"Do different metrics explain why {method_label} appears inconsistent on {task_label}?"
                ),
                claim_ids=[c.claim_id for c in selected],
                positive_claims=[c.claim_id for c in selected if c.direction == "positive"],
                negative_claims=[c.claim_id for c in selected if c.direction in ("negative", "mixed")],
                shared_entities={
                    "method": method_label or "",
                    "task": task_label or "",
                    "metrics": ", ".join(sorted(by_metric)),
                },
                varying_settings=["metric"],
            )
        )
    return out


def _detect_evidence_gap(g: nx.MultiDiGraph, claims: list[Claim]) -> list[Anomaly]:
    out: list[Anomaly] = []
    for (_, _), group in _group_by_method_task(claims).items():
        if len(group) < 2:
            continue
        positives = [c for c in group if c.direction == "positive"]
        non_positives = [c for c in group if c.direction in ("negative", "mixed")]
        if positives and non_positives:
            continue
        impact = _paper_impact_sum(g, group)
        recent = _paper_recent_activity_sum(g, group)
        if impact < 2.0 and recent < 1.0:
            continue
        method_label = _display_method_label(group[0])
        task_label = _display_task_label(group[0])
        missing = "negative/robustness evidence" if positives else "positive effectiveness evidence"
        out.append(
            Anomaly(
                anomaly_id="pending",
                type="evidence_gap",
                central_question=f"What evidence is missing around {method_label} on {task_label}?",
                claim_ids=[c.claim_id for c in group],
                positive_claims=[c.claim_id for c in positives],
                negative_claims=[c.claim_id for c in non_positives],
                shared_entities={
                    "method": method_label or "",
                    "task": task_label or "",
                    "missing_side": missing,
                },
            )
        )
    return out


def _detect_community_disconnect(citation_graph: nx.Graph, claims: list[Claim]) -> list[Anomaly]:
    by_community: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        community = _community_key(claim)
        if community:
            by_community[community].append(claim)
    candidates = [(key, group) for key, group in by_community.items() if len(group) >= 2]
    concepts_by_community = {key: _semantic_concepts(group) for key, group in candidates}
    out: list[Anomaly] = []
    seen: set[tuple[str, str]] = set()
    for (ka, ga), (kb, gb) in combinations(candidates, 2):
        pair = tuple(sorted((ka, kb)))
        if pair in seen:
            continue
        concepts_a = concepts_by_community[ka]
        concepts_b = concepts_by_community[kb]
        shared = concepts_a & concepts_b
        if len(shared) < 2:
            continue
        jaccard = len(shared) / max(1, len(concepts_a | concepts_b))
        if jaccard < 0.25 and not (len(shared) >= 4 and jaccard >= 0.12):
            continue
        if _citation_connected(citation_graph, ga, gb, max_distance=2):
            continue
        seen.add(pair)
        group = ga + gb
        out.append(
            Anomaly(
                anomaly_id="pending",
                type="community_disconnect",
                central_question=f"Could {ka} and {kb} be connected by a shared mechanism?",
                claim_ids=[c.claim_id for c in group],
                positive_claims=[c.claim_id for c in group if c.direction == "positive"],
                negative_claims=[c.claim_id for c in group if c.direction in ("negative", "mixed")],
                shared_entities={
                    "community_from": ka,
                    "community_to": kb,
                    "shared_concepts": ", ".join(sorted(shared)),
                },
            )
        )
    return out


def _detect_setting_mismatch(claims: list[Claim]) -> list[Anomaly]:
    out: list[Anomaly] = []
    for (_, _), group in _group_by_method_task(claims).items():
        positives = [c for c in group if c.direction == "positive"]
        non_positives = [c for c in group if c.direction in ("negative", "mixed")]
        if not positives or not non_positives:
            continue
        varying = _varying_setting_fields(positives, non_positives)
        if not varying:
            continue
        method_label = _display_method_label(group[0])
        task_label = _display_task_label(group[0])
        out.append(
            Anomaly(
                anomaly_id="pending",
                type="setting_mismatch",
                central_question=(
                    f"Which settings flip the effect of {method_label} on {task_label}?"
                ),
                claim_ids=[c.claim_id for c in positives + non_positives],
                positive_claims=[c.claim_id for c in positives],
                negative_claims=[c.claim_id for c in non_positives],
                shared_entities={"method": method_label, "task": task_label},
                varying_settings=varying,
            )
        )
    return out


def _varying_setting_fields(pos: list[Claim], neg: list[Claim]) -> list[str]:
    varying: list[str] = []
    for field in SETTING_FIELDS:
        pos_vals = {_norm(getattr(c.setting, field)) for c in pos}
        neg_vals = {_norm(getattr(c.setting, field)) for c in neg}
        pos_vals.discard(None)
        neg_vals.discard(None)
        if pos_vals and neg_vals and pos_vals != neg_vals:
            varying.append(field)
    return varying


def _detect_bridge_opportunity(g: nx.MultiDiGraph, claims: list[Claim]) -> list[Anomaly]:
    groups = _group_by_method_task(claims)
    out: list[Anomaly] = []
    seen: set[tuple[str, str]] = set()

    for (ka, group_a), (kb, group_b) in combinations(groups.items(), 2):
        # Skip pairs that share a method (already handled by other detectors on the same cluster).
        if ka[0] == kb[0] and ka[1] == kb[1]:
            continue
        tokens_a = _concept_tokens(ka)
        tokens_b = _concept_tokens(kb)
        overlap = tokens_a & tokens_b
        if not overlap:
            continue
        jaccard = len(overlap) / max(1, len(tokens_a | tokens_b))
        if jaccard < 0.15:
            continue
        if _clusters_already_connected(g, group_a, group_b):
            continue

        pair_key = tuple(sorted([f"{ka[0]}|{ka[1]}", f"{kb[0]}|{kb[1]}"]))
        if pair_key in seen:
            continue
        seen.add(pair_key)

        out.append(
            Anomaly(
                anomaly_id="pending",
                type="bridge_opportunity",
                central_question=(
                    f"Could the effect on {_display_task_label(group_a[0])} with {_display_method_label(group_a[0])} "
                    f"transfer to {_display_task_label(group_b[0])} with {_display_method_label(group_b[0])}?"
                ),
                claim_ids=[c.claim_id for c in group_a + group_b],
                positive_claims=[c.claim_id for c in (group_a + group_b) if c.direction == "positive"],
                negative_claims=[c.claim_id for c in (group_a + group_b) if c.direction in ("negative", "mixed")],
                shared_entities={
                    "method_from": _display_method_label(group_a[0]),
                    "task_from": _display_task_label(group_a[0]),
                    "method_to": _display_method_label(group_b[0]),
                    "task_to": _display_task_label(group_b[0]),
                    "shared_tokens": ", ".join(sorted(overlap)),
                },
            )
        )
    return out


def _clusters_already_connected(g: nx.MultiDiGraph, a: list[Claim], b: list[Claim]) -> bool:
    for ca in a:
        for cb in b:
            na, nb = f"Claim:{ca.claim_id}", f"Claim:{cb.claim_id}"
            if g.has_edge(na, nb) or g.has_edge(nb, na):
                return True
    return False


def _concept_tokens(key: tuple[str, str]) -> set[str]:
    method, task = key
    text = f"{method} {task}"
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {t for t in tokens if t not in _STOPWORDS and t not in _PLACEHOLDER_LABELS and len(t) > 2}


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    return v or None


def _community_key(claim: Claim) -> str | None:
    community = _norm(claim.domain) or _norm(_preferred_label(claim.canonical_task, claim.task))
    if community in _PLACEHOLDER_LABELS:
        return None
    return community


def _semantic_concepts(claims: list[Claim]) -> set[str]:
    fields = (
        "mechanism",
        "failure_mode",
        "evaluation_protocol",
        "assumption",
        "risk_type",
        "temporal_property",
        "data_modality",
    )
    concepts: set[str] = set()
    for claim in claims:
        for field in fields:
            value = _norm(getattr(claim, field))
            if value:
                concepts.add(value)
        concepts.update(_inferred_concepts(claim))
    return concepts


def _inferred_concepts(claim: Claim) -> set[str]:
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


def _paper_node(claim: Claim) -> str:
    return f"Paper:{claim.paper_id}"


def _unique_paper_nodes(claims: list[Claim]) -> set[str]:
    return {_paper_node(c) for c in claims}


def _paper_impact_sum(g: nx.MultiDiGraph, claims: list[Claim]) -> float:
    total = 0.0
    for node in _unique_paper_nodes(claims):
        if node in g:
            total += float(g.nodes[node].get("impact_score") or 0.0)
    return total


def _paper_recent_activity_sum(g: nx.MultiDiGraph, claims: list[Claim]) -> float:
    total = 0.0
    for node in _unique_paper_nodes(claims):
        if node in g:
            total += float(g.nodes[node].get("citation_velocity") or 0.0)
    return total


def _citation_connected(
    citation_graph: nx.Graph,
    a: list[Claim],
    b: list[Claim],
    max_distance: int,
) -> bool:
    sources = {node for node in _unique_paper_nodes(a) if node in citation_graph}
    targets = {node for node in _unique_paper_nodes(b) if node in citation_graph}
    if not sources or not targets:
        return False
    if sources & targets:
        return True
    return _reachable_within_distance(citation_graph, sources, targets, max_distance)


def _annotate_topology_scores(
    g: nx.MultiDiGraph,
    anomaly: Anomaly,
    claims_by_id: dict[str, Claim],
) -> None:
    claims = [claims_by_id[cid] for cid in anomaly.claim_ids if cid in claims_by_id]
    if not claims:
        return
    pos = [claims_by_id[cid] for cid in anomaly.positive_claims if cid in claims_by_id]
    neg = [claims_by_id[cid] for cid in anomaly.negative_claims if cid in claims_by_id]
    pos_impact = _paper_impact_sum(g, pos)
    neg_impact = _paper_impact_sum(g, neg)
    evidence_impact = _paper_impact_sum(g, claims)
    recent_activity = _paper_recent_activity_sum(g, claims)
    max_side = max(pos_impact, neg_impact)
    balance = min(pos_impact, neg_impact) / max_side if max_side else 0.0
    anomaly.evidence_impact = round(evidence_impact, 4)
    anomaly.recent_activity = round(recent_activity, 4)
    anomaly.impact_balance = round(balance, 4)
    anomaly.citation_bridge_score = 1.0 if anomaly.type in ("community_disconnect", "bridge_opportunity") else 0.0
    impact_norm = min(1.0, evidence_impact / 8.0)
    activity_norm = min(1.0, recent_activity / 4.0)
    anomaly.topology_score = round(
        min(1.0, 0.45 * impact_norm + 0.25 * activity_norm + 0.2 * balance + 0.1 * anomaly.citation_bridge_score),
        4,
    )


def _local_subgraph(g: nx.MultiDiGraph, claim_ids: list[str]) -> tuple[list[str], list[dict]]:
    seed_nodes = {f"Claim:{cid}" for cid in claim_ids if f"Claim:{cid}" in g}
    neighbors: set[str] = set(seed_nodes)
    for node in seed_nodes:
        neighbors.update(g.predecessors(node))
        neighbors.update(g.successors(node))
    paper_neighbors = {n for n in neighbors if str(n).startswith("Paper:")}
    for node in paper_neighbors:
        for _, target, data in g.out_edges(node, data=True):
            if data.get("edge_type") == "cites":
                neighbors.add(target)
        for source, _, data in g.in_edges(node, data=True):
            if data.get("edge_type") == "cites":
                neighbors.add(source)
    neighbor_set = neighbors
    edges: list[dict] = []
    append_edge = edges.append
    for source in neighbor_set:
        for target, keyed_edges in g.succ.get(source, {}).items():
            if target not in neighbor_set:
                continue
            for data in keyed_edges.values():
                append_edge(
                    {
                        "source": source,
                        "target": target,
                        "edge_type": data.get("edge_type", "related"),
                    }
                )
    return sorted(neighbor_set), edges


def _reachable_within_distance(
    citation_graph: nx.Graph,
    sources: set[str],
    targets: set[str],
    max_distance: int,
) -> bool:
    for source in sources:
        frontier = {source}
        seen = {source}
        for _ in range(max_distance):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbor in citation_graph.adj[node]:
                    if neighbor in targets:
                        return True
                    if neighbor not in seen:
                        seen.add(neighbor)
                        next_frontier.add(neighbor)
            if not next_frontier:
                break
            frontier = next_frontier
    return False
