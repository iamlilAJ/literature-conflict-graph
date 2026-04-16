"""Anomaly detection over the typed claim graph."""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

import networkx as nx

from .graph import SETTING_FIELDS
from .models import Anomaly, Claim


_STOPWORDS = {"the", "a", "an", "of", "on", "in", "for", "and", "to", "qa", "llm"}


def detect_anomalies(g: nx.MultiDiGraph, claims: list[Claim]) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    anomalies.extend(_detect_benchmark_inconsistency(claims))
    anomalies.extend(_detect_setting_mismatch(claims))
    anomalies.extend(_detect_bridge_opportunity(g, claims))

    for i, a in enumerate(anomalies, start=1):
        a.anomaly_id = f"a{i:03d}"
        nodes, edges = _local_subgraph(g, a.claim_ids)
        a.local_graph_nodes = nodes
        a.local_graph_edges = edges
    return anomalies


def _cluster_key(c: Claim) -> tuple[str, str] | None:
    """Prefer canonical fields (LLM-tagged) over free-text method/task."""
    method = (c.canonical_method or c.method or "").strip().lower()
    task = (c.canonical_task or c.task or "").strip().lower()
    if not method or not task:
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


def _detect_benchmark_inconsistency(claims: list[Claim]) -> list[Anomaly]:
    out: list[Anomaly] = []
    for (method, task), group in _group_by_method_task(claims).items():
        positives = [c for c in group if c.direction == "positive"]
        non_positives = [c for c in group if c.direction in ("negative", "mixed")]
        if not positives or not non_positives:
            continue
        method_label = group[0].canonical_method or group[0].method
        task_label = group[0].canonical_task or group[0].task
        out.append(
            Anomaly(
                anomaly_id="pending",
                type="benchmark_inconsistency",
                central_question=(
                    f"When does {method_label} help on {task_label}, "
                    "and when does it fail?"
                ),
                claim_ids=[c.claim_id for c in positives + non_positives],
                positive_claims=[c.claim_id for c in positives],
                negative_claims=[c.claim_id for c in non_positives],
                shared_entities={"method": method_label, "task": task_label},
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
        method_label = group[0].canonical_method or group[0].method
        task_label = group[0].canonical_task or group[0].task
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
                    f"Could the effect on {group_a[0].task} with {group_a[0].method} "
                    f"transfer to {group_b[0].task} with {group_b[0].method}?"
                ),
                claim_ids=[c.claim_id for c in group_a + group_b],
                positive_claims=[c.claim_id for c in (group_a + group_b) if c.direction == "positive"],
                negative_claims=[c.claim_id for c in (group_a + group_b) if c.direction in ("negative", "mixed")],
                shared_entities={
                    "method_from": group_a[0].method or "",
                    "task_from": group_a[0].task or "",
                    "method_to": group_b[0].method or "",
                    "task_to": group_b[0].task or "",
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
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _norm(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    return v or None


def _local_subgraph(g: nx.MultiDiGraph, claim_ids: list[str]) -> tuple[list[str], list[dict]]:
    seed_nodes = {f"Claim:{cid}" for cid in claim_ids if f"Claim:{cid}" in g}
    neighbors: set[str] = set(seed_nodes)
    for node in seed_nodes:
        neighbors.update(g.predecessors(node))
        neighbors.update(g.successors(node))
    sub = g.subgraph(neighbors)
    nodes = sorted(sub.nodes)
    edges = [
        {"source": u, "target": v, "edge_type": data.get("edge_type", "related")}
        for u, v, data in sub.edges(data=True)
    ]
    return nodes, edges
