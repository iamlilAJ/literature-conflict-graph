"""Per-level statistical aggregation of the typed claim graph.

Three levels:

- domain: by ``Claim.domain`` (lowercased); claims without a domain or with
  a placeholder ("other"/"unknown"/...) land in "uncategorized". Each
  domain reports paper/claim counts, top methods/tasks, anomaly-type
  distribution, and sample claim ids.
- community: Louvain partitions of the paper-only citation subgraph
  (``build_citation_graph`` + ``nx.algorithms.community.louvain_communities``,
  ``seed=42``). Singletons (size < 2) are dropped.
- cluster: by ``(canonical_method/method, canonical_task/task)`` via the
  existing ``_cluster_key`` helper.

Plus two cross-level mappings:

- ``cluster_to_community`` — majority paper-vote.
- ``anomaly_to_cluster`` — from ``anomaly.shared_entities``;
  ``community_disconnect`` anomalies are skipped (they are inter-community
  by construction).

Pure statistics; NO LLM calls. The hierarchy file is metadata + IDs, not
full claim/paper content. ~5 MB cap at 1000-paper scale.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from .anomalies import _cluster_key
from .graph import build_citation_graph
from .models import Anomaly, Claim, Paper


logger = logging.getLogger(__name__)


_UNCATEGORIZED = "uncategorized"
_MIN_COMMUNITY_SIZE = 2
_LOUVAIN_SEED = 42

# Domain values treated as missing — same set as anomalies._PLACEHOLDER_LABELS,
# inlined here to avoid coupling hierarchy bucketing to the cluster_key fallback
# (which folds into canonical_task and would mask uncategorized claims).
_PLACEHOLDER_DOMAINS = frozenset({
    "other", "unknown", "misc", "n/a", "na", "none", "null"
})


def build_hierarchy(
    claims: list[Claim],
    papers: list[Paper],
    anomalies: list[Anomaly],
    graph: nx.MultiDiGraph,
) -> dict[str, Any]:
    """Compute domain/community/cluster aggregations.

    Returns a dict with keys: ``domains``, ``communities``, ``clusters``,
    ``cluster_to_community``, ``anomaly_to_cluster``.
    """
    claims_by_id = {c.claim_id: c for c in claims}
    domains = _build_domains(claims, anomalies, claims_by_id)
    clusters = _build_clusters(claims, anomalies)
    communities, paper_to_community = _build_communities(graph)
    cluster_to_community = _cluster_to_community_map(
        clusters, claims_by_id, paper_to_community
    )
    anomaly_to_cluster = _anomaly_to_cluster_map(anomalies)
    _enrich_communities(
        communities, clusters, anomalies, cluster_to_community
    )
    return {
        "domains": domains,
        "communities": communities,
        "clusters": clusters,
        "cluster_to_community": cluster_to_community,
        "anomaly_to_cluster": anomaly_to_cluster,
    }


def _domain_label(claim: Claim) -> str:
    raw = (claim.domain or "").strip().lower()
    if not raw or raw in _PLACEHOLDER_DOMAINS:
        return _UNCATEGORIZED
    return raw


def _build_domains(
    claims: list[Claim],
    anomalies: list[Anomaly],
    claims_by_id: dict[str, Claim],
) -> dict[str, dict]:
    paper_ids: dict[str, set[str]] = defaultdict(set)
    claim_ids: dict[str, list[str]] = defaultdict(list)
    methods: dict[str, Counter] = defaultdict(Counter)
    tasks: dict[str, Counter] = defaultdict(Counter)
    anomaly_types: dict[str, Counter] = defaultdict(Counter)

    for c in claims:
        d = _domain_label(c)
        paper_ids[d].add(c.paper_id)
        claim_ids[d].append(c.claim_id)
        method = c.canonical_method or c.method
        if method:
            methods[d][method] += 1
        task = c.canonical_task or c.task
        if task:
            tasks[d][task] += 1

    for a in anomalies:
        votes: Counter = Counter()
        for cid in a.claim_ids:
            cl = claims_by_id.get(cid)
            if cl is None:
                continue
            votes[_domain_label(cl)] += 1
        if not votes:
            continue
        top = votes.most_common(1)[0][0]
        anomaly_types[top][a.type] += 1

    out: dict[str, dict] = {}
    for d in sorted(set(claim_ids) | set(paper_ids)):
        ids = sorted(claim_ids[d])
        out[d] = {
            "paper_count": len(paper_ids[d]),
            "claim_count": len(claim_ids[d]),
            "top_methods": [m for m, _ in methods[d].most_common(5)],
            "top_tasks": [t for t, _ in tasks[d].most_common(5)],
            "anomaly_type_counts": dict(anomaly_types[d]),
            "sample_claim_ids": ids[:5],
        }
    return out


def _build_clusters(
    claims: list[Claim],
    anomalies: list[Anomaly],
) -> dict[str, dict]:
    cluster_to_claims: dict[str, list[str]] = defaultdict(list)
    cluster_to_papers: dict[str, set[str]] = defaultdict(set)
    cluster_to_anomalies: dict[str, list[str]] = defaultdict(list)

    for c in claims:
        key = _cluster_key(c)
        if key is None:
            continue
        cluster_id = f"{key[0]}__{key[1]}"
        cluster_to_claims[cluster_id].append(c.claim_id)
        cluster_to_papers[cluster_id].add(c.paper_id)

    for a in anomalies:
        if a.type == "community_disconnect":
            continue
        method = (a.shared_entities or {}).get("method", "").strip().lower()
        task = (a.shared_entities or {}).get("task", "").strip().lower()
        if not method or not task:
            continue
        cluster_id = f"{method}__{task}"
        if cluster_id in cluster_to_claims:
            cluster_to_anomalies[cluster_id].append(a.anomaly_id)

    out: dict[str, dict] = {}
    for cluster_id in sorted(cluster_to_claims):
        cids = cluster_to_claims[cluster_id]
        out[cluster_id] = {
            "claim_ids": cids,
            "anomaly_ids": cluster_to_anomalies.get(cluster_id, []),
            "paper_count": len(cluster_to_papers[cluster_id]),
            "sample_claim_ids": cids[:5],
        }
    return out


def _build_communities(
    graph: nx.MultiDiGraph,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Run Louvain on the paper-only citation subgraph.

    Returns ``(communities, paper_to_community)`` where ``communities[cid]``
    has ``paper_ids``, ``paper_count``, ``top_concepts`` (filled later),
    ``anomaly_count`` (filled later); singletons are dropped.
    """
    citation = build_citation_graph(graph)
    if citation.number_of_nodes() == 0:
        return {}, {}
    try:
        partitions = nx.algorithms.community.louvain_communities(
            citation, seed=_LOUVAIN_SEED
        )
    except Exception as exc:  # pragma: no cover — fallback for older networkx
        logger.warning("louvain_communities failed (%s); falling back to connected_components", exc)
        partitions = list(nx.connected_components(citation))

    communities: dict[str, dict] = {}
    paper_to_community: dict[str, str] = {}
    idx = 0
    for members in partitions:
        if len(members) < _MIN_COMMUNITY_SIZE:
            continue
        cid = f"c{idx:03d}"
        idx += 1
        sorted_members = sorted(members)
        communities[cid] = {
            "paper_ids": sorted_members,
            "paper_count": len(sorted_members),
            "top_concepts": [],
            "anomaly_count": 0,
        }
        for m in sorted_members:
            paper_to_community[m] = cid
    return communities, paper_to_community


def _cluster_to_community_map(
    clusters: dict[str, dict],
    claims_by_id: dict[str, Claim],
    paper_to_community: dict[str, str],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for cluster_id, cdata in clusters.items():
        votes: Counter = Counter()
        for cid in cdata["claim_ids"]:
            cl = claims_by_id.get(cid)
            if cl is None:
                continue
            community_id = paper_to_community.get(f"Paper:{cl.paper_id}")
            if community_id:
                votes[community_id] += 1
        if not votes:
            continue
        top_count = votes.most_common(1)[0][1]
        winners = sorted(c for c, n in votes.items() if n == top_count)
        out[cluster_id] = winners[0]
    return out


def _anomaly_to_cluster_map(anomalies: list[Anomaly]) -> dict[str, str]:
    out: dict[str, str] = {}
    for a in anomalies:
        if a.type == "community_disconnect":
            continue
        method = (a.shared_entities or {}).get("method", "").strip().lower()
        task = (a.shared_entities or {}).get("task", "").strip().lower()
        if method and task:
            out[a.anomaly_id] = f"{method}__{task}"
    return out


def _enrich_communities(
    communities: dict[str, dict],
    clusters: dict[str, dict],
    anomalies: list[Anomaly],
    cluster_to_community: dict[str, str],
) -> None:
    cluster_counter: dict[str, Counter] = defaultdict(Counter)
    for cluster_id, community_id in cluster_to_community.items():
        weight = len(clusters.get(cluster_id, {}).get("claim_ids", []))
        cluster_counter[community_id][cluster_id] += weight

    for community_id, counter in cluster_counter.items():
        if community_id in communities:
            communities[community_id]["top_concepts"] = [
                k for k, _ in counter.most_common(5)
            ]

    for a in anomalies:
        if a.type == "community_disconnect":
            continue
        method = (a.shared_entities or {}).get("method", "").strip().lower()
        task = (a.shared_entities or {}).get("task", "").strip().lower()
        if not method or not task:
            continue
        cluster_id = f"{method}__{task}"
        community_id = cluster_to_community.get(cluster_id)
        if community_id and community_id in communities:
            communities[community_id]["anomaly_count"] += 1


def _json_default(o: Any) -> Any:
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"object of type {type(o).__name__} is not JSON serializable")


def save_hierarchy(hierarchy: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(hierarchy, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_hierarchy(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        return _empty_hierarchy()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_hierarchy()


def _empty_hierarchy() -> dict:
    return {
        "domains": {},
        "communities": {},
        "clusters": {},
        "cluster_to_community": {},
        "anomaly_to_cluster": {},
    }
