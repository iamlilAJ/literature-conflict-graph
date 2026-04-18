"""Structured scoring + MMR selection for candidate hypotheses."""

from __future__ import annotations

import re

from .models import Anomaly, Claim, Hypothesis, ScoreBreakdown


# Utility weights (tweak here).
W_EXPLAIN = 0.28
W_GROUNDING = 0.14
W_TESTABILITY = 0.18
W_NOVELTY = 0.12
W_DISCRIMINABILITY = 0.12
W_IMPACT = 0.10
W_TOPOLOGY = 0.06
W_COST = 0.15


_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "by", "with",
    "is", "are", "be", "this", "that", "it", "as", "at", "from", "than", "when",
    "which", "does", "do", "we", "they", "their", "its", "via", "per",
}

_VAGUE_PHRASES = [
    "may ", "possibly", "various factors", "complex interactions", "could ",
    "somewhat", "perhaps", "tends to", "in general",
]


def tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def explain_score(h: Hypothesis, claims_by_id: dict[str, Claim]) -> float:
    cids = [c for c in h.explains_claims if c in claims_by_id]
    if not cids:
        return 0.0
    pos = sum(1 for c in cids if claims_by_id[c].direction == "positive")
    non_pos = sum(1 for c in cids if claims_by_id[c].direction in ("negative", "mixed"))
    coverage = min(1.0, len(cids) / 4.0)
    balance = 1.0 if (pos and non_pos) else 0.5
    return coverage * balance


def grounding_score(h: Hypothesis, anomaly: Anomaly, claims_by_id: dict[str, Claim]) -> float:
    if not h.explains_claims:
        return 0.0
    exists = sum(1 for c in h.explains_claims if c in claims_by_id)
    in_anomaly = sum(1 for c in h.explains_claims if c in anomaly.claim_ids)
    base = 0.5 * (exists / len(h.explains_claims)) + 0.5 * (in_anomaly / len(h.explains_claims))
    coherence = topical_coherence_score(h, anomaly, claims_by_id)
    return base * (0.35 + 0.65 * coherence)


def testability_score(h: Hypothesis) -> float:
    has_predictions = len(h.predictions) >= 2
    has_test = bool(h.minimal_test.strip())
    return (0.6 if has_predictions else 0.0) + (0.4 if has_test else 0.0)


def novelty_score(h: Hypothesis, claims_by_id: dict[str, Claim]) -> float:
    if not claims_by_id:
        return 1.0
    h_tokens = tokenize(h.hypothesis)
    max_sim = 0.0
    for c in claims_by_id.values():
        sim = jaccard(h_tokens, tokenize(c.claim_text))
        if sim > max_sim:
            max_sim = sim
    return 1.0 - max_sim


def cost_penalty(h: Hypothesis) -> float:
    text = f"{h.hypothesis} {h.mechanism}".lower()
    hits = sum(1 for phrase in _VAGUE_PHRASES if phrase in text)
    return min(1.0, hits / 4.0)


def _claim_context_text(claim: Claim) -> str:
    parts = [
        claim.claim_text,
        claim.method,
        claim.task,
        claim.dataset,
        claim.metric,
        claim.baseline,
        claim.result,
        claim.canonical_method,
        claim.canonical_task,
        claim.domain,
        claim.mechanism,
        claim.failure_mode,
        claim.evaluation_protocol,
    ]
    if claim.setting:
        parts.extend(
            [
                claim.setting.retriever,
                claim.setting.top_k,
                claim.setting.context_length,
                claim.setting.task_type,
            ]
        )
    return " ".join(str(part or "") for part in parts if part)


def _anomaly_context_tokens(anomaly: Anomaly, claims_by_id: dict[str, Claim]) -> set[str]:
    parts = [anomaly.central_question]
    parts.extend(str(value or "") for value in (anomaly.shared_entities or {}).values())
    parts.extend(anomaly.varying_settings or [])
    for claim_id in anomaly.claim_ids:
        claim = claims_by_id.get(claim_id)
        if claim is not None:
            parts.append(_claim_context_text(claim))
    return tokenize(" ".join(parts))


def topical_coherence_score(h: Hypothesis, anomaly: Anomaly, claims_by_id: dict[str, Claim]) -> float:
    evidence_tokens = _anomaly_context_tokens(anomaly, claims_by_id)
    if not evidence_tokens:
        return 1.0
    hypothesis_tokens = tokenize(" ".join(filter(None, [h.hypothesis, h.mechanism])))
    if not hypothesis_tokens:
        return 0.0
    overlap = hypothesis_tokens & evidence_tokens
    overlap_ratio = len(overlap) / max(1, len(hypothesis_tokens))
    entity_tokens = tokenize(" ".join(str(value or "") for value in (anomaly.shared_entities or {}).values()))
    entity_overlap = len(hypothesis_tokens & entity_tokens) / max(1, len(entity_tokens)) if entity_tokens else 0.0
    score = min(1.0, overlap_ratio * 2.2 + entity_overlap * 0.6)
    if not overlap:
        return 0.0
    if overlap_ratio < 0.12 and entity_overlap == 0.0:
        score *= 0.5
    return score


def _hypothesis_tokens(h: Hypothesis) -> set[str]:
    return tokenize(" ".join([h.hypothesis, *h.predictions]))


def discriminability_score(h: Hypothesis, peers: list[Hypothesis]) -> float:
    others = [p for p in peers if p.hypothesis_id != h.hypothesis_id]
    if not others:
        return 1.0
    h_tokens = _hypothesis_tokens(h)
    sims = [jaccard(h_tokens, _hypothesis_tokens(o)) for o in others]
    mean_sim = sum(sims) / len(sims)
    return 1.0 - mean_sim


def impact_score(anomaly: Anomaly) -> float:
    return min(1.0, max(0.0, anomaly.evidence_impact / 8.0))


def topology_score(anomaly: Anomaly) -> float:
    return min(1.0, max(0.0, anomaly.topology_score))


def score_hypothesis(
    h: Hypothesis,
    anomaly: Anomaly,
    claims_by_id: dict[str, Claim],
    peers: list[Hypothesis],
) -> ScoreBreakdown:
    explain = explain_score(h, claims_by_id)
    grounding = grounding_score(h, anomaly, claims_by_id)
    testability = testability_score(h)
    novelty = novelty_score(h, claims_by_id)
    cost = cost_penalty(h)
    discrim = discriminability_score(h, peers)
    impact = impact_score(anomaly)
    topology = topology_score(anomaly)
    utility = (
        W_EXPLAIN * explain
        + W_GROUNDING * grounding
        + W_TESTABILITY * testability
        + W_NOVELTY * novelty
        + W_DISCRIMINABILITY * discrim
        + W_IMPACT * impact
        + W_TOPOLOGY * topology
        - W_COST * cost
    )
    return ScoreBreakdown(
        hypothesis_id=h.hypothesis_id,
        explain=explain,
        grounding=grounding,
        testability=testability,
        novelty=novelty,
        cost=cost,
        discriminability=discrim,
        impact=impact,
        topology=topology,
        utility=utility,
    )


def score_all(
    hypotheses: list[Hypothesis],
    anomalies: list[Anomaly],
    claims: list[Claim],
) -> dict[str, ScoreBreakdown]:
    claims_by_id = {c.claim_id: c for c in claims}
    anom_by_id = {a.anomaly_id: a for a in anomalies}
    peers_by_anom: dict[str, list[Hypothesis]] = {}
    for h in hypotheses:
        peers_by_anom.setdefault(h.anomaly_id, []).append(h)
    out: dict[str, ScoreBreakdown] = {}
    for h in hypotheses:
        anomaly = anom_by_id[h.anomaly_id]
        peers = peers_by_anom[h.anomaly_id]
        out[h.hypothesis_id] = score_hypothesis(h, anomaly, claims_by_id, peers)
    return out


def select_mmr(
    hypotheses: list[Hypothesis],
    scores: dict[str, ScoreBreakdown],
    k: int,
    lambda_: float = 0.7,
    min_anomalies: int = 1,
) -> list[Hypothesis]:
    if k <= 0 or not hypotheses:
        return []

    remaining = list(hypotheses)
    selected: list[Hypothesis] = []
    tokens_cache = {h.hypothesis_id: _hypothesis_tokens(h) for h in hypotheses}
    available_anomalies = {h.anomaly_id for h in hypotheses}
    target_anomaly_count = min(max(1, min_anomalies), k, len(available_anomalies))

    # Start with the highest-utility hypothesis.
    remaining.sort(key=lambda h: scores[h.hypothesis_id].utility, reverse=True)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < k:
        best_h = None
        best_score = float("-inf")
        selected_anomalies = {h.anomaly_id for h in selected}
        if len(selected_anomalies) < target_anomaly_count:
            candidate_pool = [h for h in remaining if h.anomaly_id not in selected_anomalies]
            if not candidate_pool:
                candidate_pool = remaining
        else:
            candidate_pool = remaining

        for h in candidate_pool:
            u = scores[h.hypothesis_id].utility
            max_sim = max(
                (jaccard(tokens_cache[h.hypothesis_id], tokens_cache[s.hypothesis_id]) for s in selected),
                default=0.0,
            )
            mmr = lambda_ * u - (1.0 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_h = h
        assert best_h is not None
        selected.append(best_h)
        remaining.remove(best_h)

    return selected
