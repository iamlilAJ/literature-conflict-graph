from aigraph.anomalies import detect_anomalies
from aigraph.graph import build_graph
from aigraph.models import Claim, Paper, Setting


def _claim(cid: str, method: str, task: str, direction: str, **setting) -> Claim:
    return Claim(
        claim_id=cid,
        paper_id=f"p{cid[1:]}",
        claim_text=f"{method} {direction} on {task}",
        method=method,
        task=task,
        dataset="DatasetX",
        metric="EM",
        direction=direction,
        setting=Setting(**setting),
    )


def test_detects_benchmark_inconsistency_and_setting_mismatch():
    claims = [
        _claim("c001", "RAG", "factual QA", "positive", top_k="5", context_length="4k"),
        _claim("c002", "RAG", "factual QA", "negative", top_k="5", context_length="128k"),
    ]
    g = build_graph(claims)
    anomalies = detect_anomalies(g, claims)

    types = {a.type for a in anomalies}
    assert "benchmark_inconsistency" in types
    assert "setting_mismatch" in types

    mismatch = next(a for a in anomalies if a.type == "setting_mismatch")
    assert mismatch.varying_settings == ["context_length"]
    assert set(mismatch.claim_ids) == {"c001", "c002"}
    # Local subgraph should include the seed claim nodes.
    assert "Claim:c001" in mismatch.local_graph_nodes
    assert "Claim:c002" in mismatch.local_graph_nodes


def test_no_anomaly_when_all_positive():
    claims = [
        _claim("c001", "RAG", "factual QA", "positive"),
        _claim("c002", "RAG", "factual QA", "positive"),
    ]
    g = build_graph(claims)
    anomalies = detect_anomalies(g, claims)
    assert all(a.type != "benchmark_inconsistency" for a in anomalies)


def test_bridge_opportunity_between_related_clusters():
    claims = [
        _claim("c001", "RAG", "factual QA", "positive"),
        _claim("c002", "entailment-filtered retrieval", "factual QA evaluation", "positive"),
    ]
    g = build_graph(claims)
    anomalies = detect_anomalies(g, claims)
    bridges = [a for a in anomalies if a.type == "bridge_opportunity"]
    assert bridges, "Expected at least one bridge_opportunity anomaly"


def test_detects_metric_mismatch():
    claims = [
        _claim("c001", "LLM", "forecasting", "positive"),
        _claim("c002", "LLM", "forecasting", "negative"),
    ]
    claims[0].metric = "accuracy"
    claims[1].metric = "calibration error"
    g = build_graph(claims)
    anomalies = detect_anomalies(g, claims)
    assert any(a.type == "metric_mismatch" for a in anomalies)


def test_detects_impact_conflict_for_high_impact_positive_and_negative_sides():
    claims = [
        _claim("c001", "DPO", "safety", "positive"),
        _claim("c002", "DPO", "safety", "negative"),
    ]
    papers = [
        Paper(paper_id="p001", title="Positive", year=2024, venue="ACL", cited_by_count=20),
        Paper(paper_id="p002", title="Negative", year=2024, venue="ACL", cited_by_count=20),
    ]
    g = build_graph(claims, papers=papers)
    anomalies = detect_anomalies(g, claims)
    conflict = next(a for a in anomalies if a.type == "impact_conflict")
    assert conflict.evidence_impact > 0
    assert conflict.impact_balance > 0


def test_detects_community_disconnect_between_shared_concept_communities():
    claims = [
        _claim("c001", "LLM", "finance forecasting", "positive"),
        _claim("c002", "LLM", "finance forecasting", "positive"),
        _claim("c003", "LLM", "time series forecasting", "positive"),
        _claim("c004", "LLM", "time series forecasting", "positive"),
    ]
    for c in claims[:2]:
        c.domain = "finance"
        c.mechanism = "event grounding"
        c.failure_mode = "temporal leakage"
        c.temporal_property = "non-stationarity"
    for c in claims[2:]:
        c.domain = "time series"
        c.mechanism = "event grounding"
        c.failure_mode = "temporal leakage"
        c.temporal_property = "non-stationarity"
    papers = [
        Paper(paper_id=f"p{i:03d}", title=f"Paper {i}", year=2025, venue="ACL", cited_by_count=20)
        for i in range(1, 5)
    ]
    g = build_graph(claims, papers=papers)
    anomalies = detect_anomalies(g, claims)
    disconnects = [a for a in anomalies if a.type == "community_disconnect"]
    assert disconnects
    assert "non-stationarity" in disconnects[0].shared_entities["shared_concepts"]


def test_filters_weak_community_disconnect_without_impact_or_activity():
    claims = [
        _claim("c001", "LLM", "scientific literature", "positive"),
        _claim("c002", "LLM", "scientific literature", "negative"),
        _claim("c003", "LLM", "scientific research", "positive"),
        _claim("c004", "LLM", "scientific research", "negative"),
    ]
    for c in claims[:2]:
        c.domain = "scientific literature"
        c.mechanism = "modality alignment"
        c.evaluation_protocol = "evaluation protocol"
        c.data_modality = "multimodal"
    for c in claims[2:]:
        c.domain = "scientific research"
        c.mechanism = "modality alignment"
        c.evaluation_protocol = "evaluation protocol"
        c.data_modality = "multimodal"
    g = build_graph(claims)
    anomalies = detect_anomalies(g, claims)
    assert not any(a.type == "community_disconnect" for a in anomalies)


def test_user_facing_labels_fall_back_from_other_to_free_text():
    claims = [
        _claim("c001", "self-consistency prompting", "reasoning", "positive"),
        _claim("c002", "self-consistency prompting", "reasoning", "negative"),
    ]
    for claim in claims:
        claim.canonical_method = "other"
        claim.canonical_task = "reasoning"
    g = build_graph(claims)
    anomalies = detect_anomalies(g, claims)
    inconsistency = next(a for a in anomalies if a.type == "benchmark_inconsistency")
    assert "other" not in inconsistency.central_question.lower()
    assert "self-consistency prompting" in inconsistency.central_question
    assert inconsistency.shared_entities["method"] == "self-consistency prompting"


def test_other_canonical_method_does_not_merge_unrelated_methods():
    claims = [
        _claim("c001", "self-consistency prompting", "reasoning", "positive"),
        _claim("c002", "tree search", "reasoning", "negative"),
    ]
    for claim in claims:
        claim.canonical_method = "other"
        claim.canonical_task = "reasoning"
    g = build_graph(claims)
    anomalies = detect_anomalies(g, claims)
    assert not any(a.type == "benchmark_inconsistency" for a in anomalies)


def test_unknown_placeholder_clusters_are_skipped_in_user_facing_anomalies():
    claims = [
        _claim("c001", "fine-tuning", "other", "positive"),
        _claim("c002", "fine-tuning", "other", "negative"),
    ]
    for claim in claims:
        claim.canonical_method = "fine-tuning"
        claim.canonical_task = "other"
        claim.task = None
    g = build_graph(claims)
    anomalies = detect_anomalies(g, claims)
    assert not any(a.type == "benchmark_inconsistency" for a in anomalies)
