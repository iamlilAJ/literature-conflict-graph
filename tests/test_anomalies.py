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


def _bridge_cluster_claims(prefix: str, method: str, task: str, n: int, *, direction: str = "positive") -> list[Claim]:
    claims: list[Claim] = []
    for i in range(n):
        cid = f"{prefix}{i:03d}"
        c = Claim(
            claim_id=cid,
            paper_id=f"{prefix}p{i:03d}",
            claim_text=f"{method} on {task}",
            method=method,
            task=task,
            direction=direction,
        )
        claims.append(c)
    return claims


def test_bridge_opportunity_between_related_clusters():
    # Cluster A: 5 papers on "noisy retrieval pipeline tuning" for "scientific document grounding".
    # Cluster B: 5 papers on "noisy retrieval calibration" for "scientific paper grounding".
    # After stripping generic tokens, the meaningful shared tokens are:
    #   {noisy, retrieval, scientific, grounding} (>=3 required, no citation overlap).
    cluster_a = _bridge_cluster_claims("a", "noisy retrieval pipeline tuning", "scientific document grounding", 5)
    cluster_b = _bridge_cluster_claims("b", "noisy retrieval calibration", "scientific paper grounding", 5)
    g = build_graph(cluster_a + cluster_b)
    anomalies = detect_anomalies(g, cluster_a + cluster_b)
    bridges = [a for a in anomalies if a.type == "bridge_opportunity"]
    assert bridges, "Expected at least one bridge_opportunity anomaly"
    # Confirm shared tokens are domain-meaningful, not generic content words.
    shared = bridges[0].shared_entities["shared_tokens"]
    for generic in ("model", "models", "reasoning", "task", "method", "approach", "framework"):
        assert generic not in shared.split(", "), f"Generic token {generic!r} leaked into shared_tokens"


def test_bridge_filtered_by_generic_tokens():
    # Three of the four overlapping tokens are generic content words; only "calibration"
    # is meaningful, which is below the >=3 threshold.
    cluster_a = _bridge_cluster_claims("a", "calibration model", "reasoning task evaluation", 5)
    cluster_b = _bridge_cluster_claims("b", "calibration approach", "reasoning task evaluation", 5)
    g = build_graph(cluster_a + cluster_b)
    anomalies = detect_anomalies(g, cluster_a + cluster_b)
    assert not any(a.type == "bridge_opportunity" for a in anomalies)


def test_bridge_filtered_by_small_community():
    # Plenty of meaningful overlap but cluster A has only 3 unique papers.
    cluster_a = _bridge_cluster_claims("a", "noisy retrieval pipeline tuning", "scientific document grounding", 3)
    cluster_b = _bridge_cluster_claims("b", "noisy retrieval calibration", "scientific paper grounding", 5)
    g = build_graph(cluster_a + cluster_b)
    anomalies = detect_anomalies(g, cluster_a + cluster_b)
    assert not any(a.type == "bridge_opportunity" for a in anomalies)


def test_bridge_filtered_by_citation_path():
    # Token + size gates pass but every paper in A cites every paper in B, so
    # the communities are heavily citation-connected and not bridge candidates.
    cluster_a = _bridge_cluster_claims("a", "noisy retrieval pipeline tuning", "scientific document grounding", 5)
    cluster_b = _bridge_cluster_claims("b", "noisy retrieval calibration", "scientific paper grounding", 5)
    papers: list[Paper] = []
    b_ids = [c.paper_id for c in cluster_b]
    a_ids = [c.paper_id for c in cluster_a]
    for c in cluster_a:
        papers.append(Paper(
            paper_id=c.paper_id, title=c.paper_id, year=2024, venue="ACL",
            cited_by_count=5, referenced_works=b_ids,
        ))
    for c in cluster_b:
        papers.append(Paper(
            paper_id=c.paper_id, title=c.paper_id, year=2024, venue="ACL",
            cited_by_count=5, referenced_works=a_ids,
        ))
    g = build_graph(cluster_a + cluster_b, papers=papers)
    anomalies = detect_anomalies(g, cluster_a + cluster_b)
    assert not any(a.type == "bridge_opportunity" for a in anomalies)


def _replication_pair(
    *,
    method: str = "chain-of-thought",
    task: str = "math reasoning",
    original_paper: str = "p_orig",
    replicating_paper: str = "p_rep",
    original_arxiv: str | None = None,
    original_title: str = "",
    baseline_raw: str = "",
    evidence_span: str = "",
) -> tuple[list[Claim], list[Paper]]:
    c_orig = Claim(
        claim_id="c_orig",
        paper_id=original_paper,
        claim_text=f"{method} helps {task}",
        method=method,
        task=task,
        direction="positive",
    )
    c_orig.canonical_method = method
    c_orig.canonical_task = task
    c_rep = Claim(
        claim_id="c_rep",
        paper_id=replicating_paper,
        claim_text=f"replicating {method} on {task} fails",
        method=method,
        task=task,
        direction="negative",
        baseline_raw=baseline_raw,
        evidence_span=evidence_span,
    )
    c_rep.canonical_method = method
    c_rep.canonical_task = task
    papers = [
        Paper(
            paper_id=original_paper,
            title=original_title or original_paper,
            year=2023,
            venue="ACL",
            cited_by_count=50,
            arxiv_id_base=original_arxiv,
        ),
        Paper(
            paper_id=replicating_paper,
            title=replicating_paper,
            year=2024,
            venue="ACL",
            cited_by_count=10,
        ),
    ]
    return [c_orig, c_rep], papers


def test_replication_arxiv_id_signal():
    claims, papers = _replication_pair(
        original_arxiv="2201.11903",
        baseline_raw="follows 2201.11903 chain-of-thought baseline",
    )
    g = build_graph(claims, papers=papers)
    anomalies = detect_anomalies(g, claims)
    reps = [a for a in anomalies if a.type == "replication_conflict"]
    assert len(reps) == 1
    assert reps[0].shared_entities["replication_signal"] == "baseline_arxiv"
    assert reps[0].shared_entities["original_paper"] == "p_orig"
    assert reps[0].shared_entities["replicating_paper"] == "p_rep"
    assert reps[0].replication_score == 1.0


def test_replication_follow_verb_signal():
    claims, papers = _replication_pair(
        evidence_span="we follow the chain-of-thought prompting approach to baseline our method",
    )
    g = build_graph(claims, papers=papers)
    anomalies = detect_anomalies(g, claims)
    reps = [a for a in anomalies if a.type == "replication_conflict"]
    assert len(reps) == 1
    assert reps[0].shared_entities["replication_signal"] == "follow_verb"


def test_replication_title_substring_signal():
    claims, papers = _replication_pair(
        original_title="Self-Consistency Improves Chain Of Thought Reasoning",
        baseline_raw="self-consistency chain thought sampling pipeline",
    )
    g = build_graph(claims, papers=papers)
    anomalies = detect_anomalies(g, claims)
    reps = [a for a in anomalies if a.type == "replication_conflict"]
    assert len(reps) == 1
    assert reps[0].shared_entities["replication_signal"] == "title_substring"


def test_replication_same_paper_excluded():
    # Same paper, opposing directions on same method/task — intra-paper, not a replication.
    claims, papers = _replication_pair(
        original_paper="p_x",
        replicating_paper="p_x",
        original_arxiv="2201.11903",
        baseline_raw="follows 2201.11903",
    )
    g = build_graph(claims, papers=papers[:1])
    anomalies = detect_anomalies(g, claims)
    assert not any(a.type == "replication_conflict" for a in anomalies)


def test_replication_same_direction_excluded():
    claims, papers = _replication_pair(
        original_arxiv="2201.11903",
        baseline_raw="follows 2201.11903",
    )
    claims[1].direction = "positive"  # both positive
    g = build_graph(claims, papers=papers)
    anomalies = detect_anomalies(g, claims)
    assert not any(a.type == "replication_conflict" for a in anomalies)


def test_replication_generic_follow_verb_excluded():
    # "follow" verb fires but canonical_method does not appear in evidence_span
    # → not a replication of this method specifically.
    claims, papers = _replication_pair(
        method="chain-of-thought",
        evidence_span="we follow the standard evaluation protocol described elsewhere",
    )
    g = build_graph(claims, papers=papers)
    anomalies = detect_anomalies(g, claims)
    assert not any(a.type == "replication_conflict" for a in anomalies)


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
