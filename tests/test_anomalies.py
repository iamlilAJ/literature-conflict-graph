from aigraph.anomalies import detect_anomalies
from aigraph.graph import build_graph
from aigraph.models import Claim, Setting


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
