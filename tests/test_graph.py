import tempfile
from pathlib import Path

from aigraph.graph import build_graph, load_graph, save_graph
from aigraph.models import Claim, Paper, Setting


def _claim(cid: str, direction: str, top_k: str = "5") -> Claim:
    return Claim(
        claim_id=cid,
        paper_id=f"p{cid[1:]}",
        claim_text=f"claim {cid}",
        method="RAG",
        task="factual QA",
        dataset="NaturalQuestions",
        metric="Exact Match",
        direction=direction,
        setting=Setting(top_k=top_k, task_type="factual"),
    )


def test_contradiction_edge_is_added_for_opposite_directions():
    claims = [_claim("c001", "positive"), _claim("c002", "negative", top_k="20")]
    g = build_graph(claims)
    edges = g.get_edge_data("Claim:c001", "Claim:c002") or {}
    edge_types = {d.get("edge_type") for d in edges.values()}
    assert "contradicts" in edge_types
    assert "setting_mismatch" in edge_types  # top_k differs
    assert "overlap" in edge_types  # same dataset+metric


def test_multi_claim_cluster_keeps_pairwise_claim_edges():
    claims = [
        _claim("c001", "positive", top_k="5"),
        _claim("c002", "positive", top_k="5"),
        _claim("c003", "negative", top_k="20"),
        _claim("c004", "mixed", top_k="30"),
    ]
    g = build_graph(claims)

    contradicts = 0
    mismatches = 0
    overlaps = 0
    for _, _, edge_data in g.edges(data=True):
        edge_type = edge_data.get("edge_type")
        if edge_type == "contradicts":
            contradicts += 1
        elif edge_type == "setting_mismatch":
            mismatches += 1
        elif edge_type == "overlap":
            overlaps += 1

    assert contradicts == 4
    assert mismatches == 4
    assert overlaps == 6


def test_method_and_task_nodes_are_shared():
    claims = [_claim("c001", "positive"), _claim("c002", "positive")]
    g = build_graph(claims)
    # The Method:rag and Task:factual qa nodes should be reused.
    assert g.has_node("Method:rag")
    assert g.has_node("Task:factual qa")
    assert g.in_degree("Method:rag") == 2
    assert g.in_degree("Task:factual qa") == 2


def test_node_link_json_roundtrips():
    claims = [_claim("c001", "positive"), _claim("c002", "negative", top_k="20")]
    g = build_graph(claims)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "graph.json"
        save_graph(g, path)
        g2 = load_graph(path)
    assert set(g.nodes) == set(g2.nodes)
    assert g.number_of_edges() == g2.number_of_edges()


def test_build_graph_adds_citation_edges_and_semantic_nodes():
    claims = [
        Claim(
            claim_id="c001",
            paper_id="openalex:W1",
            claim_text="Finance LLMs struggle with non-stationarity.",
            method="LLM",
            task="forecasting",
            direction="negative",
            domain="finance",
            mechanism="event grounding",
            failure_mode="temporal leakage",
            temporal_property="non-stationarity",
        ),
        Claim(
            claim_id="c002",
            paper_id="openalex:W2",
            claim_text="Time-series LLMs struggle with non-stationarity.",
            method="LLM",
            task="forecasting",
            direction="negative",
            domain="time series",
            mechanism="event grounding",
            failure_mode="temporal leakage",
            temporal_property="non-stationarity",
        ),
    ]
    papers = [
        Paper(
            paper_id="openalex:W1",
            title="Finance LLMs",
            year=2024,
            venue="ICLR",
            cited_by_count=9,
            referenced_works=["openalex:W2"],
            counts_by_year=[{"year": 2026, "cited_by_count": 3}],
            paper_role="survey",
            paper_role_score=0.92,
            paper_role_signals=["title:survey"],
        ),
        Paper(paper_id="openalex:W2", title="Time Series LLMs", year=2024, venue="NeurIPS", paper_role="method"),
    ]
    g = build_graph(claims, papers=papers, current_year=2026)
    assert g.has_edge("Paper:openalex:W1", "Paper:openalex:W2")
    edge_types = {d.get("edge_type") for d in (g.get_edge_data("Paper:openalex:W1", "Paper:openalex:W2") or {}).values()}
    assert "cites" in edge_types
    assert g.has_node("Domain:finance")
    assert g.has_node("Mechanism:event grounding")
    assert g.has_node("FailureMode:temporal leakage")
    assert g.has_node("TemporalProperty:non-stationarity")
    assert g.has_node("Role:survey")
    assert g.has_edge("Paper:openalex:W1", "Role:survey")
    assert g.nodes["Paper:openalex:W1"]["paper_role"] == "survey"
    assert g.nodes["Paper:openalex:W1"]["cited_by_count"] == 9
    assert g.nodes["Paper:openalex:W1"]["recent_citations"] == 3
