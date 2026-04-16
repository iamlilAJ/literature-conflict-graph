import tempfile
from pathlib import Path

from aigraph.graph import build_graph, load_graph, save_graph
from aigraph.models import Claim, Setting


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
