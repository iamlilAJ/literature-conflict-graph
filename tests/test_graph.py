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


def test_build_graph_adds_citation_edges_and_paper_attributes():
    """Semantic fields (domain, mechanism, ...) live as Claim attributes only —
    they no longer spawn per-value graph nodes. paper_role likewise stays as a
    Paper attribute, not a separate Role node."""
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
    # Semantic and Role nodes no longer exist in the graph.
    for absent in (
        "Domain:finance",
        "Mechanism:event grounding",
        "FailureMode:temporal leakage",
        "TemporalProperty:non-stationarity",
        "Role:survey",
    ):
        assert not g.has_node(absent), f"{absent} should not be in the simplified graph"
    # paper_role still travels as a Paper node attribute.
    assert g.nodes["Paper:openalex:W1"]["paper_role"] == "survey"
    assert g.nodes["Paper:openalex:W1"]["cited_by_count"] == 9
    assert g.nodes["Paper:openalex:W1"]["recent_citations"] == 3


def test_method_canonicalization_collapses_aliases_into_one_node():
    # Three papers using different surface forms of chain-of-thought should
    # collapse to a single canonical Method node, with the surface forms
    # captured in `aliases`.
    surface_forms = ["Chain Of Thought", "chain-of-thought", "CoT"]
    claims = []
    for i, surface in enumerate(surface_forms):
        c = Claim(
            claim_id=f"c{i:03d}",
            paper_id=f"p{i:03d}",
            claim_text=f"{surface} on math",
            method=surface,
            task="math reasoning",
            direction="positive",
        )
        c.canonical_method = "chain-of-thought"
        claims.append(c)
    g = build_graph(claims)
    method_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "Method"]
    assert method_nodes == ["Method:chain-of-thought"]
    node_data = g.nodes["Method:chain-of-thought"]
    # Aliases capture surface forms that DIFFER from the canonical only.
    assert set(node_data.get("aliases") or []) == {"Chain Of Thought", "CoT"}
    # All three claims hang off the same Method node.
    assert g.in_degree("Method:chain-of-thought") == 3


def test_dataset_canonicalization_uses_dataset_canonical_field():
    c = Claim(
        claim_id="c001",
        paper_id="p001",
        claim_text="x",
        method="LLM",
        task="QA",
        dataset="Natural Questions",
        direction="positive",
    )
    c.dataset_canonical = "naturalquestions"
    g = build_graph([c])
    assert g.has_node("Dataset:naturalquestions")
    assert not g.has_node("Dataset:natural questions")
    assert "Natural Questions" in (g.nodes["Dataset:naturalquestions"].get("aliases") or [])


def test_canonicalization_falls_back_to_raw_when_canonical_missing():
    c = Claim(
        claim_id="c001",
        paper_id="p001",
        claim_text="x",
        method="RAG",
        task="factual QA",
        direction="positive",
    )
    # No canonical_* fields set — graph must use the raw value.
    g = build_graph([c])
    assert g.has_node("Method:rag")
    assert g.has_node("Task:factual qa")


def test_canonicalization_ignores_placeholder_canonical_values():
    c = Claim(
        claim_id="c001",
        paper_id="p001",
        claim_text="x",
        method="self-consistency prompting",
        task="reasoning",
        direction="positive",
    )
    c.canonical_method = "other"  # placeholder, must not become the node id
    c.canonical_task = "reasoning"
    g = build_graph([c])
    assert g.has_node("Method:self-consistency prompting")
    assert not g.has_node("Method:other")


def test_bibliographic_coupling_edge_added_for_two_shared_refs():
    # P1 and P2 both cite R1 and R2 → coupling weight 2 → edge added.
    claims = [
        Claim(claim_id="c001", paper_id="P1", claim_text="x", method="m", task="t", direction="positive"),
        Claim(claim_id="c002", paper_id="P2", claim_text="x", method="m", task="t", direction="positive"),
    ]
    papers = [
        Paper(paper_id="P1", title="P1", year=2024, venue="ACL", referenced_works=["R1", "R2"]),
        Paper(paper_id="P2", title="P2", year=2024, venue="ACL", referenced_works=["R1", "R2"]),
    ]
    g = build_graph(claims, papers=papers)
    edge_data = g.get_edge_data("Paper:P1", "Paper:P2") or {}
    co_cites = [d for d in edge_data.values() if d.get("edge_type") == "co_cites"]
    assert len(co_cites) == 1
    assert co_cites[0]["weight"] == 2


def test_bibliographic_coupling_skipped_below_threshold():
    # Only 1 shared ref → below _MIN_COUPLING_WEIGHT=2 → no edge.
    claims = [
        Claim(claim_id="c001", paper_id="P1", claim_text="x", method="m", task="t", direction="positive"),
        Claim(claim_id="c002", paper_id="P2", claim_text="x", method="m", task="t", direction="positive"),
    ]
    papers = [
        Paper(paper_id="P1", title="P1", year=2024, venue="ACL", referenced_works=["R1"]),
        Paper(paper_id="P2", title="P2", year=2024, venue="ACL", referenced_works=["R1"]),
    ]
    g = build_graph(claims, papers=papers)
    edge_data = g.get_edge_data("Paper:P1", "Paper:P2") or {}
    assert not any(d.get("edge_type") == "co_cites" for d in edge_data.values())


def test_contradicts_edge_carries_impact_and_magnitude_weight():
    c_a = Claim(
        claim_id="c001",
        paper_id="P_high_impact",
        claim_text="x",
        method="m",
        task="t",
        direction="positive",
        magnitude_value=8.0,
    )
    c_b = Claim(
        claim_id="c002",
        paper_id="P_low_impact",
        claim_text="x",
        method="m",
        task="t",
        direction="negative",
        magnitude_value=-3.0,
    )
    papers = [
        Paper(paper_id="P_high_impact", title="A", year=2024, venue="ACL", cited_by_count=100),
        Paper(paper_id="P_low_impact", title="B", year=2024, venue="ACL", cited_by_count=2),
    ]
    g = build_graph([c_a, c_b], papers=papers)
    edge_data = g.get_edge_data("Claim:c001", "Claim:c002") or {}
    contradicts = [d for d in edge_data.values() if d.get("edge_type") == "contradicts"]
    assert len(contradicts) == 1
    weight = contradicts[0]["weight"]
    # log1p(100) * log1p(2) * |8 - 3| = ~4.62 * ~1.10 * 5 ≈ 25.4
    assert weight > 20.0


def test_canonical_clustering_links_aliases_with_contradicts():
    # Two claims with different surface methods but same canonical_method must
    # be linked by a contradicts edge (cluster key uses canonical).
    c_a = Claim(
        claim_id="c001",
        paper_id="p001",
        claim_text="x",
        method="Chain Of Thought",
        task="math",
        direction="positive",
    )
    c_a.canonical_method = "chain-of-thought"
    c_a.canonical_task = "math"
    c_b = Claim(
        claim_id="c002",
        paper_id="p002",
        claim_text="x",
        method="CoT",
        task="math",
        direction="negative",
    )
    c_b.canonical_method = "chain-of-thought"
    c_b.canonical_task = "math"
    g = build_graph([c_a, c_b])
    edge_data = g.get_edge_data("Claim:c001", "Claim:c002") or {}
    edge_types = {d.get("edge_type") for d in edge_data.values()}
    assert "contradicts" in edge_types
