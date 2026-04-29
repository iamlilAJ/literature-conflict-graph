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


def test_bibliographic_coupling_edge_added_for_four_shared_refs():
    # _MIN_COUPLING_WEIGHT == 4: P1 and P2 must share ≥4 refs to be coupled.
    # Below the threshold, weight-2-or-3 coincidences are noise dominated by
    # heavily-cited foundational refs (Transformer, BERT) being co-cited
    # across many corpus papers without real topical kinship.
    claims = [
        Claim(claim_id="c001", paper_id="P1", claim_text="x", method="m", task="t", direction="positive"),
        Claim(claim_id="c002", paper_id="P2", claim_text="x", method="m", task="t", direction="positive"),
    ]
    papers = [
        Paper(paper_id="P1", title="P1", year=2024, venue="ACL", referenced_works=["R1", "R2", "R3", "R4"]),
        Paper(paper_id="P2", title="P2", year=2024, venue="ACL", referenced_works=["R1", "R2", "R3", "R4"]),
    ]
    g = build_graph(claims, papers=papers)
    edge_data = g.get_edge_data("Paper:P1", "Paper:P2") or {}
    co_cites = [d for d in edge_data.values() if d.get("edge_type") == "co_cites"]
    assert len(co_cites) == 1
    assert co_cites[0]["weight"] == 4


def test_bibliographic_coupling_skipped_below_threshold():
    # 3 shared refs → below _MIN_COUPLING_WEIGHT=4 → no edge.
    claims = [
        Claim(claim_id="c001", paper_id="P1", claim_text="x", method="m", task="t", direction="positive"),
        Claim(claim_id="c002", paper_id="P2", claim_text="x", method="m", task="t", direction="positive"),
    ]
    papers = [
        Paper(paper_id="P1", title="P1", year=2024, venue="ACL", referenced_works=["R1", "R2", "R3"]),
        Paper(paper_id="P2", title="P2", year=2024, venue="ACL", referenced_works=["R1", "R2", "R3"]),
    ]
    g = build_graph(claims, papers=papers)
    edge_data = g.get_edge_data("Paper:P1", "Paper:P2") or {}
    assert not any(d.get("edge_type") == "co_cites" for d in edge_data.values())


def test_cites_edges_match_versioned_to_unversioned_refs():
    """S2 returns ArXiv externalIds unversioned (`arxiv:2201.11903`) while
    corpus paper_ids carry version suffix (`arxiv:2201.11903v3`). The graph
    must resolve the unversioned ref to the versioned in-corpus paper."""
    claims = [
        Claim(claim_id="c001", paper_id="arxiv:2303.17651v2", claim_text="x",
              method="m", task="t", direction="positive"),
        Claim(claim_id="c002", paper_id="arxiv:2201.11903v3", claim_text="x",
              method="m", task="t", direction="positive"),
    ]
    papers = [
        Paper(paper_id="arxiv:2303.17651v2", title="Self-Refine", year=2023,
              venue="arXiv",
              referenced_works=["arxiv:2201.11903"]),  # unversioned, mirrors S2
        Paper(paper_id="arxiv:2201.11903v3", title="CoT", year=2022, venue="NeurIPS"),
    ]
    g = build_graph(claims, papers=papers)
    edge_data = g.get_edge_data("Paper:arxiv:2303.17651v2", "Paper:arxiv:2201.11903v3") or {}
    cites = [d for d in edge_data.values() if d.get("edge_type") == "cites"]
    assert len(cites) == 1


def test_cites_edge_prefers_latest_version():
    """When the corpus has both v1 and v2 of the same arxiv id and another
    paper references it unversioned, the cites edge must point to the
    LATEST version, not an arbitrary one."""
    claims = [
        Claim(claim_id="c001", paper_id="arxiv:2303.17651v1", claim_text="x",
              method="m", task="t", direction="positive"),
        Claim(claim_id="c002", paper_id="arxiv:2303.17651v2", claim_text="x",
              method="m", task="t", direction="positive"),
        Claim(claim_id="c003", paper_id="arxiv:2201.11903v3", claim_text="x",
              method="m", task="t", direction="positive"),
    ]
    papers = [
        Paper(paper_id="arxiv:2303.17651v1", title="Self-Refine v1", year=2023, venue="arXiv"),
        Paper(paper_id="arxiv:2303.17651v2", title="Self-Refine v2", year=2023, venue="arXiv"),
        Paper(paper_id="arxiv:2201.11903v3", title="CoT", year=2022, venue="NeurIPS",
              referenced_works=["arxiv:2303.17651"]),
    ]
    g = build_graph(claims, papers=papers)
    # Edge to v2 must exist.
    v2_edges = g.get_edge_data("Paper:arxiv:2201.11903v3", "Paper:arxiv:2303.17651v2") or {}
    assert any(d.get("edge_type") == "cites" for d in v2_edges.values())
    # No edge to v1.
    v1_edges = g.get_edge_data("Paper:arxiv:2201.11903v3", "Paper:arxiv:2303.17651v1") or {}
    assert not any(d.get("edge_type") == "cites" for d in v1_edges.values())


def test_cites_edge_does_not_create_self_loop():
    """A paper's own unversioned id appearing in its referenced_works (e.g.
    a v2 paper referencing 'arxiv:2303.17651' which the index resolves to
    itself) must NOT produce a self-loop cites edge."""
    claims = [
        Claim(claim_id="c001", paper_id="arxiv:2303.17651v2", claim_text="x",
              method="m", task="t", direction="positive"),
    ]
    papers = [
        Paper(paper_id="arxiv:2303.17651v2", title="Self-Refine", year=2023,
              venue="arXiv",
              referenced_works=["arxiv:2303.17651"]),  # would otherwise self-loop
    ]
    g = build_graph(claims, papers=papers)
    self_edges = g.get_edge_data("Paper:arxiv:2303.17651v2", "Paper:arxiv:2303.17651v2") or {}
    assert not any(d.get("edge_type") == "cites" for d in self_edges.values())


def test_cites_edge_keeps_double_digit_version_ordering():
    """Lexicographic ordering would put 'v9' > 'v10' — the index must use
    integer comparison so v10 wins over v9."""
    claims = [
        Claim(claim_id="c001", paper_id="arxiv:2303.17651v9", claim_text="x",
              method="m", task="t", direction="positive"),
        Claim(claim_id="c002", paper_id="arxiv:2303.17651v10", claim_text="x",
              method="m", task="t", direction="positive"),
        Claim(claim_id="c003", paper_id="arxiv:2201.11903v3", claim_text="x",
              method="m", task="t", direction="positive"),
    ]
    papers = [
        Paper(paper_id="arxiv:2303.17651v9", title="v9", year=2023, venue="arXiv"),
        Paper(paper_id="arxiv:2303.17651v10", title="v10", year=2023, venue="arXiv"),
        Paper(paper_id="arxiv:2201.11903v3", title="CoT", year=2022, venue="NeurIPS",
              referenced_works=["arxiv:2303.17651"]),
    ]
    g = build_graph(claims, papers=papers)
    v10_edges = g.get_edge_data("Paper:arxiv:2201.11903v3", "Paper:arxiv:2303.17651v10") or {}
    assert any(d.get("edge_type") == "cites" for d in v10_edges.values())
    v9_edges = g.get_edge_data("Paper:arxiv:2201.11903v3", "Paper:arxiv:2303.17651v9") or {}
    assert not any(d.get("edge_type") == "cites" for d in v9_edges.values())


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


def test_build_graph_does_not_classify_stance_by_default():
    """Locks the default — build_graph must NOT make any LLM calls unless the
    caller explicitly opts in via classify_stance=True. Catches regressions
    where stance classification gets accidentally turned on as a default
    (which would crash every existing test that doesn't mock an LLM client)."""

    class _BoomClient:
        """Any attribute access blows up. If build_graph touches the LLM
        layer at all, we'll see it here."""

        def __getattr__(self, name):  # pragma: no cover - test guard
            raise AssertionError(
                f"build_graph attempted to use the LLM client (.{name}) "
                "with classify_stance defaulting to False"
            )

    papers = [
        Paper(
            paper_id="A",
            title="Foo Method",
            year=2023,
            venue="ACL",
            cited_by_count=10,
        ),
        Paper(
            paper_id="B",
            title="A study",
            year=2024,
            venue="EMNLP",
            cited_by_count=5,
            referenced_works=["A"],
            text="We extend Foo Method to a new domain.",
        ),
    ]
    claims = [
        Claim(
            claim_id="cA", paper_id="A", claim_text="x",
            method="m", task="t", direction="positive",
        ),
        Claim(
            claim_id="cB", paper_id="B", claim_text="y",
            method="m", task="t", direction="positive",
        ),
    ]

    # Pass a client that would crash if used. Default classify_stance=False
    # must mean it is never invoked.
    g = build_graph(claims, papers=papers, stance_client=_BoomClient())

    # Cites edge exists but has NO stance attribute.
    edge_data = g.get_edge_data("Paper:B", "Paper:A") or {}
    cites_attrs = [d for d in edge_data.values() if d.get("edge_type") == "cites"]
    assert cites_attrs, "expected a cites edge between B and A"
    for nd in cites_attrs:
        assert "stance" not in nd
        assert "stance_confidence" not in nd
