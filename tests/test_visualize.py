import json

from aigraph.visualize import render_visualization


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")


def test_render_visualization_creates_graph_html(tmp_path):
    input_dir = tmp_path / "run"
    input_dir.mkdir()
    _write_jsonl(input_dir / "papers.jsonl", [
        {
            "paper_id": "openalex:W1",
            "title": "Paper One",
            "year": 2024,
            "venue": "ACL",
        }
    ])
    _write_jsonl(input_dir / "claims.jsonl", [
        {
            "claim_id": "c001",
            "paper_id": "openalex:W1",
            "claim_text": "RAG improves factual QA.",
            "method": "RAG",
            "task": "factual QA",
            "direction": "positive",
        }
    ])
    _write_jsonl(input_dir / "anomalies.jsonl", [
        {
            "anomaly_id": "a001",
            "type": "benchmark_inconsistency",
            "central_question": "When does RAG help?",
            "claim_ids": ["c001"],
            "positive_claims": ["c001"],
            "negative_claims": [],
            "shared_entities": {"method": "RAG", "task": "factual QA"},
            "local_graph_nodes": ["Claim:c001", "Paper:openalex:W1"],
            "local_graph_edges": [{"source": "Paper:openalex:W1", "target": "Claim:c001", "edge_type": "makes"}],
        }
    ])
    _write_jsonl(input_dir / "hypotheses.jsonl", [
        {
            "hypothesis_id": "h001",
            "anomaly_id": "a001",
            "hypothesis": "Retrieval quality moderates the effect.",
            "mechanism": "Noise changes evidence use.",
            "explains_claims": ["c001"],
            "predictions": ["Better filtering helps."],
            "minimal_test": "Compare filtered and unfiltered retrieval.",
            "evidence_gap": "Few matched runs.",
        }
    ])
    (input_dir / "graph.json").write_text(json.dumps({
        "directed": True,
        "multigraph": True,
        "graph": {},
        "nodes": [
            {"id": "Paper:openalex:W1", "node_type": "Paper", "paper_id": "openalex:W1"},
            {"id": "Claim:c001", "node_type": "Claim", "claim_id": "c001", "claim_text": "RAG improves factual QA."},
        ],
        "edges": [{"source": "Paper:p001", "target": "Claim:c001", "edge_type": "makes", "key": 0}],
    }), encoding="utf-8")

    output = render_visualization(input_dir, input_dir / "index.html")
    html = output.read_text(encoding="utf-8")
    assert "literature conflict explorer" in html
    assert "forceSimulation" in html
    assert "When does RAG help?" in html
    assert "Retrieval quality moderates" in html
    assert "https://openalex.org/W1" in html
    assert "paperLink" in html


def test_render_visualization_handles_empty_hypotheses(tmp_path):
    input_dir = tmp_path / "run"
    input_dir.mkdir()
    _write_jsonl(input_dir / "papers.jsonl", [])
    _write_jsonl(input_dir / "claims.jsonl", [])
    _write_jsonl(input_dir / "anomalies.jsonl", [])
    _write_jsonl(input_dir / "hypotheses.jsonl", [])
    (input_dir / "graph.json").write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")

    output = render_visualization(input_dir, input_dir / "index.html")
    html = output.read_text(encoding="utf-8")
    assert "No hypotheses generated." in html
    assert '"hypotheses": 0' in html
