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
            "cited_by_count": 12,
            "selection_score": 0.74,
            "selection_reason": "high title/abstract relevance",
            "retrieval_channel": "survey",
            "paper_role": "survey",
            "paper_role_score": 0.92,
            "paper_role_signals": ["title:survey"],
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
    _write_jsonl(input_dir / "insights.jsonl", [
        {
            "insight_id": "i001",
            "type": "unifying_theory",
            "title": "Finance and time-series share temporal reasoning",
            "insight": "Both communities share non-stationarity.",
            "communities": ["finance", "time series"],
            "shared_concepts": ["non-stationarity", "temporal leakage"],
            "evidence_claims": ["c001"],
            "evidence_papers": ["openalex:W1"],
            "citation_gap": "No internal citation path was found.",
            "unifying_frame": "Language-conditioned temporal forecasting.",
            "transfer_suggestions": ["Transfer backtesting protocols."],
        }
    ])
    (input_dir / "graph.json").write_text(json.dumps({
        "directed": True,
        "multigraph": True,
        "graph": {},
        "nodes": [
            {"id": "Paper:openalex:W1", "node_type": "Paper", "paper_id": "openalex:W1", "cited_by_count": 12, "age_normalized_impact": 1.2, "paper_role": "survey"},
            {"id": "Claim:c001", "node_type": "Claim", "claim_id": "c001", "claim_text": "RAG improves factual QA."},
            {"id": "TemporalProperty:non-stationarity", "node_type": "TemporalProperty", "name": "non-stationarity"},
            {"id": "Role:survey", "node_type": "Role", "name": "Survey paper", "description": "Maps the field, taxonomy, or prior landscape."},
        ],
        "edges": [
            {"source": "Paper:openalex:W1", "target": "Claim:c001", "edge_type": "makes", "key": 0},
            {"source": "Paper:openalex:W1", "target": "Role:survey", "edge_type": "has_role", "key": 0},
        ],
    }), encoding="utf-8")

    output = render_visualization(input_dir, input_dir / "index.html")
    html = output.read_text(encoding="utf-8")
    assert "literature conflict explorer" in html
    assert "forceSimulation" in html
    assert "When does RAG help?" in html
    assert "Retrieval quality moderates" in html
    assert "https://openalex.org/W1" in html
    assert "paperLink" in html
    assert "Survey paper" in html
    assert "what this means" in html
    assert "selection reason" in html
    assert "high title/abstract relevance" in html
    assert "tool-divider" in html
    assert 'aria-label="Zoom controls"' in html
    assert 'aria-label="Evidence detail controls"' in html
    assert 'aria-label="Layout controls"' in html
    assert "zoom-fit" in html
    assert "Scroll to zoom" in html
    assert "view-hierarchy" in html
    assert "laneForNodeType" in html
    assert "detail-cluster" in html
    assert "detail-claims" in html
    assert "graphForCurrentDetail" in html
    assert "buildProjectedGraph" in html
    assert "buildClusterGraph" in html
    assert "edge_type === 'cites' ? '4 4' : (d.projected ? '3 5' : null)" in html
    assert "Topic" in html
    assert "Finance and time-series share temporal reasoning" in html
    assert "non-stationarity" in html
    assert "scrollIntoView" in html
    assert 'data-target="anomalies-section"' in html
    assert 'data-target="hypotheses-section"' in html
    assert 'data-target="insights-section"' in html
    assert ">Hypotheses</button>" in html
    assert ">Insights</span>" in html
    assert "miniNavEl.onclick = hasData ? () => {" in html
    assert "sectionEl.classList.add('open');" in html
    assert "submitGraphChat" in html
    assert "wireGraphChat();" in html
    assert "graph-chat-thread" in html
    assert "Recent context only" in html
    assert "trimChatHistory" in html
    assert "history: trimChatHistory()" in html
    assert "graph-chat-clear" in html
    assert "Ask about the whole run" in html
    assert "setSelectedContext({" in html
    assert "clearSelectedContext();" in html
    assert "syncChatContext();" in html
    assert "humanizeAnomalyLabel" in html
    assert "Claims</h2>" in html


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
    assert "No community insights generated." in html
    assert '"hypotheses": 0' in html
    assert "No hypotheses generated yet for this run." in html
    assert "No community insights generated yet for this run." in html
    assert "Not available yet" in html
    assert "fold-disabled-note" in html
    assert "miniNavEl.disabled = !hasData" in html
    assert "toggleEl.disabled = !hasData" in html
    assert "['Conflicts', s.anomalies, (s.anomalies || 0) > 0]" in html
    assert "['Hypotheses', s.hypotheses, (s.hypotheses || 0) > 0]" in html
    assert "['Insights', s.insights || 0, (s.insights || 0) > 0]" in html
    assert "Currently asking about the whole run." in html


def test_render_visualization_community_mode_uses_keyword_claim_paper_overview(tmp_path):
    input_dir = tmp_path / "_community"
    input_dir.mkdir()
    _write_jsonl(input_dir / "papers.jsonl", [
        {"paper_id": "p1", "title": "Paper 1", "year": 2025, "venue": "arXiv"},
    ])
    _write_jsonl(input_dir / "claims.jsonl", [
        {"claim_id": "run1:c001", "paper_id": "p1", "claim_text": "Claim 1", "method": "RAG", "task": "scientific QA", "direction": "positive"},
    ])
    _write_jsonl(input_dir / "anomalies.jsonl", [])
    _write_jsonl(input_dir / "hypotheses.jsonl", [])
    _write_jsonl(input_dir / "insights.jsonl", [])
    (input_dir / "graph.json").write_text(json.dumps({
        "nodes": [
            {"id": "Paper:p1", "node_type": "Paper", "paper_id": "p1"},
            {"id": "Claim:run1:c001", "node_type": "Claim", "claim_id": "run1:c001", "claim_text": "Claim 1"},
            {"id": "Method:rag", "node_type": "Method", "name": "RAG"},
            {"id": "Metric:f1", "node_type": "Metric", "name": "F1"},
        ],
        "edges": [
            {"source": "Paper:p1", "target": "Claim:run1:c001", "edge_type": "makes"},
            {"source": "Claim:run1:c001", "target": "Method:rag", "edge_type": "uses"},
            {"source": "Claim:run1:c001", "target": "Metric:f1", "edge_type": "measured_by"},
        ],
    }), encoding="utf-8")

    output = render_visualization(input_dir, input_dir / "index.html")
    html = output.read_text(encoding="utf-8")
    assert '"graph_mode": "community"' in html
    assert "const IS_COMMUNITY_GRAPH = DATA.graph_mode === 'community';" in html
    assert "label: 'Keyword'" in html
    assert "label: 'Claims'" in html
    assert "label: 'Papers'" in html
    assert "currentDetail = 'cluster'" in html
