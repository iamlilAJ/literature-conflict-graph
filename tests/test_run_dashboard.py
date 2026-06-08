"""Tests for the runs dashboard (src/aigraph/run_dashboard.py): flow
reconstruction from status.json + artifacts, and HTML rendering."""
import json

from aigraph import run_dashboard as rd


def _make_run(root, run_id="20260607-185302-abc123", **status):
    d = root / run_id
    d.mkdir(parents=True)
    base = {"topic": "memory based on-policy distillation", "status": "done",
            "stage": "complete", "source": "union", "strategy": "balanced"}
    base.update(status)
    (d / "status.json").write_text(json.dumps(base), encoding="utf-8")
    return d


def _write_lines(path, n):
    path.write_text("\n".join('{"x":1}' for _ in range(n)) + ("\n" if n else ""), encoding="utf-8")


def test_run_summary_fields(tmp_path):
    d = _make_run(tmp_path, papers=39, claims=36, anomalies=0,
                  semantic_gate={"applied": True, "kept": 12, "before": 39, "min_score": 2,
                                 "score_histogram": {"0": 27}},
                  extraction_quality="ok", direction_bias_suspected=False)
    s = rd.run_summary(d)
    assert s["topic"].startswith("memory based")
    assert s["status"] == "done" and s["source"] == "union"
    assert s["counts"]["papers"] == 39 and s["counts"]["claims"] == 36
    assert s["gate_kept"] == 12 and s["gate_before"] == 39
    assert s["created"] == "2026-06-07 18:53:02"  # parsed from run-id stamp


def test_pipeline_flow_states(tmp_path):
    d = _make_run(tmp_path, papers=39, claims=36, anomalies=0, nodes=50, edges=80,
                  hypotheses=0, open_questions=0,
                  semantic_gate={"applied": True, "kept": 12, "before": 39, "min_score": 2,
                                 "score_histogram": {"0": 27, "2": 12}},
                  extraction_quality="ok")
    _write_lines(d / "papers.jsonl", 39)
    _write_lines(d / "claims.jsonl", 36)
    _write_lines(d / "anomalies.jsonl", 0)   # ran but empty
    _write_lines(d / "graph.json", 1)
    (d / "extraction_audit.jsonl").write_text(
        json.dumps({"direction_distribution": {"positive": 30, "negative": 6},
                    "positive_bias_suspected": False}) + "\n", encoding="utf-8")
    flow = rd.pipeline_flow(d)
    by = {s["key"]: s for s in flow["stages"]}
    assert by["fetch"]["state"] == "done" and by["fetch"]["produced"] == 39
    assert by["gate"]["state"] == "done" and "kept 12/39" in by["gate"]["detail"]
    assert by["extract"]["state"] == "done"
    assert "direction=" in by["extract"]["detail"]
    # anomalies count 0 on a done run -> 'empty' (ran, produced nothing)
    assert by["anomalies"]["state"] == "empty"
    # hypotheses never produced an artifact, run done, count 0 -> empty
    assert by["hypotheses"]["state"] == "empty"


def test_active_and_pending_states(tmp_path):
    d = _make_run(tmp_path, status="running", stage="extract", papers=20)
    _write_lines(d / "papers.jsonl", 20)
    flow = rd.pipeline_flow(d)
    by = {s["key"]: s for s in flow["stages"]}
    assert by["fetch"]["state"] == "done"
    assert by["extract"]["state"] == "active"      # current stage on a running run
    assert by["render"]["state"] == "pending"      # not reached yet


def test_discover_newest_first(tmp_path):
    _make_run(tmp_path, run_id="20260607-100000-aaa")
    _make_run(tmp_path, run_id="20260607-200000-bbb")
    # a dir without status.json is ignored
    (tmp_path / "no-status").mkdir()
    # system/aux dirs (underscore-prefixed) are not user requests
    _make_run(tmp_path, run_id="_community")
    runs = rd.discover_run_summaries(tmp_path)
    assert [r["id"] for r in runs] == ["20260607-200000-bbb", "20260607-100000-aaa"]


def test_render_html_well_formed(tmp_path):
    d = _make_run(tmp_path, papers=39, claims=36, anomalies=0,
                  semantic_gate={"applied": True, "kept": 12, "before": 39})
    _write_lines(d / "papers.jsonl", 39)
    dash = rd.render_dashboard_html(tmp_path)
    assert dash.startswith("<!doctype html>") and "aigraph MCP" in dash
    assert "20260607-185302-abc123" in dash and "memory based" in dash
    assert "Fetch papers" in dash and "Semantic relevance gate" in dash  # pipeline legend
    flow = rd.render_run_flow_html(d)
    assert 'class="flow"' in flow and "Build conflict graph" in flow
    assert "kept 12/39" in flow


def test_empty_runs_root(tmp_path):
    assert rd.discover_run_summaries(tmp_path / "missing") == []
    assert "No runs yet" in rd.render_dashboard_html(tmp_path)


# ---- live fragments + richer detail --------------------------------------

def test_fragment_returns_inner_only(tmp_path):
    d = _make_run(tmp_path, papers=10)
    _write_lines(d / "papers.jsonl", 10)
    full = rd.render_dashboard_html(tmp_path, fragment=False)
    frag = rd.render_dashboard_html(tmp_path, fragment=True)
    assert full.startswith("<!doctype html>") and 'id="live"' in full and "POLL" not in frag
    assert frag.startswith("<table>") and "<!doctype" not in frag
    run_full = rd.render_run_flow_html(d, fragment=False)
    run_frag = rd.render_run_flow_html(d, fragment=True)
    assert 'id="live"' in run_full and "<!doctype" not in run_frag
    assert 'class="flow"' in run_frag


def test_running_run_has_live_poll(tmp_path):
    d = _make_run(tmp_path, status="running", stage="extract", papers=5)
    page = rd.render_run_flow_html(d)
    assert "b-running" in page and "setTimeout(tick" in page and 'id="liveind"' in page


def test_anomaly_type_breakdown(tmp_path):
    d = _make_run(tmp_path, anomalies=3, papers=5, claims=9, hypotheses=2)
    _write_lines(d / "papers.jsonl", 5)
    (d / "anomalies.jsonl").write_text(
        "\n".join(json.dumps({"type": t}) for t in
                  ["benchmark_inconsistency", "benchmark_inconsistency", "bridge_opportunity"]) + "\n",
        encoding="utf-8")
    flow = rd.pipeline_flow(d)
    anom = next(s for s in flow["stages"] if s["key"] == "anomalies")
    assert anom["state"] == "done"
    assert "benchmark_inconsistency×2" in anom["detail"] and "bridge_opportunity×1" in anom["detail"]


def test_hypotheses_selected_detail(tmp_path):
    d = _make_run(tmp_path, hypotheses=4, selected=2, anomalies=1, papers=3)
    flow = rd.pipeline_flow(d)
    hyp = next(s for s in flow["stages"] if s["key"] == "hypotheses")
    assert "4 generated" in hyp["detail"] and "2 selected" in hyp["detail"]


def test_duration_and_params_in_header(tmp_path):
    d = _make_run(tmp_path, run_id="20260607-185300-xy", source="union",
                  citation_weight=0.45, limit=40, updated_at="2026-06-07T18:54:12")
    page = rd.render_run_flow_html(d)
    assert "duration" in page and "1m12s" in page  # 18:53:00 -> 18:54:12
    assert "citation_weight" in page and "0.45" in page


# ---- idea drill-down (papers) + star graph -------------------------------

def _make_idea_run(root, run_id="20260607-185302-idea"):
    d = _make_run(root, run_id, anomalies=1, hypotheses=1, papers=2, claims=2)
    (d / "papers.jsonl").write_text(
        json.dumps({"paper_id": "arxiv:2306.13649v3", "title": "On-Policy Distillation of LMs",
                    "year": 2024, "url": "https://arxiv.org/abs/2306.13649"}) + "\n"
        + json.dumps({"paper_id": "openalex:W1", "title": "Some OA paper", "year": 2023,
                      "openalex_id": "W1"}) + "\n", encoding="utf-8")
    (d / "claims.jsonl").write_text(
        json.dumps({"claim_id": "c1", "paper_id": "arxiv:2306.13649v3", "claim_text": "OPD helps", "direction": "positive"}) + "\n"
        + json.dumps({"claim_id": "c2", "paper_id": "openalex:W1", "claim_text": "contradicts", "direction": "negative"}) + "\n",
        encoding="utf-8")
    (d / "anomalies.jsonl").write_text(
        json.dumps({"anomaly_id": "a1", "type": "impact_conflict", "claim_ids": ["c1", "c2"],
                    "central_question": "Does OPD help?"}) + "\n", encoding="utf-8")
    (d / "hypotheses.jsonl").write_text(
        json.dumps({"hypothesis_id": "h1", "anomaly_id": "a1", "hypothesis": "OPD helps when X",
                    "mechanism": "because Y", "predictions": ["p1", "p2"],
                    "explains_claims": ["c1", "c2"],
                    "novelty_audit": {"state": "unknown", "corpus_verdict": "unknown", "web_verdict": "skipped"}}) + "\n",
        encoding="utf-8")
    return d


def test_run_ideas_resolves_papers(tmp_path):
    ideas = rd.run_ideas(_make_idea_run(tmp_path))
    assert len(ideas) == 1
    it = ideas[0]
    assert it["hypothesis"] == "OPD helps when X" and it["anomaly_type"] == "impact_conflict"
    assert {p["paper_id"] for p in it["papers"]} == {"arxiv:2306.13649v3", "openalex:W1"}
    arx = next(p for p in it["papers"] if p["paper_id"].startswith("arxiv"))
    assert arx["url"] == "https://arxiv.org/abs/2306.13649"
    assert it["novelty_audit"]["state"] == "unknown"
    assert len(it["claims"]) == 2


def test_paper_link_fallbacks():
    assert rd._paper_link({"arxiv_id_base": "2306.13649"}) == "https://arxiv.org/abs/2306.13649"
    assert rd._paper_link({"openalex_id": "W123"}) == "https://openalex.org/W123"
    assert rd._paper_link({"doi": "10.1/x"}) == "https://doi.org/10.1/x"
    assert rd._paper_link({"url": "http://u"}) == "http://u"
    assert rd._paper_link({}) == ""


def test_ideas_section_renders_papers_and_graph_link(tmp_path):
    d = _make_idea_run(tmp_path)
    page = rd.render_ideas_section(rd.run_ideas(d), d.name)
    assert "Conflict-grounded hypotheses (1)" in page
    assert "OPD helps when X" in page and "Papers used (2)" in page
    assert "On-Policy Distillation of LMs" in page and "arxiv.org/abs/2306.13649" in page
    assert "星球图" in page and f'href="{d.name}/graph"' in page


def test_run_graph_shape_and_page(tmp_path):
    d = _make_idea_run(tmp_path)
    g = rd.run_graph(d)
    kinds = {n["kind"] for n in g["nodes"]}
    assert {"topic", "hypothesis", "anomaly", "paper"} <= kinds
    assert len(g["edges"]) >= 4   # topic→h1, h1→anomaly, h1→2 papers
    page = rd.render_graph_page(d)
    assert page.startswith("<!doctype html>") and "forceSimulation" in page and "星球图" in page


def test_run_cascade_ideas_and_section(tmp_path):
    d = _make_idea_run(tmp_path)
    (d / "forward_ideas.jsonl").write_text(
        json.dumps({"idea_id": "i1", "tier_label": "D · method extension", "title": "Memory-gated OPD",
                    "statement": "Use a memory buffer to ...", "mechanism": "gate tokens by ...",
                    "predictions": ["faster", "better"], "source_papers": ["arxiv:2306.13649v3"],
                    "novelty_audit": {"state": "covered", "corpus_verdict": "covered", "web_verdict": "skipped"}}) + "\n",
        encoding="utf-8")
    ideas = rd.run_cascade_ideas(d)
    assert len(ideas) == 1 and ideas[0]["title"] == "Memory-gated OPD"
    assert ideas[0]["tier"] == "D · method extension"
    assert ideas[0]["papers"][0]["title"] == "On-Policy Distillation of LMs"  # resolved from source_papers
    assert ideas[0]["novelty_audit"]["state"] == "covered"
    sec = rd.render_generated_ideas_section(ideas, d.name)
    assert "💡 Generated ideas (1)" in sec and "Memory-gated OPD" in sec
    assert "Source papers (1)" in sec and "generate_ideas" in sec
    # the full run page shows generated ABOVE the precursor hypotheses
    page = rd.render_run_flow_html(d)
    assert page.index("Generated ideas") < page.index("Conflict-grounded hypotheses")


def test_generated_ideas_empty_shows_hint(tmp_path):
    d = _make_run(tmp_path, run_id="20260607-185302-noidea")
    assert rd.run_cascade_ideas(d) == []
    sec = rd.render_generated_ideas_section([], d.name)
    assert "none yet" in sec and "generate_ideas" in sec


def test_run_ideas_empty_is_safe(tmp_path):
    d = _make_run(tmp_path, run_id="20260607-185302-empty")
    assert rd.run_ideas(d) == []
    assert rd.render_ideas_section([], d.name) == ""
    assert rd.run_graph(d)["nodes"] == [{"id": "topic", "label": "memory based on-policy distillation", "kind": "topic"}]
