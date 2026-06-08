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
