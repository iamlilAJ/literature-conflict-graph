import json

from aigraph.community import ingest_run, rebuild_community, read_community_status


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")


def test_ingest_run_builds_living_graph(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "20260417-120000-abcdef"
    run_dir.mkdir(parents=True)
    _write(run_dir / "papers.jsonl", [{"paper_id": "p1", "title": "Paper 1", "year": 2024, "venue": "ACL"}])
    _write(run_dir / "claims.jsonl", [{"claim_id": "c001", "paper_id": "p1", "claim_text": "Claim 1", "method": "RAG", "task": "QA", "direction": "positive"}])
    _write(run_dir / "insights.jsonl", [])
    status = ingest_run(run_dir, runs_dir, run_id=run_dir.name)
    assert status["runs"] == 1
    community = runs_dir / "_community"
    assert (community / "graph.json").exists()
    assert (community / "index.html").exists()
    assert read_community_status(runs_dir)["papers"] == 1


def test_rebuild_community_merges_multiple_done_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    for idx in [1, 2]:
        run_dir = runs_dir / f"20260417-12000{idx}-abcde{idx}"
        run_dir.mkdir(parents=True)
        _write(run_dir / "papers.jsonl", [{"paper_id": f"p{idx}", "title": f"Paper {idx}", "year": 2024, "venue": "ACL"}])
        _write(run_dir / "claims.jsonl", [{"claim_id": "c001", "paper_id": f"p{idx}", "claim_text": f"Claim {idx}", "method": "RAG", "task": "QA", "direction": "positive"}])
        _write(run_dir / "insights.jsonl", [])
        (run_dir / "status.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
    status = rebuild_community(runs_dir)
    assert status["runs"] == 2
    assert status["papers"] == 2
