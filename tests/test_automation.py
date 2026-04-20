import json
import subprocess
from datetime import datetime
from pathlib import Path

from aigraph.automation import build_fix_bundle, critique_runs, harvest_topics, render_crontab, run_fix_session, run_preflight_checks, run_topic_batch


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def test_harvest_topics_dedupes_and_respects_recent_cooldown(tmp_path):
    automation_dir = tmp_path / "automation"
    runs_dir = tmp_path / "runs"
    (runs_dir / "_analytics").mkdir(parents=True)
    _write_json(
        automation_dir / "topics" / "seeds.json",
        {
            "topics": [
                {"topic": "llm finance stock movement prediction time series forecasting", "priority": 0.9},
                {"topic": "rag hallucination medical qa", "priority": 0.8},
            ]
        },
    )
    _write_json(
        automation_dir / "state" / "runtime.json",
        {
            "last_harvest_at": None,
            "last_batch_at": None,
            "last_critic_at": None,
            "last_fix_bundle_at": None,
            "pending_topics": [],
            "recent_topic_runs": {"rag hallucination medical qa": datetime.now().isoformat(timespec="seconds")},
            "critiqued_runs": [],
        },
    )
    _write_jsonl(
        runs_dir / "_analytics" / "requests.jsonl",
        [
            {"event": "search_submit", "topic": "llm finance stock movement prediction time series forecasting"},
            {"event": "search_submit", "topic": "memory design language model"},
        ],
    )

    topics = harvest_topics(automation_dir, runs_dir, limit=5, cooldown_hours=48)
    labels = [row["topic"] for row in topics]
    assert "memory design language model" in labels
    assert "rag hallucination medical qa" not in labels
    assert len(labels) == len(set(labels))


def test_run_topic_batch_processes_pending_topics_with_fake_runner(tmp_path):
    automation_dir = tmp_path / "automation"
    runs_dir = tmp_path / "runs"
    _write_json(
        automation_dir / "state" / "runtime.json",
        {
            "last_harvest_at": None,
            "last_batch_at": None,
            "last_critic_at": None,
            "last_fix_bundle_at": None,
            "pending_topics": [
                {
                    "topic": "memory design language model",
                    "priority": 0.7,
                    "source": "arxiv",
                    "strategy": "balanced",
                    "limit": 10,
                    "citation_weight": 0.45,
                    "min_relevance": 0.3,
                    "insight_generator": "llm",
                }
            ],
            "recent_topic_runs": {},
            "critiqued_runs": [],
        },
    )

    def fake_runner(request, status):
        (request.run_dir / "overview.json").write_text(json.dumps({"headline": "ok"}), encoding="utf-8")
        status(
            status="done",
            stage="complete",
            progress=1.0,
            message="done",
            run_id=request.run_id,
            topic=request.topic,
            papers=4,
            claims=9,
            anomalies=2,
            insights=1,
        )

    results = run_topic_batch(automation_dir, runs_dir, batch_size=1, runner=fake_runner)
    assert len(results) == 1
    assert results[0]["status"] == "done"
    state = json.loads((automation_dir / "state" / "runtime.json").read_text(encoding="utf-8"))
    assert state["pending_topics"] == []
    assert (automation_dir / "runs" / "runs_index.jsonl").exists()


def test_critique_runs_and_fix_bundle_produce_ranked_outputs(tmp_path):
    automation_dir = tmp_path / "automation"
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "20260418-010203-abcdef"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "status.json",
        {
            "run_id": run_dir.name,
            "topic": "complex memory design for agents",
            "status": "done",
            "papers": 3,
            "claims": 7,
            "anomalies": 2,
            "insights": 1,
        },
    )
    _write_json(
        run_dir / "overview.json",
        {
            "headline": "starter map",
            "best_conflict_lines": [],
        },
    )
    _write_json(
        run_dir / "graph.json",
        {
            "nodes": [
                {"id": "Task:training document-level knowledge modules (kms)", "node_type": "Task", "name": "training document-level knowledge modules (KMs)"},
                {"id": "Paper:p1", "node_type": "Paper", "paper_id": "p1"},
            ],
            "edges": [],
        },
    )
    _write_jsonl(run_dir / "papers.jsonl", [{"paper_id": "p1", "title": "Paper 1", "year": 2025, "venue": "arXiv"}])
    _write_jsonl(run_dir / "claims.jsonl", [{"claim_id": "c1", "paper_id": "p1", "claim_text": "Claim 1"}])
    _write_jsonl(run_dir / "insights.jsonl", [{"insight_id": "i1", "title": "Scientific research and literature bridge", "type": "unifying_theory"}])

    issues = critique_runs(automation_dir, runs_dir, limit=4)
    assert issues
    assert any(issue["kind"] == "keyword_readability" for issue in issues)

    bundle = build_fix_bundle(automation_dir, runs_dir, max_issues=2)
    assert bundle["issue_count"] >= 1
    assert bundle["issues"][0]["artifacts"]["graph"].endswith("graph.json")


def test_run_fix_session_dry_run_writes_prompt_and_bundle(tmp_path):
    automation_dir = tmp_path / "automation"
    runs_dir = tmp_path / "runs"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)
    _write_json(repo_dir / "app.json", {"label": "old"})
    _git(repo_dir, "git add -A")
    _git(repo_dir, 'git commit -m "init"')
    _seed_actionable_run(automation_dir, runs_dir)
    critique_runs(automation_dir, runs_dir, limit=4)

    result = run_fix_session(automation_dir, runs_dir, repo_dir=repo_dir, dry_run=True)
    assert result["status"] == "dry_run"
    assert Path(result["prompt_path"]).exists()
    assert Path(result["bundle_path"]).exists()


def test_run_fix_session_creates_branch_and_commit_with_fake_codex(tmp_path):
    automation_dir = tmp_path / "automation"
    runs_dir = tmp_path / "runs"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _init_git_repo(repo_dir)
    file_path = repo_dir / "app.json"
    _write_json(file_path, {"label": "old"})
    _git(repo_dir, "git add -A")
    _git(repo_dir, 'git commit -m "init"')
    _seed_actionable_run(automation_dir, runs_dir)
    critique_runs(automation_dir, runs_dir, limit=4)

    def fake_runner(command: str, *, cwd: Path):
        if command.startswith("mock-codex "):
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            payload["label"] = "new"
            _write_json(file_path, payload)
            return subprocess.CompletedProcess(command, 0, stdout="mock codex ok\n", stderr="")
        return subprocess.run(command, cwd=cwd, shell=True, check=True, capture_output=True, text=True)

    result = run_fix_session(
        automation_dir,
        runs_dir,
        repo_dir=repo_dir,
        codex_command="mock-codex --bundle {bundle_path} --prompt {prompt_path}",
        test_command='python -c "print(\'ok\')"',
        command_runner=fake_runner,
    )
    assert result["status"] == "completed"
    assert result["commit"]
    assert result["branch"].startswith("codex/automation-fix-")
    assert json.loads(file_path.read_text(encoding="utf-8"))["label"] == "new"
    assert _git(repo_dir, "git rev-parse --abbrev-ref HEAD").strip() == result["branch"]


def test_render_crontab_includes_all_jobs(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    payload = render_crontab(repo_dir=repo_dir, automation_dir="automation", runs_dir="outputs/runs")
    assert "automation-harvest" in payload
    assert "automation-run-batch" in payload
    assert "automation-critic" in payload
    assert "automation-fix-bundle" in payload
    assert "automation-fix-run" in payload
    assert "AIGRAPH_CODEX_FIX_COMMAND" in payload


def test_run_preflight_checks_reports_missing_integrations(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / ".git").mkdir()
    venv_bin = repo_dir / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.delenv("AIGRAPH_CODEX_FIX_COMMAND", raising=False)
    report = run_preflight_checks(repo_dir=repo_dir, python_bin="./.venv/bin/python")
    names = {item["name"]: item for item in report["checks"]}
    assert names["repo_exists"]["ok"] is True
    assert names["python_bin"]["ok"] is True
    assert names["codex_command_env"]["ok"] is False


def _seed_actionable_run(automation_dir: Path, runs_dir: Path) -> None:
    run_dir = runs_dir / "20260418-010203-abcdef"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "status.json",
        {
            "run_id": run_dir.name,
            "topic": "complex memory design for agents",
            "status": "done",
            "papers": 3,
            "claims": 7,
            "anomalies": 2,
            "insights": 1,
        },
    )
    _write_json(
        run_dir / "overview.json",
        {
            "headline": "starter map",
            "best_conflict_lines": [],
        },
    )
    _write_json(
        run_dir / "graph.json",
        {
            "nodes": [
                {"id": "Task:training document-level knowledge modules (kms)", "node_type": "Task", "name": "training document-level knowledge modules (KMs)"},
                {"id": "Paper:p1", "node_type": "Paper", "paper_id": "p1"},
            ],
            "edges": [],
        },
    )
    _write_jsonl(run_dir / "papers.jsonl", [{"paper_id": "p1", "title": "Paper 1", "year": 2025, "venue": "arXiv"}])
    _write_jsonl(run_dir / "claims.jsonl", [{"claim_id": "c1", "paper_id": "p1", "claim_text": "Claim 1"}])
    _write_jsonl(run_dir / "insights.jsonl", [{"insight_id": "i1", "title": "Scientific research and literature bridge", "type": "unifying_theory"}])


def _init_git_repo(repo_dir: Path) -> None:
    _git(repo_dir, "git init")
    _git(repo_dir, 'git config user.email "test@example.com"')
    _git(repo_dir, 'git config user.name "Test User"')


def _git(repo_dir: Path, command: str) -> str:
    proc = subprocess.run(command, cwd=repo_dir, shell=True, check=True, capture_output=True, text=True)
    return proc.stdout
