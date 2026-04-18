"""Lightweight 24/7 automation loop for aigraph."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import textwrap
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .llm_client import build_openai_client, call_llm_text, configured_api_key, configured_model
from .paper_select import decompose_topic_query
from .server import (
    SearchRequest,
    decompose_search_topic,
    new_run_id,
    normalize_arxiv_query,
    read_jsonl_dicts,
    run_pipeline,
    write_status,
)


DEFAULT_AUTOMATION_DIR = Path("automation")
DEFAULT_RUNS_DIR = Path("outputs/runs")
DEFAULT_FIX_BRANCH_PREFIX = "codex/automation-fix"
DEFAULT_FIX_TEST_COMMAND = "./.venv/bin/pytest -q"
DEFAULT_PYTHON_BIN = "./.venv/bin/python"
SEVERITY_WEIGHT = {"low": 1, "medium": 2, "high": 3}
ISSUE_KINDS = {
    "retrieval_quality",
    "keyword_readability",
    "anomaly_usefulness",
    "insight_usefulness",
    "graph_clarity",
    "report_quality",
    "demo_worthiness",
}
GENERIC_TERMS = {"other", "research", "literature", "community", "communities"}


def ensure_automation_layout(automation_dir: Path | str = DEFAULT_AUTOMATION_DIR) -> Path:
    root = Path(automation_dir)
    for rel in ("topics", "issues", "state", "prompts", "logs", "runs"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    if not (root / "state" / "runtime.json").exists():
        _write_json(root / "state" / "runtime.json", _default_runtime_state())
    if not (root / "state" / "dashboard.json").exists():
        _write_json(root / "state" / "dashboard.json", {})
    if not (root / "topics" / "generated_topics.jsonl").exists():
        (root / "topics" / "generated_topics.jsonl").write_text("", encoding="utf-8")
    if not (root / "issues" / "issues.jsonl").exists():
        (root / "issues" / "issues.jsonl").write_text("", encoding="utf-8")
    if not (root / "runs" / "runs_index.jsonl").exists():
        (root / "runs" / "runs_index.jsonl").write_text("", encoding="utf-8")
    return root


def harvest_topics(
    automation_dir: Path | str = DEFAULT_AUTOMATION_DIR,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    *,
    limit: int = 12,
    cooldown_hours: int = 12,
) -> list[dict[str, Any]]:
    root = ensure_automation_layout(automation_dir)
    state = _load_runtime_state(root)
    now = _now_iso()
    seeds = _load_seed_topics(root)
    recent_queries = _recent_queries(Path(runs_dir))
    recent_topics = [entry["topic"] for entry in recent_queries]
    harvested: list[dict[str, Any]] = []
    for seed in seeds:
        harvested.append(_normalize_topic_record(seed, source="seed"))
    for topic in recent_topics:
        harvested.append(_normalize_topic_record({"topic": topic, "priority": 0.55}, source="user-log"))

    expansions = _generate_topic_variants(root, harvested[:6])
    harvested.extend(expansions)

    cooldown_cutoff = datetime.now() - timedelta(hours=cooldown_hours)
    recent_run_topics = {
        topic
        for topic, stamp in (state.get("recent_topic_runs") or {}).items()
        if _parse_iso(stamp) and _parse_iso(stamp) >= cooldown_cutoff
    }
    pending_topics = {item["topic"] for item in state.get("pending_topics") or []}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sorted(harvested, key=lambda row: (float(row.get("priority", 0.0)), row["topic"]), reverse=True):
        topic = item["topic"]
        if topic in seen or topic in pending_topics or topic in recent_run_topics:
            continue
        seen.add(topic)
        selected.append(
            {
                **item,
                "created_at": now,
                "status": "pending",
            }
        )
        if len(selected) >= limit:
            break

    if selected:
        _append_jsonl(root / "topics" / "generated_topics.jsonl", selected)
        state.setdefault("pending_topics", []).extend(selected)
    state["last_harvest_at"] = now
    _save_runtime_state(root, state)
    _write_dashboard(root, state, runs_dir)
    return selected


def run_topic_batch(
    automation_dir: Path | str = DEFAULT_AUTOMATION_DIR,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    *,
    batch_size: int = 3,
    runner: Callable[[SearchRequest, Callable[..., None]], None] | None = None,
) -> list[dict[str, Any]]:
    root = ensure_automation_layout(automation_dir)
    runs_root = Path(runs_dir)
    runs_root.mkdir(parents=True, exist_ok=True)
    state = _load_runtime_state(root)
    pending = list(state.get("pending_topics") or [])
    selected = pending[:batch_size]
    state["pending_topics"] = pending[batch_size:]
    runner = runner or run_pipeline
    results: list[dict[str, Any]] = []

    for item in selected:
        query_plan = decompose_search_topic(item["topic"])
        run_id = new_run_id()
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        request = SearchRequest(
            run_id=run_id,
            topic=item["topic"],
            retrieval_topic=str(query_plan.get("normalized_topic") or item["topic"]),
            retrieval_variants=[str(v) for v in (query_plan.get("retrieval_variants") or []) if str(v).strip()],
            arxiv_query=normalize_arxiv_query(str(query_plan.get("normalized_topic") or item["topic"])),
            limit=int(item.get("limit") or 10),
            insight_generator=str(item.get("insight_generator") or "llm"),
            source=str(item.get("source") or "arxiv"),
            strategy=str(item.get("strategy") or "balanced"),
            citation_weight=float(item.get("citation_weight") or 0.45),
            min_relevance=float(item.get("min_relevance") or 0.30),
            run_dir=run_dir,
            client_ip="automation",
            user_agent="aigraph-automation/1.0",
        )
        try:
            write_status(
                run_dir,
                status="queued",
                stage="queued",
                progress=0.0,
                message="Queued by automation.",
                run_id=run_id,
                topic=request.topic,
                arxiv_query=request.arxiv_query,
                source_query=request.arxiv_query if request.source == "arxiv" else request.topic,
                source=request.source,
                strategy=request.strategy,
                limit=request.limit,
                citation_weight=request.citation_weight,
                min_relevance=request.min_relevance,
            )
            runner(request, lambda **kwargs: write_status(run_dir, **kwargs))
            status = _read_json(run_dir / "status.json")
            result = {
                "run_id": run_id,
                "topic": request.topic,
                "status": status.get("status", "done"),
                "source": request.source,
                "strategy": request.strategy,
                "created_at": _now_iso(),
            }
        except Exception as exc:  # pragma: no cover - defensive top-level guard
            write_status(
                run_dir,
                status="error",
                stage="error",
                progress=1.0,
                message=f"Automation batch failed: {exc}",
                error=str(exc),
                run_id=run_id,
                topic=request.topic,
            )
            result = {
                "run_id": run_id,
                "topic": request.topic,
                "status": "error",
                "source": request.source,
                "strategy": request.strategy,
                "created_at": _now_iso(),
                "error": str(exc),
            }
        results.append(result)
        state.setdefault("recent_topic_runs", {})[request.topic] = _now_iso()

    if results:
        _append_jsonl(root / "runs" / "runs_index.jsonl", results)
    state["last_batch_at"] = _now_iso()
    _save_runtime_state(root, state)
    _write_dashboard(root, state, runs_root)
    return results


def critique_runs(
    automation_dir: Path | str = DEFAULT_AUTOMATION_DIR,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    root = ensure_automation_layout(automation_dir)
    runs_root = Path(runs_dir)
    state = _load_runtime_state(root)
    critiqued = set(state.get("critiqued_runs") or [])
    run_dirs = sorted(
        [path for path in runs_root.iterdir() if path.is_dir() and re.fullmatch(r"\d{8}-\d{6}-[a-f0-9]{6}", path.name)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    issues: list[dict[str, Any]] = []
    seen_keys = {row.get("dedupe_key") for row in _read_jsonl(root / "issues" / "issues.jsonl")}
    for run_dir in run_dirs:
        if run_dir.name in critiqued:
            continue
        status = _read_json(run_dir / "status.json")
        if status.get("status") != "done":
            continue
        run_issues = _critique_run(root, run_dir, status)
        for issue in run_issues:
            if issue["dedupe_key"] in seen_keys:
                continue
            seen_keys.add(issue["dedupe_key"])
            issues.append(issue)
        critiqued.add(run_dir.name)
        if len(critiqued) >= limit:
            break

    if issues:
        _append_jsonl(root / "issues" / "issues.jsonl", issues)
    state["critiqued_runs"] = sorted(critiqued)
    state["last_critic_at"] = _now_iso()
    _save_runtime_state(root, state)
    _write_dashboard(root, state, runs_root)
    return issues


def build_fix_bundle(
    automation_dir: Path | str = DEFAULT_AUTOMATION_DIR,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    *,
    max_issues: int = 3,
) -> dict[str, Any]:
    root = ensure_automation_layout(automation_dir)
    runs_root = Path(runs_dir)
    issues = _read_jsonl(root / "issues" / "issues.jsonl")
    ranked = _rank_issues(issues)
    chosen = ranked[:max_issues]
    bundle = {
        "created_at": _now_iso(),
        "issue_count": len(chosen),
        "issues": [],
    }
    for issue in chosen:
        run_id = issue.get("run_id", "")
        run_dir = runs_root / run_id
        bundle["issues"].append(
            {
                **issue,
                "artifacts": {
                    "status": str(run_dir / "status.json"),
                    "overview": str(run_dir / "overview.json"),
                    "graph": str(run_dir / "graph.json"),
                    "report": str(run_dir / "selected_hypotheses.md"),
                },
            }
        )
    path = root / "issues" / "fix_bundle.json"
    _write_json(path, bundle)
    state = _load_runtime_state(root)
    state["last_fix_bundle_at"] = bundle["created_at"]
    _save_runtime_state(root, state)
    _write_dashboard(root, state, runs_root)
    return bundle


def run_fix_session(
    automation_dir: Path | str = DEFAULT_AUTOMATION_DIR,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    *,
    repo_dir: Path | str = ".",
    max_issues: int = 3,
    codex_command: str | None = None,
    branch_prefix: str = DEFAULT_FIX_BRANCH_PREFIX,
    test_command: str = DEFAULT_FIX_TEST_COMMAND,
    push: bool = False,
    open_pr: bool = False,
    dry_run: bool = False,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    root = ensure_automation_layout(automation_dir)
    runs_root = Path(runs_dir)
    repo_root = Path(repo_dir).resolve()
    bundle = build_fix_bundle(root, runs_root, max_issues=max_issues)
    session_id = f"fix-{datetime.now():%Y%m%d-%H%M%S}"
    session_dir = root / "issues" / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    command_runner = command_runner or _run_shell_command
    configured_codex = codex_command or os.getenv("AIGRAPH_CODEX_FIX_COMMAND", "").strip()
    branch = f"{branch_prefix}-{datetime.now():%Y%m%d-%H%M%S}"
    prompt_text = _build_fix_prompt(root, bundle, repo_root, branch, test_command)
    prompt_path = session_dir / "codex_prompt.md"
    bundle_path = session_dir / "fix_bundle.json"
    pr_body_path = session_dir / "pr_body.md"
    result_path = session_dir / "result.json"
    prompt_path.write_text(prompt_text, encoding="utf-8")
    _write_json(bundle_path, bundle)
    pr_body_path.write_text(_build_pr_body(bundle), encoding="utf-8")

    result: dict[str, Any] = {
        "session_id": session_id,
        "created_at": _now_iso(),
        "repo_dir": str(repo_root),
        "branch": branch,
        "issue_count": bundle["issue_count"],
        "bundle_path": str(bundle_path),
        "prompt_path": str(prompt_path),
        "pr_body_path": str(pr_body_path),
        "dry_run": dry_run,
        "codex_command": configured_codex or None,
        "status": "prepared",
        "steps": [],
    }

    if bundle["issue_count"] == 0:
        result["status"] = "no_issues"
        _write_json(result_path, result)
        _update_fix_session_state(root, runs_root, result)
        return result

    if dry_run:
        result["status"] = "dry_run"
        result["next_step"] = "Run automation-fix-run again with --codex-command (or AIGRAPH_CODEX_FIX_COMMAND) and optional --push/--open-pr."
        _write_json(result_path, result)
        _update_fix_session_state(root, runs_root, result)
        return result

    if not configured_codex:
        result["status"] = "missing_codex_command"
        result["error"] = "No Codex invocation command configured."
        _write_json(result_path, result)
        _update_fix_session_state(root, runs_root, result)
        return result

    if _repo_is_dirty(repo_root, command_runner):
        result["status"] = "blocked_dirty_repo"
        result["error"] = "Repository has uncommitted changes; refusing automated fixer run."
        _write_json(result_path, result)
        _update_fix_session_state(root, runs_root, result)
        return result

    try:
        _step(result, "git_checkout_branch", command=f"git checkout -b {branch}")
        command_runner(f"git checkout -b {branch}", cwd=repo_root)

        codex_shell = configured_codex.format(
            repo_dir=_shell_quote(str(repo_root)),
            bundle_path=_shell_quote(str(bundle_path)),
            prompt_path=_shell_quote(str(prompt_path)),
            pr_body_path=_shell_quote(str(pr_body_path)),
            branch=_shell_quote(branch),
            test_command=_shell_quote(test_command),
        )
        _step(result, "codex_fix", command=codex_shell)
        command_runner(codex_shell, cwd=repo_root)

        _step(result, "run_tests", command=test_command)
        command_runner(test_command, cwd=repo_root)

        if _repo_has_no_changes(repo_root, command_runner):
            result["status"] = "no_changes"
            _write_json(result_path, result)
            _update_fix_session_state(root, runs_root, result)
            return result

        commit_message = _build_commit_message(bundle)
        _step(result, "git_add", command="git add -A")
        command_runner("git add -A", cwd=repo_root)
        _step(result, "git_commit", command=f"git commit -m {json.dumps(commit_message)}")
        command_runner(f"git commit -m {json.dumps(commit_message)}", cwd=repo_root)
        result["commit"] = _git_head(repo_root, command_runner)

        if push:
            _step(result, "git_push", command=f"git push -u origin {branch}")
            command_runner(f"git push -u origin {branch}", cwd=repo_root)
            result["pushed"] = True

        if open_pr:
            if not push:
                _step(result, "git_push", command=f"git push -u origin {branch}")
                command_runner(f"git push -u origin {branch}", cwd=repo_root)
                result["pushed"] = True
            if not shutil.which("gh"):
                raise RuntimeError("gh CLI is not installed, cannot open PR automatically.")
            pr_title = _build_pr_title(bundle)
            gh_command = (
                f"gh pr create --draft --base main --head {branch} "
                f"--title {json.dumps(pr_title)} --body-file {json.dumps(str(pr_body_path))}"
            )
            _step(result, "gh_pr_create", command=gh_command)
            pr_proc = command_runner(gh_command, cwd=repo_root)
            result["pr_url"] = (pr_proc.stdout or "").strip() or None

        result["status"] = "completed"
    except Exception as exc:  # pragma: no cover - error path exercised via test doubles
        result["status"] = "failed"
        result["error"] = str(exc)

    _write_json(result_path, result)
    _update_fix_session_state(root, runs_root, result)
    return result


def render_crontab(
    *,
    repo_dir: Path | str = ".",
    python_bin: str = DEFAULT_PYTHON_BIN,
    automation_dir: Path | str = DEFAULT_AUTOMATION_DIR,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    batch_size: int = 3,
    critic_limit: int = 8,
    max_fix_issues: int = 3,
    codex_command_env: str = "AIGRAPH_CODEX_FIX_COMMAND",
) -> str:
    repo_root = Path(repo_dir).resolve()
    automation_root = _relative_or_absolute(Path(automation_dir), repo_root)
    runs_root = _relative_or_absolute(Path(runs_dir), repo_root)
    log_dir = automation_root / "logs"
    python_ref = python_bin
    base = f"cd {shlex.quote(str(repo_root))} && {shlex.quote(python_ref)} -m aigraph.cli"

    jobs = [
        (
            "15 * * * *",
            f"{base} automation-harvest --automation-dir {shlex.quote(str(automation_root))} --runs-dir {shlex.quote(str(runs_root))}",
            log_dir / "topic_harvest.log",
        ),
        (
            "25 * * * *",
            f"{base} automation-run-batch --automation-dir {shlex.quote(str(automation_root))} --runs-dir {shlex.quote(str(runs_root))} --batch-size {batch_size}",
            log_dir / "run_batch.log",
        ),
        (
            "40 */2 * * *",
            f"{base} automation-critic --automation-dir {shlex.quote(str(automation_root))} --runs-dir {shlex.quote(str(runs_root))} --limit {critic_limit}",
            log_dir / "critic_batch.log",
        ),
        (
            "50 2 * * *",
            f"{base} automation-fix-bundle --automation-dir {shlex.quote(str(automation_root))} --runs-dir {shlex.quote(str(runs_root))} --max-issues {max_fix_issues}",
            log_dir / "fix_bundle.log",
        ),
        (
            "10 3 * * *",
            f"cd {shlex.quote(str(repo_root))} && ./automation/bin/nightly_fix.sh",
            log_dir / "nightly_fix.log",
        ),
    ]

    lines = [
        "# aigraph 24/7 automation schedule",
        f"# repo: {repo_root}",
        f"# generated: {_now_iso()}",
        f"# set {codex_command_env} before running automation-fix-run",
        (
            f"# manual fix command: {base} automation-fix-run --automation-dir {shlex.quote(str(automation_root))} "
            f"--runs-dir {shlex.quote(str(runs_root))}"
        ),
        "SHELL=/bin/bash",
        "PATH=/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "",
    ]
    for schedule, command, log_file in jobs:
        lines.append(f"{schedule} {command} >> {shlex.quote(str(log_file))} 2>&1")
    lines.append("")
    return "\n".join(lines)


def run_preflight_checks(
    *,
    repo_dir: Path | str = ".",
    python_bin: str = DEFAULT_PYTHON_BIN,
    codex_command_env: str = "AIGRAPH_CODEX_FIX_COMMAND",
) -> dict[str, Any]:
    repo_root = Path(repo_dir).resolve()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    python_path = (repo_root / python_bin).resolve() if not Path(python_bin).is_absolute() else Path(python_bin)
    add("repo_exists", repo_root.exists(), str(repo_root))
    add("python_bin", python_path.exists(), str(python_path))
    add("git_dir", (repo_root / ".git").exists(), str(repo_root / ".git"))
    add(
        "codex_command_env",
        bool(os.getenv(codex_command_env, "").strip()),
        f"{codex_command_env}={'set' if os.getenv(codex_command_env, '').strip() else 'missing'}",
    )
    add("gh_installed", shutil.which("gh") is not None, shutil.which("gh") or "gh not found")

    gh_auth_ok = False
    gh_auth_detail = "gh not installed"
    if shutil.which("gh"):
        try:
            proc = _run_shell_command("gh auth status", cwd=repo_root)
            gh_auth_ok = proc.returncode == 0
            gh_auth_detail = (proc.stdout or proc.stderr or "authenticated").strip()
        except Exception as exc:  # pragma: no cover - shell environment dependent
            gh_auth_detail = str(exc)
    add("gh_auth", gh_auth_ok, gh_auth_detail)

    git_ok = False
    git_detail = "git not available"
    if shutil.which("git"):
        try:
            proc = _run_shell_command("git remote -v", cwd=repo_root)
            git_ok = bool((proc.stdout or "").strip())
            git_detail = (proc.stdout or proc.stderr or "no remotes").strip()
        except Exception as exc:  # pragma: no cover
            git_detail = str(exc)
    add("git_remote", git_ok, git_detail)

    dirty = True
    dirty_detail = "unknown"
    try:
        dirty = _repo_is_dirty(repo_root, _run_shell_command)
        dirty_detail = "clean" if not dirty else "dirty working tree"
    except Exception as exc:  # pragma: no cover
        dirty_detail = str(exc)
    add("working_tree_clean", not dirty, dirty_detail)

    return {
        "repo_dir": str(repo_root),
        "checked_at": _now_iso(),
        "ready": all(item["ok"] for item in checks if item["name"] not in {"working_tree_clean"}),
        "checks": checks,
    }


def _critique_run(root: Path, run_dir: Path, status: dict[str, Any]) -> list[dict[str, Any]]:
    overview = _read_json(run_dir / "overview.json")
    graph = _read_json(run_dir / "graph.json")
    papers = read_jsonl_dicts(run_dir / "papers.jsonl")
    claims = read_jsonl_dicts(run_dir / "claims.jsonl")
    insights = read_jsonl_dicts(run_dir / "insights.jsonl")
    heuristic = _heuristic_issues(run_dir.name, status, overview, graph, papers, claims, insights)
    llm_issues = _llm_critic_issues(root, run_dir.name, status, overview, heuristic)
    return llm_issues or heuristic


def _heuristic_issues(
    run_id: str,
    status: dict[str, Any],
    overview: dict[str, Any],
    graph: dict[str, Any],
    papers: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    insights: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    topic = str(status.get("topic") or "")
    nodes = graph.get("nodes") or []
    long_labels = []
    for node in nodes:
        if str(node.get("node_type") or "") in {"Task", "Method", "Mechanism", "TemporalProperty"}:
            label = str(node.get("name") or node.get("value") or "")
            if len(label) > 34 or ":" in label or "(" in label:
                long_labels.append(label)
    if long_labels:
        issues.append(
            _issue(
                run_id=run_id,
                topic=topic,
                kind="keyword_readability",
                severity="medium",
                summary="Keyword labels are too technical or too long to parse quickly.",
                evidence=long_labels[0],
                suggested_fix="Add a humanized label plus linked source papers/claims in the detail panel.",
                component="visualize",
            )
        )
    if len(nodes) > 180:
        issues.append(
            _issue(
                run_id=run_id,
                topic=topic,
                kind="graph_clarity",
                severity="medium",
                summary="The graph is likely too dense for a first view.",
                evidence=f"{len(nodes)} nodes",
                suggested_fix="Keep cluster-first defaults and reduce visible low-value node families by default.",
                component="visualize",
            )
        )
    if claims and not overview.get("best_conflict_lines") and status.get("anomalies", 0) >= 1:
        issues.append(
            _issue(
                run_id=run_id,
                topic=topic,
                kind="report_quality",
                severity="medium",
                summary="The result is missing a concise main-tension line.",
                evidence="overview.best_conflict_lines is empty",
                suggested_fix="Improve curated line ranking or anomaly phrasing cleanup.",
                component="overview",
            )
        )
    generic_insight = next(
        (
            insight
            for insight in insights
            if any(term in str(insight.get("title") or "").lower() for term in GENERIC_TERMS)
        ),
        None,
    )
    if generic_insight:
        issues.append(
            _issue(
                run_id=run_id,
                topic=topic,
                kind="insight_usefulness",
                severity="high",
                summary="A community insight uses vague labels and feels hard to trust.",
                evidence=str(generic_insight.get("title") or ""),
                suggested_fix="Prune vague community labels and prefer mechanism-level bridge titles.",
                component="insights",
            )
        )
    if papers and len(papers) <= 3:
        issues.append(
            _issue(
                run_id=run_id,
                topic=topic,
                kind="retrieval_quality",
                severity="low",
                summary="The run used a very small paper pool, which may flatten the graph.",
                evidence=f"{len(papers)} papers",
                suggested_fix="Increase batch size or relax retrieval gating for under-filled queries.",
                component="paper_select",
            )
        )
    return issues


def _llm_critic_issues(root: Path, run_id: str, status: dict[str, Any], overview: dict[str, Any], heuristic: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not configured_api_key():
        return []
    prompt_path = root / "prompts" / "critic.md"
    system_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "Return strict JSON issues."
    try:
        client = build_openai_client()
        raw = call_llm_text(
            client,
            model=configured_model(),
            system=system_prompt,
            user=json.dumps(
                {
                    "run_id": run_id,
                    "topic": status.get("topic"),
                    "status_summary": {
                        "papers": status.get("papers", 0),
                        "claims": status.get("claims", 0),
                        "anomalies": status.get("anomalies", 0),
                        "insights": status.get("insights", 0),
                    },
                    "overview": overview,
                    "heuristic_findings": heuristic,
                    "output_schema": {
                        "issues": [
                            {
                                "kind": "keyword_readability|graph_clarity|retrieval_quality|insight_usefulness|report_quality|anomaly_usefulness|demo_worthiness",
                                "severity": "low|medium|high",
                                "summary": "string",
                                "evidence": "string",
                                "suggested_fix": "string",
                                "component": "string",
                            }
                        ]
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            temperature=0.0,
            max_tokens=900,
        )
        parsed = json.loads(raw)
    except Exception:
        return []
    issues = parsed.get("issues") if isinstance(parsed, dict) else None
    if not isinstance(issues, list):
        return []
    normalized: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        kind = str(issue.get("kind") or "").strip()
        severity = str(issue.get("severity") or "medium").strip().lower()
        if kind not in ISSUE_KINDS:
            continue
        if severity not in SEVERITY_WEIGHT:
            severity = "medium"
        normalized.append(
            _issue(
                run_id=run_id,
                topic=str(status.get("topic") or ""),
                kind=kind,
                severity=severity,
                summary=str(issue.get("summary") or ""),
                evidence=str(issue.get("evidence") or ""),
                suggested_fix=str(issue.get("suggested_fix") or ""),
                component=str(issue.get("component") or ""),
            )
        )
    return normalized


def _generate_topic_variants(root: Path, harvested: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not harvested:
        return []
    if configured_api_key():
        try:
            client = build_openai_client()
            prompt_path = root / "prompts" / "topic_generator.md"
            system_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "Return strict JSON."
            raw = call_llm_text(
                client,
                model=configured_model(),
                system=system_prompt,
                user=json.dumps(
                    {
                        "topics": [item["topic"] for item in harvested[:6]],
                        "goal": "Generate 4-8 retrieval-friendly adjacent topics for aigraph.",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                temperature=0.0,
                max_tokens=600,
            )
            parsed = json.loads(raw)
            generated = parsed.get("topics") if isinstance(parsed, dict) else None
            if isinstance(generated, list):
                return [
                    _normalize_topic_record(item, source="llm-expand")
                    for item in generated
                    if isinstance(item, (dict, str))
                ]
        except Exception:
            pass
    variants: list[dict[str, Any]] = []
    for item in harvested[:6]:
        decomposition = decompose_topic_query(item["topic"])
        core = decomposition.get("core_terms") or []
        modifiers = decomposition.get("modifiers") or []
        if core:
            variants.append(
                _normalize_topic_record(
                    {
                        "topic": " ".join((core + modifiers[:2])[:6]),
                        "priority": max(0.4, float(item.get("priority", 0.5)) - 0.1),
                    },
                    source="heuristic-expand",
                )
            )
    return variants


def _normalize_topic_record(item: dict[str, Any] | str, *, source: str) -> dict[str, Any]:
    if isinstance(item, str):
        topic = item
        priority = 0.5
        source_name = source
        source_pref = "arxiv"
        strategy = "balanced"
        limit = 10
        citation_weight = 0.45
        min_relevance = 0.30
    else:
        topic = str(item.get("topic") or "").strip()
        priority = float(item.get("priority") or 0.5)
        source_name = str(item.get("source") or source)
        source_pref = str(item.get("search_source") or item.get("source_pref") or item.get("source") or "arxiv")
        strategy = str(item.get("strategy") or "balanced")
        limit = int(item.get("limit") or 10)
        citation_weight = float(item.get("citation_weight") or 0.45)
        min_relevance = float(item.get("min_relevance") or 0.30)
    return {
        "topic": " ".join(topic.split()),
        "priority": max(0.1, min(1.0, priority)),
        "source_name": source_name,
        "source": source_pref if source_pref in {"arxiv", "openalex"} else "arxiv",
        "strategy": strategy if strategy in {"balanced", "high-impact", "recent"} else "balanced",
        "limit": max(5, min(30, limit)),
        "citation_weight": max(0.0, min(0.85, citation_weight)),
        "min_relevance": max(0.0, min(1.0, min_relevance)),
        "insight_generator": "llm",
    }


def _load_seed_topics(root: Path) -> list[dict[str, Any]]:
    path = root / "topics" / "seeds.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    topics = payload.get("topics") if isinstance(payload, dict) else payload
    if not isinstance(topics, list):
        return []
    return [item for item in topics if isinstance(item, (dict, str))]


def _recent_queries(runs_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    analytics_path = runs_dir / "_analytics" / "requests.jsonl"
    rows = _read_jsonl(analytics_path)
    submits = [row for row in rows if row.get("event") == "search_submit" and row.get("topic")]
    return submits[-limit:]


def _rank_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(issue.get("dedupe_key") for issue in issues)
    return sorted(
        issues,
        key=lambda issue: (
            counts.get(issue.get("dedupe_key"), 1),
            SEVERITY_WEIGHT.get(str(issue.get("severity") or "low"), 1),
            issue.get("created_at", ""),
        ),
        reverse=True,
    )


def _issue(
    *,
    run_id: str,
    topic: str,
    kind: str,
    severity: str,
    summary: str,
    evidence: str,
    suggested_fix: str,
    component: str,
) -> dict[str, Any]:
    dedupe_key = f"{kind}:{component}:{summary}".lower()
    return {
        "issue_id": f"issue-{datetime.now():%Y%m%d%H%M%S}-{abs(hash((run_id, dedupe_key))) % 100000:05d}",
        "run_id": run_id,
        "topic": topic,
        "kind": kind,
        "severity": severity,
        "summary": summary,
        "evidence": evidence,
        "suggested_fix": suggested_fix,
        "component": component,
        "dedupe_key": dedupe_key,
        "created_at": _now_iso(),
    }


def _write_dashboard(root: Path, state: dict[str, Any], runs_dir: Path | str) -> None:
    runs_root = Path(runs_dir)
    issues = _read_jsonl(root / "issues" / "issues.jsonl")
    runs_index = _read_jsonl(root / "runs" / "runs_index.jsonl")
    dashboard = {
        "last_harvest_at": state.get("last_harvest_at"),
        "last_batch_at": state.get("last_batch_at"),
        "last_critic_at": state.get("last_critic_at"),
        "last_fix_bundle_at": state.get("last_fix_bundle_at"),
        "last_fix_session_at": state.get("last_fix_session_at"),
        "last_fix_session_status": state.get("last_fix_session_status"),
        "pending_topics": len(state.get("pending_topics") or []),
        "runs_processed": len(runs_index),
        "issues_total": len(issues),
        "issues_today": sum(1 for issue in issues if str(issue.get("created_at", "")).startswith(f"{datetime.now():%Y-%m-%d}")),
        "latest_runs_dir": str(runs_root),
    }
    _write_json(root / "state" / "dashboard.json", dashboard)


def _default_runtime_state() -> dict[str, Any]:
    return {
        "last_harvest_at": None,
        "last_batch_at": None,
        "last_critic_at": None,
        "last_fix_bundle_at": None,
        "last_fix_session_at": None,
        "last_fix_session_status": None,
        "pending_topics": [],
        "recent_topic_runs": {},
        "critiqued_runs": [],
    }


def _load_runtime_state(root: Path) -> dict[str, Any]:
    path = root / "state" / "runtime.json"
    if not path.exists():
        return _default_runtime_state()
    return _read_json(path) or _default_runtime_state()


def _save_runtime_state(root: Path, state: dict[str, Any]) -> None:
    _write_json(root / "state" / "runtime.json", state)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _build_fix_prompt(root: Path, bundle: dict[str, Any], repo_root: Path, branch: str, test_command: str) -> str:
    prompt_path = root / "prompts" / "fixer.md"
    base = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
    issues = bundle.get("issues") or []
    issue_lines = []
    for idx, issue in enumerate(issues, start=1):
        issue_lines.append(
            textwrap.dedent(
                f"""
                {idx}. [{issue.get('severity','medium')}] {issue.get('kind','issue')} in `{issue.get('component','unknown')}`
                   - summary: {issue.get('summary','')}
                   - evidence: {issue.get('evidence','')}
                   - suggested fix: {issue.get('suggested_fix','')}
                   - run: {issue.get('run_id','')}
                   - artifacts:
                     - status: {issue.get('artifacts', {}).get('status', '')}
                     - overview: {issue.get('artifacts', {}).get('overview', '')}
                     - graph: {issue.get('artifacts', {}).get('graph', '')}
                     - report: {issue.get('artifacts', {}).get('report', '')}
                """
            ).strip()
        )
    return textwrap.dedent(
        f"""
        {base}

        Repository root: {repo_root}
        Working branch: {branch}
        Validation command: {test_command}

        Prioritized issues:
        {os.linesep.join(issue_lines)}

        Requirements:
        - Keep fixes tightly scoped to the listed issues.
        - Prefer small, testable edits.
        - After editing, run the validation command.
        - Do not invent product scope beyond these issues.
        - Leave the repo ready for commit and draft PR creation.
        """
    ).strip() + "\n"


def _build_pr_body(bundle: dict[str, Any]) -> str:
    issues = bundle.get("issues") or []
    bullets = []
    for issue in issues:
        bullets.append(f"- `{issue.get('kind', 'issue')}`: {issue.get('summary', '')}")
    body = ["## Summary", "", "Automated fixer session based on recent aigraph run critiques.", "", "## Issues addressed", ""]
    body.extend(bullets or ["- No issues selected."])
    body.extend(["", "## Validation", "", "- Automated tests executed by the fixer session before PR creation."])
    return "\n".join(body) + "\n"


def _build_commit_message(bundle: dict[str, Any]) -> str:
    issues = bundle.get("issues") or []
    if not issues:
        return "Apply automated aigraph fixes"
    top = issues[0]
    kind = str(top.get("kind") or "automation").replace("_", " ")
    return f"Fix aigraph {kind} issues"


def _build_pr_title(bundle: dict[str, Any]) -> str:
    issues = bundle.get("issues") or []
    if not issues:
        return "Automated aigraph fixes"
    top = issues[0]
    kind = str(top.get("kind") or "automation").replace("_", " ")
    return f"Automated fix: {kind}"


def _step(result: dict[str, Any], name: str, *, command: str) -> None:
    result.setdefault("steps", []).append({"name": name, "command": command, "at": _now_iso()})


def _repo_is_dirty(
    repo_root: Path,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> bool:
    proc = command_runner("git status --porcelain", cwd=repo_root)
    return bool((proc.stdout or "").strip())


def _repo_has_no_changes(
    repo_root: Path,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> bool:
    proc = command_runner("git status --porcelain", cwd=repo_root)
    return not bool((proc.stdout or "").strip())


def _git_head(
    repo_root: Path,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> str | None:
    proc = command_runner("git rev-parse HEAD", cwd=repo_root)
    head = (proc.stdout or "").strip()
    return head or None


def _run_shell_command(command: str, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _update_fix_session_state(root: Path, runs_root: Path, result: dict[str, Any]) -> None:
    state = _load_runtime_state(root)
    state["last_fix_session_at"] = result.get("created_at")
    state["last_fix_session_status"] = result.get("status")
    _save_runtime_state(root, state)
    _write_dashboard(root, state, runs_root)


def _relative_or_absolute(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()
