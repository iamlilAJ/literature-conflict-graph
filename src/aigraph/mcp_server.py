"""MCP (Model Context Protocol) surface for aigraph v0.7-frozen runs.

Streamable-HTTP FastMCP server. Mounted into the FastAPI app in
``web.create_app`` at ``/mcp`` (one process, one runs_root, shared
with the browser UI). Build standalone with ``build_mcp(runs_root)``.

Tools
-----
Read-only (0 LLM, sub-5s — reuse the cached query layer):
  list_runs()                          — discovered runs + counts
  get_run_summary(run)                 — metadata + anomaly-type histogram
  query_hypotheses(topic, run, k)      — structured top-K hypotheses + stats
  get_idea_report(topic, run, k)       — rendered Stage 3 markdown deliverable
  get_conflict_graph(topic, run, k)    — D3 {nodes, edges}

Run-trigger (slow, costs $ — reuse server.SearchService background worker):
  start_run(topic, max_papers, ...)    — kick off pipeline, return run_id
  get_run_status(run_id)               — poll status.json

Only ``start_run`` makes LLM calls / costs money. Everything else is a
cache read. No auth — fine for localhost / trusted LAN; gate behind a
token before exposing run-trigger publicly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
from aigraph_query import query_records  # noqa: E402

from . import web as _web  # noqa: E402


def _safe_run_dir(runs_root: Path, run_id: str) -> Optional[Path]:
    """Resolve run_id under runs_root. Returns None on path traversal."""
    run_dir = (runs_root / run_id).resolve()
    if not str(run_dir).startswith(str(runs_root.resolve())):
        return None
    return run_dir


def _coverage_banner(stats: dict) -> str:
    """A one-line coverage assessment for a topic query, so the Stage 3 reader
    (critic / downstream) knows how tightly the corpus actually covers the
    topic. Topical looseness comes from a topic the pre-built corpus doesn't
    cover well — surface that instead of silently returning loose hypotheses."""
    n = int(stats.get("n_matched", 0) or 0)
    r = int(stats.get("top_relevance", 0) or 0)
    tot = int(stats.get("n_hypotheses_total", 0) or 0)
    # Use the FRACTION matched, not absolute count, so the assessment is fair
    # across corpus sizes (a focused 45-hypothesis topic corpus shouldn't look
    # "weak" just because it has fewer total hypotheses than a 300+ one).
    frac = (n / tot) if tot else 0.0
    if n == 0:
        level, note = "none", "the corpus has no matching hypotheses — build a topic-specific corpus with `start_run`."
    elif frac >= 0.40 or (r >= 3 and n >= 80):
        level, note = "strong", "the corpus covers this topic well."
    elif frac >= 0.12 or (r >= 2 and n >= 30):
        level, note = "moderate", "partial coverage; some hypotheses may be loosely related."
    else:
        level, note = "weak", "hypotheses may not tightly address the topic — consider `start_run` for a topic-specific corpus."
    return (f"> **Corpus coverage: {level}** "
            f"({n}/{tot} hypotheses matched, top relevance {r}). {note}\n\n")


def _run_summary(run_dir: Path) -> dict[str, Any]:
    """Counts + anomaly-type histogram + provenance from run_metadata.json."""
    summary: dict[str, Any] = {"run_id": run_dir.name}
    scored = run_dir / "hypotheses_scored.jsonl"
    if not scored.exists():
        scored = run_dir / "hypotheses.jsonl"
    summary["n_hypotheses"] = sum(1 for _ in scored.open()) if scored.exists() else 0
    papers = run_dir / "papers.jsonl"
    summary["n_papers"] = sum(1 for _ in papers.open()) if papers.exists() else 0
    claims = run_dir / "claims.jsonl"
    summary["n_claims"] = sum(1 for _ in claims.open()) if claims.exists() else 0

    # anomaly-type histogram from the capped set if present
    from collections import Counter
    hist: Counter = Counter()
    for name in ("anomalies_top.jsonl", "anomalies_top_stratified.jsonl", "anomalies.jsonl"):
        p = run_dir / name
        if p.exists():
            for line in p.open():
                try:
                    hist[json.loads(line).get("type")] += 1
                except Exception:
                    pass
            break
    summary["anomaly_types"] = dict(hist)

    meta = run_dir / "run_metadata.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            for key in ("git_sha", "git_tag", "max_anomalies_cap", "model", "topic"):
                if key in m:
                    summary[key] = m[key]
        except Exception:
            pass
    return summary


def build_mcp(
    runs_root: Path | str,
    search_service: Optional[Any] = None,
    *,
    readonly: bool = False,
) -> FastMCP:
    """Build the FastMCP server.

    ``search_service`` is an optional ``aigraph.server.SearchService`` for
    the run-trigger tools; if None, those tools report run-triggering is
    disabled.

    ``readonly`` (default False): when True, the paid run-trigger tools
    (``start_run``, ``get_run_status``) are NOT registered at all. Use
    this for any network-exposed / public deployment so an anonymous
    caller cannot trigger LLM runs that cost money. The 4 read-only,
    0-LLM tools remain available.
    """
    runs_root = Path(runs_root)
    mcp = FastMCP(
        "aigraph",
        instructions=(
            "aigraph exposes a frozen literature-conflict pipeline's cached "
            "output. Use query_hypotheses to retrieve testable hypotheses "
            "(conflict explanations + cross-field bridge ideas) grounded in "
            "real paper claims for a topic, scoped to one corpus 'run'. "
            "list_runs first to see available corpora. All query_* tools are "
            "instant and free; only start_run costs money/time."
        ),
        stateless_http=True,
        streamable_http_path="/",
    )

    @mcp.tool()
    def list_runs() -> list[dict]:
        """List available corpora (runs). Each: id, n_papers, n_hypotheses.
        Pass the `id` as the `run` argument to other tools."""
        return _web._discover_runs(runs_root)

    @mcp.tool()
    def get_run_summary(run: str) -> dict:
        """Metadata for one run: paper/claim/hypothesis counts, anomaly-type
        histogram, git_sha/git_tag, model, anomaly cap."""
        run_dir = _safe_run_dir(runs_root, run)
        if run_dir is None or not run_dir.exists():
            return {"error": f"unknown run: {run}"}
        return _run_summary(run_dir)

    @mcp.tool()
    def query_hypotheses(topic: str, run: str, k: int = 5) -> dict:
        """Retrieve the top-K hypotheses most relevant to `topic` from `run`.

        0 LLM calls. Returns structured records (hypothesis text, mechanism,
        predictions, minimal_test, evidence_claims with paper titles, utility
        breakdown) plus match stats. `k` in 1..20."""
        k = max(1, min(20, int(k)))
        run_dir = _safe_run_dir(runs_root, run)
        if run_dir is None or not run_dir.exists():
            return {"error": f"unknown run: {run}"}
        try:
            records, stats = query_records(run_dir, topic, k=k, min_anomalies=1)
            return {"topic": topic, "run": run, "hypotheses": records, "stats": stats}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    @mcp.tool()
    def get_idea_report(topic: str, run: str, k: int = 8, out_path: str = "", kind: str = "critic") -> str:
        """Render the Stage 3 'Idea Generation' deliverable for `topic` as a
        COMPLETE markdown document (0 LLM, sub-second).

        `kind` selects the hypothesis type: "critic" (default — conflict
        explanations, `### h…` ids that the frontend/critic parse), "creator"
        (new-method-proposal research ideas — higher quality but use `### a…#cr…`
        ids; needs the frontend regex + Stage-3 critic updated to accept them
        before it can be the default), or "both".

        The document is a `# Stage 3: Idea Generation — <topic>` heading
        followed by the `# Selected Hypotheses` report (`### Anomaly a… —` /
        `### h… —` items grounded in real claim citations). This is the
        canonical Stage 3 format the downstream critic and the frontend
        conflict-graph renderer expect. `k` in 1..20.

        If `out_path` is given (an ABSOLUTE path to the stage deliverable, e.g.
        the project workspace's stage3_idea_generator.md), the report is WRITTEN
        there directly and only a short confirmation string is returned — so the
        caller does NOT have to re-emit the (large) document through a write
        tool. If `out_path` is empty, the full document text is returned as
        before (caller writes it verbatim)."""
        k = max(1, min(20, int(k)))
        heading = f"# Stage 3: Idea Generation — {topic}\n\n"
        run_dir = _safe_run_dir(runs_root, run)
        if run_dir is None or not run_dir.exists():
            doc = heading + f"_Invalid run:_ `{run}`\n"
        else:
            try:
                from aigraph_query import query  # markdown renderer (0 LLM)

                # Tighter selection than the query() defaults (max_hypotheses=30,
                # mmr_lambda=0.7): a smaller candidate pool + higher relevance
                # weight surfaces hypotheses that are actually on-topic instead
                # of diverse-but-generic cross-field ones (measured: RAG topic
                # overlap 0.08->0.21, code-gen 0.25->0.56).
                md, _stats = query(run_dir, topic, k=k,
                                   max_hypotheses=12, mmr_lambda=0.85,
                                   hyp_kind=kind)
                doc = heading + _coverage_banner(_stats) + md
            except Exception as exc:
                doc = heading + f"_query failed: {type(exc).__name__}: {exc}_\n"

        if out_path:
            from pathlib import Path as _P
            try:
                p = _P(out_path).expanduser()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(doc, encoding="utf-8")
                n_h = doc.count("\n### h")
                return (f"OK: wrote Stage 3 Selected Hypotheses report to {p} "
                        f"({len(doc)} bytes, {doc.count(chr(10)) + 1} lines, {n_h} hypotheses). "
                        f"Do NOT re-write it — the deliverable is already on disk.")
            except Exception as exc:
                # Fall back to returning the text so the caller can still write it.
                return doc
        return doc

    @mcp.tool()
    def get_conflict_graph(topic: str, run: str, k: int = 5, ids: str = "") -> dict:
        """Topic-filtered conflict graph for `run` as D3 {nodes, edges}.

        Nodes: topic centre, selected hypotheses, their anomalies, shared
        entities (method/task/dataset), graph-bridge targets. Optional `ids`:
        comma-separated hypothesis IDs to pin the graph to specific hypotheses
        (skips topic-relevance selection)."""
        k = max(1, min(20, int(k)))
        run_dir = _safe_run_dir(runs_root, run)
        if run_dir is None or not run_dir.exists():
            return {"error": f"unknown run: {run}"}
        id_list = None
        if ids.strip():
            id_list = [s.strip() for s in ids.split(",") if s.strip() and len(s.strip()) <= 8]
        try:
            return _web._build_topic_graph(run_dir, topic, k, id_list)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    # Paid run-trigger tools — only registered when NOT readonly, so a
    # network-exposed/public MCP cannot trigger money-spending LLM runs.
    if not readonly:

        @mcp.tool()
        def start_run(topic: str, max_papers: int = 100, generator: str = "llm") -> dict:
            """Kick off a NEW pipeline run for `topic` (fetches papers, extracts
            claims, builds graph, detects anomalies, generates hypotheses).

            SLOW (minutes-to-hours) and COSTS MONEY (LLM calls). Returns a run_id
            immediately; poll get_run_status(run_id). When done, the run appears
            in list_runs and is queryable via query_hypotheses."""
            if search_service is None:
                return {"error": "run-trigger disabled (no SearchService configured)"}
            try:
                req = search_service.submit(
                    topic=topic,
                    limit=max(1, min(500, int(max_papers))),
                    insight_generator=generator,
                )
                return {
                    "run_id": req.run_id,
                    "status": "queued",
                    "poll_with": "get_run_status",
                    "note": "LLM pipeline running in background; this is the only paid tool",
                }
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}

        @mcp.tool()
        def get_run_status(run_id: str) -> dict:
            """Poll the status of a run started with start_run. Returns the
            run's status.json (stage / progress / done / error)."""
            run_dir = _safe_run_dir(runs_root, run_id)
            if run_dir is None:
                return {"error": f"invalid run id: {run_id}"}
            status_path = run_dir / "status.json"
            if not status_path.exists():
                return {"error": f"no status for run: {run_id}"}
            try:
                return json.loads(status_path.read_text())
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}

    return mcp
