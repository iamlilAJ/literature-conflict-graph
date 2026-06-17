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
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

try:  # MCP SDK >=1.x ships DNS-rebinding protection that 421s non-localhost Host headers
    from mcp.server.transport_security import TransportSecuritySettings
except Exception:  # pragma: no cover - older SDKs have no host check
    TransportSecuritySettings = None  # type: ignore

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


def _log_query(runs_root: Path, **fields: Any) -> None:
    """Append a read-only QUERY event to ``runs_root/query_log.jsonl``.

    Read-only tools (`get_idea_report`, `query_hypotheses`, `generate_ideas`)
    answer from an existing corpus and create NO run dir / status.json, so they
    never show up in the runs dashboard. This log gives the dashboard a record
    of those grounding calls (e.g. Memento/OMC Stage 3 hitting `get_idea_report`)
    so they are no longer invisible. Best-effort — must never break a query."""
    try:
        row = {"ts": datetime.now().isoformat(timespec="seconds"), **fields}
        with (runs_root / "query_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


# Absolute matched-count floors: a corpus with only a couple of matching
# hypotheses cannot be "strong"/"moderate" however high its fraction is — a 2/2
# (frac=1.0) match on a 2-hypothesis leftover run is thin, not strong. The floor
# prevents tiny corpora from masquerading as well-covered (the fraction test
# alone called 2/2 "strong"). A focused 22/24 corpus (n=22) still reads strong.
_STRONG_N_FLOOR = 8
_MODERATE_N_FLOOR = 4


def _coverage_banner(stats: dict) -> str:
    """A one-line coverage assessment for a topic query, so the Stage 3 reader
    (critic / downstream) knows how tightly the corpus actually covers the
    topic. Topical looseness comes from a topic the pre-built corpus doesn't
    cover well — surface that instead of silently returning loose hypotheses."""
    n = int(stats.get("n_matched", 0) or 0)
    r = int(stats.get("top_relevance", 0) or 0)
    tot = int(stats.get("n_hypotheses_total", 0) or 0)
    # FRACTION matched keeps the assessment fair across corpus sizes (a focused
    # 45-hyp topic corpus shouldn't look "weak" vs a 300+ one), but gate it on an
    # absolute matched-count floor so a 2/2 thin corpus can't read "strong".
    frac = (n / tot) if tot else 0.0
    if n == 0:
        level, note = "none", "the corpus has no matching hypotheses — build a topic-specific corpus with `start_run`."
    elif (frac >= 0.40 or (r >= 3 and n >= 80)) and n >= _STRONG_N_FLOOR:
        level, note = "strong", "the corpus covers this topic well."
    elif (frac >= 0.12 or (r >= 2 and n >= 30)) and n >= _MODERATE_N_FLOOR:
        level, note = "moderate", "partial coverage; some hypotheses may be loosely related."
    else:
        level, note = "weak", "few matching hypotheses — the corpus may not tightly cover the topic; consider `start_run` for a topic-specific corpus."
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
    # Public deployment binds 0.0.0.0; the SDK's default DNS-rebinding protection
    # only trusts localhost Host headers, so external clients (e.g. a teammate
    # hitting the public IP) get HTTP 421 "Invalid Host header". This is a
    # server-to-server, query-only endpoint, so relax the Host/Origin allowlist.
    _security = None
    if TransportSecuritySettings is not None:
        _security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
            allowed_hosts=["*"],
            allowed_origins=["*"],
        )
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
        transport_security=_security,
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
            _log_query(runs_root, tool="query_hypotheses", topic=topic, run=run, k=k,
                       n_matched=stats.get("n_matched"), n_total=stats.get("n_hypotheses_total"),
                       top_relevance=stats.get("top_relevance"), returned=len(records))
            return {"topic": topic, "run": run, "hypotheses": records, "stats": stats}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    @mcp.tool()
    def get_idea_report(topic: str, run: str, k: int = 8, out_path: str = "", kind: str = "creator") -> str:
        """Render the Stage 3 'Idea Generation' deliverable for `topic` as a
        COMPLETE markdown document (0 LLM, sub-second).

        `kind` selects the hypothesis type: "creator" (default — new-method
        -proposal research ideas, `### a…#cr…` ids; these are the forward-looking
        ideas and are the better Stage-3 deliverable), "critic" (conflict
        explanations, `### h…` ids), or "both". For a run with no
        creator_hypotheses.jsonl, "creator"/"both" fall back to critic
        automatically. The frontend conflict-graph renderer and the Stage-3
        critic accept BOTH id shapes (`### h…` and `### a…#cr…`).

        The document is a `# Stage 3: Idea Generation — <topic>` heading
        followed by the `# Selected Hypotheses` report (`### Anomaly a… —` /
        `### h… —` / `### a…#cr… —` items grounded in real claim citations). This is the
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
                # min_anomalies=3 (vs default 2): with the enlarged creator pool
                # a single high-utility anomaly could fill 3 of 4 slots (measured:
                # "tree search reasoning" → 3x a016 tool ideas). Forcing 3 distinct
                # anomalies in the top-k spreads selection across the on-topic
                # anomalies (brings in a012 reasoning for that topic).
                # min_atlas_overlap=3 enables the Atlas-overlap delivery
                # filter (Methods 12, 13, 15, 28, 29 in docs/method*.md).
                # Drops the ~20% of hyps the production scorer leaks into
                # top-K that are tangential or unrelated to any known Atlas
                # bottleneck. Safe no-op for runs without an
                # atlas_overlap.jsonl sidecar (defensive: unscored hyps kept).
                md, _stats = query(run_dir, topic, k=k,
                                   max_hypotheses=12, mmr_lambda=0.85,
                                   min_anomalies=3, hyp_kind=kind,
                                   min_atlas_overlap=3, demote_weak_anomalies=True)
                doc = heading + _coverage_banner(_stats) + md
                _log_query(runs_root, tool="get_idea_report", topic=topic, run=run, k=k, kind=kind,
                           n_matched=_stats.get("n_matched"), n_total=_stats.get("n_hypotheses_total"),
                           top_relevance=_stats.get("top_relevance"), chars=len(doc))
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

    @mcp.tool()
    def generate_ideas(topic: str, run: str, min_ideas: int = 5,
                       as_markdown: bool = True) -> Any:
        """Generate research ideas for `topic` from an existing `run`, with a
        guarantee the result is NEVER empty as long as the run has >=1 paper.

        Cascades five tiers from highest-signal to most-permissive, stopping
        when it has >= min_ideas:
          A critic-conflict   (cross-paper contradictions; needs anomalies)
          B creator-newmethod (new methods grounded in open questions)
          C community-bridge  (cross-community unifying insights)
          D method-extension  (per-paper "extend this method"; LLM)
          E limitation-forward(turn limitations into directions; LLM)
          F paper-seeded      (deterministic, LLM-free terminal backstop)

        Unlike start_run, this works on a sparse/too-new corpus where no
        cross-paper anomalies form (so critic+creator are empty): it falls
        through to community bridges, then per-paper LLM ideas, then a
        deterministic abstract-seeded backstop that cannot fail. Tiers D/E are
        cached to <run>/forward_ideas.jsonl so repeat calls are 0-LLM.

        `min_ideas`: target count (clamped 1..20). `as_markdown`: return a
        rendered report string (default) or the structured {ideas, stats} dict.
        On a public/readonly deployment the paid D/E tiers are disabled and the
        cascade relies on cached + deterministic tiers."""
        min_ideas = max(1, min(20, int(min_ideas)))
        run_dir = _safe_run_dir(runs_root, run)
        if run_dir is None or not run_dir.exists():
            return {"error": f"unknown run: {run}"}
        try:
            # idea_cascade lives in scripts/, already on sys.path (line ~33).
            import idea_cascade as ic
            result = ic.generate_ideas(run_dir, topic, min_ideas=min_ideas,
                                       allow_llm=not readonly)
            if as_markdown:
                return ic.render_ideas_markdown(result)
            return result
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
                    source="union",  # multi-source recall for fresh corpora (#47)
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

        @mcp.tool()
        def research_ideas(topic: str, max_papers: int = 50, min_ideas: int = 5,
                          reuse: bool = True, wait_seconds: int = 0,
                          as_markdown: bool = True) -> Any:
            """One-shot: topic in → ideas out. Reuses a matching existing corpus
            if one exists (instant), otherwise builds a fresh one, then runs the
            generate_ideas cascade — guaranteeing a NON-EMPTY idea set.

            Flow:
              1. If `reuse`, find the best existing run whose topic matches and
                 generate ideas from it immediately (0 corpus-build cost).
              2. Otherwise kick off a new corpus build (start_run-style, SLOW +
                 paid), poll up to `wait_seconds`, and on completion generate
                 ideas. If the build is not done within the budget, returns
                 {status:"building", run_id} so you can poll get_run_status and
                 then call generate_ideas(run=run_id) yourself.

            `wait_seconds` is clamped to 0..1500; 0 = submit-and-return (no
            block). For small corpora (max_papers<=30, ~3-5 min) a wait of
            ~600 makes it truly one-shot. `min_ideas` clamped 1..20."""
            import time as _time
            min_ideas = max(1, min(20, int(min_ideas)))
            wait_seconds = max(0, min(1500, int(wait_seconds)))
            try:
                import idea_cascade as ic
            except Exception as exc:
                return {"error": f"cascade import failed: {exc}"}

            def _ideas_for(run_id: str, reused: bool) -> dict:
                rdir = _safe_run_dir(runs_root, run_id)
                result = ic.generate_ideas(rdir, topic, min_ideas=min_ideas,
                                           allow_llm=not readonly)
                out = {"status": "done", "run": run_id, "reused": reused,
                       "stats": result.get("stats", {})}
                out["ideas_markdown" if as_markdown else "ideas"] = (
                    ic.render_ideas_markdown(result) if as_markdown else result.get("ideas", []))
                return out

            # 1. reuse an existing corpus
            if reuse:
                try:
                    rid = ic.resolve_best_run(runs_root, topic)
                except Exception:
                    rid = None
                if rid:
                    try:
                        return _ideas_for(rid, reused=True)
                    except Exception as exc:
                        return {"error": f"{type(exc).__name__}: {exc}"}

            # 2. build a fresh corpus
            if search_service is None:
                return {"error": "no matching corpus and run-trigger disabled "
                                 "(no SearchService configured)"}
            try:
                req = search_service.submit(
                    topic=topic, limit=max(1, min(500, int(max_papers))),
                    insight_generator="llm", source="union")  # multi-source (#47)
            except Exception as exc:
                return {"error": f"submit failed: {type(exc).__name__}: {exc}"}
            run_id = req.run_id
            run_dir = _safe_run_dir(runs_root, run_id)

            # 3. bounded poll
            waited = 0
            while waited < wait_seconds:
                _time.sleep(5)
                waited += 5
                sp = (run_dir / "status.json") if run_dir else None
                if sp is None or not sp.exists():
                    continue
                try:
                    st = json.loads(sp.read_text())
                except Exception:
                    continue
                state = st.get("status")
                if state == "done":
                    try:
                        return _ideas_for(run_id, reused=False)
                    except Exception as exc:
                        return {"error": f"{type(exc).__name__}: {exc}", "run": run_id}
                if state == "error":
                    return {"status": "error", "run": run_id,
                            "message": st.get("message") or st.get("error") or "run failed"}

            return {"status": "building", "run_id": run_id,
                    "poll_with": "get_run_status",
                    "next": f"once status=done, call generate_ideas(run='{run_id}')",
                    "note": "corpus build runs in the background (minutes)"}

    @mcp.tool()
    def research_e2e(topic: str, max_papers: int = 30, min_ideas: int = 5,
                     k: int = 8, reuse: bool = True, wait_seconds: int = 900) -> Any:
        """ONE-SHOT end-to-end: topic in → the whole deliverable bundle out.

        Resolves-or-builds a corpus (like `research_ideas`), then assembles and
        returns IN A SINGLE CALL:
          - `idea_report_markdown` : the Stage-3 'Idea Generation' report
          - `ideas` / `ideas_markdown` : the `generate_ideas` cascade output
          - `graph` : the star-graph `{nodes, edges}`
          - `graph_html` : a SELF-CONTAINED D3 star-graph (星球图) page you can
            drop straight into a browser / iframe
          - `dashboard_url` / `graph_url` : the live dashboard + graph pages
        `wait_seconds` clamped 0..1500 (0 = submit-and-return `{status:building}`).
        The build path is paid + only effective when NOT readonly; the reuse path
        is free."""
        import time as _time
        import idea_cascade as ic
        from aigraph_query import query as _query
        from .run_dashboard import render_graph_page as _graph_page, run_graph as _run_graph
        min_ideas = max(1, min(20, int(min_ideas)))
        k = max(1, min(20, int(k)))
        wait_seconds = max(0, min(1500, int(wait_seconds)))

        def _bundle(run_id: str, reused: bool) -> dict:
            rdir = _safe_run_dir(runs_root, run_id)
            try:
                md, _stats = _query(rdir, topic, k=k, max_hypotheses=12, mmr_lambda=0.85,
                                    min_anomalies=3, hyp_kind="creator", min_atlas_overlap=3,
                                    demote_weak_anomalies=True)
                report_md = f"# Stage 3: Idea Generation — {topic}\n\n" + _coverage_banner(_stats) + md
            except Exception as exc:
                report_md, _stats = f"# Stage 3: Idea Generation — {topic}\n\n_report failed: {exc}_\n", {}
            try:
                ideas_res = ic.generate_ideas(rdir, topic, min_ideas=min_ideas, allow_llm=not readonly)
            except Exception as exc:
                ideas_res = {"ideas": [], "stats": {"error": str(exc)}}
            try:
                graph, graph_html = _run_graph(rdir), _graph_page(rdir)
            except Exception as exc:
                graph, graph_html = {"nodes": [], "edges": [], "error": str(exc)}, ""
            _log_query(runs_root, tool="research_e2e", topic=topic, run=run_id,
                       n_matched=_stats.get("n_matched"), n_total=_stats.get("n_hypotheses_total"),
                       top_relevance=_stats.get("top_relevance"), chars=len(report_md))
            return {
                "status": "done", "run": run_id, "reused": reused,
                "coverage": {"n_matched": _stats.get("n_matched"),
                             "n_total": _stats.get("n_hypotheses_total"),
                             "top_relevance": _stats.get("top_relevance")},
                "idea_report_markdown": report_md,
                "ideas_markdown": ic.render_ideas_markdown(ideas_res) if isinstance(ideas_res, dict) else "",
                "ideas": ideas_res.get("ideas", []) if isinstance(ideas_res, dict) else [],
                "graph": graph,
                "graph_html": graph_html,
                "dashboard_url": f"/dashboard/{run_id}",
                "graph_url": f"/dashboard/{run_id}/graph",
            }

        if reuse:
            try:
                rid = ic.resolve_best_run(runs_root, topic)
            except Exception:
                rid = None
            if rid:
                try:
                    return _bundle(rid, reused=True)
                except Exception as exc:
                    return {"error": f"{type(exc).__name__}: {exc}"}
        if search_service is None:
            return {"error": "no matching corpus and run-trigger disabled (no SearchService configured)"}
        try:
            req = search_service.submit(topic=topic, limit=max(1, min(500, int(max_papers))),
                                        insight_generator="llm", source="union")
        except Exception as exc:
            return {"error": f"submit failed: {type(exc).__name__}: {exc}"}
        run_id = req.run_id
        run_dir = _safe_run_dir(runs_root, run_id)
        waited = 0
        while waited < wait_seconds:
            _time.sleep(5)
            waited += 5
            sp = (run_dir / "status.json") if run_dir else None
            if sp is None or not sp.exists():
                continue
            try:
                st = json.loads(sp.read_text())
            except Exception:
                continue
            if st.get("status") == "done":
                try:
                    return _bundle(run_id, reused=False)
                except Exception as exc:
                    return {"error": f"{type(exc).__name__}: {exc}", "run": run_id}
            if st.get("status") == "error":
                return {"status": "error", "run": run_id, "message": st.get("message") or "run failed"}
        return {"status": "building", "run_id": run_id, "poll_with": "get_run_status",
                "next": f"once status=done, call research_e2e(topic='{topic}') again — reuse will hit the built corpus"}

    @mcp.tool()
    def smart_research(topic: str, run: str = "", k: int = 8, kind: str = "creator",
                       allow_build: bool = True, max_papers: int = 30,
                       min_ideas: int = 5, wait_seconds: int = 900) -> dict:
        """LLM-planned, self-correcting research over the rigid LEXICAL query layer.

        Wraps get_idea_report / research_e2e with ONE mandatory LLM call that
        rewrites whatever you type — a typo, a full sentence, a niche phrase —
        into a query optimised for the bag-of-words matcher (canonical
        hyphenated field terms + domain anchors + distinctive content words),
        then PROBES coverage (0-LLM) and self-corrects:

          1. plan_query(topic)            — mandatory LLM normalize + broader fallbacks
          2. probe each candidate query   — pick the best-coverage one (raw topic
                                            is always a candidate → never worse than baseline)
          3a strong/moderate coverage     — return the full deliverable bundle
          3b weak/none + allow_build      — build a topic-specific corpus, then bundle
          3c weak/none + not allow_build  — return the weak bundle + should_build hint

        Returns the same bundle as research_e2e (idea_report_markdown, ideas,
        graph, graph_html, dashboard/graph urls, coverage) PLUS a `query_plan`
        trace (raw→query, alternates, per-candidate coverage, decision). The
        trace is also written to query_log so the dashboard records the call.

        `run`: pin to a specific corpus; empty = auto-resolve the best match.
        `allow_build`: on weak coverage, build a fresh topic-specific corpus
        (paid; only when a SearchService is configured and not readonly).
        `wait_seconds` clamped 0..1500 for the build path (0 = submit-and-return)."""
        import time as _time
        import idea_cascade as ic
        from aigraph_query import query as _query
        from .run_dashboard import (render_graph_page as _graph_page,
                                    run_graph as _run_graph, _coverage_level)
        from .query_planner import plan_query, candidate_queries

        k = max(1, min(20, int(k)))
        min_ideas = max(1, min(20, int(min_ideas)))
        wait_seconds = max(0, min(1500, int(wait_seconds)))
        _RANK = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}

        # 1. mandatory LLM normalize of the rigid lexical input
        plan = plan_query(topic)
        cands = candidate_queries(plan)

        def _probe(run_dir: Path, q: str):
            md, st = _query(run_dir, q, k=k, max_hypotheses=12, mmr_lambda=0.85,
                            min_anomalies=3, hyp_kind=kind, min_atlas_overlap=3,
                            demote_weak_anomalies=True)
            n = int(st.get("n_matched", 0) or 0)
            tot = int(st.get("n_hypotheses_total", 0) or 0)
            r = int(st.get("top_relevance", 0) or 0)
            return md, st, _coverage_level(n, tot, r)

        def _best_over(run_dir: Path):
            """Probe every candidate query on run_dir; return (winner, probes)."""
            best = None  # (key, query, md, stats, level)
            probes: list[dict] = []
            for q in cands:
                try:
                    md, st, level = _probe(run_dir, q)
                except Exception:
                    continue
                n = int(st.get("n_matched", 0) or 0)
                r = int(st.get("top_relevance", 0) or 0)
                probes.append({"query": q, "coverage": level, "n_matched": n,
                               "n_total": int(st.get("n_hypotheses_total", 0) or 0),
                               "top_relevance": r})
                key = (_RANK.get(level, 0), n, r)
                if best is None or key > best[0]:
                    best = (key, q, md, st, level)
            return best, probes

        def _assemble(run_id, win_q, win_md, win_st, win_level, *, reused, escalated,
                      probes, decision):
            rdir = _safe_run_dir(runs_root, run_id)
            report_md = (f"# Stage 3: Idea Generation — {win_q}\n\n"
                         + _coverage_banner(win_st) + (win_md or ""))
            try:
                ideas_res = ic.generate_ideas(rdir, win_q, min_ideas=min_ideas, allow_llm=not readonly)
            except Exception as exc:
                ideas_res = {"ideas": [], "stats": {"error": str(exc)}}
            try:
                graph, graph_html = _run_graph(rdir), _graph_page(rdir)
            except Exception as exc:
                graph, graph_html = {"nodes": [], "edges": [], "error": str(exc)}, ""
            _log_query(runs_root, tool="smart_research", topic=topic, run=run_id,
                       query_used=win_q, planned=plan.get("query"), llm_applied=plan.get("applied"),
                       coverage=win_level, escalated=escalated, decision=decision,
                       n_matched=win_st.get("n_matched"), n_total=win_st.get("n_hypotheses_total"),
                       top_relevance=win_st.get("top_relevance"))
            return {
                "status": "done", "run": run_id, "reused": reused, "escalated": escalated,
                "query_plan": {
                    "raw": plan.get("raw"), "planned_query": plan.get("query"),
                    "query_used": win_q, "alternates": plan.get("alternates"),
                    "canonical_terms": plan.get("canonical_terms"),
                    "domain_anchors": plan.get("domain_anchors"),
                    "is_niche": plan.get("is_niche"), "llm_applied": plan.get("applied"),
                    "probes": probes, "decision": decision,
                },
                "coverage": {"level": win_level, "n_matched": win_st.get("n_matched"),
                             "n_total": win_st.get("n_hypotheses_total"),
                             "top_relevance": win_st.get("top_relevance")},
                "idea_report_markdown": report_md,
                "ideas_markdown": ic.render_ideas_markdown(ideas_res) if isinstance(ideas_res, dict) else "",
                "ideas": ideas_res.get("ideas", []) if isinstance(ideas_res, dict) else [],
                "graph": graph, "graph_html": graph_html,
                "dashboard_url": f"/dashboard/{run_id}", "graph_url": f"/dashboard/{run_id}/graph",
            }

        # 2. resolve the target corpus (pinned, or best lexical match on the plan)
        target = None
        if run:
            rd = _safe_run_dir(runs_root, run)
            if rd is None or not rd.exists():
                return {"error": f"unknown run: {run}",
                        "query_plan": {"planned_query": plan.get("query")}}
            target = run
        else:
            for probe_topic in (plan.get("query"), plan.get("raw"), topic):
                try:
                    rid = ic.resolve_best_run(runs_root, probe_topic or "")
                except Exception:
                    rid = None
                if rid:
                    target = rid
                    break

        # 3. probe + decide on the resolved corpus
        if target is not None:
            best, probes = _best_over(_safe_run_dir(runs_root, target))
            if best is not None:
                _key, win_q, win_md, win_st, win_level = best
                if _RANK.get(win_level, 0) >= _RANK["moderate"]:
                    return _assemble(target, win_q, win_md, win_st, win_level,
                                     reused=True, escalated=False, probes=probes,
                                     decision="reuse-coverage-ok")
                if not (allow_build and search_service is not None and not readonly):
                    out = _assemble(target, win_q, win_md, win_st, win_level,
                                    reused=True, escalated=False, probes=probes,
                                    decision="weak-no-build")
                    out["should_build"] = True
                    out["recommendation"] = (
                        "Coverage is weak on the generic corpus. Re-call with "
                        "allow_build=true (or use research_e2e) to build a "
                        f"topic-specific corpus for '{plan.get('query')}'.")
                    return out
            # no probe succeeded → fall through to build

        # 4. escalate: build a fresh topic-specific corpus
        if not (allow_build and search_service is not None and not readonly):
            return {"status": "no_corpus",
                    "query_plan": {"raw": plan.get("raw"), "planned_query": plan.get("query"),
                                   "is_niche": plan.get("is_niche"), "llm_applied": plan.get("applied")},
                    "recommendation": "No matching corpus and build is disabled "
                    "(allow_build=false / readonly / no SearchService). Call with "
                    "allow_build=true to build one."}
        build_q = plan.get("query") or topic
        try:
            req = search_service.submit(topic=build_q, limit=max(1, min(500, int(max_papers))),
                                        insight_generator="llm", source="union")
        except Exception as exc:
            return {"error": f"submit failed: {type(exc).__name__}: {exc}"}
        run_id = req.run_id
        run_dir = _safe_run_dir(runs_root, run_id)
        waited = 0
        while waited < wait_seconds:
            _time.sleep(5)
            waited += 5
            sp = (run_dir / "status.json") if run_dir else None
            if sp is None or not sp.exists():
                continue
            try:
                st = json.loads(sp.read_text())
            except Exception:
                continue
            if st.get("status") == "done":
                # Ground the freshly-built corpus's templated hypotheses before
                # bundling, so the deliverable shows evidence-grounded text rather
                # than the frozen "moderator variable" template. Best-effort.
                try:
                    from .hypothesis_enricher import enrich_run as _enrich
                    _enrich(run_dir)
                except Exception:
                    pass
                best, probes = _best_over(run_dir)
                if best is None:
                    return {"status": "done", "run": run_id, "reused": False, "escalated": True,
                            "query_plan": {"planned_query": plan.get("query"), "probes": []},
                            "note": "corpus built but no hypotheses matched any candidate query"}
                _key, win_q, win_md, win_st, win_level = best
                return _assemble(run_id, win_q, win_md, win_st, win_level,
                                 reused=False, escalated=True, probes=probes,
                                 decision="escalated-build")
            if st.get("status") == "error":
                return {"status": "error", "run": run_id, "message": st.get("message") or "run failed"}
        return {"status": "building", "run_id": run_id, "poll_with": "get_run_status",
                "query_plan": {"planned_query": plan.get("query"), "is_niche": plan.get("is_niche")},
                "next": f"once status=done, call smart_research(topic='{topic}') again — reuse will hit the built corpus"}

    @mcp.tool()
    def enrich_hypotheses(run: str, limit: int = 24, force: bool = False) -> dict:
        """Ground a run's TEMPLATED hypotheses in their real evidence claims.

        The frozen pipeline emits a generic template for every anomaly ("an
        unreported moderator variable drives the conflict…"). This runs ONE LLM
        call per hypothesis that reads the anomaly's actual evidence claims
        (paper, finding, stance, method, dataset) and rewrites the statement /
        mechanism / predictions / minimal_test into something specific to THOSE
        papers, persisting to a `hypotheses_enriched.jsonl` sidecar. Subsequent
        `get_idea_report` / `query_hypotheses` / `smart_research` calls overlay
        the grounded version automatically (0-LLM).

        Paid (LLM): up to `limit` *new* enrichments per call; already-enriched
        hypotheses are skipped unless `force`. Fresh corpora built via
        `smart_research` are enriched automatically — use this for older runs.
        Fail-open: no key / errors leave the templated text untouched."""
        run_dir = _safe_run_dir(runs_root, run)
        if run_dir is None or not run_dir.exists():
            return {"error": f"unknown run: {run}"}
        try:
            from .hypothesis_enricher import enrich_run, enricher_enabled
            if not enricher_enabled():
                return {"status": "disabled", "run": run,
                        "note": "enricher off or no API key configured (set OPENAI_API_KEY)"}
            before = sum(1 for _ in (run_dir / "hypotheses_enriched.jsonl").open()) \
                if (run_dir / "hypotheses_enriched.jsonl").exists() else 0
            enriched = enrich_run(run_dir, limit=max(1, min(100, int(limit))), force=bool(force))
            _log_query(runs_root, tool="enrich_hypotheses", run=run,
                       n_enriched=len(enriched), limit=limit, force=force)
            return {"status": "done", "run": run, "n_enriched_total": len(enriched),
                    "n_new": max(0, len(enriched) - before),
                    "sidecar": "hypotheses_enriched.jsonl",
                    "note": "grounded hypotheses now overlay automatically in get_idea_report / smart_research"}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    return mcp
