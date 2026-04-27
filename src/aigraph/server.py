"""Zero-dependency local web search server for aigraph."""

from __future__ import annotations

import html
import json
import os
import re
import secrets
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from .anomalies import detect_anomalies
from .community import community_digest, ingest_run, read_community_status
from .corpus import hydrate_papers_from_corpus
from .fetch_arxiv import fetch_arxiv_papers
from .fetch_openalex import fetch_openalex_papers
from .graph import build_graph, save_graph
from .hypotheses import TemplateGenerator, generate_hypotheses
from .insights import LLMInsightGenerator, TemplateInsightGenerator, generate_insights, prune_insights
from .io import write_jsonl
from .llm_extract import LLMClaimExtractor
from .library import curated_demos, search_prefill_href, specialty_libraries
from .llm_client import build_openai_client, call_llm_text, configured_api_key, configured_model
from .models import Anomaly, Claim, Hypothesis, Insight, Paper, ScoreBreakdown
from .overview import build_search_overview
from .paper_select import decompose_topic_query
from .paper_reader import (
    configured_reader_max_candidates,
    configured_reader_mode,
    read_paper_candidates,
)
from .report import render_report
from .scoring import score_all, select_mmr
from .visualize import render_visualization


DEFAULT_LIMIT = 20
ALLOWED_LIMITS = {10, 20, 30}
ALLOWED_SOURCES = {"arxiv", "openalex"}
ALLOWED_STRATEGIES = {"balanced", "high-impact", "recent"}
RUN_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-f0-9]{6}$")
ARXIV_SYNTAX_RE = re.compile(r"\b(all|ti|abs|au|cat|id):|(\bAND\b|\bOR\b|\bANDNOT\b)", re.IGNORECASE)
PRIMARY_PUBLIC_HOST = "graph.paper-universe.uk"
REDIRECT_HOSTS = {"paper-universe.uk", "www.paper-universe.uk"}
MAX_STORED_HYPOTHESES = 24
MAX_DISPLAY_HYPOTHESES = 10
LLM_PRUNE_TRIGGER = 32
LLM_PRUNE_SHORTLIST = 36
GRAPH_OVERVIEW_NODE_THRESHOLD = 90
GRAPH_OVERVIEW_EDGE_THRESHOLD = 120


@dataclass
class SearchRequest:
    run_id: str
    topic: str
    retrieval_topic: str
    retrieval_variants: list[str]
    arxiv_query: str
    limit: int
    insight_generator: str
    source: str
    strategy: str
    citation_weight: float
    min_relevance: float
    run_dir: Path
    client_ip: str = ""
    user_agent: str = ""


class SearchService:
    """Owns run creation, status files, and the single background worker."""

    def __init__(
        self,
        runs_dir: str | Path,
        *,
        runner: Callable[[SearchRequest, Callable[..., None]], None] | None = None,
    ):
        self.runs_dir = Path(runs_dir).resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner or run_pipeline
        self.queue: Queue[SearchRequest | None] = Queue()
        self.worker_started = False
        self.lock = threading.Lock()

    def submit(
        self,
        topic: str,
        limit: int = DEFAULT_LIMIT,
        insight_generator: str = "llm",
        source: str = "arxiv",
        strategy: str = "balanced",
        citation_weight: float = 0.45,
        min_relevance: float = 0.30,
        client_ip: str = "",
        user_agent: str = "",
    ) -> SearchRequest:
        topic = topic.strip()
        if not topic:
            raise ValueError("Please enter a research topic.")
        limit = limit if limit in ALLOWED_LIMITS else DEFAULT_LIMIT
        insight_generator = insight_generator if insight_generator in {"template", "llm"} else "llm"
        source = source if source in ALLOWED_SOURCES else "arxiv"
        strategy = strategy if strategy in ALLOWED_STRATEGIES else "balanced"
        if source == "arxiv" and strategy == "high-impact":
            strategy = "balanced"
        citation_weight = clamp_float(citation_weight, 0.0, 0.85)
        min_relevance = clamp_float(min_relevance, 0.0, 1.0)
        run_id = new_run_id()
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        query_plan = decompose_search_topic(topic)
        retrieval_topic = str(query_plan.get("normalized_topic") or topic)
        retrieval_variants = [str(item) for item in (query_plan.get("retrieval_variants") or []) if str(item).strip()]
        arxiv_query = normalize_arxiv_query(retrieval_topic)
        request = SearchRequest(
            run_id=run_id,
            topic=topic,
            retrieval_topic=retrieval_topic,
            retrieval_variants=retrieval_variants,
            arxiv_query=arxiv_query,
            limit=limit,
            insight_generator=insight_generator,
            source=source,
            strategy=strategy,
            citation_weight=citation_weight,
            min_relevance=min_relevance,
            client_ip=_clean_header(client_ip),
            user_agent=_clean_header(user_agent, max_len=300),
            run_dir=run_dir,
        )
        (run_dir / "query.txt").write_text(topic + "\n", encoding="utf-8")
        self._log_request(
            {
                "event": "search_submit",
                "run_id": run_id,
                "topic": topic,
                "retrieval_topic": retrieval_topic,
                "retrieval_variants": retrieval_variants,
                "arxiv_query": arxiv_query,
                "source": source,
                "strategy": strategy,
                "limit": limit,
                "citation_weight": citation_weight,
                "min_relevance": min_relevance,
                "client_ip": _clean_header(client_ip),
                "user_agent": _clean_header(user_agent, max_len=300),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_status(
            run_dir,
            status="queued",
            stage="queued",
            progress=0.0,
            message="Waiting for the local worker.",
            topic=topic,
            retrieval_topic=retrieval_topic,
            retrieval_variants=retrieval_variants,
            arxiv_query=arxiv_query,
            source_query=arxiv_query if source == "arxiv" else topic,
            limit=limit,
            source=source,
            strategy=strategy,
            citation_weight=citation_weight,
            min_relevance=min_relevance,
            run_id=run_id,
        )
        self.queue.put(request)
        self._ensure_worker()
        return request

    def _log_request(self, event: dict[str, Any]) -> None:
        analytics_dir = self.runs_dir / "_analytics"
        analytics_dir.mkdir(parents=True, exist_ok=True)
        path = analytics_dir / "requests.jsonl"
        with self.lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False))
                f.write("\n")

    def _ensure_worker(self) -> None:
        with self.lock:
            if self.worker_started:
                return
            thread = threading.Thread(target=self._worker_loop, name="aigraph-search-worker", daemon=True)
            thread.start()
            self.worker_started = True

    def _worker_loop(self) -> None:
        while True:
            request = self.queue.get()
            if request is None:
                return
            try:
                self.runner(request, lambda **kwargs: write_status(request.run_dir, **kwargs))
            except Exception as e:  # pragma: no cover - defensive top-level guard
                error_payload = friendly_error_payload(e, request)
                write_status(
                    request.run_dir,
                    status="error",
                    stage="error",
                    progress=1.0,
                    message=error_payload["message"],
                    error=error_payload["error"],
                    error_kind=error_payload["error_kind"],
                    error_title=error_payload["error_title"],
                    error_summary=error_payload["error_summary"],
                    error_recovery=error_payload["error_recovery"],
                    run_id=request.run_id,
                    topic=request.topic,
                    arxiv_query=request.arxiv_query,
                    source_query=request.arxiv_query if request.source == "arxiv" else request.topic,
                    limit=request.limit,
                    source=request.source,
                    strategy=request.strategy,
                    citation_weight=request.citation_weight,
                    min_relevance=request.min_relevance,
                )
            finally:
                self.queue.task_done()

    def read_status(self, run_id: str) -> dict[str, Any]:
        run_dir = safe_run_dir(self.runs_dir, run_id)
        path = run_dir / "status.json"
        if not path.exists():
            raise FileNotFoundError(run_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "done" and not data.get("overview"):
            overview = ensure_overview(run_dir, data)
            if overview:
                data["overview"] = overview
                data["overview_url"] = f"/runs/{run_id}/overview.json"
                write_status(run_dir, overview=overview, overview_url=data["overview_url"])
        return data

    def recent_runs(self, limit: int = 8) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for path in sorted(self.runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not path.is_dir() or not is_valid_run_id(path.name):
                continue
            status_path = path / "status.json"
            if not status_path.exists():
                continue
            try:
                runs.append(json.loads(status_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
            if len(runs) >= limit:
                break
        return runs

    def community_status(self) -> dict[str, Any]:
        return read_community_status(self.runs_dir)

    def community_digest(self) -> dict[str, Any]:
        return community_digest(self.runs_dir)


def _response_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def is_rate_limit_error(exc: Exception) -> bool:
    return _response_status_code(exc) == 429


def friendly_error_payload(exc: Exception, request: SearchRequest) -> dict[str, str]:
    if is_rate_limit_error(exc):
        source_name = "OpenAlex" if request.source == "openalex" else "the provider"
        return {
            "message": f"{source_name} is rate-limited right now.",
            "error_kind": "rate_limit",
            "error_title": "Rate limit hit",
            "error_summary": f"{source_name} temporarily rejected this request because too many searches are arriving at once.",
            "error_recovery": "Try again in a minute, lower the paper count, or switch to arXiv for a lighter run.",
            "error": f"{source_name} returned HTTP 429 Too Many Requests.",
        }
    return {
        "message": "The run failed.",
        "error_kind": "generic",
        "error_title": "Run failed",
        "error_summary": "Something broke while building this research map.",
        "error_recovery": "Please try the search again. If it keeps happening, narrow the topic or switch source.",
        "error": str(exc),
    }


def run_pipeline(request: SearchRequest, status: Callable[..., None]) -> None:
    run_dir = request.run_dir
    papers_path = run_dir / "papers.jsonl"
    claims_path = run_dir / "claims.jsonl"
    graph_path = run_dir / "graph.json"
    anomalies_path = run_dir / "anomalies.jsonl"
    hypotheses_path = run_dir / "hypotheses.jsonl"
    insights_path = run_dir / "insights.jsonl"
    raw_insights_path = run_dir / "raw_insights.jsonl"
    overview_path = run_dir / "overview.json"
    report_path = run_dir / "selected_hypotheses.md"
    html_path = run_dir / "index.html"

    base_status = {
        "run_id": request.run_id,
        "topic": request.topic,
        "retrieval_topic": request.retrieval_topic,
        "retrieval_variants": request.retrieval_variants,
        "arxiv_query": request.arxiv_query,
        "source": request.source,
        "strategy": request.strategy,
        "citation_weight": request.citation_weight,
        "min_relevance": request.min_relevance,
        "source_query": request.arxiv_query if request.source == "arxiv" else request.topic,
        "limit": request.limit,
        "graph_url": f"/runs/{request.run_id}/index.html",
        "report_url": f"/runs/{request.run_id}/selected_hypotheses.md",
        "data_url": f"/runs/{request.run_id}/",
    }

    source_label = "OpenAlex" if request.source == "openalex" else "arXiv"
    status(
        status="running",
        stage="fetching",
        progress=0.05,
        message=f"Searching {source_label} papers with {request.strategy} strategy.",
        **base_status,
    )
    if request.source == "openalex":
        try:
            papers = fetch_openalex_papers(
                request.retrieval_topic,
                from_year=2020,
                to_year=2026,
                limit=request.limit,
                strategy=request.strategy,
                citation_weight=request.citation_weight,
                min_relevance=request.min_relevance,
                query_variants=request.retrieval_variants,
            )
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise
            status(
                status="running",
                stage="fetching",
                progress=0.08,
                message="OpenAlex is rate-limited. Falling back to arXiv so the run can keep going.",
                source_fallback="arxiv",
                **base_status,
            )
            papers = fetch_arxiv_papers(
                request.arxiv_query,
                from_year=2020,
                to_year=2026,
                limit=request.limit,
                strategy="balanced",
            )
            base_status["source_fallback"] = "arxiv"
    else:
        papers = fetch_arxiv_papers(
            request.arxiv_query,
            from_year=2020,
            to_year=2026,
            limit=request.limit,
            strategy=request.strategy,
        )
    papers = hydrate_papers_from_corpus(papers)
    write_jsonl(papers_path, papers)
    status(
        status="running",
        stage="extracting",
        progress=0.18,
        message=f"Fetched {len(papers)} papers. Extracting claims with the LLM.",
        papers=len(papers),
        **base_status,
    )

    claims = extract_claims_with_status(papers, claims_path, status, base_status)

    status(
        status="running",
        stage="building_graph",
        progress=0.68,
        message=f"Building graph from {len(claims)} claims.",
        papers=len(papers),
        claims=len(claims),
        **base_status,
    )
    graph = build_graph(claims, papers=papers)
    save_graph(graph, graph_path)

    status(
        status="running",
        stage="detecting",
        progress=0.76,
        message="Detecting conflicts, gaps, and topology patterns.",
        papers=len(papers),
        claims=len(claims),
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        **base_status,
    )
    anomalies = detect_anomalies(graph, claims)
    write_jsonl(anomalies_path, anomalies)

    status(
        status="running",
        stage="generating",
        progress=0.84,
        message="Generating hypotheses and community insights.",
        papers=len(papers),
        claims=len(claims),
        anomalies=len(anomalies),
        **base_status,
    )
    raw_hypotheses = generate_hypotheses(anomalies, claims, generator=TemplateGenerator())

    insight_impl = LLMInsightGenerator() if request.insight_generator == "llm" else TemplateInsightGenerator()
    raw_insights = generate_insights(graph, claims, papers, anomalies, generator=insight_impl)
    write_jsonl(raw_insights_path, raw_insights)
    insights = prune_insights(raw_insights, claims)
    write_jsonl(insights_path, insights)

    status(
        status="running",
        stage="rendering",
        progress=0.93,
        message="Scoring results and rendering the report.",
        papers=len(papers),
        claims=len(claims),
        anomalies=len(anomalies),
        hypotheses=len(raw_hypotheses),
        insights=len(insights),
        **base_status,
    )
    hypotheses = prune_hypotheses(raw_hypotheses, anomalies, claims)
    write_jsonl(hypotheses_path, hypotheses)
    scores = score_all(hypotheses, anomalies, claims) if hypotheses and anomalies else {}
    selected = select_mmr(hypotheses, scores, k=8, lambda_=0.7, min_anomalies=2) if scores else []
    paper_lookup = {p.paper_id: p for p in papers}
    overview = build_search_overview(request.topic, papers, claims, anomalies, insights, selected, scores)
    overview_path.write_text(json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(selected, anomalies, claims, scores, paper_lookup=paper_lookup, insights=insights)
    report_path.write_text(report, encoding="utf-8")
    render_visualization(run_dir, html_path)

    status(
        status="done",
        stage="complete",
        progress=1.0,
        message=f"Generated {len(insights)} insight(s), {len(anomalies)} conflict/gap region(s), and {len(selected)} selected hypothesis entries.",
        papers=len(papers),
        claims=len(claims),
        anomalies=len(anomalies),
        hypotheses=len(hypotheses),
        selected=len(selected),
        insights=len(insights),
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        overview=overview,
        overview_url=f"/runs/{request.run_id}/overview.json",
        **base_status,
    )
    try:
        ingest_run(run_dir, run_dir.parent, run_id=request.run_id)
    except Exception:  # pragma: no cover - keep user run successful even if community refresh fails
        traceback.print_exc()


def extract_claims_with_status(
    papers: list[Paper],
    output: Path,
    status: Callable[..., None],
    base_status: dict[str, Any],
) -> list[Claim]:
    output.parent.mkdir(parents=True, exist_ok=True)
    extractor = LLMClaimExtractor()
    concurrency = max(1, min(4, int(os.environ.get("AIGRAPH_EXTRACT_CONCURRENCY", "2"))))
    claims: list[Claim] = []
    reader_rows: list[dict[str, Any]] = []
    reader_debug_rows: list[dict[str, Any]] = []
    reader_mode = configured_reader_mode()
    reader_max_candidates = configured_reader_max_candidates()

    def run_one(index: int, paper: Paper) -> tuple[int, Paper, list[Claim], dict[str, Any], list[dict[str, Any]]]:
        read_result = read_paper_candidates(
            paper,
            mode=reader_mode,
            max_candidates=reader_max_candidates,
        )
        local_claims = extractor.extract(paper, start_index=index * 1000, candidates=read_result.candidates)
        metrics = {
            "event": "paper_reader",
            "run_id": base_status.get("run_id"),
            "paper_id": paper.paper_id,
            "reader_mode": read_result.mode_used,
            "reader_latency_sec": float(read_result.latency_sec or 0.0),
            "reader_candidate_count": len(read_result.candidates),
            "reader_verified_claim_count": len(local_claims),
            "reader_fallback_used": bool(read_result.fallback_used),
            "reader_prefilter_count": int(read_result.prefilter_count or 0),
        }
        debug_rows = [
            {
                "run_id": base_status.get("run_id"),
                "paper_id": paper.paper_id,
                "paper_title": paper.title,
                "reader_mode": read_result.mode_used,
                "reader_fallback_used": bool(read_result.fallback_used),
                **candidate.model_dump(),
            }
            for candidate in read_result.candidates
        ]
        return index, paper, local_claims, metrics, debug_rows

    completed = 0
    results: dict[int, tuple[Paper, list[Claim], dict[str, Any], list[dict[str, Any]]]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_one, i, paper): i for i, paper in enumerate(papers)}
        for future in as_completed(futures):
            index, paper, local_claims, reader_metrics, debug_rows = future.result()
            completed += 1
            results[index] = (paper, local_claims, reader_metrics, debug_rows)
            reader_rows.append(reader_metrics)
            reader_debug_rows.extend(debug_rows)
            progress = 0.18 + 0.48 * completed / max(1, len(papers))
            status(
                status="running",
                stage="extracting",
                progress=round(progress, 4),
                message=f"Extracted claims {completed}/{len(papers)}: {paper.title[:80]}",
                papers=len(papers),
                claims=sum(len(cs) for _, cs in results.values()),
                **base_status,
            )

    with output.open("w", encoding="utf-8") as f:
        for index in sorted(results):
            _, local_claims, _, _ = results[index]
            for claim in local_claims:
                claim = claim.model_copy(update={"claim_id": f"c{len(claims) + 1:03d}"})
                f.write(claim.model_dump_json(by_alias=True))
                f.write("\n")
                claims.append(claim)
        f.flush()
    if reader_debug_rows:
        _write_jsonl_dicts(output.parent / "reader_candidates.jsonl", reader_debug_rows)
    if reader_rows:
        for row in reader_rows:
            _append_analytics_row(output.parent.parent, "reader_metrics.jsonl", row)
    status(
        status="running",
        stage="extracting",
        progress=0.66,
        message=f"Extracted {len(claims)} claims from {len(papers)} papers.",
        papers=len(papers),
        claims=len(claims),
        reader_mode=reader_mode,
        reader_candidate_count=sum(int(row.get("reader_candidate_count") or 0) for row in reader_rows),
        reader_verified_claim_count=len(claims),
        reader_fallback_count=sum(1 for row in reader_rows if row.get("reader_fallback_used")),
        **base_status,
    )
    return claims


def serve(host: str = "127.0.0.1", port: int = 7860, runs_dir: str | Path = "outputs/runs") -> None:
    service = SearchService(runs_dir)
    handler_cls = make_handler(service)
    server = ThreadingHTTPServer((host, port), handler_cls)
    print(f"aigraph search server running at http://{host}:{port}")
    print("Use Cloudflare Tunnel or ngrok to share it publicly.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive behavior
        pass
    finally:
        server.server_close()


def make_handler(service: SearchService) -> type[BaseHTTPRequestHandler]:
    class SearchHandler(BaseHTTPRequestHandler):
        server_version = "aigraph-search/0.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            redirect = public_redirect_url(self.headers.get("Host", ""), self.path)
            if redirect:
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", redirect)
                self.end_headers()
                return
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._send_html(render_home_page(service.recent_runs(limit=20), service.community_digest()))
                return
            if path == "/community" or path == "/community/":
                self._serve_community_file("index.html")
                return
            if path.startswith("/community/"):
                self._serve_community_file(unquote(path.removeprefix("/community/")))
                return
            if path.startswith("/search/"):
                run_id = path.removeprefix("/search/").strip("/")
                try:
                    status = service.read_status(run_id)
                except (FileNotFoundError, ValueError):
                    self._send_error(HTTPStatus.NOT_FOUND, "Run not found.")
                    return
                self._send_html(render_result_page(status, service.recent_runs(limit=20)))
                return
            if path.startswith("/api/runs/"):
                run_id = path.removeprefix("/api/runs/").strip("/")
                try:
                    self._send_json(service.read_status(run_id))
                except (FileNotFoundError, ValueError):
                    self._send_json({"status": "error", "error": "Run not found."}, HTTPStatus.NOT_FOUND)
                return
            if path.startswith("/runs/"):
                self._serve_run_file(path)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found.")

        def do_POST(self) -> None:  # noqa: N802 - stdlib API
            parsed = urlparse(self.path)
            if parsed.path == "/api/graph-chat":
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(length).decode("utf-8")
                try:
                    payload = json.loads(raw or "{}")
                    answer = answer_graph_chat(
                        service.runs_dir,
                        run_id=str(payload.get("run_id") or ""),
                        question=str(payload.get("question") or ""),
                        selection=payload.get("selection") or {},
                        history=payload.get("history") or [],
                    )
                    self._send_json(answer, HTTPStatus.OK)
                except ValueError as e:
                    self._send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                except FileNotFoundError:
                    self._send_json({"error": "Run not found."}, HTTPStatus.NOT_FOUND)
                return
            if parsed.path != "/api/search":
                self._send_error(HTTPStatus.NOT_FOUND, "Not found.")
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length).decode("utf-8")
            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type:
                payload = json.loads(raw or "{}")
            else:
                form = parse_qs(raw)
                payload = {key: values[0] for key, values in form.items()}
            try:
                limit = int(payload.get("limit") or DEFAULT_LIMIT)
            except (TypeError, ValueError):
                limit = DEFAULT_LIMIT
            try:
                citation_weight = float(payload.get("citation_weight") or 0.45)
            except (TypeError, ValueError):
                citation_weight = 0.45
            try:
                min_relevance = float(payload.get("min_relevance") or 0.30)
            except (TypeError, ValueError):
                min_relevance = 0.30
            try:
                req = service.submit(
                    str(payload.get("topic") or ""),
                    limit=limit,
                    insight_generator="llm",
                    source=str(payload.get("source") or "arxiv"),
                    strategy=str(payload.get("strategy") or "balanced"),
                    citation_weight=citation_weight,
                    min_relevance=min_relevance,
                    client_ip=self._client_ip(),
                    user_agent=self.headers.get("User-Agent", ""),
                )
            except ValueError as e:
                self._send_json({"status": "error", "error": str(e)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(
                {
                    "status": "queued",
                    "run_id": req.run_id,
                    "url": f"/search/{req.run_id}",
                    "graph_url": f"/runs/{req.run_id}/index.html",
                },
                HTTPStatus.CREATED,
            )

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[aigraph-search] {self.address_string()} - {fmt % args}")

        def _client_ip(self) -> str:
            forwarded = self.headers.get("CF-Connecting-IP") or self.headers.get("X-Real-IP")
            if not forwarded:
                forwarded = self.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
            return forwarded or self.client_address[0]

        def _serve_run_file(self, path: str) -> None:
            parts = path.removeprefix("/runs/").split("/", 1)
            if len(parts) != 2:
                self._send_error(HTTPStatus.NOT_FOUND, "File not found.")
                return
            run_id, rel = parts[0], unquote(parts[1])
            try:
                run_dir = safe_run_dir(service.runs_dir, run_id)
                target = safe_child_path(run_dir, rel)
            except ValueError:
                self._send_error(HTTPStatus.BAD_REQUEST, "Invalid path.")
                return
            if target.is_dir():
                target = target / "index.html"
            if not target.exists() or not target.is_file():
                self._send_error(HTTPStatus.NOT_FOUND, "File not found.")
                return
            self._send_file(target)

        def _serve_community_file(self, rel: str) -> None:
            try:
                root = service.runs_dir / "_community"
                target = safe_child_path(root, rel or "index.html")
            except ValueError:
                self._send_error(HTTPStatus.BAD_REQUEST, "Invalid path.")
                return
            if target.is_dir():
                target = target / "index.html"
            if not target.exists() or not target.is_file():
                self._send_error(HTTPStatus.NOT_FOUND, "Community graph not found yet.")
                return
            self._send_file(target)

        def _send_json(self, data: dict[str, Any], status_code: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: str, status_code: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_file(self, path: Path) -> None:
            if path.suffix.lower() == ".md":
                self._send_html(render_markdown_page(path.stem.replace("_", " "), path.read_text(encoding="utf-8")))
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type_for(path))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status_code: HTTPStatus, message: str) -> None:
            self._send_html(render_error_page(status_code, message), status_code)

    return SearchHandler


def public_redirect_url(host: str, path: str) -> str | None:
    host_only = (host or "").split(":", 1)[0].strip().lower()
    if host_only not in REDIRECT_HOSTS:
        return None
    suffix = path or "/"
    if not suffix.startswith("/"):
        suffix = "/" + suffix
    return f"https://{PRIMARY_PUBLIC_HOST}{suffix}"


def decompose_search_topic(topic: str) -> dict[str, Any]:
    plan = decompose_topic_query(topic)
    if not plan.get("needs_llm_fallback") or not configured_api_key():
        return plan
    try:
        client = build_openai_client()
        raw = call_llm_text(
            client,
            model=configured_model(),
            system=(
                "You rewrite research search topics into retrieval-friendly concepts. "
                "Use only the user's topic. Return strict JSON with keys "
                "\"normalized_topic\", \"core_terms\", \"modifiers\", and \"retrieval_variants\". "
                "Do not invent papers, venues, or citations."
            ),
            user=json.dumps(
                {
                    "topic": topic,
                    "deterministic_plan": plan,
                    "goal": "Improve retrieval quality for academic search without broadening into unrelated fields.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            temperature=0.0,
            max_tokens=450,
        )
        parsed = json.loads(raw)
    except Exception:
        return plan
    if not isinstance(parsed, dict):
        return plan
    normalized_topic = str(parsed.get("normalized_topic") or plan.get("normalized_topic") or topic).strip()
    core_terms = [str(item).strip() for item in (parsed.get("core_terms") or plan.get("core_terms") or []) if str(item).strip()]
    modifiers = [str(item).strip() for item in (parsed.get("modifiers") or plan.get("modifiers") or []) if str(item).strip()]
    retrieval_variants = [
        str(item).strip()
        for item in (parsed.get("retrieval_variants") or plan.get("retrieval_variants") or [])
        if str(item).strip()
    ]
    merged = {
        "original": topic,
        "normalized_topic": normalized_topic,
        "core_terms": core_terms[:8],
        "modifiers": modifiers[:8],
        "retrieval_variants": list(dict.fromkeys(([normalized_topic] if normalized_topic else []) + retrieval_variants))[:6],
        "needs_llm_fallback": True,
    }
    return merged


def normalize_arxiv_query(topic: str) -> str:
    topic = " ".join(topic.strip().split())
    if not topic:
        return ""
    if ARXIV_SYNTAX_RE.search(topic):
        return topic

    lower = topic.lower()
    terms: list[str] = []
    consumed: set[str] = set()
    phrase_map = {
        "large language models": 'all:"large language models"',
        "language models": 'all:"large language models"',
        "time series": 'all:"time series"',
        "machine learning": 'all:"machine learning"',
        "retrieval augmented generation": 'all:"retrieval augmented generation"',
    }
    for phrase, query in phrase_map.items():
        if phrase in lower:
            terms.append(query)
            consumed.update(phrase.split())
    words = re.findall(r"[a-zA-Z0-9+.-]+", lower)
    for word in words:
        if word in consumed or len(word) <= 1:
            continue
        if word in {"llm", "llms"}:
            query = 'all:"large language models"'
        elif word == "rag":
            query = 'all:"retrieval augmented generation"'
        else:
            query = f"all:{word}"
        if query not in terms:
            terms.append(query)
    return " AND ".join(terms) if terms else f'all:"{topic}"'


def answer_graph_chat(
    runs_dir: Path,
    *,
    run_id: str,
    question: str,
    selection: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    question = " ".join(str(question or "").split())
    if not run_id:
        raise ValueError("Missing run id.")
    if not question:
        raise ValueError("Please ask a question about this graph.")
    run_dir = safe_run_dir(runs_dir, run_id)
    context = _graph_chat_context(run_dir, selection or {})
    trimmed_history = _trim_graph_chat_history(history or [])
    if not configured_api_key():
        return {
            "answer": "Graph chat is not configured on this server yet.",
            "citations": context.get("references", [])[:6],
            "references": context.get("references", [])[:6],
        }
    client = build_openai_client()
    raw = call_llm_text(
        client,
        model=configured_model(),
        system=(
            "You analyze graph-grounded literature maps. Use only the provided JSON context. "
            "Do not invent papers or claims outside the run. Return strict JSON with keys "
            "\"answer\" and \"citations\" where citations is a list of {type, id, title} objects."
        ),
        user=json.dumps(
            {
                "question": question,
                "history": trimmed_history,
                "context": context,
                "answer_style": "short, helpful, evidence-grounded",
            },
            ensure_ascii=False,
            indent=2,
        ),
        temperature=0.0,
        max_tokens=700,
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"answer": raw.strip(), "citations": context.get("references", [])[:6]}
    answer = str(parsed.get("answer") or "").strip() or "I could not produce a grounded answer for this graph."
    allowed_refs = {(ref["type"], ref["id"]): ref for ref in context.get("references", [])}
    citations: list[dict[str, str]] = []
    for item in parsed.get("citations") or []:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("type") or ""), str(item.get("id") or ""))
        if key in allowed_refs and allowed_refs[key] not in citations:
            citations.append(allowed_refs[key])
    if not citations:
        citations = context.get("references", [])[:6]
    return {
        "answer": answer,
        "citations": citations[:6],
        "references": citations[:6],
        "history_limit": GRAPH_CHAT_HISTORY_LIMIT,
    }


def _graph_chat_context(run_dir: Path, selection: dict[str, Any]) -> dict[str, Any]:
    overview = _read_json_file(run_dir / "overview.json")
    papers = [Paper.model_validate(x) for x in read_jsonl_dicts(run_dir / "papers.jsonl")]
    claims = [Claim.model_validate(x) for x in read_jsonl_dicts(run_dir / "claims.jsonl")]
    anomalies = [Anomaly.model_validate(x) for x in read_jsonl_dicts(run_dir / "anomalies.jsonl")]
    hypotheses = [Hypothesis.model_validate(x) for x in read_jsonl_dicts(run_dir / "hypotheses.jsonl")]
    insights = [Insight.model_validate(x) for x in read_jsonl_dicts(run_dir / "insights.jsonl")]
    graph = _read_json_file(run_dir / "graph.json")

    papers_by_id = {paper.paper_id: paper for paper in papers}
    claims_by_id = {claim.claim_id: claim for claim in claims}
    anomalies_by_id = {anomaly.anomaly_id: anomaly for anomaly in anomalies}
    hypotheses_by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in hypotheses}
    insights_by_id = {insight.insight_id: insight for insight in insights}
    selection_context, references = _selection_context(
        selection,
        graph=graph,
        papers_by_id=papers_by_id,
        claims_by_id=claims_by_id,
        anomalies_by_id=anomalies_by_id,
        hypotheses_by_id=hypotheses_by_id,
        insights_by_id=insights_by_id,
    )
    return {
        "overview": {
            "headline": overview.get("headline", ""),
            "hero_line": (overview.get("hero_line") or {}).get("line", ""),
            "why_this_matters": (overview.get("why_this_matters") or {}).get("line", ""),
            "next_step": (overview.get("why_this_matters") or {}).get("next_step", ""),
        },
        "selection": selection_context,
        "top_conflicts": overview.get("top_conflicts", [])[:2],
        "hidden_bridges": overview.get("hidden_bridges", [])[:2],
        "key_explanations": overview.get("best_explanation_lines", [])[:3],
        "references": references[:6],
    }


def _selection_context(
    selection: dict[str, Any],
    *,
    graph: dict[str, Any],
    papers_by_id: dict[str, Paper],
    claims_by_id: dict[str, Claim],
    anomalies_by_id: dict[str, Anomaly],
    hypotheses_by_id: dict[str, Hypothesis],
    insights_by_id: dict[str, Insight],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    kind = str(selection.get("kind") or "").strip().lower()
    item_id = str(selection.get("id") or "").strip()
    if kind in {"anomaly", "conflict"} and item_id in anomalies_by_id:
        anomaly = anomalies_by_id[item_id]
        refs = _reference_rows(
            claim_ids=anomaly.claim_ids,
            paper_ids=[],
            claims_by_id=claims_by_id,
            papers_by_id=papers_by_id,
        )
        return {
            "kind": "anomaly",
            "id": anomaly.anomaly_id,
            "title": anomaly.central_question,
            "shared_entities": anomaly.shared_entities,
            "claim_ids": anomaly.claim_ids,
        }, refs
    if kind == "hypothesis" and item_id in hypotheses_by_id:
        hypothesis = hypotheses_by_id[item_id]
        refs = _reference_rows(
            claim_ids=hypothesis.explains_claims,
            paper_ids=[],
            claims_by_id=claims_by_id,
            papers_by_id=papers_by_id,
        )
        return {
            "kind": "hypothesis",
            "id": hypothesis.hypothesis_id,
            "title": hypothesis.hypothesis,
            "mechanism": hypothesis.mechanism,
            "claim_ids": hypothesis.explains_claims,
        }, refs
    if kind == "insight" and item_id in insights_by_id:
        insight = insights_by_id[item_id]
        refs = _reference_rows(
            claim_ids=insight.evidence_claims,
            paper_ids=insight.evidence_papers,
            claims_by_id=claims_by_id,
            papers_by_id=papers_by_id,
        )
        return {
            "kind": "insight",
            "id": insight.insight_id,
            "title": insight.title,
            "communities": insight.communities,
            "shared_concepts": insight.shared_concepts,
            "unifying_frame": insight.unifying_frame,
        }, refs
    if kind == "node" and item_id:
        return _node_selection_context(item_id, graph, claims_by_id, papers_by_id)
    refs = _reference_rows(
        claim_ids=list(claims_by_id)[:6],
        paper_ids=list(papers_by_id)[:4],
        claims_by_id=claims_by_id,
        papers_by_id=papers_by_id,
    )
    return {"kind": "graph", "id": "graph", "title": "Current run graph"}, refs


GRAPH_CHAT_HISTORY_LIMIT = 6
GRAPH_CHAT_HISTORY_CHAR_LIMIT = 600


def _trim_graph_chat_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    trimmed: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = " ".join(str(item.get("content") or "").split()).strip()
        if not content:
            continue
        trimmed.append(
            {
                "role": role,
                "content": content[:GRAPH_CHAT_HISTORY_CHAR_LIMIT],
            }
        )
    return trimmed[-GRAPH_CHAT_HISTORY_LIMIT:]


def _node_selection_context(
    node_id: str,
    graph: dict[str, Any],
    claims_by_id: dict[str, Claim],
    papers_by_id: dict[str, Paper],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    nodes = {str(node.get("id") or ""): node for node in graph.get("nodes", [])}
    node = nodes.get(node_id, {})
    claim_ids: list[str] = []
    for edge in graph.get("edges", []):
        source = str(edge.get("source") if not isinstance(edge.get("source"), dict) else edge["source"].get("id"))
        target = str(edge.get("target") if not isinstance(edge.get("target"), dict) else edge["target"].get("id"))
        if source == node_id and target.startswith("Claim:"):
            claim_ids.append(target.removeprefix("Claim:"))
        if target == node_id and source.startswith("Claim:"):
            claim_ids.append(source.removeprefix("Claim:"))
    unique_claim_ids = list(dict.fromkeys(claim_ids))
    paper_ids = [claims_by_id[cid].paper_id for cid in unique_claim_ids if cid in claims_by_id]
    refs = _reference_rows(
        claim_ids=unique_claim_ids,
        paper_ids=paper_ids,
        claims_by_id=claims_by_id,
        papers_by_id=papers_by_id,
    )
    return {
        "kind": "node",
        "id": node_id,
        "title": str(node.get("name") or node.get("value") or node.get("id") or node_id),
        "node_type": str(node.get("node_type") or ""),
        "claim_ids": unique_claim_ids[:8],
        "paper_ids": list(dict.fromkeys(paper_ids))[:8],
    }, refs


def _reference_rows(
    *,
    claim_ids: list[str],
    paper_ids: list[str],
    claims_by_id: dict[str, Claim],
    papers_by_id: dict[str, Paper],
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for claim_id in claim_ids:
        claim = claims_by_id.get(claim_id)
        if not claim or ("claim", claim_id) in seen:
            continue
        refs.append({"type": "claim", "id": claim_id, "title": claim.claim_text[:140]})
        seen.add(("claim", claim_id))
        paper = papers_by_id.get(claim.paper_id)
        if paper and ("paper", paper.paper_id) not in seen:
            refs.append({"type": "paper", "id": paper.paper_id, "title": paper.title})
            seen.add(("paper", paper.paper_id))
    for paper_id in paper_ids:
        paper = papers_by_id.get(paper_id)
        if paper and ("paper", paper.paper_id) not in seen:
            refs.append({"type": "paper", "id": paper.paper_id, "title": paper.title})
            seen.add(("paper", paper.paper_id))
    return refs


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def is_valid_run_id(run_id: str) -> bool:
    return bool(RUN_ID_RE.fullmatch(run_id))


def clamp_float(value: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, parsed))


def safe_run_dir(runs_dir: Path, run_id: str) -> Path:
    if not is_valid_run_id(run_id):
        raise ValueError("Invalid run id.")
    path = (runs_dir / run_id).resolve()
    if not path.is_dir() or not _is_relative_to(path, runs_dir.resolve()):
        raise FileNotFoundError(run_id)
    return path


def safe_child_path(root: Path, rel: str) -> Path:
    if rel.startswith("/") or "\x00" in rel:
        raise ValueError("Invalid path.")
    path = (root / rel).resolve()
    if not _is_relative_to(path, root.resolve()):
        raise ValueError("Invalid path.")
    return path


def write_status(run_dir: Path, **updates: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    current: dict[str, Any] = {}
    if status_path.exists():
        try:
            current = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
    current.update(updates)
    current["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(status_path)


def _clean_header(value: str, max_len: int = 120) -> str:
    value = " ".join(str(value or "").split())
    return value[:max_len]


def read_jsonl_dicts(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _write_jsonl_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _append_analytics_row(root: Path, filename: str, row: dict[str, Any]) -> None:
    analytics_dir = root / "_analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    path = analytics_dir / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False))
        f.write("\n")


def ensure_overview(run_dir: Path, status: dict[str, Any]) -> dict[str, Any] | None:
    overview_path = run_dir / "overview.json"
    if overview_path.exists():
        try:
            return json.loads(overview_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    try:
        papers = [Paper.model_validate(x) for x in read_jsonl_dicts(run_dir / "papers.jsonl")]
        claims = [Claim.model_validate(x) for x in read_jsonl_dicts(run_dir / "claims.jsonl")]
        anomalies = [Anomaly.model_validate(x) for x in read_jsonl_dicts(run_dir / "anomalies.jsonl")]
        insights = [Insight.model_validate(x) for x in read_jsonl_dicts(run_dir / "insights.jsonl")]
        hypotheses = [Hypothesis.model_validate(x) for x in read_jsonl_dicts(run_dir / "hypotheses.jsonl")]
    except Exception:
        return None
    overview = build_search_overview(
        str(status.get("topic") or "this topic"),
        papers,
        claims,
        anomalies,
        insights,
        hypotheses[:8],
        scores={},
    )
    overview_path.write_text(json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8")
    return overview


def prune_hypotheses(
    hypotheses: list[Hypothesis],
    anomalies: list[Anomaly],
    claims: list[Claim],
    *,
    max_keep: int = MAX_STORED_HYPOTHESES,
) -> list[Hypothesis]:
    if len(hypotheses) <= max_keep:
        return hypotheses
    scores = score_all(hypotheses, anomalies, claims)
    shortlisted = _deterministic_hypothesis_shortlist(hypotheses, scores, max_total=max(LLM_PRUNE_SHORTLIST, max_keep))
    if len(shortlisted) <= max_keep:
        return shortlisted[:max_keep]
    kept_ids = _llm_prune_hypothesis_ids(shortlisted, scores, target=max_keep)
    if kept_ids:
        by_id = {h.hypothesis_id: h for h in shortlisted}
        chosen = [by_id[hypothesis_id] for hypothesis_id in kept_ids if hypothesis_id in by_id]
        if chosen:
            chosen.sort(key=lambda h: scores[h.hypothesis_id].utility, reverse=True)
            return chosen[:max_keep]
    return shortlisted[:max_keep]


def _deterministic_hypothesis_shortlist(
    hypotheses: list[Hypothesis],
    scores: dict[str, ScoreBreakdown],
    *,
    max_total: int,
    per_anomaly: int = 2,
) -> list[Hypothesis]:
    by_anomaly: dict[str, list[Hypothesis]] = {}
    for hypothesis in hypotheses:
        by_anomaly.setdefault(hypothesis.anomaly_id, []).append(hypothesis)
    picked: list[Hypothesis] = []
    seen: set[str] = set()
    for items in by_anomaly.values():
        items.sort(key=lambda h: scores[h.hypothesis_id].utility, reverse=True)
        for hypothesis in items[:per_anomaly]:
            if hypothesis.hypothesis_id not in seen:
                seen.add(hypothesis.hypothesis_id)
                picked.append(hypothesis)
    remaining = sorted(
        [h for h in hypotheses if h.hypothesis_id not in seen],
        key=lambda h: scores[h.hypothesis_id].utility,
        reverse=True,
    )
    for hypothesis in remaining:
        if len(picked) >= max_total:
            break
        picked.append(hypothesis)
    picked.sort(key=lambda h: scores[h.hypothesis_id].utility, reverse=True)
    return picked[:max_total]


def _llm_prune_hypothesis_ids(
    hypotheses: list[Hypothesis],
    scores: dict[str, ScoreBreakdown],
    *,
    target: int,
) -> list[str]:
    if len(hypotheses) <= target or not configured_api_key():
        return []
    try:
        client = build_openai_client()
        payload = [
            {
                "hypothesis_id": h.hypothesis_id,
                "anomaly_id": h.anomaly_id,
                "hypothesis": h.hypothesis,
                "mechanism": h.mechanism,
                "utility": round(float(scores[h.hypothesis_id].utility), 3),
            }
            for h in hypotheses
        ]
        raw = call_llm_text(
            client,
            model=configured_model(),
            system=(
                "You prune noisy scientific hypothesis candidates for a user-facing demo. "
                "Return STRICT JSON only in the form {\"keep\": [\"h001\", ...]}. "
                f"Keep at most {target} ids. Prefer concrete, distinct, readable hypotheses. "
                "Penalize placeholders like 'other', generic wording, and near-duplicates."
            ),
            user=json.dumps({"target": target, "hypotheses": payload}, ensure_ascii=False, indent=2),
            temperature=0.0,
            max_tokens=500,
        )
        parsed = json.loads(raw)
        keep = parsed.get("keep") if isinstance(parsed, dict) else None
        if not isinstance(keep, list):
            return []
        allowed = {h.hypothesis_id for h in hypotheses}
        unique: list[str] = []
        for item in keep:
            hypothesis_id = str(item)
            if hypothesis_id in allowed and hypothesis_id not in unique:
                unique.append(hypothesis_id)
            if len(unique) >= target:
                break
        return unique
    except Exception:
        return []


def demo_topics() -> list[str]:
    return [
        "llm finance time series forecasting",
        "rag hallucination medical qa",
        "dpo ppo rlhf alignment safety",
        "multimodal rag scientific literature",
        "long context vs retrieval augmented generation",
    ]


def homepage_help_steps() -> list[tuple[str, str]]:
    return [
        ("Search a topic", "Type a research area or debate, like long-context vs retrieval or LLMs in finance."),
        ("Read the fault lines", "We pull representative papers, extract claims, and surface where the literature disagrees."),
        ("Open the map", "Inspect the graph, read the report, and follow the evidence behind each conflict or bridge."),
    ]


def render_home_page(recent_runs: list[dict[str, Any]], community_data: dict[str, Any] | None = None) -> str:
    examples = [
        *demo_topics()[:4],
    ]
    curated_html = render_curated_demo_cards(recent_runs)
    recent_html = render_recent_run_gallery(recent_runs)
    examples_html = "".join(f'<button type="button" class="example">{esc(x)}</button>' for x in examples)
    help_cards = "".join(
        f"""
        <article class="help-card">
          <div class="help-step">{index + 1:02d}</div>
          <h3>{esc(title)}</h3>
          <p>{esc(copy)}</p>
        </article>
        """
        for index, (title, copy) in enumerate(homepage_help_steps())
    )
    community_data = community_data or {}
    community_status = community_data.get("status") or {}
    living_html = render_living_graph_card(community_status)
    community_hero_html = render_community_hero(community_status)
    pulse_html = render_community_pulse(community_data)
    library_html = render_library_cards(recent_runs)
    return page_shell(
        "Literature Conflict Search",
        f"""
        <section class="hero">
          <div class="eyebrow">research radar</div>
          <h1>Find the fault lines in a research field.</h1>
          <p class="lead">Drop in a topic. The system collects representative papers, extracts claims, maps disagreements, and surfaces hidden bridges between communities.</p>
          <form id="search-form" class="search-box">
            <input name="topic" id="topic" autocomplete="off" placeholder="large language models finance time series forecasting" required>
            <button type="submit">Generate Map</button>
          </form>
          <details class="advanced">
            <summary>Advanced</summary>
            <label>Papers
              <select name="limit" form="search-form">
                <option value="10">10 quick</option>
                <option value="20" selected>20 balanced</option>
                <option value="30">30 slower</option>
              </select>
            </label>
            <label>Source
              <select name="source" form="search-form">
                <option value="arxiv" selected>arXiv stable</option>
                <option value="openalex">OpenAlex citations</option>
              </select>
            </label>
            <label>Paper strategy
              <select name="strategy" form="search-form">
                <option value="balanced" selected>Balanced</option>
                <option value="high-impact">High impact</option>
                <option value="recent">Recent</option>
              </select>
            </label>
            <label>Citation weight
              <select name="citation_weight" form="search-form">
                <option value="0.25">Light</option>
                <option value="0.45" selected>Primary</option>
                <option value="0.65">Dominant</option>
              </select>
            </label>
            <label>Relevance gate
              <select name="min_relevance" form="search-form">
                <option value="0.20">Loose</option>
                <option value="0.30" selected>Normal</option>
                <option value="0.45">Strict</option>
              </select>
            </label>
            <div class="hint">Citation weight only reranks papers that pass relevance. arXiv mode is citation-light. Year is a soft recency signal, not a hard filter beyond the requested year range.</div>
          </details>
          <div class="examples">{examples_html}</div>
          <div class="trust-row">
            <a href="#search-form">Impact-aware retrieval</a>
            <a href="#community-graph">Claim graph</a>
            <a href="#featured-demos">Conflict radar</a>
            <a href="#network-pulse">Bridge finder</a>
          </div>
          <div class="help-banner">
            <div class="help-label">HELP</div>
            <div>
              <strong>This tool is for finding where a research community disagrees.</strong>
              <p>Search a topic, then read the main tension, hidden bridges, representative papers, and the graph-backed report behind them.</p>
            </div>
          </div>
        </section>
        <section id="community-graph">
          <h2>Community Graph</h2>
          <p class="muted">This is what you get: one living workspace where the big clusters, conflict regions, and evidence trails stay in view at the same time.</p>
          {community_hero_html}
        </section>
        <section id="how-it-works">
          <h2>How It Works</h2>
          <div class="help-grid">{help_cards}</div>
        </section>
        <section>
          <h2>What a good output looks like</h2>
          <div class="example-callout">
            <div class="tag">example line</div>
            <blockquote>Gains attributed to the method are inflated when compared against weak baselines; the sign of the effect depends primarily on baseline choice.</blockquote>
            <p>That is the kind of line we want: short, concrete, and tied to a real conflict in the literature.</p>
          </div>
        </section>
        <section id="living-graph">
          <h2>Living Graph</h2>
          <div class="runs">{living_html}</div>
        </section>
        <section id="network-pulse">
          <h2>Network Pulse</h2>
          <div class="runs">{pulse_html}</div>
        </section>
        <section id="quick-starts">
          <h2>Quick Starts</h2>
          <p class="muted">Optional seeded topics for a fast jump-in. If a seed is thin, the system fetches fresh papers.</p>
          <div class="runs">{library_html}</div>
        </section>
        <section id="featured-demos">
          <h2>Featured demos</h2>
          <p class="muted">Open a polished example first. Every card below lands on a finished result page, not a new search.</p>
          <div class="runs">{curated_html}</div>
        </section>
        <section id="recent-maps">
          <h2>Recent maps</h2>
          <p class="muted">Fresh completed runs still show up here, but they no longer define the homepage experience.</p>
          <div class="runs">{recent_html}</div>
        </section>
        <script>
          const params = new URLSearchParams(window.location.search);
          const initialTopic = params.get('topic');
          if (initialTopic) {{
            document.getElementById('topic').value = initialTopic;
          }}
          ['source', 'strategy', 'citation_weight', 'min_relevance', 'limit'].forEach(key => {{
            const value = params.get(key);
            const el = document.querySelector(`[name="${{key}}"]`);
            if (value && el) el.value = value;
          }});
          document.querySelectorAll('.example').forEach(btn => {{
            btn.addEventListener('click', () => {{
              document.getElementById('topic').value = btn.textContent;
              document.getElementById('topic').focus();
            }});
          }});
          document.getElementById('search-form').addEventListener('submit', async event => {{
            event.preventDefault();
            const form = event.currentTarget;
            const button = form.querySelector('button');
            button.disabled = true;
            button.textContent = 'Searching...';
            const payload = Object.fromEntries(new FormData(form).entries());
            const resp = await fetch('/api/search', {{
              method: 'POST',
              headers: {{'Content-Type': 'application/json'}},
              body: JSON.stringify(payload)
            }});
            const data = await resp.json();
            if (!resp.ok) {{
              button.disabled = false;
              button.textContent = 'Search';
              alert(data.error || 'Search failed');
              return;
            }}
            window.location.href = data.url;
          }});
        </script>
        """,
    )


def render_living_graph_card(status: dict[str, Any]) -> str:
    if not status or not status.get("graph_url"):
        return """
        <div class="run">
          <strong>Living graph is empty</strong>
          <span>As completed runs come in, they will be merged into one expanding community graph.</span>
        </div>
        """
    return f"""
    <a class="run demo-run" href="{esc(status.get('graph_url'))}">
      <strong>Open the living community graph</strong>
      <span>{int(status.get('runs') or 0)} runs · {int(status.get('papers') or 0)} papers · {int(status.get('claims') or 0)} claims · {int(status.get('nodes') or 0)} nodes · {int(status.get('edges') or 0)} edges</span>
    </a>
    """


def render_community_hero(status: dict[str, Any]) -> str:
    graph_url = str(status.get("graph_url") or "/community/index.html")
    runs = int(status.get("runs") or 0)
    papers = int(status.get("papers") or 0)
    claims = int(status.get("claims") or 0)
    nodes = int(status.get("nodes") or 0)
    edges = int(status.get("edges") or 0)
    has_graph = bool(status and status.get("graph_url"))
    preview = (
        f"""
        <iframe
          class="community-preview-frame"
          src="{esc(graph_url)}"
          title="Community graph preview"
          loading="lazy"
          tabindex="-1"
          aria-hidden="true"></iframe>
        """
        if has_graph
        else """
        <div class="community-preview-fallback">
          <div class="community-preview-grid"></div>
          <div class="community-preview-node node-one"></div>
          <div class="community-preview-node node-two"></div>
          <div class="community-preview-node node-three"></div>
          <div class="community-preview-link link-one"></div>
          <div class="community-preview-link link-two"></div>
          <div class="community-preview-caption">As searches accumulate, this panel turns into the full living graph.</div>
        </div>
        """
    )
    return f"""
    <div class="community-hero-card">
      <div class="community-hero-copy">
        <div class="eyebrow">community graph</div>
        <h3>This is what you get.</h3>
        <p class="section-lead">A real map of the field, not just a search result. Start from the cluster view, trace the strongest conflict, then drill down into the paper evidence underneath it.</p>
        <div class="community-meta">
          <span>{runs} runs</span>
          <span>{papers} papers</span>
          <span>{claims} claims</span>
          <span>{nodes} nodes</span>
          <span>{edges} edges</span>
        </div>
        <div class="community-actions">
          <a class="primary" href="{esc(graph_url)}">Open Community Graph</a>
          <a href="#featured-demos">Open a finished demo first</a>
        </div>
      </div>
      <div class="community-preview-shell">
        {preview}
        <div class="community-preview-scrim"></div>
        <div class="community-callout callout-top">
          <strong>Cluster view first</strong>
          <span>The first screen is the map, not a raw dump of papers.</span>
        </div>
        <div class="community-callout callout-right">
          <strong>Conflict + bridge layer</strong>
          <span>Big tensions and cross-community bridges stay visible together.</span>
        </div>
        <div class="community-callout callout-bottom">
          <strong>Evidence drill-down</strong>
          <span>Every region can open into claims, papers, and the underlying report.</span>
        </div>
      </div>
    </div>
    """


def render_community_pulse(data: dict[str, Any]) -> str:
    newest = data.get("newest_runs") or []
    hot = data.get("hottest_topics") or []
    conflicts = data.get("biggest_conflicts") or []
    bridges = data.get("newest_bridges") or []
    blocks: list[str] = []
    if newest:
        items = "".join(
            f"<li><a href='/search/{esc(item['run_id'])}'>{esc(item['topic'])}</a> · {item['claims']} claims · {item['anomalies']} conflicts</li>"
            for item in newest[:3]
        )
        blocks.append(f"<div class='run'><strong>Newest runs</strong><span><ul>{items}</ul></span></div>")
    if hot:
        items = "".join(f"<li>{esc(item['topic'])} · {item['count']} searches</li>" for item in hot[:3])
        blocks.append(f"<div class='run'><strong>Growing topics</strong><span><ul>{items}</ul></span></div>")
    if conflicts:
        items = "".join(f"<li>{esc(item['question'])} · {item['claim_count']} claims</li>" for item in conflicts[:3])
        blocks.append(f"<div class='run'><strong>Biggest conflict clusters</strong><span><ul>{items}</ul></span></div>")
    if bridges:
        items = "".join(f"<li>{esc(item['title'])}</li>" for item in bridges[:3])
        blocks.append(f"<div class='run'><strong>Newest bridges</strong><span><ul>{items}</ul></span></div>")
    return "".join(blocks) or "<div class='run'><strong>Network pulse is quiet</strong><span>As searches accumulate, this section will show what the graph is learning.</span></div>"


def render_library_cards(recent_runs: list[dict[str, Any]]) -> str:
    done_by_topic = {str(r.get("topic")): r for r in recent_runs if r.get("status") == "done"}
    cards = []
    for lib in specialty_libraries():
        existing = done_by_topic.get(lib["topic"])
        href = (
            f"/search/{existing['run_id']}"
            if existing and existing.get("run_id")
            else search_prefill_href(
                str(lib["topic"]),
                source=str(lib.get("source") or ""),
                strategy=str(lib.get("strategy") or ""),
                citation_weight=float(lib.get("citation_weight") or 0.45),
                min_relevance=float(lib.get("min_relevance") or 0.30),
            )
        )
        status = (
            f"{int(existing.get('papers') or 0)} papers · {int(existing.get('claims') or 0)} claims"
            if existing
            else "seeded topic · fetch on demand"
        )
        pinned = ""
        if existing:
            pinned = _featured_line(existing)
        cards.append(
            f"""
            <a class="run demo-run" href="{esc(href)}">
              <strong>{esc(lib['title'])}</strong>
              <span>{esc(lib['blurb'])}</span>
              {f"<span class='run-line'>{esc(pinned)}</span>" if pinned else ""}
              <span>{esc(status)}</span>
            </a>
            """
        )
    return "".join(cards)


def render_curated_demo_cards(recent_runs: list[dict[str, Any]]) -> str:
    finished_runs = [
        r for r in recent_runs
        if r.get("status") == "done" and r.get("run_id")
    ]
    recent_by_id = {str(r.get("run_id")): r for r in finished_runs if r.get("run_id")}
    cards = []
    for demo in curated_demos():
        existing = recent_by_id.get(str(demo.get("preferred_run_id") or ""))
        if not (existing and existing.get("run_id")):
            continue
        href = f"/search/{existing['run_id']}"
        line = _featured_line(existing) or str(demo.get("pinned_line") or "")
        paper_count = int(existing.get("papers") or 0)
        claim_count = int(existing.get("claims") or 0)
        status = (
            f"{paper_count} papers · {claim_count} claims"
            if paper_count > 0 or claim_count > 0
            else "ready demo"
        )
        cards.append(
            f"""
            <a class="run demo-run curated-demo" href="{esc(href)}">
              <strong>{esc(demo['title'])}</strong>
              <span>{esc(demo['blurb'])}</span>
              {f"<span class='run-line'>{esc(line)}</span>" if line else ""}
              <span>{esc(status)}</span>
            </a>
            """
        )
    if cards:
        return "".join(cards)
    return render_recent_run_gallery(recent_runs)


def render_recent_run_gallery(recent_runs: list[dict[str, Any]]) -> str:
    finished = [r for r in recent_runs if r.get("run_id") and r.get("status") == "done"]
    if finished:
        return "".join(
            f"""
            <a class="run demo-run" href="/search/{esc(r.get('run_id', ''))}">
              <strong>{esc(r.get('topic', 'Untitled run'))}</strong>
              {f"<span class='run-line'>{esc(_featured_line(r))}</span>" if _featured_line(r) else ""}
              <span>{int(r.get('papers') or 0)} papers · {int(r.get('claims') or 0)} claims · {int(r.get('anomalies') or 0)} conflicts · {int(r.get('insights') or 0)} insights</span>
            </a>
            """
            for r in finished[:6]
        )
    return "".join(
        f"""
        <a class="run demo-topic" href="{search_prefill_href(topic)}">
          <strong>{esc(topic)}</strong>
          <span>Try this topic to generate a fresh map.</span>
        </a>
        """
        for topic in demo_topics()
    )


def _featured_line(run: dict[str, Any]) -> str:
    overview = run.get("overview") or {}
    hero = overview.get("hero_line") or {}
    if hero.get("line"):
        return str(hero["line"])
    for key in ("best_conflict_lines", "best_bridge_lines", "best_explanation_lines"):
        for item in overview.get(key) or []:
            line = str(item.get("line") or "").strip()
            if line:
                return line
    return ""

def render_result_page(status: dict[str, Any], recent_runs: list[dict[str, Any]] | None = None) -> str:
    run_id = str(status.get("run_id") or "")
    demo_html = render_curated_demo_cards(recent_runs or [])
    return page_shell(
        "Search Results",
        f"""
        <section class="result-head">
          <a class="back" href="/">← New search</a>
          <div class="eyebrow">live research map</div>
          <h1>{esc(status.get('topic', 'Untitled run'))}</h1>
          <p class="muted">source: <strong>{esc(status.get('source', 'arxiv'))}</strong> · strategy: <strong>{esc(status.get('strategy', 'balanced'))}</strong> · citation weight: <strong>{esc(status.get('citation_weight', '0.45'))}</strong> · relevance gate: <strong>{esc(status.get('min_relevance', '0.30'))}</strong></p>
          <p class="muted">query: <code>{esc(status.get('source_query') or status.get('arxiv_query', ''))}</code></p>
          <div class="space-loader" id="space-loader" style="--progress:0;">
            <div class="space-loader-bg"></div>
            <div class="space-loader-stars"></div>
            <div class="space-loader-orbit"></div>
            <div class="space-loader-progress" id="loader-progress"></div>
            <div class="space-loader-destination">
              <div class="signal-dish">
                <span class="signal-ring signal-ring-one"></span>
                <span class="signal-ring signal-ring-two"></span>
              </div>
              <span>mapped insight</span>
            </div>
            <div class="space-loader-planet">
              <div class="planet-core"></div>
              <div class="planet-ring"></div>
              <div class="planet-shadow"></div>
            </div>
            <div class="astronaut-wrap" id="astronaut-wrap">
              <svg class="astronaut-svg" viewBox="0 0 180 180" aria-hidden="true">
                <ellipse cx="66" cy="138" rx="28" ry="18" fill="rgba(102, 196, 255, .18)"/>
                <ellipse cx="91" cy="57" rx="33" ry="35" fill="#fbfcff" stroke="#0e1320" stroke-width="6"/>
                <ellipse cx="98" cy="59" rx="24" ry="25" fill="#0b1220"/>
                <circle cx="75" cy="47" r="12" fill="#f6f9ff" stroke="#0e1320" stroke-width="6"/>
                <path d="M66 81c-10 10-17 24-16 39l3 26c.5 5 4 9 9 9h38c5 0 8-4 8-9l4-28c2-15-3-28-13-38Z" fill="#fbfcff" stroke="#0e1320" stroke-width="6" stroke-linejoin="round"/>
                <path d="M60 87c11 7 24 9 40 5" fill="none" stroke="#9fd3ff" stroke-width="6" stroke-linecap="round"/>
                <path d="M54 101 33 116c-6 4-8 13-4 19l5 8" fill="none" stroke="#fbfcff" stroke-width="6" stroke-linecap="round"/>
                <path d="M102 100 126 114c6 4 8 13 4 19l-6 9" fill="none" stroke="#fbfcff" stroke-width="6" stroke-linecap="round"/>
                <path d="M68 154 55 171" fill="none" stroke="#fbfcff" stroke-width="6" stroke-linecap="round"/>
                <path d="M95 154 112 170" fill="none" stroke="#fbfcff" stroke-width="6" stroke-linecap="round"/>
                <rect x="66" y="101" width="27" height="26" rx="8" fill="#f7e8ab" opacity=".92"/>
                <path d="M91 84c8 7 18 7 28 1" fill="none" stroke="#4669ff" stroke-width="6" stroke-linecap="round"/>
                <circle cx="114" cy="84" r="8" fill="#c7edff" stroke="#3257d2" stroke-width="5"/>
                <rect x="101" y="108" width="42" height="30" rx="5" fill="#cfd4e8" stroke="#0e1320" stroke-width="5"/>
                <path d="M106 114h32" fill="none" stroke="#9ca8cb" stroke-width="4" stroke-linecap="round"/>
                <circle cx="126" cy="124" r="6" fill="#7c56ff"/>
                <ellipse cx="126" cy="124" rx="13" ry="4.6" fill="none" stroke="#ffd35d" stroke-width="3"/>
              </svg>
            </div>
            <div class="space-loader-meta">
              <span class="space-loader-label">exploration progress</span>
              <strong id="progress-label">0%</strong>
            </div>
          </div>
          <div id="status-line" class="status-line">{esc(status.get('message', 'Loading...'))}</div>
          <div id="stats" class="stats"></div>
          <div id="actions" class="actions"></div>
        </section>
        <section>
          <h2>Overview</h2>
          <div id="overview" class="overview">The map is being generated. This page updates automatically.</div>
        </section>
        <section id="demo-section">
          <h2>Explore While You Wait</h2>
          <p class="muted">Open a finished demo in another tab while this run is cooking.</p>
          <div class="runs">{demo_html}</div>
        </section>
        <script>
          const runId = {json.dumps(run_id)};
          async function poll() {{
            const resp = await fetch(`/api/runs/${{runId}}`);
            const data = await resp.json();
            const pct = Math.round((data.progress || 0) * 100);
            document.getElementById('space-loader').style.setProperty('--progress', String(pct / 100));
            document.getElementById('loader-progress').style.width = `${{pct}}%`;
            document.getElementById('progress-label').textContent = `${{pct}}%`;
            document.getElementById('status-line').textContent = `${{data.stage || data.status}} · ${{data.message || ''}}`;
            document.getElementById('stats').innerHTML = [
              ['papers', data.papers || 0, true],
              ['claims', data.claims || 0, true],
              ['conflicts', data.anomalies || 0, (data.anomalies || 0) > 0],
              ['hypotheses', data.hypotheses || 0, (data.hypotheses || 0) > 0],
              ['insights', data.insights || 0, (data.insights || 0) > 0],
            ].filter(([, , visible]) => visible).map(([label, value]) =>
              `<div class="stat"><strong>${{value}}</strong><span>${{label}}</span></div>`
            ).join('');
            if (data.status === 'done') {{
              document.getElementById('actions').innerHTML = `
                <a class="primary" href="${{data.graph_url}}">Open Graph</a>
                <a href="${{data.report_url}}">Open Report</a>
                <a href="${{data.overview_url || `/runs/${{runId}}/overview.json`}}">Overview JSON</a>
                <a href="/runs/${{runId}}/papers.jsonl">Download Data</a>
              `;
              document.getElementById('overview').innerHTML = renderOverview(data);
              return;
            }}
            if (data.status === 'error') {{
              document.getElementById('overview').innerHTML = renderErrorState(data);
              return;
            }}
            setTimeout(poll, 2000);
          }}
          function escapeHtml(s) {{
            return String(s).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
          }}
          function renderOverview(data) {{
            const ov = data.overview || {{}};
            const heroLine = ov.hero_line || null;
            const whyThisMatters = ov.why_this_matters || null;
            const bestConflictLines = (ov.best_conflict_lines || []).map(c => `
              <article class="mini-card emphasis-card">
                <div class="tag">${{escapeHtml(c.source_type || 'conflict')}}</div>
                <h3>${{escapeHtml(c.line || 'Conflict signal')}}</h3>
                <p>${{escapeHtml(c.supporting_text || c.source_id || '')}}</p>
              </article>
            `).join('');
            const bestBridgeLines = (ov.best_bridge_lines || []).map(i => `
              <article class="mini-card bridge emphasis-card">
                <div class="tag">${{escapeHtml(i.source_type || 'bridge')}}</div>
                <h3>${{escapeHtml(i.line || 'Community bridge')}}</h3>
                <p>${{escapeHtml(i.supporting_text || i.source_id || '')}}</p>
              </article>
            `).join('');
            const bestExplanationLines = (ov.best_explanation_lines || []).map(h => `
              <article class="mini-card explanation-card">
                <div class="tag">${{escapeHtml(h.source_type || 'explanation')}}</div>
                <h3>${{escapeHtml(h.line || 'Explanation')}}</h3>
                <p>${{escapeHtml(h.supporting_text || h.source_id || '')}}</p>
              </article>
            `).join('');
            const conflicts = (ov.top_conflicts || []).map(c => `
              <article class="mini-card">
                <div class="tag">${{escapeHtml(c.type || 'conflict')}}</div>
                <h3>${{escapeHtml(c.question || c.anomaly_id || 'Conflict region')}}</h3>
                <p>${{c.claim_count || 0}} claims · +${{c.positive || 0}} / -${{c.negative || 0}} · ${{c.impact_available ? `impact ${{c.impact ?? 0}}` : 'citation impact unavailable'}}</p>
              </article>
            `).join('');
            const bridges = (ov.hidden_bridges || []).map(i => `
              <article class="mini-card bridge">
                <div class="tag">${{escapeHtml(i.type || 'insight')}}</div>
                <h3>${{escapeHtml(i.title || 'Hidden bridge')}}</h3>
                <p>${{escapeHtml((i.communities || []).join(' ↔ '))}}</p>
                <p>${{(i.shared_concepts || []).map(x => `<span class="pill">${{escapeHtml(x)}}</span>`).join('')}}</p>
              </article>
            `).join('');
            const papers = (ov.top_papers || []).map(p => `
              <article class="paper-card">
                <div class="tag">${{escapeHtml(p.paper_role_label || p.retrieval_channel || 'paper')}}</div>
                <h3>${{p.url ? `<a href="${{escapeHtml(p.url)}}" target="_blank" rel="noopener">${{escapeHtml(p.title || p.paper_id)}}</a>` : escapeHtml(p.title || p.paper_id)}}</h3>
                <p>${{escapeHtml([p.venue, p.year].filter(Boolean).join(' · '))}} · ${{p.citation_available ? `${{p.citations || 0}} citations` : 'citations unavailable'}} · score ${{p.selection_score || 0}}</p>
                <p>${{escapeHtml(p.paper_role_explanation || '')}}</p>
                <p>${{escapeHtml(p.selection_reason || 'Selected as a representative paper.')}}</p>
              </article>
            `).join('');
            const path = (ov.reading_path || []).map((p, idx) => `
              <article class="path-step">
                <strong>${{idx + 1}}. ${{escapeHtml(p.step || 'Read this paper')}}</strong>
                <span>${{escapeHtml(p.title || p.paper_id)}}${{p.year ? ` (${{p.year}})` : ''}}</span>
                <p>${{escapeHtml(p.why_read || p.selection_reason || '')}}</p>
              </article>
            `).join('');
            return `
              <div class="overview-hero">
                <p><strong>${{escapeHtml((heroLine && heroLine.line) || ov.headline || 'Done.')}}</strong></p>
                ${{(heroLine && heroLine.line) ? `<p class="hero-subline">${{escapeHtml(ov.headline || '')}}</p>` : ''}}
                <p>Paper source: <strong>${{escapeHtml(data.source || 'arxiv')}}</strong>; strategy: <strong>${{escapeHtml(data.strategy || 'balanced')}}</strong>; citation weight: <strong>${{escapeHtml(data.citation_weight ?? '0.45')}}</strong>; relevance gate: <strong>${{escapeHtml(data.min_relevance ?? '0.30')}}</strong>.</p>
              </div>
              <div class="summary-band">
                <article class="summary-pill">
                  <div class="tag">Main tension</div>
                  <p>${{escapeHtml((heroLine && heroLine.line) || ov.headline || 'No clear tension yet.')}}</p>
                </article>
                <article class="summary-pill">
                  <div class="tag">Why this matters</div>
                  <p>${{escapeHtml((whyThisMatters && whyThisMatters.line) || 'The strongest signal is still emerging.')}}</p>
                </article>
                <article class="summary-pill">
                  <div class="tag">Next step</div>
                  <p>${{escapeHtml((whyThisMatters && whyThisMatters.next_step) || 'Open the graph or report to inspect the evidence.')}}</p>
                </article>
              </div>
              <h3>Main Tension</h3>
              <div class="card-grid">${{bestConflictLines || conflicts || '<div class="empty">No strong conflict detected yet.</div>'}}</div>
              <h3>Hidden Bridges</h3>
              <div class="card-grid">${{bestBridgeLines || bridges || '<div class="empty">No community bridge detected yet.</div>'}}</div>
              <h3>Key Explanations</h3>
              <div class="card-grid">${{bestExplanationLines || '<div class="empty">No high-signal explanation line yet.</div>'}}</div>
              <h3>Representative Papers</h3>
              <div class="paper-grid">${{papers || '<div class="empty">No papers available.</div>'}}</div>
              <h3>Reading Path</h3>
              <div class="path-grid">${{path || '<div class="empty">No reading path available yet.</div>'}}</div>
            `;
          }}
          function renderErrorState(data) {{
            const title = escapeHtml(data.error_title || 'Run failed');
            const summary = escapeHtml(data.error_summary || data.message || 'Something went wrong.');
            const recovery = escapeHtml(data.error_recovery || 'Please try again.');
            const raw = data.error ? `<p class="error-raw">${{escapeHtml(data.error)}}</p>` : '';
            return `
              <article class="error-card">
                <div class="tag">${{escapeHtml(data.error_kind || 'error')}}</div>
                <h3>${{title}}</h3>
                <p>${{summary}}</p>
                <p><strong>Next step:</strong> ${{recovery}}</p>
                ${{raw}}
              </article>
            `;
          }}
          poll();
        </script>
        """,
    )


def render_error_page(status_code: HTTPStatus, message: str) -> str:
    return page_shell(str(status_code.value), f"<section><h1>{status_code.value}</h1><p>{esc(message)}</p></section>")


def page_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      --ink:#e6eef3;
      --muted:#92a4b1;
      --line:rgba(120, 152, 171, 0.28);
      --bg:#091018;
      --panel:rgba(10, 18, 28, 0.82);
      --panel-strong:rgba(8, 14, 22, 0.94);
      --accent:#24d1b6;
      --accent2:#67d9ff;
      --accent3:#ff7b72;
      --soft:#0f1823;
      --glow:0 0 0 1px rgba(36,209,182,.16), 0 20px 60px rgba(0,0,0,.34);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:
      linear-gradient(115deg, rgba(36,209,182,.10), transparent 32%),
      linear-gradient(245deg, rgba(103,217,255,.10), transparent 30%),
      repeating-linear-gradient(135deg, rgba(103,217,255,.025) 0 2px, transparent 2px 26px),
      linear-gradient(180deg, #09111a 0%, #081018 46%, #070e16 100%);
      letter-spacing:0; }}
    body:before {{
      content:"";
      position:fixed;
      inset:0;
      pointer-events:none;
      background:
        linear-gradient(90deg, rgba(103,217,255,.06) 1px, transparent 1px),
        linear-gradient(0deg, rgba(36,209,182,.05) 1px, transparent 1px);
      background-size:36px 36px;
      mask-image:linear-gradient(180deg, rgba(0,0,0,.65), transparent 88%);
      opacity:.34;
    }}
    body:after {{
      content:"";
      position:fixed;
      inset:0;
      pointer-events:none;
      background:
        linear-gradient(90deg, transparent 0%, rgba(103,217,255,.035) 48%, transparent 52%, transparent 100%),
        linear-gradient(180deg, transparent 0%, rgba(36,209,182,.02) 58%, transparent 100%);
      mix-blend-mode:screen;
      opacity:.85;
    }}
    main {{ width:min(1080px, calc(100vw - 32px)); margin:0 auto; padding:34px 0 70px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:24px; margin:16px 0; box-shadow:var(--glow); backdrop-filter:blur(18px); }}
    .hero {{ margin-top:24px; padding:42px; border-color:rgba(103,217,255,.26); box-shadow:0 0 0 1px rgba(103,217,255,.1), 0 24px 80px rgba(0,0,0,.42); position:relative; overflow:hidden; background:
      linear-gradient(135deg, rgba(103,217,255,.10), transparent 34%),
      linear-gradient(225deg, rgba(36,209,182,.10), transparent 30%),
      repeating-linear-gradient(90deg, rgba(103,217,255,.025) 0 1px, transparent 1px 24px),
      linear-gradient(180deg, rgba(10,18,28,.92), rgba(8,14,22,.96)); }}
    .hero:before {{ content:""; position:absolute; inset:0; background:
      linear-gradient(90deg, rgba(103,217,255,.08) 1px, transparent 1px),
      linear-gradient(0deg, rgba(36,209,182,.07) 1px, transparent 1px);
      background-size:42px 42px; mask-image:linear-gradient(90deg, #000 0%, transparent 75%); pointer-events:none; }}
    .hero > * {{ position:relative; }}
    .eyebrow {{ color:var(--accent2); font-weight:700; font-size:14px; text-transform:uppercase; letter-spacing:.08em; }}
    h1 {{ font-size:52px; line-height:1.02; margin:10px 0 18px; max-width:900px; }}
    h2 {{ font-size:24px; margin:0 0 16px; }}
    .lead {{ color:#c1d0d9; font-size:21px; line-height:1.55; max-width:800px; margin:0 0 24px; }}
    .search-box {{ display:flex; gap:8px; margin:20px 0 12px; }}
    input, select {{ width:100%; border:1px solid var(--line); border-radius:8px; padding:14px 15px; font:inherit; background:rgba(9,16,24,.86); color:var(--ink); font-size:16px; }}
    input::placeholder {{ color:#708391; }}
    .search-box input {{ font-size:20px; padding:18px; border-color:rgba(103,217,255,.34); box-shadow:inset 0 0 0 1px rgba(103,217,255,.06), 0 0 28px rgba(36,209,182,.06); }}
    button, a {{ border-radius:8px; }}
    .search-box button, .primary {{ border:0; background:linear-gradient(135deg, var(--accent), var(--accent2)); color:#041015; padding:15px 22px; font:inherit; font-weight:800; font-size:16px; cursor:pointer; text-decoration:none; box-shadow:0 12px 34px rgba(36,209,182,.24); }}
    .advanced {{ color:var(--muted); margin:8px 0 14px; font-size:15px; }}
    .advanced label {{ display:inline-flex; align-items:center; gap:8px; margin:10px 12px 0 0; }}
    .advanced select {{ width:auto; padding:8px 10px; }}
    .hint {{ margin-top:10px; font-size:14px; color:var(--muted); line-height:1.5; }}
    .examples {{ display:flex; flex-wrap:wrap; gap:8px; margin:18px 0; }}
    .example {{ border:1px solid var(--line); background:rgba(9,16,24,.78); color:var(--ink); padding:10px 12px; font:inherit; font-size:15px; cursor:pointer; box-shadow:inset 0 0 0 1px rgba(103,217,255,.03); }}
    .trust-row {{ display:flex; flex-wrap:wrap; gap:8px; margin:18px 0 4px; }}
    .trust-row a {{ border:1px solid rgba(36,209,182,.28); background:rgba(36,209,182,.08); color:#89f4e5; border-radius:8px; padding:9px 11px; font-size:14px; font-weight:700; text-decoration:none; transition:transform .16s ease, border-color .16s ease, background .16s ease; }}
    .trust-row a:hover {{ transform:translateY(-1px); border-color:rgba(103,217,255,.34); background:rgba(103,217,255,.10); }}
    .help-banner {{ display:grid; grid-template-columns:auto 1fr; gap:14px; align-items:start; margin:18px 0 8px; padding:16px 18px; border:1px solid rgba(103,217,255,.24); border-radius:8px; background:linear-gradient(135deg, rgba(36,209,182,.08), rgba(86,156,255,.08)); box-shadow:0 10px 30px rgba(36,209,182,.08); }}
    .help-label {{ display:inline-flex; align-items:center; justify-content:center; min-width:58px; height:30px; border-radius:8px; background:rgba(103,217,255,.16); color:#9de8ff; font-size:12px; font-weight:800; letter-spacing:.08em; }}
    .help-banner strong {{ display:block; font-size:18px; margin-bottom:4px; }}
    .help-banner p {{ margin:0; color:var(--muted); font-size:15px; line-height:1.5; }}
    .help-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }}
    .help-card {{ border:1px solid var(--line); border-radius:8px; padding:14px; background:linear-gradient(180deg, rgba(14,22,34,.96), rgba(10,16,24,.96)); }}
    .help-step {{ color:#7de7d0; font-size:12px; font-weight:800; letter-spacing:.08em; margin-bottom:8px; }}
    .help-card h3 {{ margin:0 0 8px; font-size:20px; }}
    .help-card p {{ margin:0; color:var(--muted); line-height:1.6; font-size:15px; }}
    .example-callout {{ border:1px solid rgba(103,217,255,.24); border-radius:8px; padding:18px; background:linear-gradient(180deg, rgba(13,20,31,.96), rgba(9,16,24,.96)); box-shadow:0 10px 30px rgba(86,156,255,.06); }}
    .example-callout blockquote {{ margin:10px 0 8px; font-size:24px; line-height:1.48; color:#eef8ff; }}
    .example-callout p {{ margin:0; color:var(--muted); font-size:15px; line-height:1.5; }}
    .muted {{ color:var(--muted); line-height:1.6; font-size:15px; }}
    .section-lead {{ color:#d7e7ef; font-size:18px; line-height:1.55; margin:0; max-width:42ch; }}
    .community-hero-card {{ display:grid; grid-template-columns:minmax(260px, 340px) minmax(0, 1fr); gap:18px; align-items:stretch; }}
    .community-hero-copy {{ display:flex; flex-direction:column; gap:14px; }}
    .community-hero-copy h3 {{ margin:0; font-size:34px; line-height:1.05; }}
    .community-meta {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .community-meta span {{ border:1px solid rgba(103,217,255,.18); background:rgba(103,217,255,.06); border-radius:8px; padding:8px 10px; color:#d6e6ef; font-size:14px; font-weight:650; }}
    .community-actions {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:auto; }}
    .community-actions a {{ border:1px solid var(--line); background:rgba(9,16,24,.82); color:var(--ink); padding:12px 14px; text-decoration:none; font-weight:700; }}
    .community-actions .primary {{ border-color:transparent; }}
    .community-preview-shell {{ position:relative; min-height:340px; border:1px solid rgba(103,217,255,.22); border-radius:8px; overflow:hidden; background:linear-gradient(180deg, rgba(10,18,28,.96), rgba(7,13,21,.98)); box-shadow:inset 0 0 0 1px rgba(103,217,255,.04), 0 18px 40px rgba(0,0,0,.22); }}
    .community-preview-frame {{ position:absolute; inset:0; width:170%; height:170%; border:0; transform:scale(.59); transform-origin:top left; pointer-events:none; filter:saturate(.82) contrast(.98); opacity:.96; }}
    .community-preview-scrim {{ position:absolute; inset:0; background:
      linear-gradient(180deg, rgba(7,12,18,.08), rgba(7,12,18,.26)),
      linear-gradient(90deg, rgba(7,12,18,.12), transparent 25%, transparent 76%, rgba(7,12,18,.20)); pointer-events:none; }}
    .community-preview-fallback {{ position:absolute; inset:0; background:
      radial-gradient(circle at 18% 24%, rgba(103,217,255,.14), transparent 20%),
      radial-gradient(circle at 70% 30%, rgba(36,209,182,.12), transparent 18%),
      linear-gradient(180deg, rgba(10,18,28,.96), rgba(7,13,21,.98)); overflow:hidden; }}
    .community-preview-grid {{ position:absolute; inset:0; background:
      linear-gradient(90deg, rgba(103,217,255,.05) 1px, transparent 1px),
      linear-gradient(0deg, rgba(36,209,182,.05) 1px, transparent 1px); background-size:28px 28px; opacity:.45; }}
    .community-preview-node {{ position:absolute; border-radius:999px; background:radial-gradient(circle, rgba(103,217,255,.92), rgba(103,217,255,.18)); box-shadow:0 0 24px rgba(103,217,255,.18); }}
    .community-preview-node.node-one {{ width:120px; height:120px; left:64px; top:82px; }}
    .community-preview-node.node-two {{ width:84px; height:84px; left:248px; top:126px; background:radial-gradient(circle, rgba(36,209,182,.9), rgba(36,209,182,.18)); }}
    .community-preview-node.node-three {{ width:104px; height:104px; right:70px; top:96px; background:radial-gradient(circle, rgba(255,214,108,.88), rgba(255,214,108,.16)); }}
    .community-preview-link {{ position:absolute; height:2px; background:linear-gradient(90deg, rgba(103,217,255,.14), rgba(255,255,255,.44), rgba(36,209,182,.14)); transform-origin:left center; }}
    .community-preview-link.link-one {{ left:170px; top:148px; width:126px; transform:rotate(12deg); }}
    .community-preview-link.link-two {{ left:316px; top:158px; width:160px; transform:rotate(-8deg); }}
    .community-preview-caption {{ position:absolute; left:18px; bottom:16px; color:#c9d9e2; font-size:14px; max-width:32ch; line-height:1.45; }}
    .community-callout {{ position:absolute; max-width:220px; padding:10px 12px; border:1px solid rgba(103,217,255,.24); border-radius:8px; background:rgba(8,14,22,.82); box-shadow:0 12px 30px rgba(0,0,0,.28); backdrop-filter:blur(10px); }}
    .community-callout strong {{ display:block; font-size:14px; margin-bottom:4px; }}
    .community-callout span {{ display:block; color:var(--muted); font-size:13px; line-height:1.45; }}
    .community-callout.callout-top {{ left:18px; top:18px; }}
    .community-callout.callout-right {{ right:18px; top:86px; }}
    .community-callout.callout-bottom {{ left:56px; bottom:18px; max-width:250px; }}
    .runs {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:10px; }}
    .run {{ display:block; border:1px solid var(--line); background:linear-gradient(180deg, rgba(13,20,31,.92), rgba(10,17,25,.92)); border-radius:8px; padding:14px; color:var(--ink); text-decoration:none; transition:transform .16s ease, border-color .16s ease, box-shadow .16s ease; box-shadow:inset 0 0 0 1px rgba(103,217,255,.03); }}
    .run:hover {{ transform:translateY(-2px); border-color:rgba(103,217,255,.36); box-shadow:0 16px 34px rgba(0,0,0,.28), 0 0 0 1px rgba(103,217,255,.08); }}
    .run strong {{ font-size:22px; line-height:1.2; }}
    .run span {{ display:block; margin-top:6px; color:var(--muted); font-size:16px; line-height:1.35; }}
    .curated-demo {{ border-color:rgba(36,209,182,.22); box-shadow:inset 0 0 0 1px rgba(36,209,182,.06), 0 12px 28px rgba(0,0,0,.18); }}
    .run-line {{ color:#dce8ef; font-weight:650; line-height:1.45; font-size:17px !important; }}
    .space-loader {{ --progress:0; position:relative; height:180px; border:1px solid rgba(103,217,255,.18); border-radius:8px; overflow:hidden; margin:18px 0 12px; background:
      radial-gradient(circle at 18% 24%, rgba(103,217,255,.16), transparent 22%),
      radial-gradient(circle at 78% 18%, rgba(126,118,255,.2), transparent 18%),
      linear-gradient(180deg, rgba(13,20,31,.98), rgba(8,13,22,.98)); box-shadow:inset 0 0 0 1px rgba(103,217,255,.03), 0 24px 50px rgba(2,8,18,.36); }}
    .space-loader-bg {{ position:absolute; inset:0; background:
      radial-gradient(circle at 15% 80%, rgba(255,210,108,.08), transparent 20%),
      radial-gradient(circle at 64% 50%, rgba(36,209,182,.06), transparent 26%); }}
    .space-loader-stars, .space-loader-stars:before, .space-loader-stars:after {{ content:""; position:absolute; inset:0; background-image:
      radial-gradient(circle, rgba(255,255,255,.85) 0 1.2px, transparent 1.4px),
      radial-gradient(circle, rgba(157,232,255,.75) 0 1px, transparent 1.3px),
      radial-gradient(circle, rgba(255,213,93,.6) 0 1.2px, transparent 1.5px);
      background-size:130px 130px, 190px 190px, 240px 240px;
      background-position:10px 22px, 58px 70px, 110px 18px;
      opacity:.55; animation:starDrift 24s linear infinite; }}
    .space-loader-stars:before {{ opacity:.35; transform:scale(1.08); animation-duration:32s; }}
    .space-loader-stars:after {{ opacity:.28; transform:scale(.96); animation-duration:40s; }}
    .space-loader-orbit {{ position:absolute; left:34px; right:92px; top:96px; height:12px; border-radius:999px; background:linear-gradient(90deg, rgba(91,111,128,.24), rgba(91,111,128,.08)); box-shadow:inset 0 0 0 1px rgba(103,217,255,.08); }}
    .space-loader-progress {{ position:absolute; left:34px; top:96px; height:12px; width:0; border-radius:999px; background:linear-gradient(90deg, rgba(255,210,108,.35), rgba(103,217,255,.92)); box-shadow:0 0 24px rgba(103,217,255,.24); transition:width .55s cubic-bezier(.2,.8,.2,1); }}
    .space-loader-progress:after {{ content:""; position:absolute; right:-22px; top:50%; width:30px; height:30px; transform:translateY(-50%); background:radial-gradient(circle, rgba(103,217,255,.32), transparent 68%); }}
    .space-loader-planet {{ position:absolute; left:24px; bottom:14px; width:112px; height:112px; }}
    .planet-core {{ position:absolute; inset:12px; border-radius:50%; background:
      linear-gradient(180deg, #ffd868, #f0a72b 55%, #cd7d17);
      box-shadow:inset -14px -20px 0 rgba(119,70,0,.18), inset 10px 8px 0 rgba(255,245,199,.25); }}
    .planet-core:before, .planet-core:after {{ content:""; position:absolute; left:12%; right:12%; height:10px; border-radius:999px; background:rgba(152,79,7,.26); }}
    .planet-core:before {{ top:24px; transform:rotate(-11deg); }}
    .planet-core:after {{ top:56px; transform:rotate(7deg); }}
    .planet-ring {{ position:absolute; left:-8px; right:-8px; top:52px; height:24px; border-radius:999px; border:8px solid rgba(255,229,155,.72); box-shadow:inset 0 0 0 3px rgba(123,80,9,.18); transform:rotate(-12deg); }}
    .planet-shadow {{ position:absolute; right:18px; top:20px; width:30px; height:72px; border-radius:50%; background:rgba(86,53,0,.14); filter:blur(1px); }}
    .astronaut-wrap {{ position:absolute; left:calc(28px + (100% - 160px) * var(--progress)); top:24px; width:120px; height:120px; transform:translateX(-8%) rotate(-3deg); transition:left .6s cubic-bezier(.2,.8,.2,1); animation:astroFloat 3.6s ease-in-out infinite; filter:drop-shadow(0 18px 24px rgba(4,12,22,.4)); }}
    .astronaut-svg {{ width:100%; height:100%; overflow:visible; }}
    .space-loader-destination {{ position:absolute; right:24px; top:42px; display:flex; flex-direction:column; align-items:center; gap:8px; color:#dce8ef; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; opacity:.92; }}
    .signal-dish {{ position:relative; width:38px; height:38px; border-radius:50%; border:3px solid rgba(157,232,255,.8); box-shadow:0 0 18px rgba(103,217,255,.22); }}
    .signal-dish:before {{ content:""; position:absolute; inset:10px; border-radius:50%; background:linear-gradient(135deg, rgba(103,217,255,.4), rgba(157,232,255,.14)); }}
    .signal-ring {{ position:absolute; inset:-6px; border-radius:50%; border:2px solid rgba(103,217,255,.28); }}
    .signal-ring-one {{ animation:signalPulse 1.8s ease-out infinite; }}
    .signal-ring-two {{ animation:signalPulse 1.8s ease-out .9s infinite; }}
    .space-loader-meta {{ position:absolute; left:150px; bottom:18px; display:flex; align-items:end; gap:12px; }}
    .space-loader-label {{ color:#93a8b7; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.1em; }}
    .space-loader-meta strong {{ font-size:28px; line-height:1; color:#f4fbff; text-shadow:0 0 18px rgba(103,217,255,.18); }}
    .status-line {{ color:var(--muted); min-height:22px; }}
    .stats {{ display:flex; flex-wrap:wrap; gap:8px; margin:16px 0; }}
    .stat {{ border:1px solid rgba(36,209,182,.18); border-radius:8px; padding:9px 11px; min-width:92px; background:rgba(10,18,28,.82); }}
    .stat strong {{ display:block; font-size:18px; }}
    .stat span {{ color:var(--muted); font-size:12px; }}
    .actions {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }}
    .actions a {{ border:1px solid var(--line); background:rgba(9,16,24,.82); color:var(--ink); padding:10px 12px; text-decoration:none; font-weight:650; }}
    .actions .primary {{ background:linear-gradient(135deg, var(--accent), var(--accent2)); color:#041015; border-color:transparent; }}
    .overview h3 {{ margin:22px 0 10px; }}
    .overview-hero {{ border-left:4px solid var(--accent2); padding:10px 14px; background:rgba(103,217,255,.06); border-radius:8px; }}
    .hero-subline {{ color:var(--muted); margin-top:8px; }}
    .summary-band {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:10px; margin:14px 0 18px; }}
    .summary-pill {{ border:1px solid rgba(103,217,255,.18); border-radius:8px; padding:12px 13px; background:linear-gradient(180deg, rgba(12,18,28,.92), rgba(8,14,22,.92)); }}
    .summary-pill p {{ margin:8px 0 0; color:#dce8ef; line-height:1.5; font-size:15px; }}
    .card-grid, .paper-grid, .path-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:10px; margin:10px 0 18px; }}
    .mini-card, .paper-card, .path-step {{ border:1px solid var(--line); border-radius:8px; padding:13px; background:linear-gradient(180deg, rgba(13,20,31,.94), rgba(9,16,24,.94)); }}
    .emphasis-card {{ border-color:rgba(103,217,255,.24); box-shadow:inset 0 0 0 1px rgba(103,217,255,.03), 0 10px 24px rgba(0,0,0,.18); }}
    .explanation-card {{ border-color:rgba(36,209,182,.22); }}
    .mini-card h3, .paper-card h3 {{ font-size:17px; margin:5px 0 8px; line-height:1.35; }}
    .mini-card p, .paper-card p, .path-step p {{ color:var(--muted); font-size:14px; line-height:1.5; margin:7px 0 0; }}
    .error-card {{ border:1px solid rgba(255,117,138,.28); border-radius:8px; padding:16px; background:linear-gradient(180deg, rgba(41,14,22,.92), rgba(20,10,18,.94)); box-shadow:inset 0 0 0 1px rgba(255,117,138,.06), 0 14px 26px rgba(0,0,0,.18); }}
    .error-card h3 {{ margin:6px 0 10px; font-size:22px; }}
    .error-card p {{ margin:8px 0 0; color:#f2c7d0; line-height:1.55; }}
    .error-raw {{ margin-top:12px; padding:10px 12px; border-radius:8px; background:rgba(255,255,255,.04); color:#ffdce2 !important; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px !important; }}
    .paper-card a {{ color:var(--ink); text-decoration:none; }}
    .tag {{ display:inline-block; color:var(--accent2); font-size:12px; font-weight:750; text-transform:uppercase; letter-spacing:.06em; }}
    .pill {{ display:inline-block; border:1px solid rgba(103,217,255,.2); border-radius:8px; padding:2px 6px; margin:2px 4px 2px 0; font-size:12px; color:#b6c8d3; background:rgba(103,217,255,.06); }}
    .path-step strong, .path-step span {{ display:block; line-height:1.35; }}
    .path-step span {{ margin-top:4px; }}
    .empty {{ color:var(--muted); border:1px dashed rgba(103,217,255,.18); border-radius:8px; padding:14px; background:rgba(8,14,22,.74); }}
    code, pre {{ background:#0f1823; border-radius:8px; padding:2px 5px; overflow:auto; color:#d3e3ea; }}
    .back {{ color:var(--muted); text-decoration:none; font-weight:650; }}
    .markdown-body {{ line-height:1.65; color:#d9e5ec; }}
    .markdown-body h2, .markdown-body h3, .markdown-body h4 {{ margin:22px 0 10px; font-size:20px; }}
    .markdown-body p {{ margin:12px 0; color:#c7d5de; }}
    .markdown-body ul {{ margin:12px 0 12px 22px; padding:0; }}
    .markdown-body li {{ margin:6px 0; color:#c7d5de; }}
    .markdown-body strong {{ color:#f3f8fb; }}
    .markdown-body table {{ width:100%; border-collapse:collapse; margin:16px 0; font-size:14px; }}
    .markdown-body th, .markdown-body td {{ border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
    .markdown-body th {{ background:rgba(103,217,255,.08); }}
    .markdown-body a {{ color:var(--accent2); text-decoration:none; }}
    .markdown-body a:hover {{ text-decoration:underline; }}
    @keyframes astroFloat {{ 0%,100% {{ transform:translateX(-8%) translateY(0) rotate(-3deg); }} 50% {{ transform:translateX(-8%) translateY(-9px) rotate(2deg); }} }}
    @keyframes starDrift {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-36px); }} }}
    @keyframes signalPulse {{ 0% {{ transform:scale(.82); opacity:.72; }} 100% {{ transform:scale(1.45); opacity:0; }} }}
    @media (max-width: 860px) {{ .community-hero-card {{ grid-template-columns:1fr; }} .community-preview-shell {{ min-height:300px; }} .community-preview-frame {{ width:188%; height:188%; transform:scale(.53); }} }}
    @media (max-width: 720px) {{ h1 {{ font-size:38px; }} .hero {{ padding:24px; }} .search-box {{ flex-direction:column; }} .help-banner {{ grid-template-columns:1fr; }} .example-callout blockquote {{ font-size:20px; }} .run strong {{ font-size:20px; }} .run span {{ font-size:15px; }} .community-hero-copy h3 {{ font-size:28px; }} .section-lead {{ font-size:17px; }} .community-actions {{ flex-direction:column; align-items:stretch; }} .community-callout {{ position:static; max-width:none; margin:10px 12px 0; }} .community-preview-shell {{ min-height:260px; padding-bottom:12px; }} .community-preview-frame {{ width:220%; height:220%; transform:scale(.45); }} .space-loader {{ height:196px; }} .space-loader-meta {{ left:18px; bottom:16px; flex-direction:column; align-items:flex-start; gap:6px; }} .space-loader-destination {{ right:16px; top:22px; font-size:11px; }} .astronaut-wrap {{ width:100px; height:100px; top:40px; left:calc(14px + (100% - 118px) * var(--progress)); }} .space-loader-orbit, .space-loader-progress {{ left:18px; right:18px; top:116px; }} .space-loader-planet {{ width:92px; height:92px; left:10px; bottom:30px; }} }}
  </style>
</head>
<body><main>{body}</main></body>
</html>"""


def content_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".html": "text/html; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".jsonl": "application/x-ndjson; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
    }.get(suffix, "application/octet-stream")


def esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def render_markdown_page(title: str, markdown_text: str) -> str:
    body = _markdown_to_html(markdown_text)
    return page_shell(
        title,
        f"""
        <section>
          <a class="back" href="javascript:history.back()">← Back</a>
          <div class="eyebrow">rendered report</div>
          <h1>{esc(title)}</h1>
          <article class="markdown-body">{body}</article>
        </section>
        """,
    )


def _markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    out: list[str] = []
    in_list = False
    in_table = False
    table_rows: list[list[str]] = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table or not table_rows:
            return
        header = table_rows[0]
        body_rows = table_rows[1:]
        out.append("<table><thead><tr>" + "".join(f"<th>{_inline_md(cell)}</th>" for cell in header) + "</tr></thead>")
        out.append("<tbody>")
        for row in body_rows:
            out.append("<tr>" + "".join(f"<td>{_inline_md(cell)}</td>" for cell in row) + "</tr>")
        out.append("</tbody></table>")
        in_table = False
        table_rows = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            close_list()
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            in_table = True
            table_rows.append(cells)
            continue
        flush_table()
        if not stripped:
            close_list()
            continue
        if stripped.startswith("#"):
            close_list()
            level = min(4, len(stripped) - len(stripped.lstrip("#")))
            content = stripped[level:].strip()
            out.append(f"<h{level + 1}>{_inline_md(content)}</h{level + 1}>")
            continue
        if stripped.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline_md(stripped[2:].strip())}</li>")
            continue
        close_list()
        out.append(f"<p>{_inline_md(stripped)}</p>")

    close_list()
    flush_table()
    return "\n".join(out)


def _inline_md(text: str) -> str:
    text = esc(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', text)
    return text


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def reset_runs_dir(path: str | Path) -> None:
    """Test helper: remove and recreate a runs directory."""
    runs_dir = Path(path)
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
