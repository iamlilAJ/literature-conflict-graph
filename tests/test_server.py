import json
from http import HTTPStatus
from pathlib import Path

import pytest

from aigraph.server import (
    MAX_STORED_HYPOTHESES,
    PRIMARY_PUBLIC_HOST,
    SearchRequest,
    SearchService,
    _selection_context,
    _trim_graph_chat_history,
    ensure_overview,
    is_valid_run_id,
    normalize_arxiv_query,
    prune_hypotheses,
    public_redirect_url,
    render_home_page,
    render_curated_demo_cards,
    render_markdown_page,
    render_result_page,
    safe_child_path,
    safe_run_dir,
    write_status,
)
from aigraph.models import Anomaly, Claim, Hypothesis, Paper


def test_normalize_arxiv_query_expands_common_natural_language_terms():
    query = normalize_arxiv_query("llm finance time series")
    assert 'all:"large language models"' in query
    assert "all:finance" in query
    assert 'all:"time series"' in query
    assert " AND " in query


def test_normalize_arxiv_query_keeps_explicit_arxiv_syntax():
    raw = 'all:"large language models" AND all:finance'
    assert normalize_arxiv_query(raw) == raw


def test_normalize_arxiv_query_drops_generic_stopwords():
    # Generic words ("based", "for", ...) must NOT become AND constraints — they
    # silently shrink the arXiv pool (reported: 50 requested, 13 returned because
    # "memory based policy distillation" became a 4-way AND including "based").
    q = normalize_arxiv_query("memory based policy distillation")
    assert q == "all:memory AND all:policy AND all:distillation"
    assert "all:based" not in q
    assert "all:for" not in normalize_arxiv_query("agentic memory for agents")


def test_normalize_arxiv_query_dedupes_overlapping_phrases():
    # "large language models" and "language models" map to the same expansion;
    # it must appear exactly once, not twice in the AND chain.
    q = normalize_arxiv_query("large language models reasoning")
    assert q.count('all:"large language models"') == 1
    assert "all:reasoning" in q


# -- resilient multi-source retrieval (#46/#47/#48) --------------------------

def _mk_paper(pid, title, year=2024):
    from aigraph.models import Paper
    return Paper(paper_id=pid, title=title, year=year, venue="x")


def _mk_request(**over):
    from pathlib import Path
    from aigraph.server import SearchRequest
    base = dict(
        run_id="r", topic="t", retrieval_topic="memory policy distillation",
        retrieval_variants=["policy distillation"], arxiv_query="all:memory AND all:policy",
        limit=10, insight_generator="llm", source="arxiv", strategy="balanced",
        citation_weight=0.45, min_relevance=0.3, run_dir=Path("/tmp"),
    )
    base.update(over)
    return SearchRequest(**base)


def test_retrieve_corpus_unions_and_dedups_across_sources(monkeypatch):
    import aigraph.server as s
    monkeypatch.setattr(s, "fetch_arxiv_papers",
                        lambda *a, **k: [_mk_paper("arxiv:1", "Shared Title"), _mk_paper("arxiv:2", "A only")])
    monkeypatch.setattr(s, "fetch_openalex_papers",
                        lambda *a, **k: [_mk_paper("openalex:W9", "Shared Title"), _mk_paper("openalex:W8", "B only")])
    papers, rep = s._retrieve_corpus(_mk_request(source="union"), lambda **kw: None, {})
    assert sorted(p.title for p in papers) == ["A only", "B only", "Shared Title"]  # dup title collapsed
    assert {a["source"] for a in rep["retrieval_attempts"]} == {"arxiv", "openalex"}  # union hits both


def test_retrieve_corpus_arxiv_tiers_to_variants_then_falls_back(monkeypatch):
    import aigraph.server as s
    arxiv_qs = []
    def fake_arxiv(query, **k):
        arxiv_qs.append(query)
        return [_mk_paper(f"arxiv:{len(arxiv_qs)}", f"p{len(arxiv_qs)}")]  # only 1 each -> below floor
    monkeypatch.setattr(s, "fetch_arxiv_papers", fake_arxiv)
    monkeypatch.setattr(s, "fetch_openalex_papers",
                        lambda *a, **k: [_mk_paper(f"openalex:{i}", f"oa{i}") for i in range(10)])
    papers, rep = s._retrieve_corpus(
        _mk_request(source="arxiv", arxiv_query="all:primary", retrieval_variants=["alpha", "beta"]),
        lambda **kw: None, {})
    assert "all:primary" in arxiv_qs and len(arxiv_qs) >= 2  # tiered to the broader variant queries
    assert any(a["source"] == "openalex" for a in rep["retrieval_attempts"])  # low recall -> openalex fallback
    assert len(papers) >= 8 and rep["retrieval_quality"] == "ok"


def test_retrieve_corpus_skips_failing_source_without_killing_run(monkeypatch):
    import aigraph.server as s
    def boom(*a, **k):
        raise TimeoutError("slow provider")
    monkeypatch.setattr(s, "fetch_openalex_papers", boom)
    monkeypatch.setattr(s, "fetch_arxiv_papers",
                        lambda *a, **k: [_mk_paper(f"arxiv:{i}", f"t{i}") for i in range(10)])
    papers, rep = s._retrieve_corpus(_mk_request(source="openalex"), lambda **kw: None, {})
    oa = [a for a in rep["retrieval_attempts"] if a["source"] == "openalex"][0]
    assert oa["ok"] is False  # failure recorded, not raised
    assert len(papers) == 10  # arxiv fallback supplied the corpus


def test_retrieve_corpus_quality_flags(monkeypatch):
    import aigraph.server as s
    monkeypatch.setattr(s, "fetch_openalex_papers", lambda *a, **k: [])
    monkeypatch.setattr(s, "fetch_arxiv_papers", lambda *a, **k: [])
    _, rep = s._retrieve_corpus(_mk_request(source="arxiv", retrieval_variants=[]), lambda **kw: None, {})
    assert rep["retrieval_quality"] == "empty"
    monkeypatch.setattr(s, "fetch_arxiv_papers",
                        lambda *a, **k: [_mk_paper(f"a{i}", f"t{i}") for i in range(3)])
    _, rep = s._retrieve_corpus(_mk_request(source="arxiv", retrieval_variants=[]), lambda **kw: None, {})
    assert rep["retrieval_quality"] == "weak"  # 3 < recall floor (8)


def test_public_redirect_url_redirects_apex_domain_to_graph_subdomain():
    assert public_redirect_url("paper-universe.uk", "/") == f"https://{PRIMARY_PUBLIC_HOST}/"
    assert public_redirect_url("www.paper-universe.uk", "/search/demo") == f"https://{PRIMARY_PUBLIC_HOST}/search/demo"
    assert public_redirect_url("graph.paper-universe.uk", "/") is None


def test_run_id_and_path_safety(tmp_path):
    run_id = "20260416-120000-abcdef"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    assert is_valid_run_id(run_id)
    assert safe_run_dir(tmp_path, run_id) == run_dir.resolve()
    assert safe_child_path(run_dir, "index.html") == (run_dir / "index.html").resolve()
    with pytest.raises(ValueError):
        safe_run_dir(tmp_path, "../bad")
    with pytest.raises(ValueError):
        safe_child_path(run_dir, "../secret.txt")


def test_write_status_merges_updates(tmp_path):
    run_dir = tmp_path / "20260416-120000-abcdef"
    write_status(run_dir, status="queued", stage="queued", progress=0.0, run_id=run_dir.name)
    write_status(run_dir, status="running", stage="fetching", progress=0.2, papers=3)
    data = json.loads((run_dir / "status.json").read_text())
    assert data["status"] == "running"
    assert data["stage"] == "fetching"
    assert data["papers"] == 3
    assert data["run_id"] == run_dir.name


def test_html_pages_contain_search_and_polling_links():
    home = render_home_page([], {})
    assert "Living Graph" in home
    assert "Community Graph" in home
    assert "Network Pulse" in home
    assert "HELP" in home
    assert "How It Works" in home
    assert "What a good output looks like" in home
    assert "Quick Starts" in home
    assert "Literature Conflict Search" in home
    assert "/api/search" in home
    assert "Paper strategy" in home
    assert "Citation weight" in home
    assert "OpenAlex citations" in home
    assert "Featured demos" in home
    assert "Recent maps" in home
    assert "This is what you get." in home
    assert "community-preview-fallback" in home
    assert 'href="#community-graph"' in home
    assert 'href="#featured-demos"' in home
    assert "requester IPs" not in home
    assert "Generate Map" in home
    assert "Gains attributed to the method are inflated when compared against weak baselines" in home
    assert "Insight mode" not in home
    result = render_result_page(
        {
            "run_id": "20260416-120000-abcdef",
            "topic": "llm finance",
            "arxiv_query": "all:finance",
            "source": "arxiv",
            "strategy": "balanced",
            "message": "Queued",
        }
    )
    assert "/api/runs/" in result
    assert "Open Graph" in result
    assert "Open Report" in result
    assert "renderOverview" in result
    assert "Representative Papers" in result
    assert "Key Explanations" in result
    assert "Why this matters" in result
    assert "Explore While You Wait" in result
    assert "space-loader" in result
    assert "astronaut-svg" in result
    assert "mapped insight" in result
    assert "progress-label" in result
    assert "renderErrorState" in result


def test_render_markdown_page_renders_basic_markdown():
    html = render_markdown_page(
        "selected hypotheses",
        "# Title\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
    )
    assert "<article class=\"markdown-body\">" in html
    assert "<h2>Title</h2>" in html
    assert "<li>one</li>" in html
    assert "<table>" in html


def test_home_page_shows_living_graph_card():
    home = render_home_page([], {"status": {"graph_url": "/community/index.html", "runs": 3, "papers": 12, "claims": 44, "nodes": 90, "edges": 140}})
    assert "/community/index.html" in home
    assert "3 runs" in home
    assert "community-preview-frame" in home
    assert "Open Community Graph" in home


def test_home_page_shows_library_and_pulse_blocks():
    home = render_home_page(
        [{
            "run_id": "20260417-120000-abcdef",
            "topic": "large language models finance time series forecasting",
            "status": "done",
            "papers": 5,
            "claims": 10,
            "overview": {"hero_line": {"line": "Weak baselines can make the method look stronger than it is."}},
        }],
        {
            "status": {"graph_url": "/community/index.html", "runs": 1},
            "newest_runs": [{"run_id": "20260417-120000-abcdef", "topic": "large language models finance time series forecasting", "claims": 10, "anomalies": 3}],
            "hottest_topics": [{"topic": "rag hallucination medical qa", "count": 4}],
            "biggest_conflicts": [{"question": "When does RAG help?", "claim_count": 6}],
            "newest_bridges": [{"title": "Finance and time series share regime shift"}],
        },
    )
    assert "Growing topics" in home
    assert "Finance + Time Series" in home
    assert "Quick Starts" in home
    assert "Featured demos" in home
    assert "5 papers" in home
    assert "Weak baselines can make the method look stronger than it is." in home
    assert "seeded topic · fetch on demand" in home


def test_curated_demo_cards_skip_empty_counts_for_finished_demo():
    html = render_curated_demo_cards(
        [
            {
                "run_id": "20260417-170322-29ff93",
                "topic": "llm finance stock movement prediction time series forecasting",
                "status": "done",
                "papers": 0,
                "claims": 0,
                "overview": {},
            }
        ]
    )
    assert "Finance + Stock Prediction" in html
    assert "ready demo" in html
    assert "0 papers · 0 claims" not in html


def test_service_submit_uses_fake_runner_without_network(tmp_path):
    calls: list[SearchRequest] = []

    def fake_runner(req: SearchRequest, status):
        calls.append(req)
        (req.run_dir / "index.html").write_text("<html>graph</html>")
        (req.run_dir / "selected_hypotheses.md").write_text("# report")
        status(
            status="done",
            stage="complete",
            progress=1.0,
            message="fake done",
            run_id=req.run_id,
            topic=req.topic,
            arxiv_query=req.arxiv_query,
            source_query=req.arxiv_query,
            limit=req.limit,
            source=req.source,
            strategy=req.strategy,
            citation_weight=req.citation_weight,
            min_relevance=req.min_relevance,
            overview={"headline": "fake overview"},
            overview_url=f"/runs/{req.run_id}/overview.json",
            graph_url=f"/runs/{req.run_id}/index.html",
            report_url=f"/runs/{req.run_id}/selected_hypotheses.md",
        )

    service = SearchService(tmp_path, runner=fake_runner)
    req = service.submit(
        "llm finance",
        limit=10,
        insight_generator="template",
        source="openalex",
        strategy="high-impact",
        citation_weight=0.65,
        min_relevance=0.45,
        client_ip="203.0.113.7",
        user_agent="pytest browser",
    )
    service.queue.join()
    assert calls == [req]
    assert req.source == "openalex"
    assert req.strategy == "high-impact"
    assert req.citation_weight == 0.65
    assert req.min_relevance == 0.45
    status = service.read_status(req.run_id)
    assert status["status"] == "done"
    assert status["source"] == "openalex"
    assert status["strategy"] == "high-impact"
    assert status["citation_weight"] == 0.65
    assert status["min_relevance"] == 0.45
    assert status["overview"]["headline"] == "fake overview"
    assert status["graph_url"].endswith("/index.html")
    analytics = tmp_path / "_analytics" / "requests.jsonl"
    row = json.loads(analytics.read_text().splitlines()[0])
    assert row["client_ip"] == "203.0.113.7"
    assert row["topic"] == "llm finance"


def test_service_downgrades_high_impact_for_arxiv(tmp_path):
    service = SearchService(tmp_path, runner=lambda req, status: status(status="done", stage="complete"))
    req = service.submit("llm finance", limit=10, source="arxiv", strategy="high-impact")
    service.queue.join()
    assert req.strategy == "balanced"


def test_service_formats_rate_limit_errors_without_traceback(tmp_path):
    class Fake429(Exception):
        def __init__(self):
            self.response = type("Resp", (), {"status_code": 429})()
            super().__init__("rate limited")

    def failing_runner(req: SearchRequest, status):
        raise Fake429()

    service = SearchService(tmp_path, runner=failing_runner)
    req = service.submit("llm finance", limit=10, source="openalex", strategy="balanced")
    service.queue.join()
    status = service.read_status(req.run_id)
    assert status["status"] == "error"
    assert status["error_kind"] == "rate_limit"
    assert "traceback" not in status["error"].lower()
    assert "429" in status["error"]
    assert "Try again in a minute" in status["error_recovery"]


def test_ensure_overview_backfills_old_done_run(tmp_path):
    run_dir = tmp_path / "20260417-100000-abcdef"
    run_dir.mkdir()
    (run_dir / "papers.jsonl").write_text(
        '{"paper_id":"p1","title":"Survey of RAG","year":2024,"venue":"ACL","selection_score":0.8,"selection_reason":"survey signal"}\n',
        encoding="utf-8",
    )
    (run_dir / "claims.jsonl").write_text(
        '{"claim_id":"c001","paper_id":"p1","claim_text":"RAG helps QA.","direction":"positive"}\n',
        encoding="utf-8",
    )
    (run_dir / "anomalies.jsonl").write_text(
        '{"anomaly_id":"a001","type":"evidence_gap","central_question":"Where does RAG help?","claim_ids":["c001"]}\n',
        encoding="utf-8",
    )
    (run_dir / "insights.jsonl").write_text("", encoding="utf-8")
    (run_dir / "hypotheses.jsonl").write_text("", encoding="utf-8")
    overview = ensure_overview(run_dir, {"topic": "rag"})
    assert overview is not None
    assert overview["top_papers"][0]["title"] == "Survey of RAG"
    assert (run_dir / "overview.json").exists()


def test_prune_hypotheses_caps_large_candidate_set():
    anomalies = [
        Anomaly(anomaly_id="a001", type="benchmark_inconsistency", central_question="When does RAG help?", claim_ids=["c001"], positive_claims=["c001"], negative_claims=["c002"]),
        Anomaly(anomaly_id="a002", type="benchmark_inconsistency", central_question="When does prompting help?", claim_ids=["c003"], positive_claims=["c003"], negative_claims=["c004"]),
    ]
    claims = [
        Claim(claim_id="c001", paper_id="p1", claim_text="RAG helps", direction="positive"),
        Claim(claim_id="c002", paper_id="p2", claim_text="RAG hurts", direction="negative"),
        Claim(claim_id="c003", paper_id="p3", claim_text="Prompting helps", direction="positive"),
        Claim(claim_id="c004", paper_id="p4", claim_text="Prompting hurts", direction="negative"),
    ]
    hypotheses = []
    for idx in range(40):
        anomaly_id = "a001" if idx % 2 == 0 else "a002"
        hypotheses.append(
            Hypothesis(
                hypothesis_id=f"h{idx:03d}",
                anomaly_id=anomaly_id,
                hypothesis=f"Concrete hypothesis {idx} about {'RAG' if anomaly_id == 'a001' else 'prompting'} baseline mismatch.",
                mechanism="Baseline mismatch changes the apparent effect.",
                explains_claims=["c001"] if anomaly_id == "a001" else ["c003"],
                predictions=["prediction one", "prediction two"],
                minimal_test="Run a controlled ablation.",
            )
        )
    pruned = prune_hypotheses(hypotheses, anomalies, claims)
    assert len(pruned) <= MAX_STORED_HYPOTHESES
    assert {h.anomaly_id for h in pruned} == {"a001", "a002"}


def test_trim_graph_chat_history_keeps_recent_user_and_assistant_turns():
    history = [
        {"role": "system", "content": "ignore me"},
        *[
            {"role": "user" if i % 2 == 0 else "assistant", "content": f" message {i} "}
            for i in range(10)
        ],
    ]
    trimmed = _trim_graph_chat_history(history)
    assert len(trimmed) == 6
    assert trimmed[0]["content"] == "message 4"
    assert trimmed[-1]["content"] == "message 9"


def test_selection_context_accepts_conflict_alias_for_anomaly():
    anomaly = Anomaly(
        anomaly_id="a001",
        type="benchmark_inconsistency",
        central_question="When does RAG help?",
        claim_ids=["c001"],
        positive_claims=["c001"],
        negative_claims=[],
        shared_entities={"method": "RAG"},
        local_graph_nodes=[],
        local_graph_edges=[],
    )
    claim = Claim(claim_id="c001", paper_id="p1", claim_text="RAG improves factual QA.", direction="positive")
    paper = Paper(paper_id="p1", title="Paper One", year=2024, venue="ACL")
    selection, refs = _selection_context(
        {"kind": "conflict", "id": "a001"},
        graph={"nodes": [], "edges": []},
        papers_by_id={"p1": paper},
        claims_by_id={"c001": claim},
        anomalies_by_id={"a001": anomaly},
        hypotheses_by_id={},
        insights_by_id={},
    )
    assert selection["kind"] == "anomaly"
    assert selection["id"] == "a001"
    assert refs[0]["id"] == "c001"
