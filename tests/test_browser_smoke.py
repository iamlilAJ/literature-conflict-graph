import json
import re
import socket
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from aigraph.server import SearchService, make_handler, write_status
from aigraph.visualize import render_visualization


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _build_sample_run(runs_dir: Path, run_id: str = "20260420-190000-abc123") -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        run_dir / "papers.jsonl",
        [
            {
                "paper_id": "openalex:W1",
                "title": "Paper One",
                "year": 2024,
                "venue": "ACL",
                "paper_role": "survey",
                "paper_role_score": 0.92,
                "paper_role_signals": ["title:survey"],
            }
        ],
    )
    _write_jsonl(
        run_dir / "claims.jsonl",
        [
            {
                "claim_id": "c001",
                "paper_id": "openalex:W1",
                "claim_text": "RAG improves factual QA.",
                "method": "RAG",
                "task": "factual QA",
                "direction": "positive",
            }
        ],
    )
    _write_jsonl(
        run_dir / "anomalies.jsonl",
        [
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
        ],
    )
    _write_jsonl(
        run_dir / "hypotheses.jsonl",
        [
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
        ],
    )
    _write_jsonl(
        run_dir / "insights.jsonl",
        [
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
        ],
    )
    (run_dir / "overview.json").write_text(
        json.dumps(
            {
                "headline": "RAG gains are benchmark-sensitive.",
                "hero_line": {"line": "RAG helps on some QA settings, but the effect is unstable."},
                "why_this_matters": {"line": "Weak baselines can distort the apparent win.", "next_step": "Compare stronger baselines."},
                "top_conflicts": [{"type": "benchmark_inconsistency", "question": "When does RAG help?"}],
                "best_explanation_lines": [{"line": "Retrieval quality moderates the effect.", "source_type": "hypothesis"}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "graph.json").write_text(
        json.dumps(
            {
                "directed": True,
                "multigraph": True,
                "graph": {},
                "nodes": [
                    {"id": "Paper:openalex:W1", "node_type": "Paper", "paper_id": "openalex:W1", "paper_role": "survey"},
                    {"id": "Claim:c001", "node_type": "Claim", "claim_id": "c001", "claim_text": "RAG improves factual QA."},
                    {"id": "Method:rag", "node_type": "Method", "name": "RAG"},
                    {"id": "Role:survey", "node_type": "Role", "name": "Survey paper", "description": "Maps the field, taxonomy, or prior landscape."},
                ],
                "edges": [
                    {"source": "Paper:openalex:W1", "target": "Claim:c001", "edge_type": "makes", "key": 0},
                    {"source": "Claim:c001", "target": "Method:rag", "edge_type": "uses", "key": 0},
                    {"source": "Paper:openalex:W1", "target": "Role:survey", "edge_type": "has_role", "key": 0},
                ],
            }
        ),
        encoding="utf-8",
    )
    render_visualization(run_dir, run_dir / "index.html")
    write_status(
        run_dir,
        status="done",
        stage="complete",
        progress=1.0,
        run_id=run_id,
        topic="rag factual qa",
        papers=1,
        claims=1,
        anomalies=1,
        hypotheses=1,
        insights=1,
        graph_url=f"/runs/{run_id}/index.html",
        overview={"headline": "RAG gains are benchmark-sensitive."},
    )
    return run_dir


@contextmanager
def _running_server(runs_dir: Path):
    service = SearchService(runs_dir)
    handler_cls = make_handler(service)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()
    server = ThreadingHTTPServer((host, port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_explorer_browser_smoke(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = _build_sample_run(runs_dir)
    captured_payloads: list[dict] = []

    with _running_server(runs_dir) as base_url, playwright.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="chrome")
        except Exception as exc:  # pragma: no cover - environment-specific guard
            try:
                browser = pw.chromium.launch()
            except Exception as inner_exc:
                if "Executable doesn't exist" in str(inner_exc):
                    pytest.skip(
                        "No Playwright-managed browser is installed and Chrome launch failed. "
                        "Run `python -m playwright install chromium`."
                    )
                raise inner_exc from exc
        page = browser.new_page()

        def handle_chat(route):
            payload = json.loads(route.request.post_data or "{}")
            captured_payloads.append(payload)
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "answer": "The graph says the effect depends on benchmark setup.",
                        "citations": [{"type": "claim", "id": "c001", "title": "RAG improves factual QA."}],
                    }
                ),
            )

        page.route("**/api/graph-chat", handle_chat)
        page.goto(f"{base_url}/runs/{run_dir.name}/index.html", wait_until="domcontentloaded")
        page.wait_for_selector("#graph-chat-thread")
        page.wait_for_selector('button.item[data-kind="anomaly"]')

        page.click('button.item[data-kind="anomaly"]')
        expect = playwright.expect
        expect(page.locator("#chat-selection-state")).to_contain_text("Conflict #1")

        page.click('.mini-nav-btn[data-target="hypotheses-section"]')
        expect(page.locator("#hypotheses-section")).to_have_class(re.compile(r".*\bopen\b.*"))

        page.click('button.chat-chip')
        expect(page.locator("#graph-chat-thread")).to_contain_text("Why is this a conflict?")
        expect(page.locator("#graph-chat-thread")).to_contain_text("The graph says the effect depends on benchmark setup.")
        assert captured_payloads[0]["selection"]["kind"] == "conflict"

        page.click("#graph-chat-clear")
        expect(page.locator("#chat-selection-state")).to_contain_text("whole run")
        page.fill("#graph-chat-input", "Give me the short version.")
        page.press("#graph-chat-input", "Enter")
        expect(page.locator("#graph-chat-thread")).to_contain_text("Give me the short version.")
        assert captured_payloads[-1]["selection"] == {}
        assert len(captured_payloads[-1]["history"]) <= 6

        browser.close()
