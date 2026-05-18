"""HTTP frontend for v0.7-frozen runs.

Importable module. Two entry points:
  - ``app``: FastAPI ASGI app (mount it under any server)
  - ``serve(...)``: blocking helper that starts uvicorn

The app reads run directories under ``runs_root`` (default
``artifacts/runs``) and serves them via:

  GET /                  — home page with corpus picker + search box
  GET /query             — JSON: cached-hypothesis topic match
  GET /run/<id>          — landing page pre-pointed at one run
  GET /api/runs          — JSON: discovered runs + counts

No LLM calls. 0.5-5 s per query depending on cache size.
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import markdown as md
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

# Locate scripts/aigraph_query.py without making the script importable as
# a package — we just need the `query` symbol.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
from aigraph_query import query as _run_query  # noqa: E402


# Default location; can be overridden at app-creation time.
DEFAULT_RUNS_ROOT = _REPO / "artifacts" / "runs"


def _discover_runs(runs_root: Path) -> list[dict]:
    """Find run directories with hypotheses_scored.jsonl (the
    minimum aigraph_query needs)."""
    out = []
    if not runs_root.exists():
        return out
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        scored = d / "hypotheses_scored.jsonl"
        if not scored.exists():
            continue
        try:
            n = sum(1 for _ in scored.open())
        except Exception:
            n = 0
        papers = d / "papers.jsonl"
        n_papers = sum(1 for _ in papers.open()) if papers.exists() else 0
        out.append({"id": d.name, "n_hypotheses": n, "n_papers": n_papers})
    return out


def create_app(runs_root: Path | None = None) -> FastAPI:
    runs_root = Path(runs_root) if runs_root else DEFAULT_RUNS_ROOT
    app = FastAPI(title="aigraph v0.7-frozen explorer")

    @app.get("/api/runs")
    def api_runs():
        return _discover_runs(runs_root)

    @app.get("/", response_class=HTMLResponse)
    def home():
        return _render_home(runs_root, preselected_run=None, initial_topic=None)

    @app.get("/run/{run_id}", response_class=HTMLResponse)
    def run_page(run_id: str):
        # Validate run_id (path-traversal safety)
        run_dir = (runs_root / run_id).resolve()
        if not str(run_dir).startswith(str(runs_root.resolve())):
            raise HTTPException(400, "invalid run_id")
        if not (run_dir / "hypotheses_scored.jsonl").exists():
            raise HTTPException(404, f"run not found: {run_id}")
        # Pre-render selected_hypotheses.md if present (the "default" view)
        sel = run_dir / "selected_hypotheses.md"
        initial_html = (
            md.markdown(sel.read_text(), extensions=["tables", "fenced_code"])
            if sel.exists()
            else None
        )
        return _render_home(runs_root, preselected_run=run_id, initial_html=initial_html)

    @app.get("/query")
    def query_endpoint(
        topic: str = Query(..., min_length=1, max_length=500),
        run: str = Query(...),
        k: int = Query(5, ge=1, le=20),
    ):
        run_dir = (runs_root / run).resolve()
        if not str(run_dir).startswith(str(runs_root.resolve())):
            return JSONResponse({"error": f"invalid run: {run}"}, status_code=400)
        if not run_dir.exists():
            return JSONResponse({"error": f"unknown run: {run}"}, status_code=404)
        try:
            stderr_buf = StringIO()
            t0 = time.time()
            with redirect_stderr(stderr_buf):
                output_md, stats = _run_query(
                    run_dir=run_dir,
                    topic=topic,
                    k=k,
                    max_hypotheses=30,
                    mmr_lambda=0.7,
                    min_anomalies=1,
                )
            html = md.markdown(output_md, extensions=["tables", "fenced_code"])
            return JSONResponse(
                {
                    "html": html,
                    "stats": {**stats, "wall_seconds": stats.get("wall_seconds") or time.time() - t0},
                }
            )
        except Exception as exc:
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app


def _render_home(
    runs_root: Path,
    *,
    preselected_run: str | None = None,
    initial_topic: str | None = None,
    initial_html: str | None = None,
) -> str:
    runs = _discover_runs(runs_root)
    if not runs:
        runs_options = '<option value="" disabled>(no runs found)</option>'
    else:
        runs_options = "".join(
            f'<option value="{r["id"]}"{" selected" if r["id"] == preselected_run else ""}>'
            f'{r["id"]} ({r["n_papers"]}p, {r["n_hypotheses"]}h)</option>'
            for r in runs
        )
    initial_topic_val = f' value="{initial_topic}"' if initial_topic else ""
    result_placeholder = (
        initial_html
        if initial_html
        else '<em style="color:#6e7681">Enter a topic and click Search, '
        'or open a /run/&lt;id&gt; URL for that run\'s default top-K.</em>'
    )
    return _PAGE_TEMPLATE.format(
        runs_options=runs_options,
        initial_topic_val=initial_topic_val,
        result_placeholder=result_placeholder,
    )


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>aigraph v0.7-frozen explorer</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Microsoft YaHei", sans-serif;
    background: #0f1419; color: #e4e7eb;
    max-width: 920px; margin-left: auto; margin-right: auto;
  }}
  h1 {{ font-size: 20px; margin: 0 0 4px 0; }}
  .sub {{ color: #8b949e; font-size: 13px; margin-bottom: 24px; }}
  form {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; align-items: center; }}
  input[type=text] {{
    flex: 1; min-width: 320px; padding: 10px 12px;
    background: #161b22; border: 1px solid #30363d; color: #e4e7eb;
    border-radius: 6px; font-size: 14px;
  }}
  input[type=text]:focus {{ outline: 2px solid #58a6ff; outline-offset: -2px; }}
  select, input[type=number] {{
    padding: 10px 8px; background: #161b22; border: 1px solid #30363d;
    color: #e4e7eb; border-radius: 6px; font-size: 14px;
  }}
  input[type=number] {{ width: 64px; }}
  button {{
    padding: 10px 18px; background: #238636; color: white; border: 0;
    border-radius: 6px; font-size: 14px; cursor: pointer;
  }}
  button:hover {{ background: #2ea043; }}
  button:disabled {{ background: #424a53; cursor: wait; }}
  .stats {{
    font-size: 12px; color: #8b949e; padding: 8px 12px;
    background: #161b22; border-radius: 6px; margin-bottom: 16px;
    display: none;
  }}
  .stats.visible {{ display: block; }}
  #result {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 20px 24px; font-size: 14px; line-height: 1.6;
  }}
  #result h1, #result h2, #result h3 {{ color: #c9d1d9; margin-top: 1.5em; margin-bottom: 0.4em; }}
  #result h1 {{ font-size: 20px; }}
  #result h2 {{ font-size: 17px; border-bottom: 1px solid #30363d; padding-bottom: 4px; }}
  #result h3 {{ font-size: 15px; color: #79c0ff; }}
  #result code {{
    background: #1c2128; padding: 2px 6px; border-radius: 4px;
    font-size: 12.5px; color: #ffa657;
  }}
  #result strong {{ color: #f0883e; }}
  #result table {{ border-collapse: collapse; margin: 8px 0; font-size: 12px; }}
  #result th, #result td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: left; }}
  #result ul {{ padding-left: 1.5em; }}
  #result li {{ margin: 4px 0; }}
  .topic-suggestions {{ margin-top: 8px; font-size: 12px; color: #8b949e; }}
  .topic-suggestions span {{
    display: inline-block; margin-right: 8px; cursor: pointer;
    text-decoration: underline; color: #79c0ff;
  }}
</style>
</head>
<body>
<h1>aigraph v0.7-frozen explorer</h1>
<div class="sub">
  Service-mode browsing of cached hypotheses. No LLM calls at query time.
</div>

<form id="f">
  <input type="text" id="topic" name="topic" placeholder='Topic, e.g. "test-time scaling chain-of-thought"' autofocus required{initial_topic_val}>
  <select id="run" name="run">{runs_options}</select>
  <label style="font-size:12px;color:#8b949e">k <input type="number" id="k" name="k" value="5" min="1" max="20"></label>
  <button type="submit">Search</button>
</form>
<div class="topic-suggestions">Try:
  <span onclick="setTopic('chain-of-thought self-consistency')">CoT self-consistency</span>
  <span onclick="setTopic('test-time scaling inference compute')">test-time compute</span>
  <span onclick="setTopic('multi-step reasoning verifier')">verifier</span>
  <span onclick="setTopic('agentic workflow tool use')">agent tool-use</span>
  <span onclick="setTopic('process reward outcome reward')">PRM/ORM</span>
</div>

<div id="stats" class="stats"></div>
<div id="result">{result_placeholder}</div>

<script>
function setTopic(t) {{
  document.getElementById('topic').value = t;
  document.getElementById('f').requestSubmit();
}}
document.getElementById('f').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const topic = document.getElementById('topic').value.trim();
  const run = document.getElementById('run').value;
  const k = document.getElementById('k').value;
  const btn = document.querySelector('button');
  const result = document.getElementById('result');
  const stats = document.getElementById('stats');
  btn.disabled = true; btn.textContent = 'Searching…';
  result.innerHTML = '<em style="color:#6e7681">Querying…</em>';
  stats.classList.remove('visible');
  try {{
    const resp = await fetch(`/query?topic=${{encodeURIComponent(topic)}}&run=${{run}}&k=${{k}}`);
    const data = await resp.json();
    if (data.error) {{
      result.innerHTML = `<div style="color:#f85149">Error: ${{data.error}}</div>`;
    }} else {{
      result.innerHTML = data.html;
      stats.textContent = `Matched ${{data.stats.n_matched}}/${{data.stats.n_hypotheses_total}} hypotheses across ${{data.stats.n_candidates}} candidates → selected ${{data.stats.n_selected}}. Wall ${{data.stats.wall_seconds.toFixed(2)}}s. LLM calls: ${{data.stats.llm_calls}}.`;
      stats.classList.add('visible');
    }}
  }} catch (err) {{
    result.innerHTML = `<div style="color:#f85149">Request failed: ${{err}}</div>`;
  }} finally {{
    btn.disabled = false; btn.textContent = 'Search';
  }}
}});
</script>
</body>
</html>"""


def serve(host: str = "127.0.0.1", port: int = 8000, runs_root: Path | None = None) -> None:
    """Blocking helper: start uvicorn with the FastAPI app."""
    import uvicorn

    app = create_app(runs_root)
    print(f"aigraph web explorer on http://{host}:{port}")
    print(f"  runs_root: {Path(runs_root) if runs_root else DEFAULT_RUNS_ROOT}")
    discovered = _discover_runs(Path(runs_root) if runs_root else DEFAULT_RUNS_ROOT)
    print(f"  {len(discovered)} run(s): {', '.join(r['id'] for r in discovered)}")
    uvicorn.run(app, host=host, port=port, log_level="info")
