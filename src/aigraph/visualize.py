"""Static HTML visualization for aigraph output directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_visualization(input_dir: str | Path, output: str | Path) -> Path:
    """Render a self-contained HTML graph explorer.

    Args:
        input_dir: Directory containing papers/claims/graph/anomalies/hypotheses files.
        output: HTML file to write.

    Returns:
        The output path.
    """
    input_path = Path(input_dir)
    output_path = Path(output)
    payload = _load_payload(input_path)
    html = _render_html(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _load_payload(input_dir: Path) -> dict[str, Any]:
    graph = _read_json(input_dir / "graph.json", default={"nodes": [], "edges": []})
    overview = _read_json(input_dir / "overview.json", default={})
    papers = [_augment_paper_links(p) for p in _read_jsonl(input_dir / "papers.jsonl")]
    claims = _read_jsonl(input_dir / "claims.jsonl")
    anomalies = _read_jsonl(input_dir / "anomalies.jsonl")
    hypotheses = _read_jsonl(input_dir / "hypotheses.jsonl")
    insights = _read_jsonl(input_dir / "insights.jsonl")
    graph_mode = "community" if input_dir.name == "_community" else "run"

    return {
        "run_id": "" if graph_mode == "community" else input_dir.name,
        "graph_mode": graph_mode,
        "overview": overview,
        "summary": {
            "papers": len(papers),
            "claims": len(claims),
            "nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
            "anomalies": len(anomalies),
            "hypotheses": len(hypotheses),
            "insights": len(insights),
        },
        "graph": graph,
        "papers": papers,
        "claims": claims,
        "anomalies": anomalies,
        "hypotheses": hypotheses,
        "insights": insights,
    }


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _augment_paper_links(paper: dict[str, Any]) -> dict[str, Any]:
    out = dict(paper)
    if not out.get("url"):
        paper_id = str(out.get("paper_id", ""))
        if paper_id.startswith("openalex:"):
            suffix = paper_id.split(":", 1)[1]
            if suffix:
                out["url"] = f"https://openalex.org/{suffix}"
    if not out.get("doi_url") and out.get("doi"):
        doi = str(out["doi"]).removeprefix("https://doi.org/")
        out["doi_url"] = f"https://doi.org/{doi}"
    return out


def _json_script(payload: dict[str, Any]) -> str:
    # Avoid accidentally closing the JSON script tag.
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def _render_html(payload: dict[str, Any]) -> str:
    data = _json_script(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>literature conflict explorer</title>
  <style>
    :root {{
      --paper: #94a3b8;
      --claim: #60a5fa;
      --method: #24d1b6;
      --task: #f59e0b;
      --dataset: #f472b6;
      --metric: #fb7185;
      --baseline: #a78bfa;
      --setting: #22d3ee;
      --edge: rgba(171, 190, 204, 0.42);
      --ink: #e7eef5;
      --muted: #8da0ad;
      --line: rgba(120, 152, 171, 0.24);
      --panel: rgba(10, 18, 28, 0.88);
      --bg: #081019;
      --highlight: #ff7b72;
      --soft: rgba(103, 217, 255, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(36,209,182,.16), transparent 26%),
        radial-gradient(circle at top right, rgba(103,217,255,.14), transparent 22%),
        linear-gradient(180deg, #0a121c 0%, #081019 52%, #071018 100%);
      letter-spacing: 0;
    }}
    body:before {{
      content:"";
      position:fixed;
      inset:0;
      pointer-events:none;
      background:
        linear-gradient(90deg, rgba(103,217,255,.05) 1px, transparent 1px),
        linear-gradient(0deg, rgba(36,209,182,.04) 1px, transparent 1px);
      background-size:34px 34px;
      opacity:.32;
      mask-image:linear-gradient(180deg, rgba(0,0,0,.72), transparent 90%);
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: rgba(7, 12, 18, 0.86);
      backdrop-filter: blur(18px);
      padding: 14px 18px;
      display: flex;
      gap: 18px;
      align-items: center;
      flex-wrap: wrap;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 700;
    }}
    .summary {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .stat {{
      border: 1px solid var(--line);
      background: rgba(10, 18, 28, 0.76);
      border-radius: 8px;
      padding: 7px 9px;
      min-width: 86px;
    }}
    .stat strong {{
      display: block;
      font-size: 16px;
      line-height: 1.1;
    }}
    .stat span {{
      color: var(--muted);
      font-size: 12px;
    }}
    main {{
      min-height: calc(100vh - 82px);
      display: grid;
      grid-template-columns: 300px minmax(420px, 1fr) 360px;
    }}
    aside, section, .detail {{
      min-width: 0;
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: rgba(7, 12, 18, 0.82);
      padding: 14px;
      overflow: auto;
      max-height: calc(100vh - 82px);
      backdrop-filter: blur(16px);
    }}
    .detail {{
      border-left: 1px solid var(--line);
      background: rgba(7, 12, 18, 0.82);
      padding: 14px;
      overflow: auto;
      max-height: calc(100vh - 82px);
      backdrop-filter: blur(16px);
    }}
    h2 {{
      font-size: 14px;
      margin: 16px 0 8px;
      text-transform: uppercase;
      color: #a9bcc8;
    }}
    h2:first-child {{ margin-top: 0; }}
    .side-intro {{
      position: sticky;
      top: 0;
      z-index: 3;
      background: rgba(7, 12, 18, 0.96);
      padding-bottom: 10px;
      margin-bottom: 8px;
      border-bottom: 1px solid rgba(255,255,255,.05);
    }}
    .side-mini-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }}
    .mini-nav-btn, .fold-toggle {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(10, 18, 28, 0.88);
      color: var(--ink);
      font: inherit;
      cursor: pointer;
    }}
    .mini-nav-btn {{
      padding: 7px 9px;
      font-size: 12px;
      color: #cfe2ec;
    }}
    .fold-section {{
      border: 1px solid rgba(255,255,255,.05);
      border-radius: 8px;
      margin: 10px 0;
      background: rgba(9, 16, 24, 0.62);
      overflow: hidden;
    }}
    .fold-toggle {{
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 11px;
      font-weight: 650;
      border: 0;
      border-bottom: 1px solid rgba(255,255,255,.04);
      background: rgba(11, 18, 27, 0.9);
    }}
    .fold-meta {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }}
    .fold-body {{
      display: none;
      padding: 8px 10px 10px;
    }}
    .fold-section.open .fold-body {{
      display: block;
    }}
    button.item {{
      display: block;
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(14, 21, 31, 0.96), rgba(9, 16, 24, 0.96));
      color: var(--ink);
      border-radius: 8px;
      padding: 9px;
      margin: 7px 0;
      cursor: pointer;
      font: inherit;
    }}
    button.item:hover, button.item.active {{
      border-color: rgba(103,217,255,.36);
      background: var(--soft);
      box-shadow: 0 0 0 1px rgba(103,217,255,.06), 0 12px 28px rgba(0,0,0,.24);
    }}
    .item-title {{
      font-weight: 650;
      font-size: 13px;
      line-height: 1.25;
    }}
    .item-meta {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 5px;
      line-height: 1.3;
    }}
    #graph-wrap {{
      position: relative;
      min-height: calc(100vh - 82px);
      background:
        radial-gradient(circle at 50% 0%, rgba(103,217,255,.08), transparent 28%),
        linear-gradient(180deg, #08111a, #071018);
    }}
    .graph-tools {{
      position: absolute;
      right: 12px;
      top: 12px;
      display: flex;
      gap: 6px;
      padding: 6px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(7, 12, 18, 0.88);
      backdrop-filter: blur(16px);
      z-index: 2;
    }}
    .graph-tools button {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(10, 18, 28, 0.88);
      color: var(--ink);
      font: inherit;
      font-size: 12px;
      line-height: 1;
      padding: 7px 9px;
      cursor: pointer;
    }}
    .graph-tools button:hover {{
      border-color: rgba(103,217,255,.42);
      background: var(--soft);
    }}
    .graph-tools button.active {{
      border-color: rgba(103,217,255,.46);
      background: rgba(103,217,255,.12);
      color: #8de4ff;
    }}
    .zoom-hint {{
      position: absolute;
      right: 12px;
      top: 58px;
      color: var(--muted);
      font-size: 12px;
      background: rgba(7, 12, 18, 0.82);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 5px 7px;
      z-index: 2;
    }}
    #graph {{
      display: block;
      width: 100%;
      height: calc(100vh - 82px);
      cursor: grab;
    }}
    #graph:active {{
      cursor: grabbing;
    }}
    .legend {{
      position: absolute;
      left: 12px;
      bottom: 12px;
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      max-width: calc(100% - 24px);
      background: rgba(7, 12, 18, 0.86);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
    }}
    .legend span {{
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .dot {{
      width: 9px;
      height: 9px;
      display: inline-block;
      border-radius: 50%;
      margin-right: 4px;
      vertical-align: middle;
    }}
    .node circle {{
      stroke: rgba(255,255,255,.18);
      stroke-width: 1.5px;
      cursor: pointer;
      filter: drop-shadow(0 0 8px rgba(103,217,255,.08));
    }}
    .node text {{
      pointer-events: none;
      font-size: 10px;
      fill: #dbe6ee;
      paint-order: stroke;
      stroke: rgba(6, 10, 16, 0.92);
      stroke-width: 4px;
      stroke-linejoin: round;
    }}
    .node text.hidden-label {{
      display: none;
    }}
    .link {{
      stroke: var(--edge);
      stroke-opacity: 0.55;
    }}
    .lane rect {{
      fill: rgba(255, 255, 255, 0.03);
      stroke: rgba(103,217,255,.12);
      stroke-width: 1;
      rx: 8;
      ry: 8;
    }}
    .lane text {{
      fill: #8dc6d7;
      font-size: 12px;
      font-weight: 700;
      text-anchor: middle;
      letter-spacing: 0;
    }}
    .link.highlight {{
      stroke: var(--highlight);
      stroke-width: 2.4px;
      stroke-opacity: 0.95;
    }}
    .node.highlight circle {{
      stroke: var(--highlight);
      stroke-width: 3px;
    }}
    .node.dim, .link.dim {{
      opacity: 0.16;
    }}
    .field {{
      margin: 10px 0;
      padding-bottom: 10px;
      border-bottom: 1px solid rgba(255,255,255,.06);
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 3px;
    }}
    .value {{
      font-size: 13px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .value a {{
      color: #8de4ff;
      text-decoration: none;
      font-weight: 600;
    }}
    .value a:hover {{
      text-decoration: underline;
    }}
    .pill {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 2px 6px;
      margin: 2px;
      font-size: 12px;
      color: #d7e5ed;
      background: rgba(103,217,255,.08);
    }}
    .empty {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .empty.compact {{
      font-size: 12px;
      margin: 0;
    }}
    .muted-inline {{
      color: var(--muted);
      font-size: 12px;
    }}
    .chat-panel {{
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid rgba(255,255,255,.07);
    }}
    .chat-starters {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 10px 0;
    }}
    .chat-chip {{
      border: 1px solid rgba(103,217,255,.2);
      border-radius: 8px;
      background: rgba(103,217,255,.07);
      color: #dce8ef;
      font: inherit;
      font-size: 12px;
      padding: 6px 8px;
      cursor: pointer;
    }}
    .chat-form {{
      display: grid;
      gap: 8px;
      margin: 10px 0;
    }}
    .chat-form textarea {{
      width: 100%;
      resize: vertical;
      min-height: 92px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(10, 18, 28, 0.88);
      color: var(--ink);
      font: inherit;
      padding: 10px;
    }}
    .chat-form button {{
      justify-self: start;
      border: 1px solid rgba(103,217,255,.26);
      border-radius: 8px;
      background: rgba(103,217,255,.12);
      color: var(--ink);
      font: inherit;
      font-weight: 650;
      padding: 8px 12px;
      cursor: pointer;
    }}
    .chat-answer {{
      border: 1px solid rgba(103,217,255,.14);
      border-radius: 8px;
      padding: 10px;
      background: rgba(103,217,255,.05);
      line-height: 1.5;
    }}
    .chat-answer p {{
      margin: 0;
    }}
    @media (max-width: 980px) {{
      main {{
        grid-template-columns: 1fr;
      }}
      aside, .detail {{
        max-height: none;
        border: 0;
        border-bottom: 1px solid var(--line);
      }}
      #graph, #graph-wrap {{
        height: 620px;
        min-height: 620px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>literature conflict explorer</h1>
    <div class="summary" id="summary"></div>
  </header>
  <main>
    <aside>
      <div class="side-intro">
        <div class="side-mini-nav">
          <button type="button" class="mini-nav-btn" data-target="anomalies-section">Conflicts</button>
          <button type="button" class="mini-nav-btn" data-target="hypotheses-section">Explanations</button>
          <button type="button" class="mini-nav-btn" data-target="insights-section">Insights</button>
        </div>
        <p class="empty compact">Start with the graph, then open only the section you need.</p>
      </div>
      <div class="fold-section" id="anomalies-section">
        <button type="button" class="fold-toggle" data-section="anomaly-list">
          <span>Conflicts & Gaps</span>
          <span class="fold-meta" id="anomaly-count"></span>
        </button>
        <div class="fold-body" id="anomaly-list"></div>
      </div>
      <div class="fold-section" id="hypotheses-section">
        <button type="button" class="fold-toggle" data-section="hypothesis-list">
          <span>Possible Explanations</span>
          <span class="fold-meta" id="hypothesis-count"></span>
        </button>
        <div class="fold-body" id="hypothesis-list"></div>
      </div>
      <div class="fold-section" id="insights-section">
        <button type="button" class="fold-toggle" data-section="insight-list">
          <span>Community Insights</span>
          <span class="fold-meta" id="insight-count"></span>
        </button>
        <div class="fold-body" id="insight-list"></div>
      </div>
    </aside>
    <section id="graph-wrap" aria-label="Graph visualization">
      <div class="graph-tools" aria-label="Graph controls">
        <button type="button" id="zoom-out">−</button>
        <button type="button" id="zoom-in">+</button>
        <button type="button" id="zoom-fit">Fit</button>
        <button type="button" id="zoom-reset">Reset</button>
        <button type="button" id="detail-cluster">Clusters</button>
        <button type="button" id="detail-claims">Claims</button>
        <button type="button" id="detail-full">Full</button>
        <button type="button" id="view-hierarchy">Hierarchy</button>
        <button type="button" id="view-free">Free</button>
      </div>
      <div class="zoom-hint">Scroll to zoom · drag background to pan · start with Clusters, then open Claims and Full when you want more evidence detail</div>
      <svg id="graph"></svg>
      <div class="legend" id="legend"></div>
    </section>
    <div class="detail">
      <h2>Details</h2>
      <div id="detail" class="empty">Click a node, anomaly, hypothesis, or insight.</div>
      <div class="chat-panel" id="chat-panel">
        <h2>Ask This Graph</h2>
        <p class="empty compact" id="chat-context">Ask about the whole run, or click a node first for selection-aware analysis.</p>
        <div class="chat-starters" id="chat-starters">
          <button type="button" class="chat-chip">Why is this a conflict?</button>
          <button type="button" class="chat-chip">Which papers support this keyword?</button>
          <button type="button" class="chat-chip">What would resolve this disagreement?</button>
        </div>
        <form id="graph-chat-form" class="chat-form">
          <textarea id="graph-chat-input" rows="3" placeholder="Ask a question about this graph..." {"" if payload.get("run_id") else "disabled"}></textarea>
          <button type="submit" id="graph-chat-send" {"" if payload.get("run_id") else "disabled"}>Ask</button>
        </form>
        <div id="graph-chat-response" class="empty compact">{'Graph chat is available for completed runs.' if payload.get("run_id") else 'Graph chat is disabled for the community-wide aggregate map.'}</div>
      </div>
    </div>
  </main>
  <script type="application/json" id="aigraph-data">{data}</script>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script>
    const DATA = JSON.parse(document.getElementById('aigraph-data').textContent);
    const IS_COMMUNITY_GRAPH = DATA.graph_mode === 'community';
    const RUN_ID = DATA.run_id || '';
    const papersById = Object.fromEntries(DATA.papers.map(d => [d.paper_id, d]));
    const claimsById = Object.fromEntries(DATA.claims.map(d => [d.claim_id, d]));
    const anomaliesById = Object.fromEntries(DATA.anomalies.map(d => [d.anomaly_id, d]));
    const insightsById = Object.fromEntries((DATA.insights || []).map(d => [d.insight_id, d]));
    let selectedContext = null;

    const typeColor = {{
      Paper: '#6b7280',
      Claim: '#2563eb',
      Method: '#059669',
      Model: '#16a34a',
      Task: '#d97706',
      Dataset: '#7c3aed',
      Metric: '#dc2626',
      Baseline: '#9333ea',
      Setting: '#0891b2',
      Domain: '#0f766e',
      DataModality: '#64748b',
      Mechanism: '#be123c',
      FailureMode: '#b91c1c',
      EvaluationProtocol: '#4f46e5',
      Assumption: '#525252',
      RiskType: '#ea580c',
      TemporalProperty: '#0369a1'
    }};

    function esc(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }}[ch]));
    }}

    function paperFromNode(node) {{
      return papersById[node.paper_id] || papersById[String(node.id || '').replace(/^Paper:/, '')] || null;
    }}

    function paperLink(paper, fallbackId = '') {{
      if (!paper) return esc(fallbackId);
      const title = paper.title || fallbackId || paper.paper_id || 'paper';
      const year = paper.year ? ` (${{paper.year}})` : '';
      const label = `${{title}}${{year}}`;
      const url = paper.url || paper.doi_url;
      if (!url) return esc(label);
      return `<a href="${{esc(url)}}" target="_blank" rel="noopener noreferrer">${{esc(label)}}</a>`;
    }}

    function shortTitle(title, maxLen = 38) {{
      const text = String(title || '').trim();
      if (!text) return '';
      return text.length > maxLen ? `${{text.slice(0, maxLen - 3)}}...` : text;
    }}

    function nodeLabel(node) {{
      if (node.node_type === 'Paper') {{
        const paper = paperFromNode(node);
        if (paper && paper.title) return shortTitle(paper.title);
      }}
      if (node.node_type === 'Claim') return node.claim_id || node.id;
      if (node.name) return node.name;
      if (node.value) return `${{node.field}}=${{node.value}}`;
      return String(node.id).split(':').slice(1).join(':') || node.id;
    }}

    function field(label, value) {{
      if (value === undefined || value === null || value === '') return '';
      return `<div class="field"><div class="label">${{esc(label)}}</div><div class="value">${{esc(value)}}</div></div>`;
    }}

    function fieldHtml(label, html) {{
      if (!html) return '';
      return `<div class="field"><div class="label">${{esc(label)}}</div><div class="value">${{html}}</div></div>`;
    }}

    function pills(obj) {{
      if (!obj || !Object.keys(obj).length) return '';
      return Object.entries(obj).map(([k, v]) => `<span class="pill">${{esc(k)}}=${{esc(v)}}</span>`).join('');
    }}

    function showDetail(html) {{
      document.getElementById('detail').className = '';
      document.getElementById('detail').innerHTML = html;
    }}

    function renderSummary() {{
      const s = DATA.summary;
      const items = [
        ['Papers', s.papers], ['Claims', s.claims], ['Nodes', s.nodes],
        ['Edges', s.edges], ['Anomalies', s.anomalies], ['Hypotheses', s.hypotheses],
        ['Insights', s.insights || 0]
      ];
      document.getElementById('summary').innerHTML = items.map(([label, value]) =>
        `<div class="stat"><strong>${{value}}</strong><span>${{label}}</span></div>`
      ).join('');
    }}

    function renderLists() {{
      const anomalyList = document.getElementById('anomaly-list');
      anomalyList.innerHTML = DATA.anomalies.length ? DATA.anomalies.map(a => `
        <button class="item" data-kind="anomaly" data-id="${{esc(a.anomaly_id)}}">
          <div class="item-title">${{esc(a.anomaly_id)}} · ${{esc(a.type)}}</div>
          <div class="item-meta">${{esc(a.central_question)}}</div>
          <div class="item-meta">${{a.claim_ids.length}} claims · +${{a.positive_claims.length}} / -${{a.negative_claims.length}}</div>
        </button>
      `).join('') : '<div class="empty">No anomalies found.</div>';

      const hypothesisList = document.getElementById('hypothesis-list');
      const shownHypotheses = DATA.hypotheses.slice(0, Math.min(DATA.hypotheses.length, 10));
      const hiddenHypothesisCount = Math.max(0, DATA.hypotheses.length - shownHypotheses.length);
      hypothesisList.innerHTML = DATA.hypotheses.length ? shownHypotheses.map(h => `
        <button class="item" data-kind="hypothesis" data-id="${{esc(h.hypothesis_id)}}">
          <div class="item-title">${{esc(h.hypothesis_id)}} · ${{esc(h.anomaly_id)}}</div>
          <div class="item-meta">${{esc(h.hypothesis)}}</div>
        </button>
      `).join('') + (hiddenHypothesisCount ? `<div class="empty">Showing ${{shownHypotheses.length}} of ${{DATA.hypotheses.length}} hypotheses to keep the map readable.</div>` : '') : '<div class="empty">No hypotheses generated.</div>';

      const insightList = document.getElementById('insight-list');
      insightList.innerHTML = (DATA.insights || []).length ? DATA.insights.map(i => `
        <button class="item" data-kind="insight" data-id="${{esc(i.insight_id)}}">
          <div class="item-title">${{esc(i.insight_id)}} · ${{esc(i.type)}}</div>
          <div class="item-meta">${{esc(i.title)}}</div>
          <div class="item-meta">${{(i.communities || []).map(esc).join(' ↔ ')}}</div>
        </button>
      `).join('') : '<div class="empty">No community insights generated.</div>';

      document.querySelectorAll('button.item').forEach(btn => {{
        btn.addEventListener('click', () => {{
          document.querySelectorAll('button.item').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          if (btn.dataset.kind === 'anomaly') selectAnomaly(btn.dataset.id);
          if (btn.dataset.kind === 'hypothesis') selectHypothesis(btn.dataset.id);
          if (btn.dataset.kind === 'insight') selectInsight(btn.dataset.id);
        }});
      }});
    }}

    function renderDetailNode(node) {{
      const id = node.id || '';
      let html = `<h2>${{esc(node.node_type || 'Node')}}</h2>`;
      if (node.node_type === 'Paper') {{
        const paper = paperFromNode(node) || {{}};
        html += field('title', paper.title || id);
        html += field('year', paper.year);
        html += field('venue', paper.venue);
        html += field('citations', paper.cited_by_count ?? node.cited_by_count);
        html += field('retrieval channel', paper.retrieval_channel);
        html += field('selection score', paper.selection_score ? Number(paper.selection_score).toFixed(2) : '');
        html += field('selection reason', paper.selection_reason);
        html += field('recent citations', node.recent_citations);
        html += field('citation velocity', node.citation_velocity ? Number(node.citation_velocity).toFixed(2) : '');
        html += field('impact score', node.age_normalized_impact ? Number(node.age_normalized_impact).toFixed(2) : '');
        html += fieldHtml('paper link', paperLink(paper, id));
        html += field('abstract', paper.abstract);
      }} else if (node.node_type === 'Claim') {{
        const claim = claimsById[node.claim_id] || {{}};
        const paper = papersById[claim.paper_id] || {{}};
        html += field('claim', claim.claim_text);
        html += field('direction', claim.direction);
        html += fieldHtml('paper', paperLink(paper, claim.paper_id));
        html += field('method', claim.method);
        html += field('task', claim.task || claim.canonical_task);
        html += field('dataset', claim.dataset);
        html += field('metric', claim.metric);
        html += field('domain', claim.domain);
        html += field('mechanism', claim.mechanism);
        html += field('failure mode', claim.failure_mode);
        html += field('evaluation protocol', claim.evaluation_protocol);
        html += field('temporal property', claim.temporal_property);
        html += field('evidence', claim.evidence_span);
      }} else {{
        html += field('id', id);
        html += field('label', nodeLabel(node));
      }}
      showDetail(html);
    }}

    function renderAnomaly(a) {{
      const claimRows = a.claim_ids.map(cid => claimsById[cid]).filter(Boolean).map(c => {{
        const p = papersById[c.paper_id] || {{}};
        return `<div class="field"><div class="label">${{esc(c.claim_id)}} · ${{esc(c.direction)}}</div><div class="value">${{paperLink(p, c.paper_id)}}<br>${{esc(c.claim_text)}}</div></div>`;
      }}).join('');
      showDetail(`
        <h2>${{esc(a.anomaly_id)}} · ${{esc(a.type)}}</h2>
        ${{field('question', a.central_question)}}
        <div class="field"><div class="label">shared entities</div><div class="value">${{pills(a.shared_entities)}}</div></div>
        ${{a.varying_settings.length ? field('varying settings', a.varying_settings.join(', ')) : ''}}
        ${{field('evidence impact', a.evidence_impact ? Number(a.evidence_impact).toFixed(2) : '')}}
        ${{field('recent activity', a.recent_activity ? Number(a.recent_activity).toFixed(2) : '')}}
        ${{field('topology score', a.topology_score ? Number(a.topology_score).toFixed(2) : '')}}
        <h2>Evidence Claims</h2>
        ${{claimRows || '<div class="empty">No claim details.</div>'}}
      `);
    }}

    function renderHypothesis(h) {{
      const a = anomaliesById[h.anomaly_id];
      const claims = (h.explains_claims || []).map(cid => claimsById[cid]).filter(Boolean).map(c =>
        `<div class="field"><div class="label">${{esc(c.claim_id)}} · ${{esc(c.direction)}}</div><div class="value">${{esc(c.claim_text)}}</div></div>`
      ).join('');
      const preds = (h.predictions || []).map(p => `<li>${{esc(p)}}</li>`).join('');
      showDetail(`
        <h2>${{esc(h.hypothesis_id)}} · Hypothesis</h2>
        ${{field('anomaly', a ? `${{a.anomaly_id}} · ${{a.type}}` : h.anomaly_id)}}
        ${{field('hypothesis', h.hypothesis)}}
        ${{field('mechanism', h.mechanism)}}
        <div class="field"><div class="label">predictions</div><div class="value"><ul>${{preds}}</ul></div></div>
        ${{field('minimal test', h.minimal_test)}}
        ${{field('evidence gap', h.evidence_gap)}}
        <h2>Explained Claims</h2>
        ${{claims || '<div class="empty">No linked claims.</div>'}}
      `);
    }}

    function renderInsight(i) {{
      const concepts = (i.shared_concepts || []).map(c => `<span class="pill">${{esc(c)}}</span>`).join('');
      const papers = (i.evidence_papers || []).map(pid => {{
        const p = papersById[pid] || {{}};
        const meta = [p.retrieval_channel, p.selection_score ? `score ${{Number(p.selection_score).toFixed(2)}}` : '', p.selection_reason].filter(Boolean).join(' · ');
        return `<div class="field"><div class="label">${{esc(pid)}}</div><div class="value">${{paperLink(p, pid)}}${{meta ? '<br><span class="muted">' + esc(meta) + '</span>' : ''}}</div></div>`;
      }}).join('');
      const suggestions = (i.transfer_suggestions || []).map(s => `<li>${{esc(s)}}</li>`).join('');
      showDetail(`
        <h2>${{esc(i.insight_id)}} · ${{esc(i.type)}}</h2>
        ${{field('title', i.title)}}
        ${{field('communities', (i.communities || []).join(' ↔ '))}}
        <div class="field"><div class="label">shared concepts</div><div class="value">${{concepts}}</div></div>
        ${{field('insight', i.insight)}}
        ${{field('unifying frame', i.unifying_frame)}}
        ${{field('citation gap', i.citation_gap)}}
        <div class="field"><div class="label">transfer suggestions</div><div class="value"><ul>${{suggestions}}</ul></div></div>
        ${{field('scores', `impact=${{Number(i.impact_score || 0).toFixed(2)}}, topology=${{Number(i.topology_score || 0).toFixed(2)}}, confidence=${{Number(i.confidence_score || 0).toFixed(2)}}`)}}
        <h2>Evidence Papers</h2>
        ${{papers || '<div class="empty">No linked papers.</div>'}}
      `);
    }}

    let svg, viewport, zoomBehavior, nodeSel, linkSel, labelSel, currentWidth = 0, currentHeight = 0;
    let currentLayout = 'hierarchy';
    let currentDetail = 'cluster';

    const hierarchyLanes = IS_COMMUNITY_GRAPH
      ? [
          {{ key: 'keyword', label: 'Keyword', types: ['Domain', 'Task', 'TemporalProperty', 'DataModality', 'RiskType', 'Method', 'Model', 'Mechanism'] }},
          {{ key: 'claim', label: 'Claims', types: ['Claim'] }},
          {{ key: 'paper', label: 'Papers', types: ['Paper'] }},
        ]
      : [
          {{ key: 'topic', label: 'Topic', types: ['Domain', 'Task', 'TemporalProperty', 'DataModality', 'RiskType'] }},
          {{ key: 'method', label: 'Method', types: ['Method', 'Model', 'Mechanism', 'Baseline', 'Setting'] }},
          {{ key: 'evidence', label: 'Evidence', types: ['Paper', 'Claim'] }},
          {{ key: 'evaluation', label: 'Evaluation', types: ['Dataset', 'Metric', 'FailureMode', 'EvaluationProtocol', 'Assumption'] }},
        ];
    const hierarchyTypeToLane = Object.fromEntries(
      hierarchyLanes.flatMap((lane, index) => lane.types.map(type => [type, index]))
    );
    const clusterNodeTypes = new Set([
      'Method',
      'Model',
      'Task',
      'Domain',
      'Mechanism',
      'TemporalProperty',
      'DataModality',
      'RiskType'
    ]);
    const claimsNodeTypes = IS_COMMUNITY_GRAPH
      ? new Set([
          'Paper',
          'Claim',
          'Method',
          'Model',
          'Task',
          'Domain',
          'Mechanism',
          'TemporalProperty',
          'DataModality',
          'RiskType'
        ])
      : new Set([
          'Paper',
          'Claim',
          'Method',
          'Task',
          'Domain',
          'Mechanism',
          'TemporalProperty',
          'DataModality',
          'RiskType'
        ]);

    function edgeKey(e) {{
      const s = typeof e.source === 'object' ? e.source.id : e.source;
      const t = typeof e.target === 'object' ? e.target.id : e.target;
      return `${{s}}|${{t}}|${{e.edge_type || ''}}`;
    }}

    function endpointId(value) {{
      return typeof value === 'object' ? value.id : value;
    }}

    function clearHighlight() {{
      nodeSel.classed('highlight', false).classed('dim', false);
      linkSel.classed('highlight', false).classed('dim', false);
    }}

    function highlight(nodeIds, edgeKeys = new Set()) {{
      const ids = new Set(nodeIds);
      nodeSel.classed('highlight', d => ids.has(d.id)).classed('dim', d => ids.size && !ids.has(d.id));
      linkSel.classed('highlight', d => edgeKeys.has(edgeKey(d)))
        .classed('dim', d => ids.size && !(ids.has(typeof d.source === 'object' ? d.source.id : d.source) && ids.has(typeof d.target === 'object' ? d.target.id : d.target)));
    }}

    function updateLabelVisibility(transform = d3.zoomTransform(svg.node())) {{
      if (!labelSel) return;
      labelSel.classed('hidden-label', d => transform.k < 0.55 && d.node_type !== 'Claim');
    }}

    function updateViewButtons() {{
      document.getElementById('detail-cluster')?.classList.toggle('active', currentDetail === 'cluster');
      document.getElementById('detail-claims')?.classList.toggle('active', currentDetail === 'claims');
      document.getElementById('detail-full')?.classList.toggle('active', currentDetail === 'full');
      document.getElementById('view-hierarchy')?.classList.toggle('active', currentLayout === 'hierarchy');
      document.getElementById('view-free')?.classList.toggle('active', currentLayout === 'free');
    }}

    function buildProjectedGraph(rawNodes, rawLinks, visibleTypeSet) {{
      const visibleNodes = rawNodes.filter(node => visibleTypeSet.has(node.node_type));
      const visibleById = new Map(visibleNodes.map(node => [node.id, node]));
      const links = [];
      const seen = new Set();

      function addLink(source, target, edgeType, extras = {{}}) {{
        if (!source || !target || source === target) return;
        if (!visibleById.has(source) || !visibleById.has(target)) return;
        const key = `${{source}}|${{target}}|${{edgeType || ''}}|${{extras.projected ? extras.via || 'projected' : 'direct'}}`;
        if (seen.has(key)) return;
        seen.add(key);
        links.push({{
          source,
          target,
          edge_type: edgeType || 'related',
          projected: Boolean(extras.projected),
          via: extras.via || null,
        }});
      }}

      rawLinks.forEach(link => {{
        const source = endpointId(link.source);
        const target = endpointId(link.target);
        if (visibleById.has(source) && visibleById.has(target)) {{
          addLink(source, target, link.edge_type, {{ projected: false }});
        }}
      }});

      const claims = new Map();
      function ensureClaim(claimId) {{
        if (!claims.has(claimId)) {{
          claims.set(claimId, {{
            incomingVisible: new Set(),
            outgoingVisible: new Set(),
          }});
        }}
        return claims.get(claimId);
      }}

      rawLinks.forEach(link => {{
        const source = endpointId(link.source);
        const target = endpointId(link.target);
        if (source.startsWith('Claim:') && visibleById.has(target)) {{
          ensureClaim(source).outgoingVisible.add(target);
        }}
        if (target.startsWith('Claim:') && visibleById.has(source)) {{
          ensureClaim(target).incomingVisible.add(source);
        }}
      }});

      claims.forEach((entry, claimId) => {{
        entry.incomingVisible.forEach(source => {{
          entry.outgoingVisible.forEach(target => {{
            addLink(source, target, 'projected', {{ projected: true, via: claimId }});
          }});
        }});

        const entityNeighbors = [...entry.outgoingVisible];
        for (let i = 0; i < entityNeighbors.length; i += 1) {{
          for (let j = i + 1; j < entityNeighbors.length; j += 1) {{
            const left = visibleById.get(entityNeighbors[i]);
            const right = visibleById.get(entityNeighbors[j]);
            if (!left || !right) continue;
            if (left.node_type === right.node_type) continue;
            addLink(left.id, right.id, 'context', {{ projected: true, via: claimId }});
          }}
        }}
      }});

      const linkedIds = new Set();
      links.forEach(link => {{
        linkedIds.add(endpointId(link.source));
        linkedIds.add(endpointId(link.target));
      }});

      return {{
        nodes: visibleNodes.filter(node => linkedIds.has(node.id)).map(node => ({{ ...node }})),
        links: links.map(link => ({{ ...link }})),
      }};
    }}

    function buildClusterGraph(rawNodes, rawLinks) {{
      const projected = buildProjectedGraph(rawNodes, rawLinks, clusterNodeTypes);
      const degree = new Map(projected.nodes.map(node => [node.id, 0]));
      projected.links.forEach(link => {{
        const source = endpointId(link.source);
        const target = endpointId(link.target);
        degree.set(source, (degree.get(source) || 0) + 1);
        degree.set(target, (degree.get(target) || 0) + 1);
      }});
      const limit = IS_COMMUNITY_GRAPH ? 18 : 14;
      const ranked = [...projected.nodes].sort((left, right) => {{
        const leftPriority = left.node_type === 'Task' || left.node_type === 'Method' ? 1 : 0;
        const rightPriority = right.node_type === 'Task' || right.node_type === 'Method' ? 1 : 0;
        return (
          (rightPriority - leftPriority)
          || ((degree.get(right.id) || 0) - (degree.get(left.id) || 0))
          || (String(left.name || left.value || '').length - String(right.name || right.value || '').length)
        );
      }});
      const kept = new Set(ranked.slice(0, limit).map(node => node.id));
      const links = projected.links.filter(link => kept.has(endpointId(link.source)) && kept.has(endpointId(link.target)));
      const linkedIds = new Set();
      links.forEach(link => {{
        linkedIds.add(endpointId(link.source));
        linkedIds.add(endpointId(link.target));
      }});
      const nodes = ranked.filter(node => linkedIds.has(node.id) || kept.has(node.id)).slice(0, limit);
      return {{
        nodes: nodes.map(node => ({{ ...node }})),
        links: links.map(link => ({{ ...link }})),
      }};
    }}

    function graphForCurrentDetail() {{
      const rawNodes = DATA.graph.nodes || [];
      const rawLinks = DATA.graph.edges || [];
      if (currentDetail === 'cluster') {{
        return buildClusterGraph(rawNodes, rawLinks);
      }}
      if (currentDetail === 'claims') {{
        return buildProjectedGraph(rawNodes, rawLinks, claimsNodeTypes);
      }}
      return {{
        nodes: rawNodes.map(node => ({{ ...node }})),
        links: rawLinks.map(link => ({{ ...link }})),
      }};
    }}

    function laneForNodeType(nodeType) {{
      return hierarchyTypeToLane[nodeType] ?? Math.min(hierarchyLanes.length - 1, 2);
    }}

    function hashString(value) {{
      let hash = 0;
      const text = String(value || '');
      for (let i = 0; i < text.length; i += 1) {{
        hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
      }}
      return Math.abs(hash);
    }}

    function hierarchyTargetX(node, width) {{
      const laneIndex = laneForNodeType(node.node_type);
      const laneWidth = width / hierarchyLanes.length;
      return laneWidth * laneIndex + laneWidth / 2;
    }}

    function hierarchyTargetY(node, height) {{
      const paddingTop = 84;
      const paddingBottom = 44;
      const usable = Math.max(180, height - paddingTop - paddingBottom);
      const seed = hashString(`${{node.id}}|${{node.node_type}}`);
      return paddingTop + (seed % 1000) / 1000 * usable;
    }}

    function renderHierarchyLanes(width, height) {{
      const laneLayer = viewport.append('g')
        .attr('class', 'lane-layer')
        .style('display', currentLayout === 'hierarchy' ? null : 'none');
      const laneWidth = width / hierarchyLanes.length;
      const lanes = laneLayer.selectAll('g')
        .data(hierarchyLanes)
        .join('g')
        .attr('class', 'lane');
      lanes.append('rect')
        .attr('x', (_, i) => i * laneWidth + 12)
        .attr('y', 16)
        .attr('width', Math.max(80, laneWidth - 24))
        .attr('height', Math.max(220, height - 32));
      lanes.append('text')
        .attr('x', (_, i) => i * laneWidth + laneWidth / 2)
        .attr('y', 40)
        .text(d => d.label);
    }}

    function fitGraph(duration = 650) {{
      if (!svg || !viewport || !nodeSel || !nodeSel.size()) return;
      const bounds = viewport.node().getBBox();
      if (!bounds.width || !bounds.height) return;
      const margin = 54;
      const scale = Math.max(0.18, Math.min(1.45, Math.min(
        (currentWidth - margin * 2) / bounds.width,
        (currentHeight - margin * 2) / bounds.height
      )));
      const x = currentWidth / 2 - scale * (bounds.x + bounds.width / 2);
      const y = currentHeight / 2 - scale * (bounds.y + bounds.height / 2);
      const transform = d3.zoomIdentity.translate(x, y).scale(scale);
      svg.transition().duration(duration).call(zoomBehavior.transform, transform);
      updateLabelVisibility(transform);
    }}

    function resetGraph(duration = 450) {{
      if (!svg) return;
      const transform = d3.zoomIdentity;
      svg.transition().duration(duration).call(zoomBehavior.transform, transform);
      updateLabelVisibility(transform);
    }}

    function selectAnomaly(id) {{
      const a = anomaliesById[id];
      if (!a) return;
      const edgeKeys = new Set((a.local_graph_edges || []).map(e => `${{e.source}}|${{e.target}}|${{e.edge_type || ''}}`));
      highlight(a.local_graph_nodes || [], edgeKeys);
      renderAnomaly(a);
    }}

    function selectHypothesis(id) {{
      const h = DATA.hypotheses.find(x => x.hypothesis_id === id);
      if (!h) return;
      highlight((h.explains_claims || []).map(cid => `Claim:${{cid}}`));
      renderHypothesis(h);
    }}

    function selectInsight(id) {{
      const i = insightsById[id];
      if (!i) return;
      const ids = new Set();
      (i.evidence_claims || []).forEach(cid => ids.add(`Claim:${{cid}}`));
      (i.evidence_papers || []).forEach(pid => ids.add(`Paper:${{pid}}`));
      const shared = new Set((i.shared_concepts || []).map(c => String(c).toLowerCase()));
      (DATA.graph.nodes || []).forEach(n => {{
        const label = String(n.name || n.value || '').toLowerCase();
        if (shared.has(label)) ids.add(n.id);
      }});
      highlight([...ids]);
      renderInsight(i);
    }}

    function drawGraph() {{
      const wrap = document.getElementById('graph-wrap');
      const width = Math.max(420, wrap.clientWidth);
      const height = Math.max(560, wrap.clientHeight);
      currentWidth = width;
      currentHeight = height;
      const graphData = graphForCurrentDetail();
      const nodes = graphData.nodes;
      const links = graphData.links;

      svg = d3.select('#graph').attr('viewBox', [0, 0, width, height]);
      svg.selectAll('*').remove();
      viewport = svg.append('g').attr('class', 'viewport');
      renderHierarchyLanes(width, height);

      zoomBehavior = d3.zoom()
        .scaleExtent([0.12, 4])
        .on('zoom', event => {{
          viewport.attr('transform', event.transform);
          updateLabelVisibility(event.transform);
        }});
      svg.call(zoomBehavior);

      linkSel = viewport.append('g')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('class', 'link')
        .attr('stroke-width', d => d.edge_type === 'contradicts' ? 2.2 : (d.edge_type === 'cites' ? 0.9 : (d.projected ? 1.0 : 1.2)))
        .attr('stroke-dasharray', d => d.edge_type === 'cites' ? '4 4' : (d.projected ? '3 5' : null))
        .attr('stroke-opacity', d => d.edge_type === 'cites' ? 0.28 : (d.projected ? 0.34 : 0.55));

      linkSel.append('title').text(d => `${{d.source}} → ${{d.target}} · ${{d.edge_type || 'related'}}`);

      const node = viewport.append('g')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .attr('class', 'node')
        .call(d3.drag()
          .on('start', dragstarted)
          .on('drag', dragged)
          .on('end', dragended));

      node.append('circle')
        .attr('r', d => {{
          if (d.node_type === 'Paper') return 5.5 + Math.min(8, Number(d.age_normalized_impact || d.impact_score || 0) * 1.6);
          return d.node_type === 'Claim' ? 7 : 5.5;
        }})
        .attr('fill', d => typeColor[d.node_type] || '#404040')
        .on('click', (event, d) => {{
          clearHighlight();
          d3.select(event.currentTarget.parentNode).classed('highlight', true);
          renderDetailNode(d);
        }});

      node.append('title').text(d => `${{d.node_type || 'Node'}} · ${{nodeLabel(d)}}`);
      node.append('text')
        .attr('x', 8)
        .attr('y', 3)
        .text(d => nodeLabel(d).slice(0, 30));

      nodeSel = node;
      labelSel = node.selectAll('text');

      const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(d => d.edge_type === 'makes' ? 54 : (d.edge_type === 'cites' ? 96 : 82)))
        .force('charge', d3.forceManyBody().strength(currentDetail === 'cluster' ? -280 : (currentDetail === 'claims' ? -240 : -320)))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => d.node_type === 'Paper' ? 34 : 26))
        .force('x', d3.forceX(d => currentLayout === 'hierarchy' ? hierarchyTargetX(d, width) : width / 2)
          .strength(currentLayout === 'hierarchy' ? 0.34 : 0.05))
        .force('y', d3.forceY(d => currentLayout === 'hierarchy' ? hierarchyTargetY(d, height) : height / 2)
          .strength(currentLayout === 'hierarchy' ? 0.12 : 0.05));
      simulation.alphaDecay(currentDetail === 'full' ? 0.06 : 0.09);
      simulation.velocityDecay(currentDetail === 'full' ? 0.34 : 0.42);

      let didFit = false;
      simulation.on('tick', () => {{
        linkSel
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);
        nodeSel.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
        if (!didFit && simulation.alpha() < 0.18) {{
          didFit = true;
          fitGraph(500);
        }}
      }});
      simulation.on('end', () => fitGraph(350));

      function dragstarted(event, d) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      }}
      function dragged(event, d) {{
        d.fx = event.x;
        d.fy = event.y;
      }}
      function dragended(event, d) {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      }}

      document.getElementById('zoom-in').onclick = () => svg.transition().duration(250).call(zoomBehavior.scaleBy, 1.25);
      document.getElementById('zoom-out').onclick = () => svg.transition().duration(250).call(zoomBehavior.scaleBy, 0.8);
      document.getElementById('zoom-fit').onclick = () => fitGraph(450);
      document.getElementById('zoom-reset').onclick = () => resetGraph(350);
      document.getElementById('detail-cluster').onclick = () => {{
        currentDetail = 'cluster';
        updateViewButtons();
        drawGraph();
      }};
      document.getElementById('detail-claims').onclick = () => {{
        currentDetail = 'claims';
        updateViewButtons();
        drawGraph();
      }};
      document.getElementById('detail-full').onclick = () => {{
        currentDetail = 'full';
        updateViewButtons();
        drawGraph();
      }};
      document.getElementById('view-hierarchy').onclick = () => {{
        currentLayout = 'hierarchy';
        updateViewButtons();
        drawGraph();
      }};
      document.getElementById('view-free').onclick = () => {{
        currentLayout = 'free';
        updateViewButtons();
        drawGraph();
      }};
      updateViewButtons();
    }}

    function renderLegend() {{
      const types = ['Paper', 'Claim', 'Method', 'Task', 'Metric', 'Domain', 'Mechanism', 'FailureMode', 'TemporalProperty'];
      document.getElementById('legend').innerHTML = types.map(t =>
        `<span><i class="dot" style="background:${{typeColor[t]}}"></i>${{t}}</span>`
      ).join('') + '<span>-- citation edge</span>';
    }}

    renderSummary();
    renderLists();
    renderLegend();
    updateViewButtons();
    drawGraph();
    window.addEventListener('resize', () => drawGraph());
    if (DATA.anomalies.length) selectAnomaly(DATA.anomalies[0].anomaly_id);
  </script>
</body>
</html>
"""
