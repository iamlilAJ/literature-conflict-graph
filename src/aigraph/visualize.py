"""Static HTML visualization for aigraph output directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paper_select import paper_role_explanation, paper_role_label


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
    role = str(out.get("paper_role") or "other")
    out["paper_role"] = role
    out["paper_role_label"] = paper_role_label(role)
    out["paper_role_explanation"] = paper_role_explanation(role)
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
    .mini-nav-btn[disabled], .fold-toggle[disabled] {{
      cursor: not-allowed;
      opacity: 0.52;
      border-color: rgba(120, 152, 171, 0.16);
      color: #79909e;
      background: rgba(8, 14, 22, 0.62);
      box-shadow: none;
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
    .fold-section.disabled {{
      opacity: 0.82;
      border-color: rgba(120, 152, 171, 0.14);
      background: rgba(8, 14, 22, 0.52);
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
    .fold-disabled-note {{
      display: none;
      padding: 0 11px 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
    .fold-section.disabled .fold-disabled-note {{
      display: block;
    }}
    .fold-body {{
      display: none;
      padding: 8px 10px 10px;
    }}
    .fold-section.open .fold-body {{
      display: block;
    }}
    .fold-note {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
      line-height: 1.45;
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
    .tool-group {{
      display: flex;
      gap: 6px;
      align-items: center;
    }}
    .tool-divider {{
      width: 1px;
      align-self: stretch;
      background: rgba(255,255,255,.08);
      margin: 0 2px;
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
      line-height: 1.45;
      overflow-wrap: anywhere;
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
      display: grid;
      gap: 10px;
    }}
    .chat-header {{
      display: grid;
      gap: 8px;
    }}
    .chat-toolbar {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
    }}
    .chat-toolbar-actions {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }}
    .chat-selection {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      min-height: 18px;
      max-width: 260px;
    }}
    .chat-limit {{
      color: var(--muted);
      font-size: 11px;
      line-height: 1.4;
      text-align: right;
    }}
    .chat-reset {{
      border: 1px solid rgba(255,255,255,.10);
      border-radius: 8px;
      background: rgba(10, 18, 28, 0.88);
      color: var(--ink);
      font: inherit;
      font-size: 12px;
      font-weight: 650;
      padding: 6px 8px;
      cursor: pointer;
      white-space: nowrap;
    }}
    .chat-reset[disabled] {{
      cursor: not-allowed;
      opacity: 0.52;
      color: #79909e;
    }}
    .chat-thread {{
      display: grid;
      gap: 10px;
      min-height: 240px;
      max-height: 360px;
      overflow-y: auto;
      padding: 12px;
      border: 1px solid rgba(103,217,255,.14);
      border-radius: 8px;
      background: rgba(103,217,255,.04);
    }}
    .chat-message {{
      display: flex;
      gap: 10px;
    }}
    .chat-message.user {{
      justify-content: flex-end;
    }}
    .chat-message.assistant,
    .chat-message.system {{
      justify-content: flex-start;
    }}
    .chat-bubble {{
      max-width: 92%;
      border-radius: 8px;
      padding: 10px 12px;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }}
    .chat-message.user .chat-bubble {{
      background: rgba(103,217,255,.16);
      border: 1px solid rgba(103,217,255,.22);
      color: var(--ink);
    }}
    .chat-message.assistant .chat-bubble {{
      background: rgba(15, 24, 36, 0.96);
      border: 1px solid rgba(255,255,255,.08);
      color: var(--ink);
    }}
    .chat-message.system .chat-bubble {{
      background: rgba(103,217,255,.07);
      border: 1px dashed rgba(103,217,255,.22);
      color: var(--muted);
      font-size: 12px;
    }}
    .chat-bubble p {{
      margin: 0;
    }}
    .chat-meta {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
    }}
    .chat-citations {{
      margin: 8px 0 0;
      padding-left: 18px;
      color: #cfe0ea;
      font-size: 12px;
    }}
    .chat-starters {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
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
      padding: 10px;
      border: 1px solid rgba(255,255,255,.06);
      border-radius: 8px;
      background: rgba(10, 18, 28, 0.94);
    }}
    .chat-form textarea {{
      width: 100%;
      resize: none;
      min-height: 50px;
      max-height: 160px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(5, 10, 18, 0.96);
      color: var(--ink);
      font: inherit;
      padding: 10px;
    }}
    .chat-form-footer {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
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
    .chat-form button[disabled] {{
      opacity: 0.6;
      cursor: progress;
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
          <button type="button" class="mini-nav-btn" data-target="hypotheses-section">Hypotheses</button>
          <button type="button" class="mini-nav-btn" data-target="insights-section">Insights</button>
        </div>
        <p class="empty compact">Start with the graph. When hypotheses or insights are available, they will open below.</p>
      </div>
      <div class="fold-section" id="anomalies-section">
        <button type="button" class="fold-toggle" data-section="anomaly-list">
          <span>Conflicts</span>
          <span class="fold-meta" id="anomaly-count"></span>
        </button>
        <div class="fold-disabled-note" id="anomaly-disabled-note"></div>
        <div class="fold-body" id="anomaly-list"></div>
      </div>
      <div class="fold-section" id="hypotheses-section">
        <button type="button" class="fold-toggle" data-section="hypothesis-list">
          <span>Hypotheses</span>
          <span class="fold-meta" id="hypothesis-count"></span>
        </button>
        <div class="fold-disabled-note" id="hypothesis-disabled-note"></div>
        <div class="fold-body" id="hypothesis-list"></div>
      </div>
      <div class="fold-section" id="insights-section">
        <button type="button" class="fold-toggle" data-section="insight-list">
          <span>Insights</span>
          <span class="fold-meta" id="insight-count"></span>
        </button>
        <div class="fold-disabled-note" id="insight-disabled-note"></div>
        <div class="fold-body" id="insight-list"></div>
      </div>
    </aside>
    <section id="graph-wrap" aria-label="Graph visualization">
      <div class="graph-tools" aria-label="Graph controls">
        <div class="tool-group" aria-label="Zoom controls">
          <button type="button" id="zoom-out">−</button>
          <button type="button" id="zoom-in">+</button>
          <button type="button" id="zoom-fit">Fit</button>
          <button type="button" id="zoom-reset">Reset</button>
        </div>
        <div class="tool-divider" aria-hidden="true"></div>
        <div class="tool-group" aria-label="Evidence detail controls">
          <button type="button" id="detail-cluster">Clusters</button>
          <button type="button" id="detail-claims">Claims</button>
          <button type="button" id="detail-full">Full</button>
        </div>
        <div class="tool-divider" aria-hidden="true"></div>
        <div class="tool-group" aria-label="Layout controls">
          <button type="button" id="view-hierarchy">Hierarchy</button>
          <button type="button" id="view-free">Free</button>
        </div>
      </div>
      <div class="zoom-hint">Scroll to zoom · drag background to pan · start with Clusters, then open Claims and Full when you want more evidence detail</div>
      <svg id="graph"></svg>
      <div class="legend" id="legend"></div>
    </section>
    <div class="detail">
      <h2>Details</h2>
      <div id="detail" class="empty">Click a node, anomaly, hypothesis, or insight.</div>
      <div class="chat-panel" id="chat-panel">
        <div class="chat-header">
          <h2>Ask This Graph</h2>
          <p class="empty compact" id="chat-context">Ask about the whole run, or click a node first to make the answer selection-aware.</p>
          <div class="chat-toolbar">
            <div class="chat-selection" id="chat-selection-state">Currently asking about the whole run.</div>
            <div class="chat-toolbar-actions">
              <div class="chat-limit" id="chat-limit-note">Recent context only · latest 6 messages stay in scope.</div>
              <button type="button" class="chat-reset" id="graph-chat-clear" disabled>Ask about the whole run</button>
            </div>
          </div>
        </div>
        <div id="graph-chat-thread" class="chat-thread">
          <div class="chat-message system">
            <div class="chat-bubble">I only use this run's graph evidence and the most recent conversation turns, so the answers stay tight and grounded.</div>
          </div>
        </div>
        <div class="chat-starters" id="chat-starters">
          <button type="button" class="chat-chip">Why is this a conflict?</button>
          <button type="button" class="chat-chip">Which papers support this keyword?</button>
          <button type="button" class="chat-chip">What would resolve this disagreement?</button>
        </div>
        <form id="graph-chat-form" class="chat-form">
          <textarea id="graph-chat-input" rows="2" placeholder="Message this graph..." {"" if payload.get("run_id") else "disabled"}></textarea>
          <div class="chat-form-footer">
            <div class="empty compact" id="graph-chat-status">{'Graph chat is available for completed runs.' if payload.get("run_id") else 'Graph chat is disabled for the community-wide aggregate map.'}</div>
            <button type="submit" id="graph-chat-send" {"" if payload.get("run_id") else "disabled"}>Send</button>
          </div>
        </form>
      </div>
    </div>
  </main>
  <script type="application/json" id="aigraph-data">{data}</script>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script>
    const DATA = JSON.parse(document.getElementById('aigraph-data').textContent);
    const IS_COMMUNITY_GRAPH = DATA.graph_mode === 'community';
    const RUN_ID = DATA.run_id || '';
    const papersById = Object.fromEntries((DATA.papers || []).map(d => [d.paper_id, d]));
    const claimsById = Object.fromEntries((DATA.claims || []).map(d => [d.claim_id, d]));
    const anomaliesById = Object.fromEntries((DATA.anomalies || []).map(d => [d.anomaly_id, d]));
    const insightsById = Object.fromEntries((DATA.insights || []).map(d => [d.insight_id, d]));
    let selectedContext = null;
    let chatMessages = [];
    let chatPending = false;
    const CHAT_CONTEXT_WINDOW = 6;

    const typeColor = {{
      Paper: '#6b7280',
      Claim: '#2563eb',
      Method: '#059669',
      Task: '#d97706',
      Dataset: '#7c3aed',
      Metric: '#dc2626'
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

    function humanizeNodeType(type) {{
      return ({{
        Task: 'Research task',
        Method: 'Method family',
        Dataset: 'Dataset or resource',
        Metric: 'Evaluation metric'
      }})[type] || 'Concept';
    }}

    function nodeGloss(node) {{
      const glosses = {{
        Task: 'A problem setting or task that papers evaluate.',
        Method: 'A method family used to solve the task.',
        Dataset: 'A dataset or benchmark resource used by linked papers.',
        Metric: 'A metric used to score results across linked claims.'
      }};
      return glosses[node.node_type] || 'A concept node connected to claims and papers in this run.';
    }}

    function paperRoleMeta(paper) {{
      if (!paper) return '';
      const label = paper.paper_role_label || (paper.paper_role ? paper.paper_role.replace(/^./, c => c.toUpperCase()) : '');
      if (!label) return '';
      const why = paper.paper_role_explanation || '';
      return why ? `${{label}} · ${{why}}` : label;
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

    function linkedClaimsForNode(nodeId) {{
      const claimIds = new Set();
      for (const edge of (DATA.graph.edges || [])) {{
        const sourceId = endpointId(edge.source);
        const targetId = endpointId(edge.target);
        if (targetId === nodeId && String(sourceId).startsWith('Claim:')) {{
          claimIds.add(String(sourceId).replace(/^Claim:/, ''));
        }}
        if (sourceId === nodeId && String(targetId).startsWith('Claim:')) {{
          claimIds.add(String(targetId).replace(/^Claim:/, ''));
        }}
      }}
      return Array.from(claimIds).map(cid => claimsById[cid]).filter(Boolean);
    }}

    function pills(obj) {{
      if (!obj || !Object.keys(obj).length) return '';
      return Object.entries(obj).map(([k, v]) => `<span class="pill">${{esc(k)}}=${{esc(v)}}</span>`).join('');
    }}

    function showDetail(html) {{
      document.getElementById('detail').className = '';
      document.getElementById('detail').innerHTML = html;
    }}

    function trimChatHistory() {{
      return chatMessages
        .filter(msg => msg.role === 'user' || msg.role === 'assistant')
        .slice(-CHAT_CONTEXT_WINDOW)
        .map(msg => ({{
          role: msg.role,
          content: msg.content
        }}));
    }}

    function renderChatThread() {{
      const threadEl = document.getElementById('graph-chat-thread');
      if (!threadEl) return;
      const welcome = `<div class="chat-message system"><div class="chat-bubble">I only use this run's graph evidence and the most recent conversation turns, so the answers stay tight and grounded.</div></div>`;
      const body = chatMessages.map(msg => {{
        const refs = (msg.citations || []).map(ref => {{
          const label = ref.title || ref.claim_id || ref.paper_id || ref.id || 'reference';
          return `<li>${{esc(label)}}</li>`;
        }}).join('');
        const meta = msg.meta ? `<div class="chat-meta">${{esc(msg.meta)}}</div>` : '';
        const citations = refs ? `<ul class="chat-citations">${{refs}}</ul>` : '';
        return `<div class="chat-message ${{esc(msg.role)}}"><div class="chat-bubble"><p>${{esc(msg.content)}}</p>${{meta}}${{citations}}</div></div>`;
      }}).join('');
      threadEl.innerHTML = welcome + body;
      threadEl.scrollTop = threadEl.scrollHeight;
    }}

    function pushChatMessage(role, content, options = {{}}) {{
      chatMessages.push({{
        role,
        content,
        citations: options.citations || [],
        meta: options.meta || ''
      }});
      if (chatMessages.length > 18) {{
        chatMessages = chatMessages.slice(-18);
      }}
      renderChatThread();
    }}

    function updatePendingState(pending) {{
      chatPending = pending;
      const sendBtn = document.getElementById('graph-chat-send');
      const input = document.getElementById('graph-chat-input');
      const statusEl = document.getElementById('graph-chat-status');
      if (sendBtn) {{
        sendBtn.disabled = pending || !RUN_ID;
        sendBtn.textContent = pending ? 'Thinking…' : 'Send';
      }}
      if (input) input.disabled = pending || !RUN_ID;
      if (statusEl) {{
        statusEl.textContent = pending
          ? 'Reading only the selected evidence and recent chat turns...'
          : (RUN_ID ? 'Graph chat is available for completed runs.' : 'Graph chat is disabled for the community-wide aggregate map.');
      }}
    }}

    function syncChatContext() {{
      const contextEl = document.getElementById('chat-context');
      const selectionEl = document.getElementById('chat-selection-state');
      const clearBtn = document.getElementById('graph-chat-clear');
      const limitNote = document.getElementById('chat-limit-note');
      if (contextEl) {{
        contextEl.textContent = selectedContext?.label
          ? `Selection-aware mode is on for ${{selectedContext.label}}.`
          : 'Ask about the whole run, or click a node first to make the answer selection-aware.';
      }}
      if (selectionEl) {{
        selectionEl.textContent = selectedContext?.label
          ? `Currently asking about ${{selectedContext.label}}.`
          : 'Currently asking about the whole run.';
      }}
      if (clearBtn) {{
        clearBtn.disabled = !selectedContext;
      }}
      if (limitNote) {{
        limitNote.textContent = selectedContext?.label
          ? `Recent context only · latest ${{CHAT_CONTEXT_WINDOW}} messages + selected evidence.`
          : `Recent context only · latest ${{CHAT_CONTEXT_WINDOW}} messages stay in scope.`;
      }}
    }}

    function setSelectedContext(context) {{
      selectedContext = context || null;
      syncChatContext();
    }}

    function clearSelectedContext() {{
      selectedContext = null;
      document.querySelectorAll('button.item').forEach(btn => btn.classList.remove('active'));
      clearHighlight();
      syncChatContext();
    }}

    function titleCase(text) {{
      return String(text || '')
        .replace(/_/g, ' ')
        .toLowerCase()
        .replace(/\\b\\w/g, char => char.toUpperCase());
    }}

    function indexLabel(prefix, id, fallbackIndex) {{
      const match = String(id || '').match(/(\\d+)/);
      const number = match ? Number(match[1]) : fallbackIndex;
      return `${{prefix}} #${{number}}`;
    }}

    function humanizeAnomalyLabel(anomaly) {{
      return `${{indexLabel('Conflict', anomaly.anomaly_id, 1)}} · ${{titleCase(anomaly.type || 'Conflict')}}`;
    }}

    function humanizeHypothesisLabel(hypothesis) {{
      return `${{indexLabel('Hypothesis', hypothesis.hypothesis_id, 1)}}`;
    }}

    function humanizeInsightLabel(insight) {{
      return `${{indexLabel('Insight', insight.insight_id, 1)}} · ${{titleCase(insight.type || 'Insight')}}`;
    }}

    function renderSummary() {{
      const s = DATA.summary;
      const items = [
        ['Papers', s.papers, true],
        ['Claims', s.claims, true],
        ['Nodes', s.nodes, true],
        ['Edges', s.edges, true],
        ['Conflicts', s.anomalies, (s.anomalies || 0) > 0],
        ['Hypotheses', s.hypotheses, (s.hypotheses || 0) > 0],
        ['Insights', s.insights || 0, (s.insights || 0) > 0]
      ].filter(([, , visible]) => visible);
      document.getElementById('summary').innerHTML = items.map(([label, value]) =>
        `<div class="stat"><strong>${{value}}</strong><span>${{label}}</span></div>`
      ).join('');
    }}

    function renderLists() {{
      const sectionState = {{
        anomalies: {{
          count: DATA.anomalies.length,
          emptyMessage: 'No anomalies found.',
          disabledMessage: 'No conflicts were detected in this run.'
        }},
        hypotheses: {{
          count: DATA.hypotheses.length,
          emptyMessage: 'No hypotheses generated.',
          disabledMessage: 'No hypotheses generated yet for this run.'
        }},
        insights: {{
          count: (DATA.insights || []).length,
          emptyMessage: 'No community insights generated.',
          disabledMessage: 'No community insights generated yet for this run.'
        }}
      }};
      const anomalyList = document.getElementById('anomaly-list');
      anomalyList.innerHTML = DATA.anomalies.length ? DATA.anomalies.map(a => `
        <button class="item" data-kind="anomaly" data-id="${{esc(a.anomaly_id)}}">
          <div class="item-title">${{esc(humanizeAnomalyLabel(a))}}</div>
          <div class="item-meta">${{esc(a.central_question)}}</div>
          <div class="item-meta">${{a.claim_ids.length}} claims · +${{a.positive_claims.length}} / -${{a.negative_claims.length}}</div>
        </button>
      `).join('') : `<div class="empty">${{sectionState.anomalies.emptyMessage}}</div>`;

      const hypothesisList = document.getElementById('hypothesis-list');
      const shownHypotheses = DATA.hypotheses.slice(0, Math.min(DATA.hypotheses.length, 10));
      const hiddenHypothesisCount = Math.max(0, DATA.hypotheses.length - shownHypotheses.length);
      hypothesisList.innerHTML = DATA.hypotheses.length ? shownHypotheses.map(h => `
        <button class="item" data-kind="hypothesis" data-id="${{esc(h.hypothesis_id)}}">
          <div class="item-title">${{esc(humanizeHypothesisLabel(h))}}</div>
          <div class="item-meta">${{esc(h.hypothesis)}}</div>
        </button>
      `).join('') + (hiddenHypothesisCount ? `<div class="empty">Showing ${{shownHypotheses.length}} of ${{DATA.hypotheses.length}} hypotheses to keep the map readable.</div>` : '') : `<div class="empty">${{sectionState.hypotheses.emptyMessage}}</div>`;

      const insightList = document.getElementById('insight-list');
      insightList.innerHTML = (DATA.insights || []).length ? DATA.insights.map(i => `
        <button class="item" data-kind="insight" data-id="${{esc(i.insight_id)}}">
          <div class="item-title">${{esc(humanizeInsightLabel(i))}}</div>
          <div class="item-meta">${{esc(i.title)}}</div>
          <div class="item-meta">${{(i.communities || []).map(esc).join(' ↔ ')}}</div>
        </button>
      `).join('') : `<div class="empty">${{sectionState.insights.emptyMessage}}</div>`;

      const sections = [
        {{
          key: 'anomalies',
          sectionId: 'anomalies-section',
          sectionLabel: 'Conflicts',
          countId: 'anomaly-count',
          noteId: 'anomaly-disabled-note',
          miniNavTarget: 'anomalies-section',
          disabledMessage: sectionState.anomalies.disabledMessage
        }},
        {{
          key: 'hypotheses',
          sectionId: 'hypotheses-section',
          sectionLabel: 'Hypotheses',
          countId: 'hypothesis-count',
          noteId: 'hypothesis-disabled-note',
          miniNavTarget: 'hypotheses-section',
          disabledMessage: sectionState.hypotheses.disabledMessage
        }},
        {{
          key: 'insights',
          sectionId: 'insights-section',
          sectionLabel: 'Insights',
          countId: 'insight-count',
          noteId: 'insight-disabled-note',
          miniNavTarget: 'insights-section',
          disabledMessage: sectionState.insights.disabledMessage
        }}
      ];

      sections.forEach(section => {{
        const meta = sectionState[section.key];
        const sectionEl = document.getElementById(section.sectionId);
        const countEl = document.getElementById(section.countId);
        const noteEl = document.getElementById(section.noteId);
        const toggleEl = sectionEl?.querySelector('.fold-toggle');
        const bodyEl = sectionEl?.querySelector('.fold-body');
        const miniNavEl = document.querySelector(`.mini-nav-btn[data-target="${{section.miniNavTarget}}"]`);
        const hasData = meta.count > 0;
        if (countEl) countEl.textContent = hasData ? `${{meta.count}} ready` : 'Not available yet';
        if (noteEl) noteEl.textContent = hasData ? '' : section.disabledMessage;
        if (sectionEl) {{
          sectionEl.classList.toggle('open', hasData && section.key === 'anomalies');
          sectionEl.classList.toggle('disabled', !hasData);
        }}
        if (toggleEl) {{
          toggleEl.disabled = !hasData;
          toggleEl.title = hasData ? `Open ${{section.sectionLabel}}` : section.disabledMessage;
          toggleEl.setAttribute('aria-disabled', String(!hasData));
          toggleEl.onclick = hasData ? () => {{
            sectionEl.classList.toggle('open');
          }} : null;
        }}
        if (miniNavEl) {{
          miniNavEl.disabled = !hasData;
          miniNavEl.title = hasData ? `Jump to ${{section.sectionLabel}}` : section.disabledMessage;
          miniNavEl.setAttribute('aria-disabled', String(!hasData));
          miniNavEl.onclick = hasData ? () => {{
            sectionEl.classList.add('open');
            sectionEl.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
          }} : null;
        }}
      }});

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
      setSelectedContext({{
        kind: 'node',
        id: node.id,
        label: nodeLabel(node)
      }});
      const id = node.id || '';
      let html = `<h2>${{esc(node.node_type || 'Node')}}</h2>`;
      if (node.node_type === 'Paper') {{
        const paper = paperFromNode(node) || {{}};
        html += field('title', paper.title || id);
        html += field('paper role', paper.paper_role_label || '');
        html += field('role context', paper.paper_role_explanation || '');
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
        const linkedClaims = linkedClaimsForNode(id);
        const linkedPapers = Array.from(new Map(
          linkedClaims
            .map(c => papersById[c.paper_id])
            .filter(Boolean)
            .map(p => [p.paper_id, p])
        ).values());
        const claimRows = linkedClaims.map(c => {{
          const p = papersById[c.paper_id] || {{}};
          const meta = paperRoleMeta(p);
          return `<div class="field"><div class="label">${{esc(c.claim_id)}} · ${{esc(c.direction)}}</div><div class="value">${{esc(c.claim_text)}}<br><span class="muted-inline">${{paperLink(p, c.paper_id)}}${{meta ? ' · ' + esc(meta) : ''}}</span></div></div>`;
        }}).join('');
        const paperRows = linkedPapers.map(p => {{
          const meta = [paperRoleMeta(p), p.retrieval_channel || '', p.selection_score ? `score ${{Number(p.selection_score).toFixed(2)}}` : ''].filter(Boolean).join(' · ');
          return `<div class="field"><div class="label">${{esc(p.paper_id)}}</div><div class="value">${{paperLink(p, p.paper_id)}}${{meta ? '<br><span class="muted-inline">' + esc(meta) + '</span>' : ''}}</div></div>`;
        }}).join('');
        html += field('label', nodeLabel(node));
        html += field('what this means', nodeGloss(node));
        html += field('node type', humanizeNodeType(node.node_type));
        html += field('raw id', id);
        html += `<h2>Connected Claims</h2>${{claimRows || '<div class="empty">No connected claims found.</div>'}}`;
        html += `<h2>Source Papers</h2>${{paperRows || '<div class="empty">No linked papers found.</div>'}}`;
      }}
      showDetail(html);
    }}

    function renderAnomaly(a) {{
      setSelectedContext({{
        kind: 'conflict',
        id: a.anomaly_id,
        label: humanizeAnomalyLabel(a)
      }});
      const groupedByPaper = new Map();
      a.claim_ids.map(cid => claimsById[cid]).filter(Boolean).forEach(c => {{
        const key = c.paper_id || 'unknown-paper';
        if (!groupedByPaper.has(key)) groupedByPaper.set(key, []);
        groupedByPaper.get(key).push(c);
      }});
      const claimRows = Array.from(groupedByPaper.entries()).map(([paperId, groupedClaims]) => {{
        const p = papersById[paperId] || {{}};
        const claimList = groupedClaims.map(c =>
          `<li><strong>${{esc(c.direction || 'claim')}}</strong> · ${{esc(c.claim_text)}}${{c.evidence_span ? `<br><span class="muted-inline">${{esc(c.evidence_span)}}</span>` : ''}}</li>`
        ).join('');
        return `<div class="field"><div class="label">${{paperLink(p, paperId)}}${{groupedClaims.length > 1 ? ` · ${{groupedClaims.length}} claims` : ' · 1 claim'}}</div><div class="value"><ul>${{claimList}}</ul></div></div>`;
      }}).join('');
      showDetail(`
        <h2>${{esc(humanizeAnomalyLabel(a))}}</h2>
        ${{field('question', a.central_question)}}
        <div class="field"><div class="label">shared entities</div><div class="value">${{pills(a.shared_entities)}}</div></div>
        ${{a.varying_settings.length ? field('varying settings', a.varying_settings.join(', ')) : ''}}
        ${{field('evidence impact', a.evidence_impact ? Number(a.evidence_impact).toFixed(2) : '')}}
        ${{field('recent activity', a.recent_activity ? Number(a.recent_activity).toFixed(2) : '')}}
        ${{field('topology score', a.topology_score ? Number(a.topology_score).toFixed(2) : '')}}
        ${{field('raw id', a.anomaly_id)}}
        <h2>Claims</h2>
        ${{claimRows || '<div class="empty">No claim details.</div>'}}
      `);
    }}

    function renderHypothesis(h) {{
      setSelectedContext({{
        kind: 'hypothesis',
        id: h.hypothesis_id,
        label: humanizeHypothesisLabel(h)
      }});
      const a = anomaliesById[h.anomaly_id];
      const claims = (h.explains_claims || []).map(cid => claimsById[cid]).filter(Boolean).map(c =>
        `<div class="field"><div class="label">${{esc(c.claim_id)}} · ${{esc(c.direction)}}</div><div class="value">${{esc(c.claim_text)}}</div></div>`
      ).join('');
      const preds = (h.predictions || []).map(p => `<li>${{esc(p)}}</li>`).join('');
      showDetail(`
        <h2>${{esc(humanizeHypothesisLabel(h))}}</h2>
        ${{field('conflict', a ? humanizeAnomalyLabel(a) : h.anomaly_id)}}
        ${{field('hypothesis', h.hypothesis)}}
        ${{field('mechanism', h.mechanism)}}
        <div class="field"><div class="label">predictions</div><div class="value"><ul>${{preds}}</ul></div></div>
        ${{field('minimal test', h.minimal_test)}}
        ${{field('evidence gap', h.evidence_gap)}}
        ${{field('raw id', h.hypothesis_id)}}
        <h2>Claims</h2>
        ${{claims || '<div class="empty">No linked claims.</div>'}}
      `);
    }}

    function renderInsight(i) {{
      setSelectedContext({{
        kind: 'insight',
        id: i.insight_id,
        label: humanizeInsightLabel(i)
      }});
      const concepts = (i.shared_concepts || []).map(c => `<span class="pill">${{esc(c)}}</span>`).join('');
      const papers = (i.evidence_papers || []).map(pid => {{
        const p = papersById[pid] || {{}};
        const meta = [paperRoleMeta(p), p.retrieval_channel, p.selection_score ? `score ${{Number(p.selection_score).toFixed(2)}}` : '', p.selection_reason].filter(Boolean).join(' · ');
        return `<div class="field"><div class="label">${{esc(pid)}}</div><div class="value">${{paperLink(p, pid)}}${{meta ? '<br><span class="muted">' + esc(meta) + '</span>' : ''}}</div></div>`;
      }}).join('');
      const suggestions = (i.transfer_suggestions || []).map(s => `<li>${{esc(s)}}</li>`).join('');
      showDetail(`
        <h2>${{esc(humanizeInsightLabel(i))}}</h2>
        ${{field('title', i.title)}}
        ${{field('communities', (i.communities || []).join(' ↔ '))}}
        <div class="field"><div class="label">shared concepts</div><div class="value">${{concepts}}</div></div>
        ${{field('insight', i.insight)}}
        ${{field('unifying frame', i.unifying_frame)}}
        ${{field('citation gap', i.citation_gap)}}
        <div class="field"><div class="label">transfer suggestions</div><div class="value"><ul>${{suggestions}}</ul></div></div>
        ${{field('scores', `impact=${{Number(i.impact_score || 0).toFixed(2)}}, topology=${{Number(i.topology_score || 0).toFixed(2)}}, confidence=${{Number(i.confidence_score || 0).toFixed(2)}}`)}}
        ${{field('raw id', i.insight_id)}}
        <h2>Evidence Papers</h2>
        ${{papers || '<div class="empty">No linked papers.</div>'}}
      `);
    }}

    async function submitGraphChat(question) {{
      if (!RUN_ID) return;
      pushChatMessage('user', question, {{
        meta: selectedContext?.label ? `Context: ${{selectedContext.label}}` : 'Context: whole run'
      }});
      updatePendingState(true);
      try {{
        const resp = await fetch('/api/graph-chat', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            run_id: RUN_ID,
            question,
            selection: selectedContext || {{}},
            history: trimChatHistory()
          }})
        }});
        let data = null;
        let parseError = null;
        try {{
          data = await resp.json();
        }} catch (err) {{
          parseError = err;
        }}
        updatePendingState(false);
        if (!resp.ok) {{
          const serverMsg = (data && data.error) || `Graph chat failed (HTTP ${{resp.status}}).`;
          pushChatMessage('assistant', serverMsg, {{
            meta: 'The graph context stayed unchanged.'
          }});
          return;
        }}
        if (parseError || !data) {{
          pushChatMessage('assistant', 'Graph chat returned an unreadable response.', {{
            meta: 'The graph context stayed unchanged.'
          }});
          return;
        }}
        pushChatMessage('assistant', data.answer || 'No answer available.', {{
          citations: data.citations || data.references || [],
          meta: selectedContext?.label ? `Grounded in ${{selectedContext.label}}` : 'Grounded in the whole run'
        }});
        syncChatContext();
      }} catch (error) {{
        updatePendingState(false);
        pushChatMessage('assistant', 'Graph chat failed.', {{
          meta: 'The graph context stayed unchanged.'
        }});
      }}
    }}

    function wireGraphChat() {{
      const form = document.getElementById('graph-chat-form');
      const input = document.getElementById('graph-chat-input');
      const clearBtn = document.getElementById('graph-chat-clear');
      // Always normalize chip state, even when chat is disabled: in community
      // mode the textarea is statically `disabled`, but the chip buttons would
      // otherwise stay clickable and fire a request that always fails.
      document.querySelectorAll('.chat-chip').forEach(btn => {{
        const label = btn.textContent.trim();
        if (label) btn.setAttribute('aria-label', `Ask: ${{label}}`);
        if (!RUN_ID) {{
          btn.disabled = true;
          btn.setAttribute('aria-disabled', 'true');
          btn.title = 'Graph chat is disabled for the community-wide aggregate map.';
        }}
      }});
      if (!form || !input || !RUN_ID) return;
      syncChatContext();
      renderChatThread();
      updatePendingState(false);
      input.addEventListener('input', () => {{
        input.style.height = 'auto';
        input.style.height = `${{Math.min(input.scrollHeight, 160)}}px`;
      }});
      input.addEventListener('keydown', async event => {{
        if (event.key === 'Enter' && !event.shiftKey) {{
          event.preventDefault();
          if (chatPending) return;
          const question = input.value.trim();
          if (!question) return;
          input.value = '';
          input.style.height = 'auto';
          await submitGraphChat(question);
        }}
      }});
      form.addEventListener('submit', async event => {{
        event.preventDefault();
        if (chatPending) return;
        const question = input.value.trim();
        if (!question) return;
        input.value = '';
        input.style.height = 'auto';
        await submitGraphChat(question);
      }});
      document.querySelectorAll('.chat-chip').forEach(btn => {{
        btn.addEventListener('click', async () => {{
          if (btn.disabled || chatPending) return;
          const question = btn.textContent.trim();
          if (!question) return;
          input.value = '';
          input.style.height = 'auto';
          await submitGraphChat(question);
        }});
      }});
      if (clearBtn) {{
        clearBtn.addEventListener('click', () => {{
          clearSelectedContext();
          input.focus();
        }});
      }}
    }}

    let svg, viewport, zoomBehavior, nodeSel, linkSel, labelSel, currentWidth = 0, currentHeight = 0;
    let currentLayout = 'hierarchy';
    let currentDetail = 'cluster';

    const hierarchyLanes = IS_COMMUNITY_GRAPH
      ? [
          {{ key: 'keyword', label: 'Keyword', types: ['Task', 'Method'] }},
          {{ key: 'claim', label: 'Claims', types: ['Claim'] }},
          {{ key: 'paper', label: 'Papers', types: ['Paper'] }},
        ]
      : [
          {{ key: 'topic', label: 'Topic', types: ['Task'] }},
          {{ key: 'method', label: 'Method', types: ['Method'] }},
          {{ key: 'evidence', label: 'Evidence', types: ['Paper', 'Claim'] }},
          {{ key: 'evaluation', label: 'Evaluation', types: ['Dataset', 'Metric'] }},
        ];
    const hierarchyTypeToLane = Object.fromEntries(
      hierarchyLanes.flatMap((lane, index) => lane.types.map(type => [type, index]))
    );
    const clusterNodeTypes = new Set(['Method', 'Task']);
    const claimsNodeTypes = new Set(['Paper', 'Claim', 'Method', 'Task']);

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
      svg.on('click', event => {{
        const target = event.target;
        if (target?.closest?.('.node')) return;
        clearSelectedContext();
      }});

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

      // Scale force parameters with graph size so large graphs (>100 nodes)
      // get enough repulsion + collision padding to actually converge.
      const sizeScale = Math.max(1, Math.sqrt(Math.max(1, nodes.length) / 50));
      const baseCharge = currentDetail === 'cluster' ? -280 : (currentDetail === 'claims' ? -240 : -320);
      const collisionPad = Math.min(18, Math.round((sizeScale - 1) * 6));
      const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(d => d.edge_type === 'makes' ? 54 : (d.edge_type === 'cites' ? 96 : 82)))
        .force('charge', d3.forceManyBody().strength(baseCharge * sizeScale))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => (d.node_type === 'Paper' ? 34 : 26) + collisionPad))
        .force('x', d3.forceX(d => currentLayout === 'hierarchy' ? hierarchyTargetX(d, width) : width / 2)
          .strength(currentLayout === 'hierarchy' ? 0.34 : 0.05))
        .force('y', d3.forceY(d => currentLayout === 'hierarchy' ? hierarchyTargetY(d, height) : height / 2)
          .strength(currentLayout === 'hierarchy' ? 0.12 : 0.05));
      // Bigger graphs need a slightly faster cool-down so the layout actually
      // settles within the user's attention span.
      const decayBoost = Math.min(0.04, (sizeScale - 1) * 0.02);
      simulation.alphaDecay((currentDetail === 'full' ? 0.06 : 0.09) + decayBoost);
      simulation.velocityDecay((currentDetail === 'full' ? 0.34 : 0.42) + Math.min(0.1, (sizeScale - 1) * 0.04));

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
      const types = ['Paper', 'Claim', 'Method', 'Task', 'Dataset', 'Metric'];
      document.getElementById('legend').innerHTML = types.map(t =>
        `<span><i class="dot" style="background:${{typeColor[t]}}"></i>${{t}}</span>`
      ).join('') + '<span>-- citation edge</span>';
    }}

    renderSummary();
    renderLists();
    renderLegend();
    updateViewButtons();
    wireGraphChat();
    syncChatContext();
    drawGraph();
    window.addEventListener('resize', () => drawGraph());
    if (DATA.anomalies.length) selectAnomaly(DATA.anomalies[0].anomaly_id);
  </script>
</body>
</html>
"""
