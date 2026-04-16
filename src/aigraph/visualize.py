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
    papers = [_augment_paper_links(p) for p in _read_jsonl(input_dir / "papers.jsonl")]
    claims = _read_jsonl(input_dir / "claims.jsonl")
    anomalies = _read_jsonl(input_dir / "anomalies.jsonl")
    hypotheses = _read_jsonl(input_dir / "hypotheses.jsonl")

    return {
        "summary": {
            "papers": len(papers),
            "claims": len(claims),
            "nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
            "anomalies": len(anomalies),
            "hypotheses": len(hypotheses),
        },
        "graph": graph,
        "papers": papers,
        "claims": claims,
        "anomalies": anomalies,
        "hypotheses": hypotheses,
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
      --paper: #6b7280;
      --claim: #2563eb;
      --method: #059669;
      --task: #d97706;
      --dataset: #7c3aed;
      --metric: #dc2626;
      --baseline: #9333ea;
      --setting: #0891b2;
      --edge: #a3a3a3;
      --ink: #171717;
      --muted: #666666;
      --line: #d9d9d9;
      --panel: #ffffff;
      --bg: #f7f7f7;
      --highlight: #ef4444;
      --soft: #f1f5f9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      letter-spacing: 0;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: #ffffff;
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
      background: #ffffff;
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
      background: #ffffff;
      padding: 14px;
      overflow: auto;
      max-height: calc(100vh - 82px);
    }}
    .detail {{
      border-left: 1px solid var(--line);
      background: #ffffff;
      padding: 14px;
      overflow: auto;
      max-height: calc(100vh - 82px);
    }}
    h2 {{
      font-size: 14px;
      margin: 16px 0 8px;
      text-transform: uppercase;
      color: #3f3f46;
    }}
    h2:first-child {{ margin-top: 0; }}
    button.item {{
      display: block;
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      border-radius: 8px;
      padding: 9px;
      margin: 7px 0;
      cursor: pointer;
      font: inherit;
    }}
    button.item:hover, button.item.active {{
      border-color: #2563eb;
      background: var(--soft);
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
      background: #fbfbfb;
    }}
    #graph {{
      display: block;
      width: 100%;
      height: calc(100vh - 82px);
    }}
    .legend {{
      position: absolute;
      left: 12px;
      bottom: 12px;
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      max-width: calc(100% - 24px);
      background: rgba(255, 255, 255, 0.92);
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
      stroke: #ffffff;
      stroke-width: 1.5px;
      cursor: pointer;
    }}
    .node text {{
      pointer-events: none;
      font-size: 10px;
      fill: #262626;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 3px;
      stroke-linejoin: round;
    }}
    .link {{
      stroke: var(--edge);
      stroke-opacity: 0.55;
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
      border-bottom: 1px solid #eeeeee;
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
      color: #2563eb;
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
      color: #262626;
      background: #ffffff;
    }}
    .empty {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
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
      <h2>Conflicts & Gaps</h2>
      <div id="anomaly-list"></div>
      <h2>Possible Explanations</h2>
      <div id="hypothesis-list"></div>
    </aside>
    <section id="graph-wrap" aria-label="Graph visualization">
      <svg id="graph"></svg>
      <div class="legend" id="legend"></div>
    </section>
    <div class="detail">
      <h2>Details</h2>
      <div id="detail" class="empty">Click a node, anomaly, or hypothesis.</div>
    </div>
  </main>
  <script type="application/json" id="aigraph-data">{data}</script>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <script>
    const DATA = JSON.parse(document.getElementById('aigraph-data').textContent);
    const papersById = Object.fromEntries(DATA.papers.map(d => [d.paper_id, d]));
    const claimsById = Object.fromEntries(DATA.claims.map(d => [d.claim_id, d]));
    const anomaliesById = Object.fromEntries(DATA.anomalies.map(d => [d.anomaly_id, d]));

    const typeColor = {{
      Paper: '#6b7280',
      Claim: '#2563eb',
      Method: '#059669',
      Model: '#16a34a',
      Task: '#d97706',
      Dataset: '#7c3aed',
      Metric: '#dc2626',
      Baseline: '#9333ea',
      Setting: '#0891b2'
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
        ['Edges', s.edges], ['Anomalies', s.anomalies], ['Hypotheses', s.hypotheses]
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
      hypothesisList.innerHTML = DATA.hypotheses.length ? DATA.hypotheses.map(h => `
        <button class="item" data-kind="hypothesis" data-id="${{esc(h.hypothesis_id)}}">
          <div class="item-title">${{esc(h.hypothesis_id)}} · ${{esc(h.anomaly_id)}}</div>
          <div class="item-meta">${{esc(h.hypothesis)}}</div>
        </button>
      `).join('') : '<div class="empty">No hypotheses generated.</div>';

      document.querySelectorAll('button.item').forEach(btn => {{
        btn.addEventListener('click', () => {{
          document.querySelectorAll('button.item').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          if (btn.dataset.kind === 'anomaly') selectAnomaly(btn.dataset.id);
          if (btn.dataset.kind === 'hypothesis') selectHypothesis(btn.dataset.id);
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

    let svg, nodeSel, linkSel, labelSel;

    function edgeKey(e) {{
      const s = typeof e.source === 'object' ? e.source.id : e.source;
      const t = typeof e.target === 'object' ? e.target.id : e.target;
      return `${{s}}|${{t}}|${{e.edge_type || ''}}`;
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

    function drawGraph() {{
      const wrap = document.getElementById('graph-wrap');
      const width = Math.max(420, wrap.clientWidth);
      const height = Math.max(560, wrap.clientHeight);
      const nodes = (DATA.graph.nodes || []).map(d => ({{...d}}));
      const links = (DATA.graph.edges || []).map(d => ({{...d}}));

      svg = d3.select('#graph').attr('viewBox', [0, 0, width, height]);
      svg.selectAll('*').remove();

      linkSel = svg.append('g')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('class', 'link')
        .attr('stroke-width', d => d.edge_type === 'contradicts' ? 2.2 : 1.2);

      linkSel.append('title').text(d => `${{d.source}} → ${{d.target}} · ${{d.edge_type || 'related'}}`);

      const node = svg.append('g')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .attr('class', 'node')
        .call(d3.drag()
          .on('start', dragstarted)
          .on('drag', dragged)
          .on('end', dragended));

      node.append('circle')
        .attr('r', d => d.node_type === 'Claim' ? 7 : 5.5)
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
        .force('link', d3.forceLink(links).id(d => d.id).distance(d => d.edge_type === 'makes' ? 42 : 72))
        .force('charge', d3.forceManyBody().strength(-230))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(24));

      simulation.on('tick', () => {{
        linkSel
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);
        nodeSel.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
      }});

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
    }}

    function renderLegend() {{
      const types = ['Paper', 'Claim', 'Method', 'Task', 'Dataset', 'Metric', 'Baseline', 'Setting'];
      document.getElementById('legend').innerHTML = types.map(t =>
        `<span><i class="dot" style="background:${{typeColor[t]}}"></i>${{t}}</span>`
      ).join('');
    }}

    renderSummary();
    renderLists();
    renderLegend();
    drawGraph();
    window.addEventListener('resize', () => drawGraph());
    if (DATA.anomalies.length) selectAnomaly(DATA.anomalies[0].anomaly_id);
  </script>
</body>
</html>
"""
