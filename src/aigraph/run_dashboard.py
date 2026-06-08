"""Runs dashboard for the aigraph MCP web UI (NON-FROZEN).

Answers three operator questions about the MCP:
  * which requests/runs exist (each ``start_run`` writes a run dir),
  * how each one ran (the pipeline stages, counts, and quality flags it
    produced), and
  * what the flow is (the canonical pipeline, stage by stage).

It reconstructs each run's flow from the merged ``status.json`` plus the
artifacts present in the run dir (``papers.jsonl``, ``claims.jsonl``,
``anomalies.jsonl``, the #49/#51 artifacts, …). Pure stdlib + HTML strings (no
``markdown`` import) so it stays unit-testable without the web stack.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Canonical pipeline order (mirrors server.run_pipeline). Each stage maps to the
# artifact it writes and/or the status count it reports, so we can show what each
# stage actually produced for a given run.
_STAGES: list[dict[str, Any]] = [
    {"key": "fetch", "label": "Fetch papers", "artifact": "papers.jsonl", "count": "papers",
     "note": "arXiv / OpenAlex union retrieval (#46/#47/#48)"},
    {"key": "gate", "label": "Semantic relevance gate", "artifact": None, "count": None,
     "note": "LLM 0-3 topical filter + re-rank (#58)"},
    {"key": "extract", "label": "Extract claims", "artifact": "claims.jsonl", "count": "claims",
     "note": "per-paper LLM extraction + retry (#49)"},
    {"key": "canonicalize", "label": "Run-local taxonomy", "artifact": None, "count": None,
     "note": "re-canonicalize 'other' method/task (#50)"},
    {"key": "graph", "label": "Build conflict graph", "artifact": "graph.json", "count": "nodes",
     "note": "claims → entity/claim graph"},
    {"key": "anomalies", "label": "Detect anomalies", "artifact": "anomalies.jsonl", "count": "anomalies",
     "note": "conflicts / gaps / bridges"},
    {"key": "open_questions", "label": "Open questions", "artifact": "open_questions.jsonl", "count": "open_questions",
     "note": "from papers' limitations (#51)"},
    {"key": "creator", "label": "Creator hypotheses", "artifact": "creator_hypotheses.jsonl", "count": "creator_hypotheses",
     "note": "anomalies + open questions → methods (#51)"},
    {"key": "hypotheses", "label": "Generate hypotheses", "artifact": "hypotheses.jsonl", "count": "hypotheses",
     "note": "templated from anomalies"},
    {"key": "insights", "label": "Community insights", "artifact": "insights.jsonl", "count": "insights",
     "note": "topology digests"},
    {"key": "render", "label": "Score & render", "artifact": "selected_hypotheses.md", "count": "selected",
     "note": "MMR selection + report"},
]

_COUNT_FIELDS = [
    "papers", "claims", "anomalies", "hypotheses", "open_questions",
    "creator_hypotheses", "insights", "selected", "nodes", "edges",
]


def _read_status(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "status.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _count_lines(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return 0


def _run_created(run_id: str) -> str:
    """Best-effort creation time from the run-id stamp ``YYYYMMDD-HHMMSS-...``."""
    parts = run_id.split("-")
    if len(parts) >= 2 and len(parts[0]) == 8 and parts[0].isdigit() and len(parts[1]) == 6 and parts[1].isdigit():
        d, t = parts[0], parts[1]
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]} {t[0:2]}:{t[2:4]}:{t[4:6]}"
    return ""


def run_summary(run_dir: Path) -> dict[str, Any]:
    """One-line summary of a run for the dashboard table."""
    st = _read_status(run_dir)
    counts = {k: st.get(k) for k in _COUNT_FIELDS if st.get(k) is not None}
    gate = st.get("semantic_gate") or {}
    return {
        "id": run_dir.name,
        "topic": st.get("topic") or st.get("retrieval_topic") or "",
        "status": st.get("status") or "unknown",
        "stage": st.get("stage") or "",
        "progress": st.get("progress"),
        "source": st.get("source") or "",
        "strategy": st.get("strategy") or "",
        "created": _run_created(run_dir.name),
        "updated": st.get("updated_at") or "",
        "counts": counts,
        "retrieval_quality": st.get("retrieval_quality"),
        "extraction_quality": st.get("extraction_quality"),
        "gate_kept": gate.get("kept") if isinstance(gate, dict) else None,
        "gate_before": gate.get("before") if isinstance(gate, dict) else None,
        "direction_bias_suspected": st.get("direction_bias_suspected"),
        "error": st.get("error_title") or st.get("error"),
        "message": st.get("message") or "",
    }


def pipeline_flow(run_dir: Path) -> dict[str, Any]:
    """Reconstruct the stage-by-stage flow of one run."""
    st = _read_status(run_dir)
    run_status = st.get("status") or "unknown"

    # pass 1: per-stage signals
    meta = []
    for idx, spec in enumerate(_STAGES):
        artifact = spec["artifact"]
        apath = run_dir / artifact if artifact else None
        exists = bool(apath and apath.exists())
        lines = _count_lines(apath) if exists else None
        count = st.get(spec["count"]) if spec["count"] else None
        produced = count if count is not None else lines
        has_output = bool((count is not None and count > 0) or (lines is not None and lines > 0))
        countable = bool(artifact or spec["count"])          # gate/canon aren't countable
        ran_marker = bool(exists or count is not None)       # left an (possibly empty) trace
        meta.append({"spec": spec, "exists": exists, "produced": produced,
                     "has_output": has_output, "countable": countable, "ran_marker": ran_marker})

    # how far the run demonstrably reached: furthest stage with real output, and
    # (for a running run) everything before the current stage.
    last_output = max((i for i, m in enumerate(meta) if m["has_output"]), default=-1)
    cur_key = st.get("stage")
    cur_idx = next((i for i, m in enumerate(meta) if m["spec"]["key"] == cur_key), -1) if run_status == "running" else -1
    reached = max(last_output, cur_idx - 1)

    stages = []
    for i, m in enumerate(meta):
        spec = m["spec"]
        if m["has_output"]:
            state = "done"
        elif i <= reached:
            # something downstream completed → this stage ran. An empty trace on a
            # countable stage = 'empty'; a no-artifact passthrough (gate/canon) = 'done'.
            state = "empty" if (m["countable"] and m["ran_marker"]) else "done"
        elif run_status == "running" and i == cur_idx:
            state = "active"
        elif run_status == "done":
            state = "empty"   # run finished; this stage yielded nothing
        else:
            state = "pending"
        stages.append({
            "key": spec["key"], "label": spec["label"], "note": spec["note"],
            "state": state, "produced": m["produced"],
            "artifact": spec["artifact"] if m["exists"] else None,
            "detail": _stage_detail(spec["key"], st, run_dir),
        })
    return {"id": run_dir.name, "summary": run_summary(run_dir), "stages": stages, "raw_status": st}


def _stage_detail(key: str, st: dict[str, Any], run_dir: Path) -> str:
    """A short human note on what this stage did for this run."""
    if key == "fetch":
        rq = st.get("retrieval_quality")
        attempts = st.get("retrieval_attempts") or []
        srcs = ", ".join(sorted({a.get("source", "?") for a in attempts if isinstance(a, dict)})) if attempts else ""
        before = st.get("papers_before_gate")
        bits = []
        if rq:
            bits.append(f"quality={rq}")
        if srcs:
            bits.append(f"sources={srcs}")
        if before is not None:
            bits.append(f"pre-gate={before}")
        return " · ".join(bits)
    if key == "gate":
        g = st.get("semantic_gate") or {}
        if isinstance(g, dict) and g.get("applied"):
            return f"kept {g.get('kept')}/{g.get('before')} (min_score={g.get('min_score')}, hist={g.get('score_histogram')})"
        return "not applied (no LLM key or disabled)" if g else ""
    if key == "extract":
        eq = st.get("extraction_quality")
        audit = _read_first_jsonl(run_dir / "extraction_audit.jsonl")
        bits = []
        if eq:
            bits.append(f"quality={eq}")
        if audit:
            bits.append(f"direction={audit.get('direction_distribution')}")
            if audit.get("positive_bias_suspected"):
                bits.append("⚠ positive-bias suspected")
        return " · ".join(bits)
    if key == "canonicalize":
        return "re-maps only 'other' method/task (fail-open)"
    if key == "graph":
        return f"nodes={st.get('nodes')} · edges={st.get('edges')}"
    if key == "anomalies":
        types = _count_by_field(run_dir / "anomalies.jsonl", "type")
        return ("types: " + ", ".join(f"{t}×{n}" for t, n in types)) if types else ""
    if key == "hypotheses":
        sel = st.get("selected")
        base = f"{st.get('hypotheses', 0)} generated"
        return base + (f" · {sel} selected" if sel is not None else "")
    if key == "render":
        sel = st.get("selected")
        return f"{sel} selected → selected_hypotheses.md" if sel is not None else ""
    return ""


def _count_by_field(path: Path, field: str) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    try:
        with path.open(encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                v = str(json.loads(ln).get(field) or "?")
                counts[v] = counts.get(v, 0) + 1
    except (OSError, json.JSONDecodeError):
        return []
    return sorted(counts.items(), key=lambda kv: -kv[1])


def _read_first_jsonl(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    return json.loads(ln)
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
                if limit and len(rows) >= limit:
                    break
    except OSError:
        return []
    return rows


def _paper_link(p: dict[str, Any]) -> str:
    """Best external link for a paper."""
    if p.get("url"):
        return str(p["url"])
    for k in ("arxiv_id_base", "arxiv_id_full"):
        if p.get(k):
            return f"https://arxiv.org/abs/{p[k]}"
    oa = p.get("openalex_id")
    if oa:
        return str(oa) if str(oa).startswith("http") else f"https://openalex.org/{oa}"
    if p.get("doi"):
        return f"https://doi.org/{p['doi']}"
    return ""


def run_ideas(run_dir: Path, limit: int = 24) -> list[dict[str, Any]]:
    """The run's hypotheses, each resolved to the source PAPERS it used
    (explains_claims → claim.paper_id → paper) + its anomaly + novelty audit."""
    hyps = _load_jsonl(run_dir / "hypotheses.jsonl") or _load_jsonl(run_dir / "hypotheses_scored.jsonl")
    claims = {c.get("claim_id"): c for c in _load_jsonl(run_dir / "claims.jsonl")}
    papers = {p.get("paper_id"): p for p in _load_jsonl(run_dir / "papers.jsonl")}
    anoms = {a.get("anomaly_id"): a for a in _load_jsonl(run_dir / "anomalies.jsonl")}
    out: list[dict[str, Any]] = []
    for h in hyps[:limit]:
        used_papers: list[dict[str, Any]] = []
        used_claims: list[dict[str, Any]] = []
        seen: set[str] = set()
        for cid in (h.get("explains_claims") or []):
            c = claims.get(cid)
            if not c:
                continue
            pid = c.get("paper_id")
            used_claims.append({"claim_id": cid, "text": (c.get("claim_text") or "")[:220],
                                "direction": c.get("direction"), "paper_id": pid})
            if pid and pid not in seen:
                seen.add(pid)
                p = papers.get(pid) or {"paper_id": pid}
                used_papers.append({"paper_id": pid, "title": p.get("title") or pid,
                                    "year": p.get("year"), "url": _paper_link(p)})
        an = anoms.get(h.get("anomaly_id")) or {}
        out.append({
            "id": h.get("hypothesis_id"), "hypothesis": h.get("hypothesis") or "",
            "mechanism": h.get("mechanism") or "", "predictions": h.get("predictions") or [],
            "minimal_test": h.get("minimal_test") or "", "novelty_audit": h.get("novelty_audit"),
            "anomaly_type": an.get("type") or "", "central_question": an.get("central_question") or "",
            "papers": used_papers, "claims": used_claims,
        })
    return out


def run_graph(run_dir: Path) -> dict[str, Any]:
    """Star graph (星球图) data: topic in the centre, hypotheses around it, each
    linked to its anomaly and the papers it used. Self-contained {nodes,edges}."""
    ideas = run_ideas(run_dir, limit=40)
    topic = _read_status(run_dir).get("topic") or run_dir.name
    nodes: list[dict[str, Any]] = [{"id": "topic", "label": topic[:60], "kind": "topic"}]
    seen = {"topic"}
    edges: list[dict[str, Any]] = []

    def add(nid: str, label: str, kind: str) -> None:
        if nid not in seen:
            seen.add(nid)
            nodes.append({"id": nid, "label": label, "kind": kind})

    for idea in ideas:
        hid = str(idea["id"])
        add(hid, (idea["hypothesis"] or hid)[:48], "hypothesis")
        edges.append({"source": "topic", "target": hid})
        if idea["anomaly_type"]:
            aid = "anom:" + (idea["central_question"] or idea["anomaly_type"])[:48]
            add(aid, idea["anomaly_type"], "anomaly")
            edges.append({"source": hid, "target": aid})
        for p in idea["papers"][:6]:
            pid = "p:" + str(p["paper_id"])
            add(pid, (p["title"] or "")[:42], "paper")
            edges.append({"source": hid, "target": pid})
    return {"nodes": nodes, "edges": edges, "topic": topic, "id": run_dir.name}


def discover_run_summaries(runs_root: Path) -> list[dict[str, Any]]:
    """All runs that have a status.json, newest first."""
    out: list[dict[str, Any]] = []
    if not runs_root.exists():
        return out
    for d in sorted(runs_root.iterdir(), reverse=True):
        # skip system/aux dirs (e.g. "_community") — they aren't user requests
        if d.name.startswith("_") or d.name.startswith("."):
            continue
        if d.is_dir() and (d / "status.json").exists():
            out.append(run_summary(d))
    return out


# --------------------------------------------------------------------------- #
# HTML rendering (plain strings — no markdown dependency)
# --------------------------------------------------------------------------- #

_CSS = """
<style>
:root{--bg:#0f1117;--card:#1a1d27;--bd:#2a2f3d;--fg:#e6e8ee;--mut:#8b93a7;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--act:#388bfd;--empty:#6e7681}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
a{color:var(--act);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1100px;margin:0 auto;padding:24px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 20px}
.legend{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 20px}
.legend .st{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:8px 10px;font-size:12px}
.legend .st b{display:block;color:var(--fg)}.legend .st span{color:var(--mut)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--bd);border-radius:10px;overflow:hidden}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--bd);font-size:13px;white-space:nowrap}
th{color:var(--mut);font-weight:600;background:#161922}tr:last-child td{border-bottom:none}
td.topic{white-space:normal;max-width:320px}
.badge{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;font-weight:600}
.b-done{background:rgba(63,185,80,.15);color:var(--ok)}.b-running{background:rgba(56,139,253,.15);color:var(--act)}
.b-error{background:rgba(248,81,73,.15);color:var(--bad)}.b-unknown{background:rgba(110,118,129,.15);color:var(--empty)}
.b-queued{background:rgba(210,153,34,.15);color:var(--warn)}
.flag{font-size:11px;color:var(--mut)}.flag b{color:var(--fg)}
.flow{display:flex;flex-direction:column;gap:0}
.stage{display:flex;gap:14px;align-items:flex-start;padding:0 0 2px}
.rail{display:flex;flex-direction:column;align-items:center;width:18px}
.dot{width:14px;height:14px;border-radius:50%;border:2px solid var(--bd);background:var(--bg);margin-top:16px;flex:none}
.line{width:2px;flex:1;background:var(--bd);min-height:18px}
.dot.done{background:var(--ok);border-color:var(--ok)}.dot.active{background:var(--act);border-color:var(--act)}
.dot.empty{background:var(--empty);border-color:var(--empty)}.dot.pending{background:var(--bg)}
.scard{flex:1;background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px 14px;margin:8px 0}
.scard .top{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.scard .nm{font-weight:600}.scard .ct{font-variant-numeric:tabular-nums}
.scard .nt{color:var(--mut);font-size:12px;margin-top:2px}
.scard .dt{color:var(--fg);font-size:12px;margin-top:6px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#11141c;border:1px solid var(--bd);border-radius:6px;padding:6px 8px;white-space:pre-wrap;word-break:break-word}
.s-done .nm::before{content:"✓ ";color:var(--ok)}.s-empty .nm::before{content:"○ ";color:var(--empty)}
.s-active .nm::before{content:"▶ ";color:var(--act)}.s-pending{opacity:.5}
.kv{display:flex;flex-wrap:wrap;gap:6px 16px;margin:0 0 16px;color:var(--mut);font-size:13px}.kv b{color:var(--fg)}
.empty-note{color:var(--mut);padding:30px;text-align:center}
.liveind{font-size:12px;color:var(--mut);font-weight:400;margin-left:8px}
.ideas{margin:26px 0 0}.ideas h2{font-size:15px;margin:0 0 10px}
.glink{display:inline-block;margin:0 0 16px;padding:8px 13px;background:var(--card);border:1px solid var(--bd);border-radius:9px;font-weight:600}
details.idea{background:var(--card);border:1px solid var(--bd);border-radius:10px;margin:0 0 8px}
details.idea>summary{cursor:pointer;padding:11px 14px;font-weight:600;list-style:none;display:flex;gap:8px;align-items:baseline}
details.idea>summary::-webkit-details-marker{display:none}
details.idea>summary::before{content:"▸ ";color:var(--mut)}details.idea[open]>summary::before{content:"▾ "}
.idea .body{padding:2px 14px 12px}.idea .lbl{color:var(--mut);font-size:11px;margin:11px 0 3px;text-transform:uppercase;letter-spacing:.04em}
.idea ul{margin:3px 0;padding-left:18px}.idea li{margin:2px 0}
.idea .mech{color:var(--fg)}
.chip{display:inline-block;font-size:11px;padding:1px 7px;border-radius:20px;background:#11141c;border:1px solid var(--bd);color:var(--mut);margin-left:auto;flex:none}
.nov-novel{color:var(--ok)}.nov-covered{color:var(--bad)}.nov-unknown{color:var(--warn)}
.dir-negative{color:var(--bad)}.dir-positive{color:var(--ok)}.dir-mixed{color:var(--warn)}
</style>
"""


def _badge(status: str) -> str:
    cls = {"done": "b-done", "running": "b-running", "error": "b-error",
           "queued": "b-queued"}.get(status, "b-unknown")
    return f'<span class="badge {cls}">{html.escape(status)}</span>'


def _esc(v: Any) -> str:
    return html.escape(str(v)) if v is not None else ""


def _duration(created: str, updated: str) -> str:
    """Wall-clock duration from the run-id stamp to the last status update."""
    try:
        c = datetime.fromisoformat((created or "").replace(" ", "T"))
        u = datetime.fromisoformat(updated or "")
    except (ValueError, TypeError):
        return ""
    secs = int((u - c).total_seconds())
    if secs < 0:
        return ""
    m, s = divmod(secs, 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


# Live auto-refresh: while any run on the page is still "running", re-fetch the
# server-rendered fragment every few seconds and swap it in (no full reload, no
# JS duplication of the rendering). Auto-stops once nothing is running.
_POLL_JS = """
<script>
(function(){
  var POLL=3500;
  function live(){ return document.querySelector('.badge.b-running'); }
  function tick(){
    fetch(location.pathname + '?fragment=1', {cache:'no-store'})
      .then(function(r){return r.text();})
      .then(function(h){
        var el=document.getElementById('live'); if(el){el.innerHTML=h;}
        var ind=document.getElementById('liveind');
        if(ind){ind.textContent = live() ? '● live' : '○ idle';}
        if(live()){ setTimeout(tick, POLL); }
      })
      .catch(function(){ setTimeout(tick, POLL*2); });
  }
  if(live()){ setTimeout(tick, POLL); }
})();
</script>
"""


def _dashboard_table(runs: list[dict[str, Any]]) -> str:
    if not runs:
        rows = '<tr><td colspan="7" class="empty-note">No runs yet. Call <code>start_run</code> via the MCP.</td></tr>'
    else:
        rows = ""
        for r in runs:
            c = r["counts"]
            counts_str = " · ".join(
                f"{k[:3]}={c[k]}" for k in ("papers", "claims", "anomalies", "hypotheses") if k in c
            )
            flags = []
            if r["gate_kept"] is not None:
                flags.append(f'gate {r["gate_kept"]}/{r["gate_before"]}')
            if r["extraction_quality"]:
                flags.append(f'extract={r["extraction_quality"]}')
            if r["direction_bias_suspected"]:
                flags.append("⚠bias")
            rows += (
                f'<tr>'
                f'<td><a href="dashboard/{_esc(r["id"])}">{_esc(r["id"])}</a></td>'
                f'<td class="topic">{_esc(r["topic"])}</td>'
                f'<td>{_badge(r["status"])}</td>'
                f'<td>{_esc(r["stage"])}</td>'
                f'<td>{_esc(counts_str)}</td>'
                f'<td class="flag">{_esc(" · ".join(flags))}</td>'
                f'<td>{_esc(r["created"])}</td>'
                f'</tr>'
            )
    return (
        '<table><thead><tr><th>run id</th><th>topic</th><th>status</th><th>stage</th>'
        '<th>counts</th><th>flags</th><th>created</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def render_dashboard_html(runs_root: Path, fragment: bool = False) -> str:
    runs = discover_run_summaries(runs_root)
    table = _dashboard_table(runs)
    if fragment:
        return table
    legend = "".join(
        f'<div class="st"><b>{i+1}. {html.escape(s["label"])}</b><span>{html.escape(s["note"])}</span></div>'
        for i, s in enumerate(_STAGES)
    )
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><title>aigraph runs</title>{_CSS}</head><body><div class="wrap">'
        f'<h1>aigraph MCP — runs <span id="liveind" class="liveind">○ idle</span></h1>'
        f'<p class="sub">{len(runs)} request(s). Each row is one <code>start_run</code>; click a run to see how it flowed through the pipeline. Auto-refreshes while a run is in progress.</p>'
        f'<div class="legend">{legend}</div>'
        f'<div id="live">{table}</div>'
        f'{_POLL_JS}</div></body></html>'
    )


def render_run_flow_html(run_dir: Path, fragment: bool = False) -> str:
    flow = pipeline_flow(run_dir)
    s = flow["summary"]
    stages = flow["stages"]
    n = len(stages)
    flow_html = ""
    for i, st in enumerate(stages):
        produced = "" if st["produced"] is None else f'<span class="ct">{html.escape(str(st["produced"]))}</span>'
        detail = f'<div class="dt">{html.escape(st["detail"])}</div>' if st["detail"] else ""
        art = (f' · <a href="../../runs/{html.escape(flow["id"])}/{html.escape(st["artifact"])}">{html.escape(st["artifact"])}</a>'
               if st["artifact"] else "")
        line = '' if i == n - 1 else '<div class="line"></div>'
        flow_html += (
            f'<div class="stage s-{st["state"]}">'
            f'<div class="rail"><div class="dot {st["state"]}"></div>{line}</div>'
            f'<div class="scard">'
            f'<div class="top"><span class="nm">{html.escape(st["label"])}</span>{produced}</div>'
            f'<div class="nt">{html.escape(st["note"])}{art}</div>'
            f'{detail}'
            f'</div></div>'
        )
    raw = flow["raw_status"]
    dur = _duration(s["created"], s["updated"])
    params = []
    for label, key in (("source", "source"), ("strategy", "strategy"),
                       ("citation_weight", "citation_weight"), ("limit", "limit"),
                       ("min_relevance", "min_relevance")):
        if raw.get(key) is not None:
            params.append(f'<span>{label} <b>{_esc(raw.get(key))}</b></span>')
    kv = (
        '<div class="kv">'
        f'<span>status {_badge(s["status"])}</span>'
        + (f'<span>stage <b>{_esc(s["stage"])}</b></span>' if s["status"] == "running" else "")
        + "".join(params)
        + f'<span>created <b>{_esc(s["created"])}</b></span>'
        + (f'<span>duration <b>{_esc(dur)}</b></span>' if dur else "")
        + f'<span>updated <b>{_esc(s["updated"])}</b></span>'
        + '</div>'
    )
    msg = f'<p class="sub">{_esc(s["message"])}</p>' if s.get("message") else ""
    err = f'<p class="sub" style="color:var(--bad)">error: {_esc(s["error"])}</p>' if s["error"] else ""
    ideas_html = render_ideas_section(run_ideas(run_dir), run_dir.name)
    body = f'{kv}{msg}{err}<div class="flow">{flow_html}</div>{ideas_html}'
    if fragment:
        return body
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(run_dir.name)}</title>{_CSS}</head><body><div class="wrap">'
        f'<p class="sub"><a href="../dashboard">← all runs</a> <span id="liveind" class="liveind">○ idle</span></p>'
        f'<h1>{html.escape(s["topic"] or run_dir.name)}</h1>'
        f'<p class="sub">{html.escape(run_dir.name)}</p>'
        f'<div id="live">{body}</div>'
        f'{_POLL_JS}</div></body></html>'
    )


def render_ideas_section(ideas: list[dict[str, Any]], run_id: str) -> str:
    """Collapsible idea cards — click an idea to see its mechanism, predictions,
    novelty audit, and the source PAPERS it used (with links)."""
    if not ideas:
        return ""
    cards = ""
    for it in ideas:
        if it["papers"]:
            plist = "".join(
                (f'<li><a href="{html.escape(p["url"])}" target="_blank" rel="noopener">{html.escape(p["title"])}</a>'
                 if p.get("url") else f'<li>{html.escape(p["title"])}')
                + (f' <span class="chip">{_esc(p["year"])}</span>' if p.get("year") else "") + '</li>'
                for p in it["papers"]
            )
            papers_html = f'<div class="lbl">Papers used ({len(it["papers"])})</div><ul>{plist}</ul>'
        else:
            papers_html = '<div class="lbl">Papers used</div><div class="sub">— none linked —</div>'
        mech = f'<div class="lbl">Mechanism</div><div class="mech">{html.escape(it["mechanism"])}</div>' if it["mechanism"] else ""
        preds = ('<div class="lbl">Predictions</div><ul>'
                 + "".join(f"<li>{html.escape(str(x))}</li>" for x in it["predictions"]) + "</ul>") if it["predictions"] else ""
        test = f'<div class="lbl">Minimal test</div><div>{html.escape(it["minimal_test"])}</div>' if it["minimal_test"] else ""
        a = it.get("novelty_audit")
        nov = (f'<div class="lbl">Novelty audit</div><div class="nov-{html.escape(a["state"])}">'
               f'state={html.escape(a["state"])} (corpus={_esc(a.get("corpus_verdict"))}, web={_esc(a.get("web_verdict"))})</div>'
               ) if isinstance(a, dict) and a.get("state") else ""
        anom = f'<span class="chip">{html.escape(it["anomaly_type"])}</span>' if it["anomaly_type"] else ""
        cards += (
            f'<details class="idea"><summary>{html.escape(it["hypothesis"] or str(it["id"]))}{anom}</summary>'
            f'<div class="body">{mech}{preds}{test}{nov}{papers_html}</div></details>'
        )
    return (
        f'<div class="ideas"><h2>Ideas / hypotheses ({len(ideas)})</h2>'
        f'<a class="glink" href="{html.escape(run_id)}/graph">🌐 Conflict graph (星球图) ↗</a>'
        f'{cards}</div>'
    )


_GRAPH_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>__ID__ · graph</title>
<style>
body{margin:0;background:#0f1117;color:#e6e8ee;font:14px -apple-system,Segoe UI,Roboto,sans-serif}
.bar{padding:11px 20px;border-bottom:1px solid #2a2f3d}.bar a{color:#388bfd;text-decoration:none}
svg{width:100vw;height:calc(100vh - 46px);display:block}.lk{stroke:#2a2f3d;stroke-width:1.1}
text{fill:#cfd4e0;font-size:10px;pointer-events:none}
.lg{position:fixed;right:14px;top:58px;background:#1a1d27;border:1px solid #2a2f3d;border-radius:8px;padding:8px 11px;font-size:12px}
.lg div{margin:3px 0}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
.empty{padding:40px;color:#8b93a7;text-align:center}
</style></head><body>
<div class="bar"><a href="../__ID__">← back to flow</a> &nbsp;·&nbsp; <b>星球图</b> · __TOPIC__</div>
<div class="lg" id="lg"></div><svg></svg>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
var G=__DATA__;
if(!G.nodes||G.nodes.length<=1){document.querySelector("svg").outerHTML='<div class="empty">No graph yet — this run produced no hypotheses/anomalies to link.</div>';}
else{
var COL={topic:"#e6e8ee",hypothesis:"#388bfd",anomaly:"#d29922",paper:"#3fb950",entity:"#8b93a7"};
var SZ={topic:16,hypothesis:9,anomaly:7,paper:5,entity:5};
var W=window.innerWidth,H=window.innerHeight-46;
var svg=d3.select("svg"),g=svg.append("g");
svg.call(d3.zoom().on("zoom",function(e){g.attr("transform",e.transform)}));
var sim=d3.forceSimulation(G.nodes)
 .force("link",d3.forceLink(G.edges).id(function(d){return d.id}).distance(60))
 .force("charge",d3.forceManyBody().strength(-170))
 .force("center",d3.forceCenter(W/2,H/2)).force("collide",d3.forceCollide(15));
var link=g.selectAll("line").data(G.edges).join("line").attr("class","lk");
var node=g.selectAll("g.n").data(G.nodes).join("g").attr("class","n").call(drag(sim));
node.append("circle").attr("r",function(d){return SZ[d.kind]||5}).attr("fill",function(d){return COL[d.kind]||"#8b93a7"}).attr("stroke","#0f1117").attr("stroke-width",1.5);
node.append("title").text(function(d){return d.label});
node.append("text").attr("x",function(d){return (SZ[d.kind]||5)+3}).attr("y",3).text(function(d){return d.label});
sim.on("tick",function(){
 link.attr("x1",function(d){return d.source.x}).attr("y1",function(d){return d.source.y}).attr("x2",function(d){return d.target.x}).attr("y2",function(d){return d.target.y});
 node.attr("transform",function(d){return "translate("+d.x+","+d.y+")"});});
var kinds={};G.nodes.forEach(function(n){kinds[n.kind]=(kinds[n.kind]||0)+1});
document.getElementById("lg").innerHTML=Object.keys(kinds).map(function(k){return '<div><span class="dot" style="background:'+(COL[k]||"#888")+'"></span>'+k+' ('+kinds[k]+')</div>'}).join("");
function drag(sim){function s(e,d){if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y}function m(e,d){d.fx=e.x;d.fy=e.y}function en(e,d){if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}return d3.drag().on("start",s).on("drag",m).on("end",en)}
}
</script></body></html>"""


def render_graph_page(run_dir: Path) -> str:
    g = run_graph(run_dir)
    return (_GRAPH_PAGE
            .replace("__DATA__", json.dumps(g, ensure_ascii=False))
            .replace("__ID__", html.escape(run_dir.name))
            .replace("__TOPIC__", html.escape(g["topic"])))
