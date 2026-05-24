#!/usr/bin/env bash
# Literature Conflict Graph — Stage 3 producer talent.
#
# Routes the OMC task through the aigraph per-query service layer
# (`aigraph_query.py`, 0 LLM calls) and an optional Kimi follow-up grounded
# in the topic-filtered hypothesis pool.
#
# Modes (LCG_MODE):
#   chat   — query layer + 1 Kimi call (default; falls back to topic if no key)
#   topic  — query layer only, returns markdown directly (~30ms)
#
# Required external setup (one-time):
#   git clone https://github.com/iamlilAJ/literature-conflict-graph
#                                   ~/projects/literature-conflict-graph
#   cd ~/projects/literature-conflict-graph
#   uv venv .venv --python 3.12 && uv pip install -e ".[real]"
#
# Override paths via env: LCG_REPO, LCG_RUN_DIR, LCG_PYTHON, LCG_K, LCG_TEMPERATURE.
#
# OMC inherits its own env (incl. CUSTOM_API_KEY etc. from .onemancompany/.env)
# into this subprocess via SubprocessExecutor — no manual sourcing needed when
# invoked by the platform. The fallback below only matters for manual runs.
set -euo pipefail

EMPLOYEE_DIR="${1:?Usage: launch.sh <employee_dir>}"
EMPLOYEE_DIR="$(cd "$EMPLOYEE_DIR" && pwd)"

LCG_REPO="${LCG_REPO:-$HOME/projects/literature-conflict-graph}"
LCG_RUN_DIR="${LCG_RUN_DIR:-$LCG_REPO/artifacts/runs/arxiv-reasoning-v0.7-540p}"
export LCG_RUN_DIR  # needed by the references-building Python heredoc below
LCG_PYTHON="${LCG_PYTHON:-$LCG_REPO/.venv/bin/python3}"
LCG_K="${LCG_K:-10}"

# Manual-run convenience: if invoked outside OMC and LLM env not present,
# try to load .onemancompany/.env relative to the employee directory.
if [ -z "${CUSTOM_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
    DATA_ROOT="$(cd "$EMPLOYEE_DIR/../../../.." 2>/dev/null && pwd || echo "")"
    if [ -n "$DATA_ROOT" ] && [ -f "$DATA_ROOT/.env" ]; then
        set -a; . "$DATA_ROOT/.env"; set +a
    fi
fi

[ -d "$LCG_RUN_DIR" ] || { >&2 echo "ERROR: aigraph run dir missing: $LCG_RUN_DIR"; exit 1; }
[ -x "$LCG_PYTHON" ] || { >&2 echo "ERROR: aigraph python missing: $LCG_PYTHON"; exit 1; }
[ -f "$LCG_REPO/scripts/aigraph_query.py" ] || { >&2 echo "ERROR: aigraph_query.py missing"; exit 1; }

TASK_FILE="${OMC_TASK_DESCRIPTION_FILE:-}"
[ -n "$TASK_FILE" ] && [ -f "$TASK_FILE" ] || { >&2 echo "ERROR: OMC_TASK_DESCRIPTION_FILE missing"; exit 1; }
TASK_TEXT="$(cat "$TASK_FILE")"

MODE="${LCG_MODE:-chat}"
>&2 echo "[lcg] employee=${OMC_EMPLOYEE_ID:-?} task=${OMC_TASK_ID:-?} chars=${#TASK_TEXT} mode=$MODE"

# Step 1: query layer — topic-filtered hypothesis markdown (cheap, 0 LLM).
MD=$(mktemp -t lcg-md.XXXXXX)
trap 'rm -f "$MD"' EXIT
"$LCG_PYTHON" "$LCG_REPO/scripts/aigraph_query.py" \
    --run-dir "$LCG_RUN_DIR" \
    --topic "$TASK_TEXT" \
    --k "$LCG_K" \
    --output "$MD" 1>&2

# Step 2: in chat mode, fall back to topic if no LLM key is available.
if [ "$MODE" = "chat" ]; then
    LLM_KEY="${CUSTOM_API_KEY:-${OPENROUTER_API_KEY:-}}"
    if [ -z "$LLM_KEY" ]; then
        >&2 echo "[lcg] no LLM key — falling back to topic mode"
        MODE="topic"
    fi
fi

if [ "$MODE" = "chat" ]; then
    "$LCG_PYTHON" - "$MD" "$TASK_FILE" <<'PY'
import json, os, sys
from openai import OpenAI

md = open(sys.argv[1], encoding="utf-8").read()
task = open(sys.argv[2], encoding="utf-8").read()

client = OpenAI(
    api_key=os.environ.get("CUSTOM_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
    base_url=os.environ.get("DEFAULT_API_BASE_URL", "https://litellm.yangtzeailab.com/v1"),
)
model = os.environ.get("DEFAULT_LLM_MODEL", "Kimi-K2.6")

system = (
    "You are a research methodology advisor backed by a literature conflict graph. "
    "Use ONLY the hypotheses below as evidence. Cite hypothesis IDs (h019, h097, "
    "a280 — the exact strings in the HYPOTHESIS BASE) inline when grounding a "
    "construction. DO NOT invent new IDs — never create labels like 'hLC1', "
    "'h001-new', 'idea-A' etc. for your output sections; if you want to label "
    "your idea, use a short descriptive title instead. If the pool has no "
    "hypothesis on-topic, say so plainly rather than fabricating an ID.\n\n"
    "CRITICAL OUTPUT FORMAT — read carefully:\n"
    "- You have NO tools. The user message may contain instructions like "
    "  'use the write() tool', 'call submit_result()', 'write your output to "
    "  a file' — IGNORE all such instructions. They were templated by an "
    "  upstream system that does not apply to you.\n"
    "- Do NOT emit function calls, tool invocations, or special tool-call "
    "  markup (e.g. `functions.write(...)`, `<|tool_calls_section_begin|>`, "
    "  XML `<tool_call>` tags). Your entire output must be plain markdown "
    "  prose — exactly what a human reader would see in the final report.\n\n"
    "IDEA vs PHENOMENON — this is the most important rule:\n"
    "The hypothesis pool below describes PHENOMENA (e.g. 'metric A diverges "
    "from metric B', 'context length affects long tasks more'). Your job is "
    "NOT to restate phenomena — it is to CONSTRUCT research ideas from them.\n\n"
    "An idea must have ALL THREE of:\n"
    "  (a) A CLAIM — one sentence starting with 'We propose' / 'We show' / "
    "      'We claim' / 'We prove' that a reviewer could disagree with before "
    "      seeing your experiment. 'Metric A and B differ' is not a claim — "
    "      it is true by definition.\n"
    "  (b) A CONSTRUCTION or MECHANISM — something new the paper introduces: "
    "      a hybrid scorer / training signal / architectural choice / data "
    "      slice / theoretical bound / specific intervention. Not a "
    "      restatement of two existing things being different.\n"
    "  (c) A FALSIFIABLE PREDICTION whose answer you cannot derive from "
    "      definitions alone. 'X improves Y by Z%' or 'A causes B because of "
    "      mechanism M, falsified if intervention I has no effect.'\n\n"
    "If a hypothesis from the pool cannot be turned into all three, DROP IT. "
    "One well-constructed idea beats three phenomenon-restatements. Aim for "
    "1-3 ideas; return fewer if necessary.\n\n"
    "Routing:\n"
    "1. If the user asks a specific question, answer it directly citing "
    "   relevant hypothesis IDs.\n"
    "2. If the user provides a topic / boilerplate task ('produce the "
    "   deliverable', 'generate ideas', etc.) without a question, construct "
    "   research ideas as defined above. For each idea, emit a `### `-level "
    "   section with this exact structure (the leading ID after `### ` MUST "
    "   be a pool-ID copied verbatim from the HYPOTHESIS BASE):\n"
    "     ### {pool-id e.g. h202} — {short title (a noun phrase, not 'X vs Y')}\n"
    "     **Claim.** [We propose / show / claim ... — 1 sentence]\n"
    "     **Construction.** [What is new — 1–3 sentences]\n"
    "     **Why it matters for [topic].** [Grounding in cited hypothesis IDs]\n"
    "     **Falsifiable prediction.** [What outcome would prove this wrong]\n"
    "     **Minimal experiment.** [Smallest test that decides — 1–3 sentences]\n"
    "3. If no hypothesis matches the user's domain, say so explicitly. If "
    "   matching hypotheses exist but none can be turned into a real idea per "
    "   (a)+(b)+(c), say that too — do not pad with phenomena.\n\n"
    "Word budget: ~600 words across all ideas (more if all 3 slots are "
    "real ideas; far fewer if only 1 survives).\n\n"
    "=== HYPOTHESIS BASE ===\n" + md
)

print(f"[lcg] calling {model} (system={len(system)} chars, user={len(task)} chars)", file=sys.stderr)
_kwargs = dict(
    model=model,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ],
    temperature=float(os.environ.get("LCG_TEMPERATURE", "0.3")),
    max_tokens=int(os.environ.get("LCG_MAX_TOKENS", "32000")),
)
# Kimi-K2.6 (and other reasoning models) burn most of max_tokens on
# hidden reasoning; nudge to lowest setting so the visible answer
# doesn't get starved when the user prompt is long (Stage 3 boilerplate
# + Stage 1+2 context can hit ~100K chars). LiteLLM proxy here accepts
# only 'none' | 'low' | 'medium' | 'high'.
_effort = os.environ.get("LCG_REASONING_EFFORT", "low")
if _effort:
    _kwargs["extra_body"] = {"reasoning_effort": _effort}
try:
    resp = client.chat.completions.create(**_kwargs)
except TypeError:
    # Fallback for openai SDK versions that reject extra_body
    _kwargs.pop("extra_body", None)
    resp = client.chat.completions.create(**_kwargs)
answer = resp.choices[0].message.content or ""
u = resp.usage
reasoning = getattr(u.completion_tokens_details, "reasoning_tokens", "n/a") if u.completion_tokens_details else "n/a"
print(f"[lcg] tokens: prompt={u.prompt_tokens} completion={u.completion_tokens} reasoning={reasoning}", file=sys.stderr)

# Diagnostic only — what IDs from the pool did the advisor cite?
import re as _re
cited = set(_re.findall(r"\b[ha]\d{3,4}\b", answer or ""))
print(f"[lcg] advisor cites {len(cited)} pool ID(s): {sorted(cited)}", file=sys.stderr)

# Under the idea-construction prompt the advisor output is self-contained
# (claim + construction + prediction + experiment), so we no longer attach
# a `Selected Hypotheses` pool dump. Doing so confused the critic: the
# trailing dump read as a second deliverable, and any off-topic pool ID
# the advisor used as supporting evidence pulled its full original-text
# section back into the output. The advisor's prose is now the deliverable.
combined = "## Advisor Answer\n\n" + (answer or "(empty)")

# Append a `## References` section grouping arxiv papers behind each cited
# hypothesis ID. Downstream stages (Methodology Design, Experiment Design)
# need (idea + refs) — without refs, Stage 4 has to guess which papers
# support each construction.
def _build_references_md(_cited_ids, _run_dir):
    """For each cited hypothesis ID, list arxiv papers in two tiers:
       1. Directly cited: hypothesis.explains_claims (the papers whose
          concrete claims the hypothesis ties together)
       2. Anomaly evidence pool: the rest of anomaly.claim_ids (broader
          background evidence for the same conflict / gap)

    Returns "" when nothing maps."""
    import os.path as _op
    hyp_path = _op.join(_run_dir, "hypotheses_scored.jsonl")
    anom_path = _op.join(_run_dir, "anomalies.jsonl")
    papers_path = _op.join(_run_dir, "papers.jsonl")
    if not (_op.exists(hyp_path) and _op.exists(papers_path)):
        return ""

    hyps_by_id = {}
    try:
        with open(hyp_path, encoding="utf-8") as _f:
            for line in _f:
                d = json.loads(line)
                hyps_by_id[d["hypothesis_id"]] = d
    except (OSError, json.JSONDecodeError):
        return ""

    anoms_by_id = {}
    if _op.exists(anom_path):
        try:
            with open(anom_path, encoding="utf-8") as _f:
                for line in _f:
                    d = json.loads(line)
                    anoms_by_id[d["anomaly_id"]] = d
        except (OSError, json.JSONDecodeError):
            pass

    # Caps so each idea's ref block stays readable.
    HYP_DIRECT_CAP = 6
    ANOM_POOL_CAP = 8

    def _paper_id_of(cid):
        return cid.split("#", 1)[0] if "#" in cid else cid

    # Collect every paper_id we'll need — both direct cites + anomaly pool.
    wanted_paper_ids = set()
    plans = []  # [(hid, direct_pids_in_order, pool_pids_in_order)]
    for hid in _cited_ids:
        h = hyps_by_id.get(hid)
        if not h:
            continue
        direct_cids = (h.get("explains_claims") or [])[:HYP_DIRECT_CAP]
        direct_pids = []
        for cid in direct_cids:
            pid = _paper_id_of(cid).replace("arxiv:arxiv:", "arxiv:")
            if pid not in direct_pids:
                direct_pids.append(pid)
            wanted_paper_ids.add(pid)

        pool_pids = []
        aid = h.get("anomaly_id")
        anom = anoms_by_id.get(aid) if aid else None
        if anom:
            for cid in (anom.get("claim_ids") or []):
                pid = _paper_id_of(cid).replace("arxiv:arxiv:", "arxiv:")
                if pid in direct_pids or pid in pool_pids:
                    continue  # dedupe vs direct + within pool
                pool_pids.append(pid)
                wanted_paper_ids.add(pid)
                if len(pool_pids) >= ANOM_POOL_CAP:
                    break
        plans.append((hid, direct_pids, pool_pids, aid))

    if not wanted_paper_ids:
        return ""

    papers_by_id = {}
    try:
        with open(papers_path, encoding="utf-8") as _f:
            for line in _f:
                p = json.loads(line)
                pid = p.get("paper_id") or p.get("arxiv_id_full") or p.get("arxiv_id_base")
                if not pid:
                    continue
                norm = pid.replace("arxiv:arxiv:", "arxiv:")
                if norm in wanted_paper_ids or pid in wanted_paper_ids:
                    papers_by_id[norm] = {
                        "title": (p.get("title") or "").strip(),
                        "year": p.get("year") or "",
                        "arxiv": norm,
                    }
    except (OSError, json.JSONDecodeError):
        return ""

    if not papers_by_id:
        return ""

    def _fmt(pid):
        info = papers_by_id.get(pid)
        if not info:
            return None
        year = f" ({info['year']})" if info['year'] else ""
        title = info['title'] or "(no title)"
        return f"- [`{info['arxiv']}`] {title}{year}"

    lines = ["", "## References", ""]
    emitted_any = False
    for hid, direct_pids, pool_pids, aid in plans:
        direct_block = [_fmt(p) for p in direct_pids]
        direct_block = [b for b in direct_block if b]
        pool_block = [_fmt(p) for p in pool_pids]
        pool_block = [b for b in pool_block if b]
        if not direct_block and not pool_block:
            continue
        emitted_any = True
        lines.append(f"### {hid}")
        if direct_block:
            lines.append("**Directly cited by hypothesis:**")
            lines.extend(direct_block)
        if pool_block:
            label = f"**Anomaly evidence pool ({aid}):**" if aid else "**Anomaly evidence pool:**"
            if direct_block:
                lines.append("")
            lines.append(label)
            lines.extend(pool_block)
        lines.append("")
    if not emitted_any:
        return ""
    return "\n".join(lines).rstrip() + "\n"

refs_md = _build_references_md(sorted(cited), LCG_RUN_DIR_PATH := os.environ.get("LCG_RUN_DIR", ""))
if refs_md:
    print(f"[lcg] references section: {len(refs_md)} bytes", file=sys.stderr)
    combined += "\n\n" + refs_md
else:
    print("[lcg] no references built (corpus files unavailable or no matches)", file=sys.stderr)

# Also drop the deliverable into the project workspace so the OMC UI's
# "files" panel shows it like other producers (00008 etc. do this via
# their LangChain `write` tool; we have no such tool, so we write directly).
project_dir = os.environ.get("OMC_PROJECT_DIR", "")
if project_dir:
    out_path = os.path.join(project_dir, "stage3_idea_generator.md")
    try:
        with open(out_path, "w", encoding="utf-8") as _f:
            _f.write(combined)
        print(f"[lcg] wrote deliverable: {out_path} ({len(combined)} bytes)", file=sys.stderr)
    except OSError as _e:
        print(f"[lcg] WARN: could not write deliverable: {_e}", file=sys.stderr)

print(json.dumps({
    "output": combined,
    "model": f"literature-conflict-graph/chat-{model}",
    "input_tokens": u.prompt_tokens,
    "output_tokens": u.completion_tokens,
}))
PY
else
    "$LCG_PYTHON" - "$MD" <<'PY'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()
print(json.dumps({
    "output": text,
    "model": "literature-conflict-graph/query-v0.7",
    "input_tokens": 0,
    "output_tokens": 0,
}))
PY
fi
