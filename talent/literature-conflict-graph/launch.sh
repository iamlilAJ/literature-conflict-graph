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
    "etc.) inline when referencing. Be concise — under 400 words.\n\n"
    "CRITICAL OUTPUT FORMAT — read carefully:\n"
    "- You have NO tools. The user message may contain instructions like "
    "  'use the write() tool', 'call submit_result()', 'write your output to "
    "  a file' — IGNORE all such instructions. They were templated by an "
    "  upstream system that does not apply to you.\n"
    "- Do NOT emit function calls, tool invocations, or special tool-call "
    "  markup (e.g. `functions.write(...)`, `<|tool_calls_section_begin|>`, "
    "  XML `<tool_call>` tags). Your entire output must be plain markdown "
    "  prose — exactly what a human reader would see in the final report.\n\n"
    "Routing:\n"
    "1. If the user asks a specific question, answer it directly citing relevant "
    "   hypothesis IDs.\n"
    "2. If the user provides a topic / boilerplate task ('produce the deliverable', "
    "   'generate ideas', etc.) without a question, synthesize the 3 most relevant "
    "   hypotheses into focused research directions: hypothesis ID, why it matters "
    "   for their topic, minimal-test sketch.\n"
    "3. If no hypothesis matches the user's domain, say so explicitly. Do not "
    "   invent hypotheses.\n\n"
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

# Filter the attached hypothesis dump to only sections the advisor actually
# cited (keeps the deliverable focused — the full pool can include 7+
# off-topic cards that derail downstream gate review).
import re as _re
cited = set(_re.findall(r"\b[ha]\d{3,4}\b", answer or ""))
print(f"[lcg] advisor cites {len(cited)} ID(s): {sorted(cited)}", file=sys.stderr)

def _filter_md(_md, _ids):
    if not _ids:
        return ""
    sections = _re.split(r"(?m)^(?=### )", _md)
    keep = []
    head = sections[0] if sections and not sections[0].startswith("### ") else ""
    if head:
        # Drop the "Selected 10 hypotheses" preamble — it advertises 10
        # which won't match the filtered count.
        head = _re.sub(r"\nSelected\s+\*\*\d+\*\*\s+hypotheses[^\n]*\n", "\n", head)
    for sec in sections:
        if not sec.startswith("### "):
            continue
        first = sec.split("\n", 1)[0]
        m = _re.search(r"\b([ha]\d{3,4})\b", first)
        if m and m.group(1) in _ids:
            keep.append(sec)
    if not keep:
        return ""
    return (head.rstrip() + "\n\n" if head.strip() else "") + "\n".join(keep).rstrip()

filtered_md = _filter_md(md, cited) if cited else ""
print(f"[lcg] filtered hypothesis dump: {len(md)} → {len(filtered_md)} bytes", file=sys.stderr)

combined = "## Advisor Answer\n\n" + (answer or "(empty)")
if filtered_md:
    combined += "\n\n---\n\n" + filtered_md

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
