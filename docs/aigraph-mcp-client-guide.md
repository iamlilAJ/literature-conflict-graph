# aigraph MCP Server — Client Integration Guide

The aigraph MCP server exposes the v0.7-frozen literature-conflict pipeline as
a Model Context Protocol service, allowing LLM agents to invoke its cached
hypothesis-generation surface as first-class tools. This document describes
how to connect to the deployed instance, the available tools, and operational
procedures for client integrators.

## 1. Service overview

| Attribute | Value |
|---|---|
| Host | `8.208.118.99` |
| Endpoint (same-host) | `http://127.0.0.1:8765/mcp/` |
| Endpoint (local client) | SSH tunnel → `http://localhost:18765/mcp/` (see §2.3) |
| Transport | Streamable HTTP (MCP spec) |
| Process supervisor | `tmux` session `aigraph` |
| Repository | `github.com/iamlilAJ/literature-conflict-graph`, branch `feat/lcg-tfidf-relevance` |
| Server-side install | `~/aigraph/` (**git clone** since 2026-06-01; deploy via `git pull` — see §5.4) |
| LLM endpoint | DeepSeek-V4-Flash via `litellm.yangtzeailab.com` (`AIGRAPH_BASE_URL` in `~/aigraph/.env`) — **live** |
| Reference run | `arxiv-reasoning-v0.7-540p-thaw1` (474 papers, 300 hypotheses, 330-record `atlas_overlap.jsonl` sidecar) |

The MCP, browser UI, query endpoint, and conflict-graph endpoint are served by
a single uvicorn process bound to `0.0.0.0:8765`. CORS is open. The `/mcp/`
endpoint enforces HTTP Host validation (§4) so only same-host / tunnelled
clients reach the MCP tools; the REST endpoints (`/query`, `/query/graph`,
`/api/runs`) are reachable from the public internet without auth (§4).

### 1.1 Two-phase mental model

Everything below maps onto two halves — understand these and the tools make
sense:

- **① Build a corpus (write; heavy LLM; minutes).** `start_run` /
  `research_ideas(reuse=false)` fetch papers → extract claims (1 LLM call per
  paper) → build graph → detect anomalies → generate hypotheses + insights,
  and persist everything to a run directory under `artifacts/runs/<run_id>/`.
- **② Serve ideas (read; mostly 0 LLM; sub-second).** `query_hypotheses`,
  `get_idea_report`, `get_conflict_graph`, `generate_ideas` read those cached
  files, topic-filter + rank, and return. A corpus is built once and queried
  many times. See `docs/idea-generation-pipeline.md` for the full data flow.

## 2. Connection patterns

### 2.1 Direct JSON-RPC over the SSH loopback

The lowest-dependency client is a `curl` against `127.0.0.1` from the host
itself. Streamable HTTP responses use SSE framing (`data: …` prefix per
chunk).

```bash
ssh admin@8.208.118.99 'curl -s -X POST http://127.0.0.1:8765/mcp/ \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}"'
```

A `tools/list` call returns the seven registered tools enumerated in §3.

### 2.2 Production client — `langchain-mcp-adapters`

This is the integration path used by the OMC Stage 3 producer agent
(employee `00008`). It must run from a host that can reach the loopback
endpoint (i.e. on the server itself, inside the same network namespace as
`aigraph`).

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient({
        "aigraph": {
            "url": "http://localhost:8765/mcp/",
            "transport": "streamable_http",
        }
    })
    tools = await client.get_tools()
    print("registered tools:", [t.name for t in tools])

    get_idea_report = next(t for t in tools if t.name == "get_idea_report")
    result = await get_idea_report.ainvoke({
        "topic": "chain of thought reasoning",
        "run":   "arxiv-reasoning-v0.7-540p-thaw1",
        "k":     3,
    })
    text = result if isinstance(result, str) else result[0]["text"]
    print(text[:500])

asyncio.run(main())
```

Run from `~/onemancompany/.venv` (which already provides
`langchain-mcp-adapters` and `openai`); the aigraph venv is intentionally
minimal and does not include the client adapters.

### 2.3 Remote clients

The endpoint is bound to `0.0.0.0:8765`, but the `/mcp/` transport returns
`421 Misdirected Request` for any request whose `Host` header is not in its
allow-list (the MCP SDK's DNS-rebinding protection — see §4). This is a
deliberate boundary. For supported external integration:

- **SSH tunnel (recommended for local IDE / Mac clients)** —
  ```bash
  ssh -L 18765:127.0.0.1:8765 admin@8.208.118.99
  ```
  Then point any local MCP client at `http://localhost:18765/mcp/`. The
  Host header is `localhost`, which the transport accepts. Picking a non-8765
  local port (e.g. 18765) avoids colliding with other local services.
  For a backgrounded tunnel: add `-fN` flags. Verified working:
  `curl http://localhost:18765/mcp/ → 200 OK`.
- **Bastion access** — `ssh` into the host and invoke either pattern above
  (loopback curl from inside the server itself).
- **Reverse proxy** — terminate TLS in front of the endpoint and rewrite the
  `Host` header to `localhost`; expose only the rewritten origin.
- **Widen the allow-list** — pass `TransportSecuritySettings(allowed_hosts=…)`
  when building the FastMCP app (mcp_server.py), or disable
  `enable_dns_rebinding_protection`. Not currently configured.

The OMC frontend's conflict-graph component (`frontend/src/lcg-graph.js`)
already follows the same-host pattern; see §6.

## 3. Tool reference

Nine tools follow the JSON-RPC 2.0 envelope. The five read-only tools (§3.1)
make zero LLM calls and respond sub-second. The two run-trigger tools (§3.2)
build corpora via the now-live LLM endpoint. The two idea-generation tools
(§3.3) cascade with a non-empty guarantee, mostly 0-LLM.

### 3.1 Read-only (0 LLM)

#### `list_runs`

```json
{"name": "list_runs", "arguments": {}}
```

Returns `list[{id, n_papers, n_hypotheses}]` for every run discovered under
`--runs-root`. Use the returned `id` as the `run` argument to subsequent
tools.

#### `get_run_summary(run)`

```json
{"name": "get_run_summary",
 "arguments": {"run": "arxiv-reasoning-v0.7-540p-thaw1"}}
```

Returns counts (papers, claims, hypotheses), an anomaly-type histogram, the
`git_sha` / `git_tag` / `model` / `topic` / cap from `run_metadata.json`.

#### `query_hypotheses(topic, run, k)`

```json
{"name": "query_hypotheses",
 "arguments": {"topic": "retrieval augmented generation",
               "run":   "arxiv-reasoning-v0.7-540p-thaw1",
               "k":     5}}
```

Structured top-K. Each record contains `hypothesis_id, anomaly_id,
anomaly_type, central_question, hypothesis, mechanism, predictions,
minimal_test, scope_conditions, evidence_gap, graph_bridge`, an
`evidence_claims` array (`{claim_id, paper_id, title, year, direction,
claim_text}`), and a `utility` score breakdown. `stats` reports match counts,
candidate count, selection size, and wall time. `k ∈ [1, 20]`.

#### `get_idea_report(topic, run, k)`

```json
{"name": "get_idea_report",
 "arguments": {"topic": "chain of thought reasoning",
               "run":   "arxiv-reasoning-v0.7-540p-thaw1",
               "k":     8}}
```

Returns the canonical Stage 3 deliverable as a markdown string:
a `# Stage 3: Idea Generation — <topic>` heading followed by a
`# Selected Hypotheses` report with `### Anomaly a… —` and `### h… —`
sections grounded in real claim citations. The OMC frontend's conflict-graph
renderer parses this exact format; downstream consumers should treat the
returned text as opaque and write it verbatim. `k ∈ [1, 20]`.

#### `get_conflict_graph(topic, run, k, ids)`

```json
{"name": "get_conflict_graph",
 "arguments": {"topic": "agentic reasoning",
               "run":   "arxiv-reasoning-v0.7-540p-thaw1",
               "k":     8}}
```

Returns a D3-friendly `{nodes, edges, stats}` payload for the topic-filtered
subgraph. The optional `ids` argument accepts a comma-separated list of
hypothesis IDs to pin the graph to a specific selection.

### 3.2 Run-trigger (paid; phase ①, only when NOT readonly)

#### `start_run(topic, max_papers, generator)`

```json
{"name": "start_run",
 "arguments": {"topic":      "sparse mixture of experts",
               "max_papers": 50,
               "generator":  "llm"}}
```

Spawns a full corpus-build pipeline via `SearchService.submit` (the phase ①
build of §1.1). Returns `{run_id, status: "queued", poll_with:
"get_run_status"}` immediately; the work runs on a single background worker
thread. The configured LLM endpoint (`AIGRAPH_BASE_URL` →
`litellm.yangtzeailab.com`, model `DeepSeek-V4-Flash`) **is live**. Two
caveats observed in practice:

- The build is **LLM-heavy** (1 claim-extraction call per paper) and will fail
  the whole run if the shared endpoint **rate-limits or times out**
  (`status:"error"`, `error:"the provider is rate-limited right now"` /
  `"read operation timed out"`). Retry, lower `max_papers`, or use a less
  contended endpoint. This is an endpoint-capacity issue, not a code fault.
- The build runs **critic-mode only** (template generator) and is
  **anomaly-gated** — a too-new/too-homogeneous corpus forms 0 anomalies and
  produces 0 hypotheses (an empty `selected_hypotheses.md`). Use
  `generate_ideas` (§3.3) to get a non-empty result from such a corpus.

#### `get_run_status(run_id)`

```json
{"name": "get_run_status", "arguments": {"run_id": "20260530-…"}}
```

Reads `<run_dir>/status.json`. Returns `{status, stage, progress, papers,
claims, …}`. Poll until `status` is `"done"` or `"error"`.

### 3.3 Idea-generation cascade (phase ②; non-empty guarantee)

These two tools solve the anomaly-gating problem: they fall through to
community bridges, per-paper LLM extensions, and a deterministic
abstract-seeded backstop, so the result is **never empty** as long as the run
has ≥1 paper. See `docs/idea-generation-pipeline.md` for the tier mechanics.

#### `generate_ideas(topic, run, min_ideas, as_markdown)`

```json
{"name": "generate_ideas",
 "arguments": {"topic":     "agentic memory for self-evolving agent",
               "run":       "20260602-133529-6a7d4f",
               "min_ideas": 5}}
```

Runs the six-tier cascade (A critic → B creator → C community-bridge →
D method-extension → E limitation-forward → F deterministic backstop) on an
**existing** run, stopping at `min_ideas` (clamped 1–20). Tiers A/B/C are
0-LLM reads of cached artifacts; D/E call the LLM and cache results to
`<run>/forward_ideas.jsonl` (repeat calls are then 0-LLM, sub-second). With
`as_markdown:true` (default) returns a rendered report string; with `false`
returns `{topic, run, ideas:[…], stats:{n_ideas, by_tier, tiers_used,
guaranteed_nonempty}}`. Does **not** build a corpus — pair with `start_run`.

#### `research_ideas(topic, max_papers, min_ideas, reuse, wait_seconds, as_markdown)`

```json
{"name": "research_ideas",
 "arguments": {"topic":        "bayesian memory",
               "max_papers":   20,
               "min_ideas":    5,
               "wait_seconds": 600}}
```

One-shot wrapper: **resolve-or-build → cascade**. With `reuse:true` (default)
it finds the best existing corpus whose topic matches (token coverage ≥ 0.34,
must have `claims.jsonl`, prefers `status:"done"` runs) and runs
`generate_ideas` immediately. Otherwise it submits a fresh build and polls up
to `wait_seconds` (clamped 0–1500; `0` = submit-and-return
`{status:"building", run_id}`). On a completed build it returns
`{status:"done", run, reused, stats, ideas_markdown|ideas}`. Registered only
when NOT readonly (the build path is paid). For a fresh topic, set
`wait_seconds≈600` (small corpora finish in ~3–5 min) or poll with
`get_run_status` then call `generate_ideas`.

> **Reuse threshold caveat:** a 2-token topic that shares one content word
> with an existing corpus (e.g. "bayesian **memory**" vs "agentic **memory**…")
> scores 0.5 ≥ 0.34 and will reuse the wrong corpus. Pass `reuse:false` to
> force a fresh build when the topic is genuinely new.

## 4. HTTP Host validation (and what is NOT protected)

The **`/mcp/` endpoint** rejects requests whose `Host` is not in the MCP
transport's allow-list. From outside the host:

```text
HTTP/1.1 421 Misdirected Request
Invalid Host header
```

This is the **MCP SDK's transport-security layer**
(`mcp/server/transport_security.py`: `enable_dns_rebinding_protection` +
`allowed_hosts`), applied by the streamable-HTTP transport to everything under
the `app.mount("/mcp", …)` sub-app. It is **not** uvicorn and **not** a
Starlette `TrustedHostMiddleware` (web.py adds only `CORSMiddleware`).
Spoofing the `Host` header does not bypass it. So the MCP tools are reachable
only same-host or via the SSH tunnel (§2.3). Empirically verified:
`/mcp/` → 421 from the public IP, `/mcp/` → 200 over the loopback/tunnel.

**The REST endpoints are NOT Host-protected.** `GET /`, `/api/runs`,
`/query?...`, and `/query/graph?...` answer `200` to any public client at
`http://8.208.118.99:8765`. They are read-only (no `start_run`), but they do
expose cached hypotheses + the run list with no auth. The paid `start_run` is
only reachable through `/mcp/`, which the Host check protects. If the REST
surface needs locking down, put a reverse proxy / IP allowlist in front, or
set the env gate documented in `web.py`. Port `8765` is otherwise open on the
public IP.

## 5. Operations

### 5.1 Health check

```bash
ssh admin@8.208.118.99 \
  'ss -ltnp | grep ":8765 "
   curl -s -m 5 -o /dev/null -w "%{http_code}\n" \
     -X POST http://127.0.0.1:8765/mcp/ \
     -H "Content-Type: application/json" \
     -H "Accept: application/json, text/event-stream" \
     -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}"'
```

Expected: one Python listener on `0.0.0.0:8765`; `tools/list` returns `200`.

### 5.2 Restart

```bash
ssh admin@8.208.118.99 '
tmux kill-session -t aigraph 2>/dev/null
sleep 1
cd ~/aigraph && tmux new-session -d -s aigraph \
  "cd ~/aigraph && .venv/bin/python -m aigraph.cli web \
     --host 0.0.0.0 --port 8765 --runs-root artifacts/runs \
     2>&1 | tee /tmp/aigraph_mcp.log"
sleep 6
ss -ltnp | grep ":8765 "
'
```

### 5.3 Logs

```bash
ssh admin@8.208.118.99 'tail -30 /tmp/aigraph_mcp.log'
# Or attach to the session (Ctrl-b d to detach):
ssh -t admin@8.208.118.99 'tmux attach -t aigraph'
```

### 5.4 Deploying code changes

`~/aigraph/` is a git clone of `feat/lcg-tfidf-relevance` (since 2026-06-01).
To roll out a change pushed to that branch:

```bash
ssh admin@8.208.118.99 'cd ~/aigraph && git pull --ff-only'
# any change under src/aigraph/*.py also requires the restart in §5.2
```

Server-only files (`.venv`, `MCP_README.md`, run-dir data files, `findings/`,
`outputs/`) are preserved by the clone — they live alongside the tracked
files. If a run dir needs new data files (`papers.jsonl`, `claims.jsonl`,
…), scp those individually; the sidecar `atlas_overlap.jsonl` is tracked
in the repo.

For backward reference, the older `scp scripts/aigraph_query.py …` flow
worked while `~/aigraph` was not git-tracked; that workflow is deprecated.

## 6. OMC Stage 3 integration

The Idea Generator (employee `00008`) consumes this MCP automatically:

- The client lives at
  `~/onemancompany/src/onemancompany/agents/aigraph_mcp_tools.py`. At OMC
  startup it loads five read-only tools (`list_runs`, `get_run_summary`,
  `query_hypotheses`, `get_idea_report`, `get_conflict_graph`) and registers
  them, scoped to employee `00008` only. `start_run` and `get_run_status`
  are deliberately not registered.
- The agent's instructions are at
  `~/onemancompany/.onemancompany/company/human_resource/employees/00008/skills/idea_generator/SKILL.md`
  with `autoload: true`. It directs the agent to call `get_idea_report` and
  write the returned text verbatim to `stage3_idea_generator.md`.
- The conflict-graph frontend (`onemancompany/frontend/src/lcg-graph.js`)
  resolves `window.LCG_GRAPH_BASE` from the page origin and calls
  `<origin>:8765/query/graph`. Same host, no Host-header issue.

**Start ordering matters**: aigraph must be running before the OMC process
imports `common_tools`, otherwise the MCP tool registration silently skips
and Stage 3 has no `get_idea_report` to call.

## 7. Known limitations

| Symptom | Cause | Remediation |
|---|---|---|
| `Invalid Host header` (421) from remote curl on `/mcp/` | MCP SDK transport-security DNS-rebinding check (`allowed_hosts`) | Use loopback / SSH-tunnel access patterns (§2.3) |
| `start_run` fails with `provider is rate-limited` / `read operation timed out` | Shared LLM endpoint (`litellm.yangtzeailab.com`, `DeepSeek-V4-Flash`) is throttled or slow under the per-paper claim-extraction load | Retry, lower `max_papers`, or point `AIGRAPH_BASE_URL` at a less contended endpoint |
| `start_run` produces 0 hypotheses / empty `selected_hypotheses.md` | Corpus formed 0 anomalies (too-new/too-homogeneous; both critic+creator are anomaly-gated) | Use `generate_ideas` / `research_ideas` (§3.3) — non-empty guarantee |
| Stage 3 reports "no tool `get_idea_report`" | OMC started before aigraph; tool registration skipped | Restart in order — aigraph, then OMC |
| Source change not reflected at runtime | `~/aigraph/` is not git-managed | `scp` + restart (§5.4, §5.2) |
| Delivered output contains "X on X" self-conflict | Pre-`68f3aa5` aigraph_query | Redeploy `scripts/aigraph_query.py` + restart |

## 8. Prerequisites (provisioning new hosts)

These are already installed on the current host; only relevant when
standing up a new deployment.

```bash
ssh admin@<host> 'cd ~/aigraph && .venv/bin/pip install \
  mcp fastapi "uvicorn[standard]" markdown \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/'
```

For broader operational context (initial deployment, Docker, Cloudflare
tunnel, corpus daemon), see `~/aigraph/MCP_README.md` on the host and
`DEPLOY.md` in the repository.
