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
| Endpoint | `http://127.0.0.1:8765/mcp/` (loopback only — see §4) |
| Transport | Streamable HTTP (MCP spec) |
| Process supervisor | `tmux` session `aigraph` |
| Repository | `github.com/iamlilAJ/literature-conflict-graph`, branch `stable/v0.7-runner-local` |
| Server-side install | `~/aigraph/` (scp-deployed; not a git clone) |
| Available run | `arxiv-reasoning-v0.7-540p-thaw1` (474 papers, 2 011 claims, 300 hypotheses) |

The MCP, browser UI, query endpoint, and conflict-graph endpoint are served by
a single uvicorn process bound to `0.0.0.0:8765`. CORS is open; there is no
authentication layer in front of the MCP — exposure is restricted by HTTP
Host validation (§4).

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

The endpoint is bound to `0.0.0.0:8765` but uvicorn returns
`421 Misdirected Request` for any request whose `Host` header is not the
loopback name. This is a deliberate boundary. For supported external
integration:

- **SSH tunnel (recommended for local IDE / Mac clients)** —
  ```bash
  ssh -L 18765:127.0.0.1:8765 admin@8.208.118.99
  ```
  Then point any local MCP client at `http://localhost:18765/mcp/`. The
  Host header is `localhost`, which uvicorn accepts. Picking a non-8765
  local port (e.g. 18765) avoids colliding with other local services.
  For a backgrounded tunnel: add `-fN` flags. Verified working:
  `curl http://localhost:18765/mcp/ → 200 OK`.
- **Bastion access** — `ssh` into the host and invoke either pattern above
  (loopback curl from inside the server itself).
- **Reverse proxy** — terminate TLS in front of the endpoint and rewrite the
  `Host` header to `localhost`; expose only the rewritten origin.
- **Re-binding** — start uvicorn with `--forwarded-allow-ips '*'` and adjust
  trusted hosts in `web.py`. Not currently configured.

The OMC frontend's conflict-graph component (`frontend/src/lcg-graph.js`)
already follows the same-host pattern; see §6.

## 3. Tool reference

All seven tools follow the JSON-RPC 2.0 envelope. The five read-only tools
make zero LLM calls and respond sub-second; the two run-trigger tools depend
on a configured LLM endpoint and are currently inert (§7).

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

### 3.2 Run-trigger (paid; currently inert)

#### `start_run(topic, max_papers, generator)`

```json
{"name": "start_run",
 "arguments": {"topic":      "sparse mixture of experts",
               "max_papers": 50,
               "generator":  "llm"}}
```

Spawns a full pipeline run via `SearchService.submit`. Returns
`{run_id, status: "queued", poll_with: "get_run_status"}` immediately. The
underlying LLM provider declared in `aigraph/.env`
(`sub2api.us.justafish.top`) is currently unreachable, so submitted runs will
fail in their fetch/extract stage. Restoring this surface requires repointing
`AIGRAPH_BASE_URL` to a working endpoint (the OMC LiteLLM proxy is one
option).

#### `get_run_status(run_id)`

```json
{"name": "get_run_status", "arguments": {"run_id": "20260530-…"}}
```

Reads `<run_dir>/status.json`. Returns the run's stage/progress/done state.
Only meaningful once `start_run` is functional.

## 4. HTTP Host validation

Direct requests to `http://8.208.118.99:8765/mcp/` from outside the host
return:

```text
HTTP/1.1 400 Bad Request
Content-Length: 19
Invalid Host header
```

This is uvicorn's default trusted-host enforcement. Spoofing the `Host`
header on a remote curl does not bypass it, because uvicorn applies the
check before request routing. Supported access paths are enumerated in
§2.3.

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
| `Invalid Host header` from remote curl | uvicorn TrustedHost check | Use loopback access patterns (§2.3) |
| `start_run` fails or hangs | Configured LLM endpoint (`sub2api.us.justafish.top`) is unreachable | Repoint `AIGRAPH_BASE_URL` to a working provider |
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
