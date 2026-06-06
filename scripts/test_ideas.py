#!/usr/bin/env python3
"""Test the aigraph idea-generation MCP tools — friendly, no-dependency client.

Run this ON the aigraph server (it talks to 127.0.0.1:8765 by default), or
anywhere with an SSH tunnel (`ssh -L 18765:127.0.0.1:8765 admin@<host>` then
set AIGRAPH_MCP=http://localhost:18765/mcp/).

Usage:
  python3 test_ideas.py                      # smoke test: health + list runs + ideas on an existing run
  python3 test_ideas.py "your topic"         # research_ideas on a topic (reuse a matching corpus if any)
  python3 test_ideas.py "new topic" --build  # FRESH build: start_run -> poll progress -> ideas (resumable)
  python3 test_ideas.py "t" --build --detach # submit the build, print run_id, don't wait
  python3 test_ideas.py --resume <run_id>    # reconnect to a build: poll to done, then ideas
  python3 test_ideas.py "t" --run <run_id>   # generate_ideas on a specific existing run
  python3 test_ideas.py --list               # just list available runs

Env:
  AIGRAPH_MCP   MCP endpoint (default http://127.0.0.1:8765/mcp/)
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

MCP = os.environ.get("AIGRAPH_MCP", "http://127.0.0.1:8765/mcp/")
_ID = 0


def mcp_call(method, params, timeout=120):
    """POST a JSON-RPC call to the MCP, parse the SSE reply, return the result."""
    global _ID
    _ID += 1
    body = json.dumps({"jsonrpc": "2.0", "id": _ID, "method": method,
                       "params": params}).encode("utf-8")
    req = urllib.request.Request(
        MCP, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"})
    raw = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")
    # streamable-HTTP frames each chunk as "data: {json}"
    m = re.search(r"data:\s*(\{.*\})", raw, re.DOTALL)
    payload = m.group(1) if m else raw
    obj = json.loads(payload)
    if "error" in obj:
        raise RuntimeError(f"MCP error: {obj['error']}")
    return obj.get("result", obj)


def tool_call(name, arguments, timeout=120):
    """Call a tool and unwrap its content into a Python value (dict or str)."""
    res = mcp_call("tools/call", {"name": name, "arguments": arguments}, timeout)
    sc = res.get("structuredContent")
    if isinstance(sc, dict) and sc:
        # FastMCP wraps a bare value as {"result": value}; unwrap that
        return sc.get("result", sc)
    content = res.get("content") or []
    if content and content[0].get("type") == "text":
        text = content[0]["text"]
        try:
            return json.loads(text)
        except Exception:
            return text  # a plain markdown string
    return res


def list_runs():
    runs = tool_call("list_runs", {}, timeout=30)
    if not isinstance(runs, list):
        runs = runs.get("result", runs) if isinstance(runs, dict) else []
    return runs


def print_result(out):
    """Pretty-print whatever a tool returned (string, ideas dict, or status)."""
    if isinstance(out, str):
        print(out)
        return
    if not isinstance(out, dict):
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    status = out.get("status")
    if status == "building":
        print(f"⏳ corpus is BUILDING — run_id: {out.get('run_id')}")
        print(f"   poll:  python3 {sys.argv[0]} --status {out.get('run_id')}")
        print(f"   then:  python3 {sys.argv[0]} \"<topic>\" --run {out.get('run_id')}")
        return
    if status == "error":
        # Distinguish failure source so operators don't chase the wrong thing:
        # a paper-retrieval provider (arxiv/openalex) 429 is NOT the LLM endpoint
        # being rate-limited (issue #42).
        src = str(out.get("source") or "").lower()
        kind = str(out.get("error_kind") or "").lower()
        msg = out.get("message") or out.get("error") or "unknown error"
        print(f"❌ build failed: {msg}")
        if kind == "rate_limit" and src in ("arxiv", "openalex"):
            print(f"   (the {src} PAPER-RETRIEVAL provider returned HTTP 429 — this is NOT the")
            print("    shared LLM endpoint. Wait and retry, lower --max-papers, or switch source.)")
        elif kind == "rate_limit":
            print("   (the shared LLM endpoint is rate-limited — retry or lower --max-papers)")
        else:
            print(f"   (failure source: {src or 'unknown'}, kind: {kind or 'error'})")
        return
    md = out.get("ideas_markdown")
    stats = out.get("stats", {})
    if status:
        print(f"✅ status={status}  run={out.get('run')}  reused={out.get('reused')}")
    if stats:
        print(f"   {stats.get('n_ideas')} ideas | tiers={stats.get('tiers_used')} "
              f"| by_tier={stats.get('by_tier')} | nonempty={stats.get('guaranteed_nonempty')}")
    print()
    if md:
        print(md)
    elif out.get("ideas"):
        for i, idea in enumerate(out["ideas"], 1):
            print(f"{i}. [{idea.get('tier_label')}] {idea.get('title')}")
            print(f"   {idea.get('statement','')[:300]}")
            if idea.get("minimal_test"):
                print(f"   test: {idea['minimal_test'][:160]}")
            print()
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


def poll_until_done(run_id, *, interval=8, max_wait=1800):
    """Poll get_run_status with SHORT requests until the run finishes (issue #53).

    This replaces a single long blocking MCP request: a dropped connection or
    timeout no longer looks like a failed run, the user gets live progress, and
    the run can be reconnected with --resume. Returns the final status dict."""
    waited = 0
    last = None
    while True:
        try:
            st = tool_call("get_run_status", {"run_id": run_id}, timeout=30)
        except Exception as e:
            # a transient poll error is not a run failure — keep polling
            print(f"  (poll hiccup: {str(e)[:60]} — retrying; run continues server-side)")
            time.sleep(interval); waited += interval
            if waited >= max_wait:
                return {"status": "building", "run_id": run_id}
            continue
        if not isinstance(st, dict):
            return {"status": "unknown", "raw": st}
        status, stage = st.get("status"), st.get("stage")
        key = (status, stage, st.get("papers"))
        if key != last:
            print(f"  [{status}/{stage}] papers={st.get('papers')} "
                  f"quality={st.get('retrieval_quality')} {str(st.get('message') or '')[:72]}")
            last = key
        if status in ("done", "complete", "error"):
            return st
        if waited >= max_wait:
            print(f"  (still running after {max_wait}s — it continues server-side; "
                  f"reconnect: python3 {sys.argv[0]} --resume {run_id})")
            return st
        time.sleep(interval); waited += interval


def build_async(topic, *, max_papers, min_ideas, max_wait, detach):
    """submit (start_run) → poll → generate_ideas. Resumable, non-blocking (#53)."""
    print(f"→ start_run(topic={topic!r}, max_papers={max_papers}) ...")
    sub = tool_call("start_run", {"topic": topic, "max_papers": max_papers}, timeout=60)
    run_id = sub.get("run_id") if isinstance(sub, dict) else None
    if not run_id:
        print("  start_run did not return a run_id:")
        print_result(sub)
        return 1
    print(f"  run_id: {run_id}")
    print(f"  reconnect anytime:  python3 {sys.argv[0]} --resume {run_id}\n")
    if detach:
        print("  (--detach: submitted; not waiting. Use --resume to pick up the ideas.)")
        return 0
    return finish_run(topic, run_id, min_ideas=min_ideas, max_wait=max_wait)


def finish_run(topic, run_id, *, min_ideas, max_wait):
    """Poll a run to completion (if needed) then generate ideas from it."""
    st = tool_call("get_run_status", {"run_id": run_id}, timeout=30)
    if isinstance(st, dict) and st.get("status") not in ("done", "complete", "error"):
        st = poll_until_done(run_id, max_wait=max_wait)
    if isinstance(st, dict) and st.get("status") == "error":
        print_result(st)
        return 1
    if isinstance(st, dict) and st.get("status") not in ("done", "complete"):
        return 0  # still building — user can --resume later
    print(f"\n→ generate_ideas(run={run_id}, topic={topic!r}) ...\n")
    out = tool_call("generate_ideas",
                    {"topic": topic, "run": run_id, "min_ideas": min_ideas, "as_markdown": True},
                    timeout=300)
    print_result(out)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Test aigraph idea-generation tools")
    ap.add_argument("topic", nargs="?", help="research topic")
    ap.add_argument("--run", help="generate_ideas on this existing run_id")
    ap.add_argument("--build", action="store_true",
                    help="force a fresh corpus build (start_run + poll + generate_ideas)")
    ap.add_argument("--resume", metavar="RUN_ID",
                    help="reconnect to an in-progress/finished build: poll to done, then generate ideas")
    ap.add_argument("--detach", action="store_true",
                    help="with --build: submit and print run_id without waiting (resume later)")
    ap.add_argument("--wait", type=int, default=0,
                    help="max seconds to poll a build before backgrounding it (0=default 1800)")
    ap.add_argument("--max-papers", type=int, default=20)
    ap.add_argument("--min-ideas", type=int, default=5)
    ap.add_argument("--status", metavar="RUN_ID", help="poll a run's build status once")
    ap.add_argument("--list", action="store_true", help="list available runs and exit")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args()

    print(f"# MCP endpoint: {MCP}\n")

    # health
    try:
        tools = mcp_call("tools/list", {}, timeout=15).get("tools", [])
        names = [t["name"] for t in tools]
        print(f"✓ MCP up — {len(names)} tools: {', '.join(names)}\n")
        for need in ("generate_ideas", "research_ideas"):
            if need not in names:
                print(f"⚠ tool '{need}' not registered — server may be on old code\n")
    except Exception as e:
        print(f"✗ cannot reach MCP at {MCP}: {e}")
        print("  On the server use the default (127.0.0.1:8765). From your laptop open an SSH")
        print("  tunnel first:  ssh -L 18765:127.0.0.1:8765 admin@8.208.118.99")
        print("  then:          AIGRAPH_MCP=http://localhost:18765/mcp/ python3 test_ideas.py")
        return 1

    if args.status:
        print_result(tool_call("get_run_status", {"run_id": args.status}, timeout=30))
        return 0

    if args.resume:
        max_wait = args.wait if args.wait > 0 else 1800
        topic = args.topic
        if not topic:
            try:
                st0 = tool_call("get_run_status", {"run_id": args.resume}, timeout=30)
                topic = (st0.get("topic") if isinstance(st0, dict) else None) or "research ideas"
            except Exception:
                topic = "research ideas"
        return finish_run(topic, args.resume, min_ideas=args.min_ideas, max_wait=max_wait)

    runs = list_runs()
    print(f"✓ {len(runs)} run(s) available:")
    for r in runs[:10]:
        print(f"    {r.get('id')}  ({r.get('n_papers')} papers, {r.get('n_hypotheses')} hyps)")
    print()
    if args.list:
        return 0

    # 1) explicit run → generate_ideas
    if args.run:
        topic = args.topic or "research ideas"
        print(f"→ generate_ideas(run={args.run}, topic={topic!r}) ...\n")
        out = tool_call("generate_ideas",
                        {"topic": topic, "run": args.run,
                         "min_ideas": args.min_ideas, "as_markdown": True},
                        timeout=300)
        print(json.dumps(out, ensure_ascii=False, indent=2) if args.json else "")
        print_result(out)
        return 0

    # 2) topic given
    if args.topic:
        if args.build:
            # fresh build via the async submit+poll+resume flow (#53)
            max_wait = args.wait if args.wait > 0 else 1800
            return build_async(args.topic, max_papers=args.max_papers,
                               min_ideas=args.min_ideas, max_wait=max_wait, detach=args.detach)
        # reuse path: research_ideas returns fast when a matching corpus exists.
        # Cap the blocking wait so a would-be build doesn't hang the request; if it
        # reports building, point the user at the async --resume flow.
        wait = min(args.wait, 120)
        print(f"→ research_ideas(topic={args.topic!r}, reuse=True) ...\n")
        out = tool_call("research_ideas",
                        {"topic": args.topic, "max_papers": args.max_papers,
                         "min_ideas": args.min_ideas, "reuse": True,
                         "wait_seconds": wait, "as_markdown": True},
                        timeout=max(60, wait + 30))
        print(json.dumps(out, ensure_ascii=False, indent=2) if args.json else "")
        print_result(out)
        if isinstance(out, dict) and out.get("status") == "building" and out.get("run_id"):
            print(f"\n  building a fresh corpus — reconnect: "
                  f"python3 {sys.argv[0]} --resume {out['run_id']}")
        return 0

    # 3) no args → smoke test on the first available run
    if not runs:
        print("no runs to test on — pass a topic to build one:  python3 test_ideas.py \"some topic\" --build --wait 600")
        return 0
    run = runs[0]
    topic = "research ideas"
    try:
        # use the run's own topic if we can read it via get_run_summary
        summ = tool_call("get_run_summary", {"run": run["id"]}, timeout=30)
        if isinstance(summ, dict):
            topic = summ.get("topic") or topic
    except Exception:
        pass
    print(f"→ smoke test: generate_ideas(run={run['id']}, topic={topic!r}) ...\n")
    out = tool_call("generate_ideas",
                    {"topic": topic, "run": run["id"],
                     "min_ideas": args.min_ideas, "as_markdown": False},
                    timeout=300)
    print_result(out)
    print("\n# OK — to test your own topic:  python3 test_ideas.py \"your topic\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
