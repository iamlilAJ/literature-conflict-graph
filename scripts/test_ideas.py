#!/usr/bin/env python3
"""Test the aigraph idea-generation MCP tools — friendly, no-dependency client.

Run this ON the aigraph server (it talks to 127.0.0.1:8765 by default), or
anywhere with an SSH tunnel (`ssh -L 18765:127.0.0.1:8765 admin@<host>` then
set AIGRAPH_MCP=http://localhost:18765/mcp/).

Usage:
  python3 test_ideas.py                      # smoke test: health + list runs + ideas on an existing run
  python3 test_ideas.py "your topic"         # research_ideas on a topic (reuse a matching corpus if any)
  python3 test_ideas.py "new topic" --build  # force a FRESH corpus build (slow, paid, may rate-limit)
  python3 test_ideas.py "t" --build --wait 600   # build and block up to 600s for ideas
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
        print(f"❌ build failed: {out.get('message')}")
        print("   (usually the shared LLM endpoint is rate-limited — retry or lower --max-papers)")
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


def main():
    ap = argparse.ArgumentParser(description="Test aigraph idea-generation tools")
    ap.add_argument("topic", nargs="?", help="research topic")
    ap.add_argument("--run", help="generate_ideas on this existing run_id")
    ap.add_argument("--build", action="store_true",
                    help="force a fresh corpus build (reuse=false)")
    ap.add_argument("--wait", type=int, default=0,
                    help="seconds to block waiting for a fresh build (0=submit-and-return)")
    ap.add_argument("--max-papers", type=int, default=20)
    ap.add_argument("--min-ideas", type=int, default=5)
    ap.add_argument("--status", metavar="RUN_ID", help="poll a run's build status")
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

    # 2) topic given → research_ideas (resolve-or-build)
    if args.topic:
        print(f"→ research_ideas(topic={args.topic!r}, reuse={not args.build}, "
              f"wait_seconds={args.wait}) ...\n")
        out = tool_call("research_ideas",
                        {"topic": args.topic, "max_papers": args.max_papers,
                         "min_ideas": args.min_ideas, "reuse": not args.build,
                         "wait_seconds": args.wait, "as_markdown": True},
                        timeout=max(60, args.wait + 30))
        print(json.dumps(out, ensure_ascii=False, indent=2) if args.json else "")
        print_result(out)
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
