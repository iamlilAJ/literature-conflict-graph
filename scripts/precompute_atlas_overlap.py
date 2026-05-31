"""End-to-end Atlas overlap precompute for a run directory.

Wraps the 3-step Method 13 operator workflow as one command:

  1. LOCAL: For each hypothesis in <run>/hypotheses_scored.jsonl (and
     optionally creator_hypotheses.jsonl), retrieve top-K Atlas bottleneck
     quotes via TF-IDF and write a judge-input JSONL.
  2. REMOTE: scp judge input to <host>, kick off run_method3_judge.py in
     a tmux session, poll until completion.
  3. LOCAL: pull judge output back, convert to sidecar format, write
     <run>/atlas_overlap.jsonl alongside hypotheses_scored.jsonl. If the
     sidecar already exists, merge (preserving any prior records).

After this completes, query-time consumers (scripts/aigraph_query.py with
--min-atlas-overlap 3, or the MCP get_idea_report) automatically use the
new sidecar. The aigraph production server pulls fresh sidecars via the
caller's normal scp procedure — this script handles local-side compute.

Usage:
    python scripts/precompute_atlas_overlap.py \\
        --run artifacts/runs/<run-id> \\
        [--host admin@8.208.118.99] \\
        [--remote-script /tmp/run_method3_judge.py] \\
        [--include-creator]
"""
from __future__ import annotations
import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_QUOTES = _REPO / "artifacts/atlas_test/method3_atlas_quotes.jsonl"
DEFAULT_HOST = "admin@8.208.118.99"
DEFAULT_REMOTE_SCRIPT = "/tmp/run_method3_judge.py"


# --- TF-IDF retrieval (same as method3_build_retrieval.py) ----------------
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_STOP = set("the a an and or of to in for on at by with from is are was were be been being this that these those it its as which whether also we us our paper papers method methods using uses used result results approach approaches model models task tasks based introduce introduces introduced propose proposes proposed work works show shows demonstrated demonstrate evaluate evaluation training train inference learning learned learn data dataset datasets large new novel can will may could would should they them their not but if then than so such all any more most some many one two three four five however moreover thus therefore while when where what who why because between within across over under above below into onto off out up down again here there now only own same other different study studies present existing state art baseline baselines benchmark benchmarks".split())


def toks(t: str) -> list[str]:
    return [s.lower() for s in _TOKEN.findall(t or "")
            if s.lower() not in _STOP and len(s) > 2]


def tf(t: list[str]) -> dict[str, float]:
    c = Counter(t); n = max(1, sum(c.values()))
    return {k: v / n for k, v in c.items()}


def build_idf(docs: list[list[str]]) -> dict[str, float]:
    df: Counter = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    N = len(docs)
    return {t: math.log((N + 1) / (v + 1)) + 1 for t, v in df.items()}


def tfidf(t: list[str], idf: dict[str, float]) -> dict[str, float]:
    f = tf(t); d = math.log(2)
    return {k: v * idf.get(k, d) for k, v in f.items()}


def cos(a: dict, b: dict) -> float:
    c = set(a) & set(b)
    if not c:
        return 0.0
    num = sum(a[k] * b[k] for k in c)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def hyp_text(h: dict) -> str:
    sc = h.get("scope_conditions") or {}
    return " ".join(filter(None, [
        h.get("hypothesis", ""),
        h.get("mechanism", ""),
        " ".join(str(v) for v in sc.values() if v),
        (h.get("minimal_test") or "")[:200],
    ]))


# --- Pipeline stages -------------------------------------------------------


def build_judge_input(run_dir: Path, include_creator: bool,
                      quotes_path: Path, k_top: int = 5) -> tuple[Path, list[dict]]:
    """Stage 1 — LOCAL. Returns (output_jsonl_path, atlas_quotes_corpus)."""
    quotes = [json.loads(l) for l in open(quotes_path)]
    qt = [toks(q["quote"]) for q in quotes]
    idf = build_idf(qt)
    qvecs = [tfidf(t, idf) for t in qt]

    paths = [run_dir / "hypotheses_scored.jsonl"]
    if include_creator and (run_dir / "creator_hypotheses.jsonl").exists():
        paths.append(run_dir / "creator_hypotheses.jsonl")

    hyps = []
    for p in paths:
        kind = "creator" if p.name.startswith("creator_") else "critic"
        for l in open(p):
            l = l.strip()
            if not l:
                continue
            h = json.loads(l)
            h["_kind"] = kind
            hyps.append(h)
    print(f"[stage 1] retrieving top-{k_top} Atlas quotes for {len(hyps)} hyps "
          f"(corpus: {len(quotes)} quotes)", file=sys.stderr)

    out = []
    for h in hyps:
        v = tfidf(toks(hyp_text(h)), idf)
        sims = sorted(((cos(v, qv), i) for i, qv in enumerate(qvecs)), reverse=True)
        top = [{"idx": i, "sim": round(s, 3),
                "dim": quotes[i]["dimension"],
                "severity": quotes[i]["severity"],
                "quote": quotes[i]["quote"][:400]}
               for s, i in sims[:k_top]]
        out.append({
            "hyp_id": h.get("hypothesis_id"),
            "anomaly_id": h.get("anomaly_id"),
            "pop": h["_kind"],
            "hypothesis_text": hyp_text(h)[:800],
            "top_k_atlas": top,
        })

    out_path = run_dir / "atlas_judge_input.jsonl"
    with open(out_path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[stage 1] wrote {len(out)} judge inputs → {out_path}", file=sys.stderr)
    return out_path, quotes


def run_remote_judge(input_path: Path, host: str, remote_script: str,
                     workers: int = 6) -> Path:
    """Stage 2 — REMOTE. scp input to /tmp, run judge in tmux, poll for ALL_DONE,
    pull output back. Returns local output path."""
    remote_input = "/tmp/atlas_judge_input.jsonl"
    remote_output = "/tmp/atlas_judge_output.jsonl"
    remote_stderr = "/tmp/atlas_judge.stderr"
    print(f"[stage 2] scp {input_path} → {host}:{remote_input}", file=sys.stderr)
    subprocess.run(["scp", "-q", str(input_path),
                    f"{host}:{remote_input}"], check=True)

    tmux_cmd = (
        f"tmux kill-session -t atlas_judge 2>/dev/null; "
        f"tmux new-session -d -s atlas_judge "
        f"'cd /tmp && /home/admin/onemancompany/.venv/bin/python3 {remote_script} "
        f"{remote_input} {remote_output} {workers} 2>{remote_stderr}; "
        f"echo ALL_DONE >>{remote_stderr}'"
    )
    subprocess.run(["ssh", host, tmux_cmd], check=True)
    print(f"[stage 2] judge tmux session armed; polling for ALL_DONE...", file=sys.stderr)
    while True:
        r = subprocess.run(["ssh", host, f"grep -q '^ALL_DONE' {remote_stderr} 2>/dev/null && echo done || echo wait"],
                           capture_output=True, text=True)
        if "done" in r.stdout:
            break
        time.sleep(30)
    print(f"[stage 2] judge completed; pulling output", file=sys.stderr)

    out_path = input_path.parent / "atlas_judge_output.jsonl"
    subprocess.run(["scp", "-q", f"{host}:{remote_output}", str(out_path)],
                   check=True)
    return out_path


def write_sidecar(judge_output: Path, run_dir: Path) -> Path:
    """Stage 3 — LOCAL. Convert judge output to sidecar format, merging
    with any existing sidecar (the existing record wins for back-compat)."""
    sidecar_path = run_dir / "atlas_overlap.jsonl"
    existing = {}
    if sidecar_path.exists():
        for l in open(sidecar_path):
            try:
                r = json.loads(l)
                existing[r["hypothesis_id"]] = r
            except Exception:
                continue
    n_existing = len(existing)
    n_added = 0
    n_overwritten = 0
    for l in open(judge_output):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        if "error" in r or not r.get("_judge_ok"):
            continue
        rec = {
            "hypothesis_id": r["hyp_id"],
            "anomaly_id": r["anomaly_id"],
            "atlas_overlap": r["atlas_overlap"],
            "forward_looking": r["forward_looking"],
            "named_mechanism": r["named_mechanism"],
            "single_variable_test": r["single_variable_test"],
            "specific_scope": r["specific_scope"],
            "closest_quote": r.get("closest_quote", ""),
            "closest_dim": r.get("closest_dim", ""),
            "why": r.get("why", ""),
            "kind": r.get("pop", "unknown"),
        }
        if rec["hypothesis_id"] in existing:
            n_overwritten += 1
        else:
            n_added += 1
        existing[rec["hypothesis_id"]] = rec

    with open(sidecar_path, "w") as f:
        for r in existing.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[stage 3] sidecar {sidecar_path}: "
          f"{len(existing)} total ({n_added} new, {n_overwritten} overwritten, "
          f"{n_existing} prior)", file=sys.stderr)

    overlap_dist = Counter(r["atlas_overlap"] for r in existing.values())
    print(f"[stage 3] overlap distribution: {dict(sorted(overlap_dist.items()))}",
          file=sys.stderr)
    return sidecar_path


def main():
    ap = argparse.ArgumentParser(
        description="Compute per-hypothesis Atlas overlap scores for a run.")
    ap.add_argument("--run", required=True, type=Path,
                    help="Path to artifacts/runs/<run-id>/")
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help=f"SSH host running the Likert judge (default: {DEFAULT_HOST})")
    ap.add_argument("--remote-script", default=DEFAULT_REMOTE_SCRIPT,
                    help=f"Path to run_method3_judge.py on the host "
                         f"(default: {DEFAULT_REMOTE_SCRIPT}). Verify with: "
                         f"ssh <host> 'ls {DEFAULT_REMOTE_SCRIPT}'")
    ap.add_argument("--quotes", default=str(DEFAULT_QUOTES), type=Path,
                    help="Atlas bottleneck quotes corpus (built by Method 3)")
    ap.add_argument("--k-top", type=int, default=5,
                    help="Top-K TF-IDF Atlas quotes per hypothesis to feed the judge")
    ap.add_argument("--workers", type=int, default=6,
                    help="Remote LLM workers (Kimi rate limits at ~6-8)")
    ap.add_argument("--include-creator", action="store_true",
                    help="Also score creator_hypotheses.jsonl (default off; "
                         "the MCP defaults kind=creator so usually want this on)")
    ap.add_argument("--skip-judge", action="store_true",
                    help="Skip stages 2-3 — only build the judge input. For "
                         "ops that prefer to run the judge manually.")
    args = ap.parse_args()

    if not args.run.exists():
        print(f"error: run dir {args.run} does not exist", file=sys.stderr)
        return 1
    if not args.quotes.exists():
        print(f"error: atlas quotes corpus {args.quotes} does not exist; "
              f"run scripts/method3_build_retrieval.py first to build it",
              file=sys.stderr)
        return 1

    judge_input, _ = build_judge_input(args.run, args.include_creator,
                                       args.quotes, args.k_top)
    if args.skip_judge:
        print("--skip-judge set; not running stages 2-3", file=sys.stderr)
        return 0

    judge_output = run_remote_judge(judge_input, args.host, args.remote_script,
                                     args.workers)
    sidecar = write_sidecar(judge_output, args.run)
    print(f"\nOK. Sidecar at {sidecar}.")
    print("To activate in production: scp this file to the matching path on the "
          "aigraph server (~/aigraph/artifacts/runs/<run>/atlas_overlap.jsonl).",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
