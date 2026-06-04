"""HindSight-style label-free quality oracle for aigraph hypotheses.

For each hypothesis, find the closest paper among those submitted *after* a
temporal cutoff (default Jan 2025), using TF-IDF cosine over title+abstract.
A high best-match means the hypothesis is shaped like a real later-published
research direction.

Honest scope: our hypotheses were generated with full-corpus access, so this
is *retrospective* HindSight — a relative quality signal between two hyp
populations, not a true prospective test. For a true prospective test, the
generator would have to be cut off from post-2501 papers at run time.

Usage:
  python scripts/score_hindsight.py
  python scripts/score_hindsight.py --run arxiv-reasoning-v0.7-540p-thaw1 --cutoff 2501
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

_REPO = Path(__file__).resolve().parent.parent

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_STOP = set("""
the a an and or of to in for on at by with from is are was were be been being
this that these those it its as which whether also we us our paper papers
method methods using uses used result results approach approaches model
models task tasks based introduce introduces introduced propose proposes
proposed work works show shows demonstrated demonstrate evaluate evaluation
training train inference learning learned learn data dataset datasets large
new novel can will may could would should they them their not but if then
than so such all any more most some many one two three four five however
moreover thus therefore while when where what who why because between
within across over under above below into onto off out up down again here
there now only own same other different study studies present existing
state art baseline baselines benchmark benchmarks
""".split())


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if t.lower() not in _STOP and len(t) > 2]


def tf(tokens: list[str]) -> dict[str, float]:
    c = Counter(tokens)
    n = max(1, sum(c.values()))
    return {t: v / n for t, v in c.items()}


def build_idf(doc_tokens: list[list[str]]) -> dict[str, float]:
    df: Counter = Counter()
    for d in doc_tokens:
        for t in set(d):
            df[t] += 1
    N = len(doc_tokens)
    return {t: math.log((N + 1) / (v + 1)) + 1 for t, v in df.items()}


def tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    t = tf(tokens)
    default_idf = math.log(2)
    return {k: v * idf.get(k, default_idf) for k, v in t.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    num = sum(a[k] * b[k] for k in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def yymm(paper: dict) -> int | None:
    aid = paper.get("arxiv_id_base") or paper.get("arxiv_id_full") or ""
    if "." in aid:
        try:
            return int(aid.split(".")[0])
        except ValueError:
            return None
    return None


def paper_text(p: dict) -> str:
    return f"{p.get('title', '')}. {p.get('abstract', '')}"


def hyp_text(h: dict) -> str:
    sc = h.get("scope_conditions") or {}
    parts = [
        h.get("hypothesis", ""),
        h.get("mechanism", ""),
        " ".join(str(v) for v in sc.values() if v),
        (h.get("minimal_test") or "")[:200],
    ]
    return " ".join(p for p in parts if p)


def summarise(label: str, rows: list[dict]) -> dict:
    ss = [r["best_sim"] for r in rows]
    if not ss:
        print(f"\n=== {label} === empty")
        return {}
    out = {
        "n": len(ss),
        "mean": mean(ss),
        "sd": stdev(ss) if len(ss) > 1 else 0.0,
        "median": median(ss),
        "frac_gt_25": sum(1 for s in ss if s > 0.25) / len(ss),
        "frac_gt_35": sum(1 for s in ss if s > 0.35) / len(ss),
        "frac_gt_45": sum(1 for s in ss if s > 0.45) / len(ss),
    }
    print(f"\n=== {label} ({out['n']} hyps) ===")
    print(f"  best-sim:  mean={out['mean']:.3f}±{out['sd']:.3f}  median={out['median']:.3f}")
    print(f"  fraction with best-sim > 0.25: {out['frac_gt_25']:.1%}")
    print(f"  fraction with best-sim > 0.35: {out['frac_gt_35']:.1%}")
    print(f"  fraction with best-sim > 0.45: {out['frac_gt_45']:.1%}")
    top = sorted(rows, key=lambda r: -r["best_sim"])[:5]
    print("  top-5 matches:")
    for r in top:
        title = (r["best_paper_title"] or "")[:90]
        print(f"    {r['hyp_id'][:18]:18}  sim={r['best_sim']:.3f}  → {r['best_paper_id']}: {title}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="arxiv-reasoning-v0.7-540p-thaw1",
                    help="Run dir name under artifacts/runs/")
    ap.add_argument("--cutoff", type=int, default=2501,
                    help="arxiv YYMM cutoff; papers >= cutoff are 'future'")
    ap.add_argument("--forward", default="artifacts/atlas_test/forward_hyps.jsonl",
                    help="Forward-framing hyps to compare (optional)")
    ap.add_argument("--out", default="artifacts/atlas_test/hindsight_scores.jsonl")
    args = ap.parse_args()

    run_dir = _REPO / "artifacts/runs" / args.run
    papers = [json.loads(l) for l in open(run_dir / "papers.jsonl") if l.strip()]
    src = [p for p in papers if (yymm(p) or 9999) < args.cutoff]
    fut = [p for p in papers if (yymm(p) or 0) >= args.cutoff]
    print(f"corpus: {len(papers)} total, {len(src)} pre-{args.cutoff}, {len(fut)} post-{args.cutoff}")
    if not fut:
        print("no future papers — nothing to match against", file=sys.stderr)
        return 1

    fut_tokens = [tokenize(paper_text(p)) for p in fut]
    idf = build_idf(fut_tokens)
    fut_vecs = [tfidf(toks, idf) for toks in fut_tokens]

    frozen = [json.loads(l) for l in open(run_dir / "hypotheses_scored.jsonl") if l.strip()]
    forward = []
    fwd_path = _REPO / args.forward
    if fwd_path.exists():
        forward = [json.loads(l) for l in open(fwd_path) if l.strip()]
    print(f"hyps: {len(frozen)} frozen | {len(forward)} forward")

    def score(hyps: list[dict]) -> list[dict]:
        rows = []
        for h in hyps:
            q = tfidf(tokenize(hyp_text(h)), idf)
            best_i, best_s = -1, -1.0
            for i, v in enumerate(fut_vecs):
                s = cosine(q, v)
                if s > best_s:
                    best_s, best_i = s, i
            rows.append({
                "hyp_id": h.get("hypothesis_id", "?"),
                "anomaly_id": h.get("anomaly_id", "?"),
                "best_sim": best_s,
                "best_paper_id": fut[best_i].get("paper_id") if best_i >= 0 else None,
                "best_paper_title": fut[best_i].get("title") if best_i >= 0 else None,
            })
        return rows

    rows_fz = score(frozen)
    summarise("FROZEN", rows_fz)
    rows_fw = []
    if forward:
        rows_fw = score(forward)
        summarise("FORWARD", rows_fw)

        # Permutation test on mean delta
        import random
        random.seed(0)
        a = [r["best_sim"] for r in rows_fw]
        b = [r["best_sim"] for r in rows_fz]
        delta = mean(a) - mean(b)
        pool = a + b
        n_a = len(a)
        worse = 0
        ITER = 2000
        for _ in range(ITER):
            random.shuffle(pool)
            d = mean(pool[:n_a]) - mean(pool[n_a:])
            if abs(d) >= abs(delta):
                worse += 1
        p = worse / ITER
        print(f"\n=== forward − frozen Δmean = {delta:+.3f}  permutation p = {p:.3f} (n_iter={ITER})")

    out_path = _REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows_fz:
            f.write(json.dumps({**r, "pop": "frozen"}) + "\n")
        for r in rows_fw:
            f.write(json.dumps({**r, "pop": "forward"}) + "\n")
    print(f"\nwrote {len(rows_fz)+len(rows_fw)} per-hyp scores → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
