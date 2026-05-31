"""Method 29 — atlas_overlap on creator_hypotheses.jsonl.

The MCP get_idea_report defaults kind="creator" (forward-looking research
ideas; ### a…#cr… IDs) — NOT kind="critic" (conflict explanations).
All prior Method 3/7/8/11/12/13/14/15/22/23/24/26 testing used critic hyps.
The shipped filter is therefore operating on creator hyps in production
without prior validation.

This builds the judge input for the 78 creator hyps; the next step is to
run run_method3_judge.py on the server and pull back results.
"""
from __future__ import annotations
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
RUN = _REPO / "artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1"
QUOTES_OUT = _REPO / "artifacts/atlas_test/method3_atlas_quotes.jsonl"

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_STOP = set("the a an and or of to in for on at by with from is are was were be been being this that these those it its as which whether also we us our paper papers method methods using uses used result results approach approaches model models task tasks based introduce introduces introduced propose proposes proposed work works show shows demonstrated demonstrate evaluate evaluation training train inference learning learned learn data dataset datasets large new novel can will may could would should they them their not but if then than so such all any more most some many one two three four five however moreover thus therefore while when where what who why because between within across over under above below into onto off out up down again here there now only own same other different study studies present existing state art baseline baselines benchmark benchmarks".split())


def toks(t):
    return [s.lower() for s in _TOKEN.findall(t or "") if s.lower() not in _STOP and len(s) > 2]


def tf(t):
    c = Counter(t); n = max(1, sum(c.values()))
    return {k: v / n for k, v in c.items()}


def idf_(docs):
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    N = len(docs)
    return {t: math.log((N + 1) / (v + 1)) + 1 for t, v in df.items()}


def tfidf(t, idf):
    f = tf(t); d = math.log(2)
    return {k: v * idf.get(k, d) for k, v in f.items()}


def cos(a, b):
    c = set(a) & set(b)
    if not c:
        return 0.0
    num = sum(a[k] * b[k] for k in c)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def hyp_text(h):
    sc = h.get("scope_conditions") or {}
    return " ".join(filter(None, [
        h.get("hypothesis", ""),
        h.get("mechanism", ""),
        " ".join(str(v) for v in sc.values() if v),
        (h.get("minimal_test") or "")[:200],
    ]))


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    quotes = [json.loads(l) for l in open(QUOTES_OUT)]
    qt = [toks(q["quote"]) for q in quotes]
    idf = idf_(qt)
    qvecs = [tfidf(t, idf) for t in qt]

    hyps = [json.loads(l) for l in open(RUN / "creator_hypotheses.jsonl") if l.strip()]
    print(f"creator hyps: {len(hyps)} | atlas quotes corpus: {len(quotes)}",
          file=sys.stderr)

    out = []
    for h in hyps:
        v = tfidf(toks(hyp_text(h)), idf)
        sims = sorted(((cos(v, qv), i) for i, qv in enumerate(qvecs)), reverse=True)
        top = [{"idx": i, "sim": round(s, 3),
                "dim": quotes[i]["dimension"],
                "severity": quotes[i]["severity"],
                "quote": quotes[i]["quote"][:400]}
               for s, i in sims[:K]]
        out.append({
            "hyp_id": h.get("hypothesis_id"),
            "anomaly_id": h.get("anomaly_id"),
            "pop": "creator",
            "hypothesis_text": hyp_text(h)[:800],
            "top_k_atlas": top,
        })

    op = _REPO / "artifacts/atlas_test/method29_judge_input.jsonl"
    with open(op, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} creator-hyp judge inputs → {op}", file=sys.stderr)


if __name__ == "__main__":
    main()
