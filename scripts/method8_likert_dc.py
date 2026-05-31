"""Method 8 — Re-judge Workflow D vs Workflow C under the Likert rubric.

Method 1's verdict (workflow-d-vs-c-verdict.md) was inconclusive because
the binary 4-axis rubric ceiling-saturated both arms at 4/4. Method 3's
Likert judge gives 1-5 per axis + atlas_overlap. This re-judges the same
14 hC and 14 hD outputs from wfd_out2.jsonl under that Likert.

Reuses run_method3_judge.py unchanged (same input format).
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
WFD_OUT = _REPO / "artifacts/atlas_test/wfd_out2.jsonl"
QUOTES_OUT = _REPO / "artifacts/atlas_test/method3_atlas_quotes.jsonl"

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_STOP = set("the a an and or of to in for on at by with from is are was were be been being this that these those it its as which whether also we us our paper papers method methods using uses used result results approach approaches model models task tasks based introduce introduces introduced propose proposes proposed work works show shows demonstrated demonstrate evaluate evaluation training train inference learning learned learn data dataset datasets large new novel can will may could would should they them their not but if then than so such all any more most some many one two three four five however moreover thus therefore while when where what who why because between within across over under above below into onto off out up down again here there now only own same other different study studies present existing state art baseline baselines benchmark benchmarks".split())


def tokenize(t):
    return [s.lower() for s in _TOKEN.findall(t or "") if s.lower() not in _STOP and len(s) > 2]


def tf(toks):
    c = Counter(toks); n = max(1, sum(c.values()))
    return {t: v / n for t, v in c.items()}


def build_idf(docs):
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    N = len(docs)
    return {t: math.log((N + 1) / (v + 1)) + 1 for t, v in df.items()}


def tfidf(toks, idf):
    t = tf(toks); di = math.log(2)
    return {k: v * idf.get(k, di) for k, v in t.items()}


def cosine(a, b):
    common = set(a) & set(b)
    if not common:
        return 0.0
    num = sum(a[k] * b[k] for k in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def main():
    k_top = 5
    quotes = [json.loads(l) for l in open(QUOTES_OUT)]
    qtoks = [tokenize(q["quote"]) for q in quotes]
    idf = build_idf(qtoks)
    qvecs = [tfidf(t, idf) for t in qtoks]

    rows = [json.loads(l) for l in open(WFD_OUT) if l.strip()]
    ok = [r for r in rows if "error" not in r and r.get("hC") and r.get("hD")]
    print(f"D vs C pairs available: {len(ok)}", file=sys.stderr)

    out = []
    for r in ok:
        for label in ("C", "D"):
            text = r[f"h{label}"]
            hv = tfidf(tokenize(text), idf)
            sims = sorted(((cosine(hv, v), i) for i, v in enumerate(qvecs)), reverse=True)
            top = [{"idx": i, "sim": round(s, 3),
                    "dim": quotes[i]["dimension"],
                    "severity": quotes[i]["severity"],
                    "quote": quotes[i]["quote"][:400]}
                   for s, i in sims[:k_top]]
            out.append({
                "hyp_id": f"{label.lower()}_{r['anomaly_id']}",
                "anomaly_id": r["anomaly_id"],
                "pop": f"workflow_{label}",  # workflow_C or workflow_D
                "hypothesis_text": text[:800],
                "top_k_atlas": top,
            })

    op = _REPO / "artifacts/atlas_test/method8_judge_input.jsonl"
    with open(op, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} judge-input records ({len(ok)} pairs) → {op}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
