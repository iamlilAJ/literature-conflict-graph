"""Method 3, stage 1 (local) — build the judge input.

Extract Atlas bottleneck quotes from joint_anomalies → corpus.
For each test hypothesis, find top-K most-similar Atlas quotes by TF-IDF.
Output one judge-input record per hypothesis with its top-K candidates.

The next stage (server-side run_method3_judge.py) reads this and runs
a single LLM-Likert call per hyp asking "how much does this hyp address
the same question as the closest Atlas bottleneck?"

This isolates Atlas's value as an EVALUATOR — not as a prompt context
(Method 2 showed prompt-injection hurts) or anomaly source (Method 6,
deferred). Anti-novelty: high Likert = hyp restates known bottleneck;
low Likert = hyp tackles a question Atlas doesn't already have on file.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
JOINT = _REPO / "artifacts/atlas_test/joint_anomalies.jsonl"
FWD = _REPO / "artifacts/atlas_test/forward_hyps.jsonl"
FROZEN_RUN = _REPO / "artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1"

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
    return [t.lower() for t in _TOKEN.findall(text or "")
            if t.lower() not in _STOP and len(t) > 2]


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


def hyp_text(h: dict) -> str:
    sc = h.get("scope_conditions") or {}
    parts = [
        h.get("hypothesis", ""),
        h.get("mechanism", ""),
        " ".join(str(v) for v in sc.values() if v),
        (h.get("minimal_test") or "")[:200],
    ]
    return " ".join(p for p in parts if p)


def main() -> int:
    n_per_pop = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    k_top = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    random.seed(43)

    # --- 1. Build Atlas bottleneck quotes corpus -------------------------
    joint = [json.loads(l) for l in open(JOINT) if l.strip()]
    quotes = []
    for j in joint:
        for sig in (j.get("bottleneck_signals") or []):
            q = (sig.get("quote") or "").strip()
            if len(q) > 40:  # drop fragments
                quotes.append({
                    "anomaly_id": j["anomaly_id"],
                    "dimension": sig.get("dimension"),
                    "severity": sig.get("severity"),
                    "source_title": sig.get("source_title", ""),
                    "quote": q,
                })
    # Dedup by quote text (some shared across anomalies)
    seen = set()
    unique = []
    for q in quotes:
        key = q["quote"][:200]
        if key in seen:
            continue
        seen.add(key)
        unique.append(q)
    quotes = unique
    print(f"Atlas quotes corpus: {len(quotes)} unique quotes "
          f"from {len(joint)} joint anomalies", file=sys.stderr)

    quote_tokens = [tokenize(q["quote"]) for q in quotes]
    idf = build_idf(quote_tokens)
    quote_vecs = [tfidf(t, idf) for t in quote_tokens]

    # --- 2. Sample test hyps ---------------------------------------------
    fwd_raw = [json.loads(l) for l in open(FWD) if l.strip()]
    random.shuffle(fwd_raw)
    forward_sample = fwd_raw[:n_per_pop]
    for h in forward_sample:
        h["_pop"] = "forward"

    frozen_path = FROZEN_RUN / "hypotheses_scored.jsonl"
    frozen_raw = [json.loads(l) for l in open(frozen_path) if l.strip()]
    random.shuffle(frozen_raw)
    frozen_sample = frozen_raw[:n_per_pop]
    for h in frozen_sample:
        h["_pop"] = "frozen"

    hyps = forward_sample + frozen_sample
    print(f"test hyps: {len(forward_sample)} forward + {len(frozen_sample)} frozen", file=sys.stderr)

    # --- 3. For each hyp, retrieve top-K Atlas quotes --------------------
    out = []
    for h in hyps:
        q_vec = tfidf(tokenize(hyp_text(h)), idf)
        sims = [(cosine(q_vec, v), i) for i, v in enumerate(quote_vecs)]
        sims.sort(reverse=True)
        top = [{"idx": i, "sim": round(s, 3),
                "dim": quotes[i]["dimension"],
                "severity": quotes[i]["severity"],
                "quote": quotes[i]["quote"][:400]}
               for s, i in sims[:k_top]]
        out.append({
            "hyp_id": h.get("hypothesis_id", "?"),
            "anomaly_id": h.get("anomaly_id"),
            "pop": h["_pop"],
            "hypothesis_text": hyp_text(h)[:800],
            "top_k_atlas": top,
        })

    out_path = _REPO / "artifacts/atlas_test/method3_judge_input.jsonl"
    with open(out_path, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Also stash the quotes corpus so the judge can show provenance
    quotes_path = _REPO / "artifacts/atlas_test/method3_atlas_quotes.jsonl"
    with open(quotes_path, "w") as fh:
        for q in quotes:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"wrote {len(out)} judge-input records → {out_path}", file=sys.stderr)
    print(f"wrote {len(quotes)} Atlas quotes → {quotes_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
