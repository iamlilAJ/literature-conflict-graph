"""Method 3, stage 2 (server-side) — LLM-Likert judge.

For each judge-input record (one hyp + top-K Atlas bottleneck quotes),
ask Kimi:
  1. Which quote is closest to addressing the same research question?
  2. Likert 1-5: how much does the hyp address the SAME question as that
     closest quote? (1 = different question; 5 = exact restatement of a
     known bottleneck)
  3. Brief justification.

Also gives a Likert 1-5 for our 4 rubric criteria (forward_looking,
named_mechanism, single_variable_test, specific_scope) — this prototypes
the rubric upgrade (task #31) and gives us a same-call multi-axis read.

Inputs:
  method3_judge_input.jsonl  (from method3_build_retrieval.py)
Outputs:
  method3_judge_output.jsonl

Usage on server:
  python3 run_method3_judge.py method3_judge_input.jsonl method3_judge_output.jsonl 4
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from statistics import mean

from openai import OpenAI  # type: ignore


ENV_PATH = "/home/admin/onemancompany/.onemancompany/.env"
if os.path.exists(ENV_PATH):
    for ln in open(ENV_PATH):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, _, v = ln.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

_CLI = OpenAI(
    api_key=os.environ["CUSTOM_API_KEY"],
    base_url=os.environ["DEFAULT_API_BASE_URL"],
    timeout=420,
    max_retries=3,
)
_MODEL = os.environ["DEFAULT_LLM_MODEL"]


SYS_JUDGE = (
    "You evaluate a research hypothesis against a list of KNOWN BOTTLENECKS "
    "(direct quotes from prior papers identifying open problems in this area). "
    "Output STRICT JSON, no markdown, no fences:\n"
    '{"closest_idx": <0..K-1 index of the bottleneck most semantically '
    'similar to the hypothesis>,\n'
    ' "atlas_overlap": <1..5 Likert — how much the hypothesis addresses the '
    'SAME research question as the closest bottleneck. 1 = totally different '
    'question; 3 = related area, distinct mechanism; 5 = essentially restates '
    'a known open question>,\n'
    ' "forward_looking": <1..5 Likert — proposes a new mechanism/direction '
    'to test (5) vs merely restates a known disagreement (1)>,\n'
    ' "named_mechanism": <1..5 — names a concrete causal mechanism (5) vs '
    'vague (1)>,\n'
    ' "single_variable_test": <1..5 — implies a controlled experiment '
    'varying ONE thing (5) vs no test or many-variable (1)>,\n'
    ' "specific_scope": <1..5 — names a concrete method+task (5) vs generic (1)>,\n'
    ' "why": "one sentence justifying atlas_overlap and the closest_idx pick"}'
)


def _chat(system: str, user: str, mt: int = 5000, temp: float = 0.0) -> str:
    for _ in range(2):
        try:
            r = _CLI.chat.completions.create(
                model=_MODEL, max_tokens=mt, temperature=temp,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            txt = (r.choices[0].message.content or "").strip()
            if txt:
                return txt
        except Exception:
            time.sleep(6)
    return ""


def _parse(raw: str) -> dict:
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0:
        return {}
    try:
        return json.loads(raw[s:e + 1])
    except Exception:
        return {}


def _to_int(v, lo=1, hi=5) -> int:
    try:
        n = int(v)
        return max(lo, min(hi, n))
    except Exception:
        return lo


def judge_one(rec: dict) -> dict:
    top = rec["top_k_atlas"]
    quotes_block = "\n".join(
        f"  [{i}] dim={q['dim']} severity={q['severity']} sim={q['sim']}: \"{q['quote']}\""
        for i, q in enumerate(top)
    )
    user = (
        f"HYPOTHESIS:\n{rec['hypothesis_text']}\n\n"
        f"KNOWN BOTTLENECKS (top-{len(top)} most-similar Atlas open questions):\n"
        f"{quotes_block}"
    )
    raw = _chat(SYS_JUDGE, user, mt=4000)
    o = _parse(raw)
    out = {
        "hyp_id": rec["hyp_id"],
        "anomaly_id": rec["anomaly_id"],
        "pop": rec["pop"],
        "closest_idx": _to_int(o.get("closest_idx", 0), 0, len(top) - 1) if top else None,
        "atlas_overlap": _to_int(o.get("atlas_overlap")),
        "forward_looking": _to_int(o.get("forward_looking")),
        "named_mechanism": _to_int(o.get("named_mechanism")),
        "single_variable_test": _to_int(o.get("single_variable_test")),
        "specific_scope": _to_int(o.get("specific_scope")),
        "_judge_ok": bool(o),
        "why": (o.get("why") or "")[:240],
    }
    if out["closest_idx"] is not None and top:
        cq = top[out["closest_idx"]]
        out["closest_quote"] = cq["quote"][:240]
        out["closest_dim"] = cq["dim"]
    return out


def main() -> None:
    inp, outp = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    recs = [json.loads(l) for l in open(inp) if l.strip()]
    print(f"Method 3 judge on {len(recs)} hyps, {_MODEL}, {workers} workers",
          file=sys.stderr)
    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(judge_one, r) for r in recs]
        for i, f in enumerate(as_completed(futures)):
            try:
                out.append(f.result())
            except Exception as ex_:
                out.append({"error": f"{type(ex_).__name__}"})
            if i % 4 == 0:
                print(f"  {i}/{len(recs)} ({time.time() - t0:.0f}s)",
                      file=sys.stderr)
    with open(outp, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in out if "error" not in r and r.get("_judge_ok")]
    if ok:
        keys = ("atlas_overlap", "forward_looking", "named_mechanism",
                "single_variable_test", "specific_scope")
        def mean_pop(pop: str, key: str) -> float:
            vs = [r[key] for r in ok if r["pop"] == pop]
            return round(mean(vs), 2) if vs else 0.0

        print(json.dumps({
            "n_ok": len(ok),
            "n_err": len(out) - len(ok),
            "by_pop": {
                "forward": {k: mean_pop("forward", k) for k in keys},
                "frozen":  {k: mean_pop("frozen", k) for k in keys},
            },
            "atlas_overlap_dist": dict(Counter(r["atlas_overlap"] for r in ok)),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
