"""Phase 2 Step 3: A/B classify both OLD and NEW 100p method extractions.

Use same 5-class judge as Step 1. Compare purity.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv

load_dotenv("/Users/liuanjie/Documents/New project/hypothesis_generation/.env")
from openai import OpenAI

OLD = Path("/tmp/100p_claims_OLD.jsonl")
NEW = Path("/tmp/100p_claims_NEW.jsonl")
OUT = Path("/Users/liuanjie/Documents/New project/hypothesis_generation/artifacts/phase2_ab_comparison.json")

JUDGE_MODEL = "gpt-5.4-mini"
SEED = 17

SYSTEM_PROMPT = (
    "You are auditing an ML claim extractor that mistakenly puts non-method strings "
    "into the 'method' field. Given a (method_string, task_context) extracted from "
    "an AI paper, classify it as exactly ONE of:\n\n"
    "- real_method: a genuine ML method name (e.g. 'RAG', 'Chain-of-Thought', "
    "'Self-Refine', 'Tree-of-Thought', 'DPO'). Includes novel methods with "
    "specific names ('SpatialRGPT', 'KD-Encoder').\n"
    "- model_name: a model name not a method (e.g. 'GPT-4', 'Llama-70B', "
    "'Qwen2.5-VL', 'o1-preview', 'Claude Sonnet 4').\n"
    "- metric: an evaluation metric (e.g. 'Top-1 accuracy', 'F1', 'BLEU', "
    "'pass@k').\n"
    "- paraphrase: a descriptive phrase NOT a method name "
    "(e.g. 'comprehensive evaluation', 'LLM-based planner', "
    "'multi-agent self-correction approach', 'fine-tuning' alone).\n"
    "- garbage: extraction error, gibberish, or text not classifiable above.\n\n"
    "Output STRICT JSON: {\"class\": \"<one_of_5>\", \"rationale\": \"<one sentence>\"}"
)


def load_methods(path):
    rows = []
    for line in path.open():
        c = json.loads(line)
        m = c.get("method")
        if isinstance(m, str) and m.strip():
            rows.append({"method": m.strip(), "task": c.get("task") or ""})
    return rows


def classify(rows, label):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["AIGRAPH_BASE_URL"])
    out = []
    for i, r in enumerate(rows):
        if i % 50 == 0:
            print(f"  [{label}] {i}/{len(rows)}", file=sys.stderr)
        user = f"method: {r['method']!r}\ntask: {r['task']!r}\n\nClassify."
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_tokens=150, temperature=0,
            )
            raw = resp.choices[0].message.content
            s, e = raw.find("{"), raw.rfind("}")
            obj = json.loads(raw[s:e+1]) if s != -1 else {}
            out.append({**r, "class": obj.get("class"), "rationale": obj.get("rationale")})
        except Exception as exc:
            out.append({**r, "class": "ERROR", "rationale": str(exc)[:100]})
    return out


def main():
    random.seed(SEED)
    old_rows = load_methods(OLD)
    new_rows = load_methods(NEW)
    print(f"OLD methods: {len(old_rows)} from /tmp/100p_claims_OLD.jsonl", file=sys.stderr)
    print(f"NEW methods: {len(new_rows)} from /tmp/100p_claims_NEW.jsonl", file=sys.stderr)

    # Classify both fully (no sampling — purity is the metric)
    print("\n=== classifying OLD ===", file=sys.stderr)
    old_cls = classify(old_rows, "OLD")
    print("=== classifying NEW ===", file=sys.stderr)
    new_cls = classify(new_rows, "NEW")

    old_c = Counter(r["class"] for r in old_cls)
    new_c = Counter(r["class"] for r in new_cls)

    print("\n========================================", file=sys.stderr)
    print("A/B PURITY COMPARISON (100p mini cohort)", file=sys.stderr)
    print("========================================", file=sys.stderr)
    classes = ["real_method", "model_name", "metric", "paraphrase", "garbage", "ERROR"]
    print(f"{'class':<14} {'OLD n':>8} {'OLD %':>8} {'NEW n':>8} {'NEW %':>8} {'Δ':>8}", file=sys.stderr)
    for cls in classes:
        o = old_c.get(cls, 0); n = new_c.get(cls, 0)
        op = 100*o/max(1, len(old_cls)); np = 100*n/max(1, len(new_cls))
        d = np - op
        print(f"{cls:<14} {o:>8} {op:>7.1f}% {n:>8} {np:>7.1f}% {d:>+7.1f}", file=sys.stderr)

    OUT.write_text(json.dumps({
        "model": JUDGE_MODEL,
        "old_total": len(old_cls), "new_total": len(new_cls),
        "old_distribution": dict(old_c), "new_distribution": dict(new_c),
        "old_rows": old_cls, "new_rows": new_cls,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}", file=sys.stderr)

    old_real = old_c.get("real_method", 0) / max(1, len(old_cls))
    new_real = new_c.get("real_method", 0) / max(1, len(new_cls))
    delta = (new_real - old_real) * 100
    print(f"\n=== HEADLINE ===", file=sys.stderr)
    print(f"real_method purity:  OLD {100*old_real:.1f}%  ->  NEW {100*new_real:.1f}%  (Δ {delta:+.1f} pp)", file=sys.stderr)
    if delta >= 10:
        print(f"PASS: improvement ≥ 10pp", file=sys.stderr)
    elif delta >= 5:
        print(f"BORDERLINE: improvement {delta:+.1f}pp — discuss before commit", file=sys.stderr)
    else:
        print(f"FAIL: improvement {delta:+.1f}pp — do not commit", file=sys.stderr)


if __name__ == "__main__":
    main()
