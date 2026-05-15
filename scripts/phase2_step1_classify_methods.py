"""Phase 2 Step 1: classify 540p claim.method values into 5 classes.

Classes: {real_method, model_name, metric, paraphrase, garbage}

Output: histogram + verbatim per-class samples.
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

SRC = Path("/Users/liuanjie/Documents/New project/hypothesis_generation/artifacts/runs/arxiv-reasoning-v0.7-540p/claims.jsonl")
OUT = Path(__file__).parent.parent / "artifacts/phase2_method_classification.json"
JUDGE_MODEL = "gpt-5.4-mini"
SEED = 17
N_SAMPLE = 300

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


def main():
    random.seed(SEED)

    # Load all claim.method values
    rows = []
    for line in SRC.open():
        c = json.loads(line)
        m = c.get("method")
        if isinstance(m, str) and m.strip():
            rows.append({
                "claim_id": c.get("claim_id"),
                "paper_id": c.get("paper_id"),
                "method": m.strip(),
                "task": c.get("task") or "",
            })
    print(f"540p claims with method: {len(rows)}", file=sys.stderr)
    sample = random.sample(rows, min(N_SAMPLE, len(rows)))
    print(f"Sampling {len(sample)} for classification", file=sys.stderr)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["AIGRAPH_BASE_URL"])

    classified = []
    for i, r in enumerate(sample):
        if i % 25 == 0:
            print(f"  classify {i}/{len(sample)}", file=sys.stderr)
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
            r["class"] = obj.get("class")
            r["rationale"] = obj.get("rationale")
        except Exception as exc:
            r["class"] = "ERROR"
            r["rationale"] = str(exc)[:100]
        classified.append(r)

    counter = Counter(r["class"] for r in classified)
    print("\n=== 5-class distribution (N=300) ===", file=sys.stderr)
    for cls, n in counter.most_common():
        print(f"  {cls}: {n}/{len(classified)} = {100*n/len(classified):.1f}%", file=sys.stderr)

    # Per-class verbatim samples (5 per class)
    by_cls = {}
    for r in classified:
        by_cls.setdefault(r["class"], []).append(r)
    print("\n=== verbatim samples per class ===", file=sys.stderr)
    for cls, items in by_cls.items():
        print(f"\n{cls} ({len(items)} total):", file=sys.stderr)
        for it in items[:5]:
            print(f"  - method={it['method']!r:50}  task={it['task'][:40]!r}", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "model": JUDGE_MODEL, "seed": SEED, "n": len(classified),
        "distribution": dict(counter),
        "rows": classified,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
