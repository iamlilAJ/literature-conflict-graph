"""Q1: Method-namespace coverage probe.

Sample 200 random claims from 540p run, fuzzy-match each claim's `method`
field against Atlas V_M (8155 method_names from data/intern_atlas/data/paper_methods/).

For unmatched ones, LLM judge with gpt-5.4-mini.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, "/Users/liuanjie/Documents/New project/hypothesis_generation/src")

from dotenv import load_dotenv

load_dotenv("/Users/liuanjie/Documents/New project/hypothesis_generation/.env")
from openai import OpenAI

SRC_CLAIMS = Path(
    "/Users/liuanjie/Documents/New project/hypothesis_generation/artifacts/runs/arxiv-reasoning-v0.7-540p/claims.jsonl"
)
ATLAS_PM = Path(
    "/Users/liuanjie/Documents/New project/hypothesis_generation/data/intern_atlas/data/paper_methods"
)
OUT_JSON = Path(__file__).parent / "Q1_method_map.json"

JUDGE_MODEL = "gpt-5.4-mini"
NORM_THRESHOLD = 0.92  # ratio
TOKEN_THRESHOLD = 0.85
SEED = 17
N_SAMPLE = 200


def main():
    random.seed(SEED)

    # 1. Load Atlas V_M
    pm = pl.scan_parquet(str(ATLAS_PM / "*.parquet"))
    methods_df = pm.select("method_id", "method_name").unique("method_id").collect()
    atlas_names = methods_df["method_name"].to_list()
    atlas_ids = methods_df["method_id"].to_list()
    name_to_id = dict(zip(atlas_names, atlas_ids))
    print(f"Atlas V_M: {len(atlas_names)} unique method names", file=sys.stderr)

    # 2. Sample 200 claims with method field
    all_claims_with_method = []
    for line in SRC_CLAIMS.open():
        c = json.loads(line)
        m = c.get("method")
        if isinstance(m, str) and m.strip():
            all_claims_with_method.append({
                "claim_id": c.get("claim_id"),
                "paper_id": c.get("paper_id"),
                "method": m.strip(),
                "canonical_method": c.get("canonical_method"),
                "task": c.get("task"),
            })
    sample = random.sample(all_claims_with_method, min(N_SAMPLE, len(all_claims_with_method)))
    print(f"Sampled {len(sample)} claims (from {len(all_claims_with_method)} with method)", file=sys.stderr)

    # 3. Fuzzy match each claim's method against Atlas
    # Use rapidfuzz.process.extractOne with two scorer functions:
    #   - fuzz.ratio (normalized Levenshtein-ish ratio)
    #   - fuzz.token_set_ratio (token-bag match)
    # Also exact match by lowercased equality.
    atlas_lower_to_idx = {n.lower(): i for i, n in enumerate(atlas_names)}

    rows = []
    for c in sample:
        m = c["method"]
        m_lower = m.lower().strip()
        match_type = None
        match_score = 0.0
        matched_name = None
        matched_id = None

        # Exact match (case-insensitive)
        if m_lower in atlas_lower_to_idx:
            i = atlas_lower_to_idx[m_lower]
            match_type = "exact"
            match_score = 1.0
            matched_name = atlas_names[i]
            matched_id = atlas_ids[i]
        else:
            # ratio match
            r = process.extractOne(m, atlas_names, scorer=fuzz.ratio)
            if r and r[1] >= NORM_THRESHOLD * 100:
                matched_name = r[0]
                matched_id = name_to_id[r[0]]
                match_type = "ratio"
                match_score = r[1] / 100.0
            else:
                # token_set match
                tsr = process.extractOne(m, atlas_names, scorer=fuzz.token_set_ratio)
                if tsr and tsr[1] >= TOKEN_THRESHOLD * 100:
                    matched_name = tsr[0]
                    matched_id = name_to_id[tsr[0]]
                    match_type = "token_set"
                    match_score = tsr[1] / 100.0

        rows.append({
            **c,
            "match_type": match_type,  # None if unmatched
            "match_score": round(match_score, 4),
            "matched_atlas_name": matched_name,
            "matched_atlas_id": matched_id,
            "judge_label": None,
            "judge_rationale": None,
        })

    # 4. Stats so far
    from collections import Counter
    counter = Counter(r["match_type"] for r in rows)
    print(f"match types: {dict(counter)}", file=sys.stderr)
    unmatched = [r for r in rows if r["match_type"] is None]
    print(f"unmatched: {len(unmatched)} (will judge with LLM)", file=sys.stderr)

    # 5. LLM judge unmatched
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["AIGRAPH_BASE_URL"],
    )
    JUDGE_PROMPT = (
        "You are auditing whether a method name extracted from an AI paper claim "
        "should appear in a canonical method registry of 8,155 ML methods.\n\n"
        "Given the method name and the surrounding task context, classify it as ONE of:\n\n"
        "- novel_method: a genuine method that's likely a real specific ML technique "
        "(e.g. 'TaSR-RAG', 'K-RagRec') and not in our registry\n"
        "- alias_not_in_Atlas: a known method but expressed in non-canonical form "
        "(e.g. 'Retrieval-Augmented Generation' for what Atlas might call 'RAG')\n"
        "- too_generic: descriptive phrase, not a method name proper "
        "(e.g. 'fine-tuning', 'chain-of-thought prompting' — generic technique families)\n"
        "- garbage: extraction error, model name only, or non-method string "
        "(e.g. 'GPT-4', 'Llama' — those are models not methods, or pure noise)\n\n"
        "Output STRICT JSON: {\"label\": \"<one_of_4>\", \"rationale\": \"<one sentence>\"}"
    )
    def JUDGE_USER(m, t):
        task_str = repr(t) if t else "(none)"
        return f"Method: {m!r}\nTask context: {task_str}\n\nClassify."

    for i, r in enumerate(unmatched):
        if i % 20 == 0:
            print(f"  judging {i}/{len(unmatched)}", file=sys.stderr)
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": JUDGE_USER(r["method"], r["task"])},
                ],
                max_tokens=150,
                temperature=0,
            )
            raw = resp.choices[0].message.content
            # Tolerant parse: find first { ... }
            s = raw.find("{")
            e = raw.rfind("}")
            obj = json.loads(raw[s : e + 1]) if s != -1 else {}
            r["judge_label"] = obj.get("label")
            r["judge_rationale"] = obj.get("rationale")
        except Exception as exc:
            r["judge_label"] = "ERROR"
            r["judge_rationale"] = f"call failed: {str(exc)[:100]}"

    # 6. Write artifact
    payload = {
        "model": JUDGE_MODEL,
        "endpoint_note": "OpenAI-style endpoint; brief specified Haiku-4.5 but endpoint blocks Anthropic models",
        "thresholds": {"normalized_ratio": NORM_THRESHOLD, "token_set": TOKEN_THRESHOLD},
        "n_sample": len(rows),
        "n_atlas_methods": len(atlas_names),
        "seed": SEED,
        "match_type_counts": dict(counter),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT_JSON}", file=sys.stderr)

    # 7. Print summary
    print("\n=== MATCH RATE ===", file=sys.stderr)
    n_total = len(rows)
    n_matched = sum(1 for r in rows if r["match_type"] is not None)
    print(f"  matched (any method): {n_matched}/{n_total} = {100*n_matched/n_total:.1f}%", file=sys.stderr)
    for tt in ("exact", "ratio", "token_set"):
        n = counter.get(tt, 0)
        print(f"    {tt}: {n}", file=sys.stderr)
    print(f"  unmatched: {counter.get(None,0)} ({100*counter.get(None,0)/n_total:.1f}%)", file=sys.stderr)
    judge_counter = Counter(r["judge_label"] for r in rows if r["judge_label"])
    print(f"  judge labels (on unmatched): {dict(judge_counter)}", file=sys.stderr)


if __name__ == "__main__":
    main()
