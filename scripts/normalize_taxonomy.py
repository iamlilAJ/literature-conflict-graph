"""Domain-adaptive taxonomy normalizer — the P1 fix for fragmented anomalies.

Problem (friend's review-62): on a new domain the frozen claim extractor's
static CANONICAL_METHODS/TASKS don't fit, so most claims get canonical=other.
The frozen anomaly detector then falls back to the raw, one-off method/task
labels (CluE, FORGE, MobEvolve…) and groups by EXACT (method, task) — so every
paper becomes its own singleton group and no anomalies form.

Fix WITHOUT touching frozen code: re-canonicalize the claims into mid-level
buckets *derived from this corpus* (one LLM call clusters the raw labels), write
the buckets back into the writable canonical_method/canonical_task fields, then
re-run the unchanged frozen detector on the improved data. Fix the data, not
the detector.

Unlike a hardcoded memory-specific list, the buckets are mined from the
corpus's own vocabulary, so this works for any domain (memory, robotics, …).

Usage (run on the LLM host, or set the AIGRAPH_* env):
  python scripts/normalize_taxonomy.py --run artifacts/runs/<id>            # dry-run report
  python scripts/normalize_taxonomy.py --run artifacts/runs/<id> --apply    # write claims_normalized.jsonl
  python scripts/normalize_taxonomy.py --run artifacts/runs/<id> --apply --redetect  # + re-run anomalies
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from aigraph.io import read_jsonl, write_jsonl  # noqa: E402
from aigraph.models import Claim  # noqa: E402
from aigraph.llm_client import build_openai_client, call_llm_text, configured_model  # noqa: E402
from aigraph.anomalies import _group_by_method_task, _cluster_key  # noqa: E402

_PLACEHOLDER = {"other", "unknown", "misc", "n/a", "na", "none", "null", "", None}


def _raw_label(canon, raw):
    """The label the frozen detector would actually group on today."""
    if canon and str(canon).lower() not in _PLACEHOLDER:
        return str(canon).strip()
    if raw and str(raw).lower() not in _PLACEHOLDER:
        return str(raw).strip()
    return None


_SYS_METHOD = (
    "You are organizing a research corpus's taxonomy. Given a list of raw METHOD "
    "names extracted from papers on a topic (many are one-off names for the same "
    "underlying technique family), cluster them into 8-15 STABLE MID-LEVEL method "
    "categories that a reviewer would use to compare papers. Each category is a "
    "short kebab-case label (e.g. 'experience-replay', 'graph-memory', "
    "'self-evolution', 'memory-poisoning'). Output STRICT JSON mapping EVERY raw "
    'label to one category: {"<raw label>": "<mid-level-category>", ...}. '
    "Group aggressively — different paper-specific names for the same idea MUST "
    "share a category. Use 'other' only for genuinely uncategorizable labels."
)
_SYS_TASK = _SYS_METHOD.replace("METHOD", "TASK").replace("method categories",
    "task categories").replace("'experience-replay', 'graph-memory', "
    "'self-evolution', 'memory-poisoning'",
    "'agentic-memory', 'memory-consolidation', 'lifelong-adaptation', "
    "'gui-automation'")


def _build_mapping(client, model, topic, labels, system):
    """One LLM call → {raw_label: mid_level}. Falls back to identity on failure."""
    if not labels:
        return {}
    # Show labels with frequency so the model groups the important ones well.
    listed = "\n".join(f"- {lab}  (x{cnt})" for lab, cnt in labels)
    user = f"Topic: {topic}\n\nRaw labels (with frequency):\n{listed}"
    try:
        raw = call_llm_text(client, model=model, system=system, user=user,
                            max_tokens=4000)
    except Exception as exc:
        print(f"  [warn] mapping LLM call failed: {exc}", file=sys.stderr)
        return {}
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0:
        return {}
    try:
        m = json.loads(raw[s:e + 1])
    except Exception:
        return {}
    # normalize keys/values
    out = {}
    for k, v in m.items():
        if isinstance(v, str) and v.strip():
            out[str(k).strip().lower()] = v.strip().lower()
    return out


def _group_hist(claims):
    groups = _group_by_method_task(claims)
    sizes = Counter(len(v) for v in groups.values())
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    # how many multi-claim groups have a +/- or mixed split (anomaly-eligible)
    eligible = 0
    for v in multi.values():
        dirs = {c.direction for c in v}
        if "positive" in dirs and ("negative" in dirs or "mixed" in dirs):
            eligible += 1
    return {
        "n_groups": len(groups),
        "singletons": sizes.get(1, 0),
        "size2": sizes.get(2, 0),
        "size3plus": sum(n for s, n in sizes.items() if s >= 3),
        "multi_groups": len(multi),
        "anomaly_eligible_groups": eligible,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--topic", default=None, help="defaults to the run's query.txt")
    ap.add_argument("--apply", action="store_true", help="write claims_normalized.jsonl")
    ap.add_argument("--redetect", action="store_true", help="re-run detect_anomalies on normalized claims")
    ap.add_argument("--top-labels", type=int, default=120, help="max distinct labels per axis to send to the LLM")
    args = ap.parse_args()

    run = args.run
    claims = read_jsonl(run / "claims.jsonl", Claim)
    topic = args.topic or (run / "query.txt").read_text(encoding="utf-8").strip() if (run / "query.txt").exists() else (args.topic or run.name)
    print(f"run={run.name}  claims={len(claims)}  topic={topic!r}\n")

    # --- before ---
    before = _group_hist(claims)
    methods_raw = Counter(_raw_label(c.canonical_method, c.method) for c in claims)
    tasks_raw = Counter(_raw_label(c.canonical_task, c.task) for c in claims)
    methods_raw.pop(None, None); tasks_raw.pop(None, None)
    print(f"BEFORE: distinct methods={len(methods_raw)}  distinct tasks={len(tasks_raw)}")
    print(f"        groups={before['n_groups']}  singletons={before['singletons']} "
          f"size2={before['size2']} size3+={before['size3plus']} "
          f"multi={before['multi_groups']}  anomaly-eligible={before['anomaly_eligible_groups']}\n")

    # --- build mid-level mappings from the corpus vocabulary ---
    client, model = build_openai_client(), configured_model()
    print("clustering raw labels into mid-level taxonomy (2 LLM calls)...", file=sys.stderr)
    mmap = _build_mapping(client, model, topic, methods_raw.most_common(args.top_labels), _SYS_METHOD)
    tmap = _build_mapping(client, model, topic, tasks_raw.most_common(args.top_labels), _SYS_TASK)
    print(f"  method buckets: {len(set(mmap.values()))}  task buckets: {len(set(tmap.values()))}")

    # --- apply: rewrite canonical fields ---
    normalized = []
    for c in claims:
        d = c.model_dump()
        ml = _raw_label(c.canonical_method, c.method)
        tl = _raw_label(c.canonical_task, c.task)
        if ml and ml.lower() in mmap:
            d["canonical_method"] = mmap[ml.lower()]
        if tl and tl.lower() in tmap:
            d["canonical_task"] = tmap[tl.lower()]
        normalized.append(Claim(**d))

    after = _group_hist(normalized)
    nm = Counter(_raw_label(c.canonical_method, c.method) for c in normalized)
    nt = Counter(_raw_label(c.canonical_task, c.task) for c in normalized)
    nm.pop(None, None); nt.pop(None, None)
    print(f"\nAFTER:  distinct methods={len(nm)}  distinct tasks={len(nt)}")
    print(f"        groups={after['n_groups']}  singletons={after['singletons']} "
          f"size2={after['size2']} size3+={after['size3plus']} "
          f"multi={after['multi_groups']}  anomaly-eligible={after['anomaly_eligible_groups']}")
    print(f"\n  ANOMALY-ELIGIBLE GROUPS: {before['anomaly_eligible_groups']} → "
          f"{after['anomaly_eligible_groups']}  "
          f"(multi-claim groups: {before['multi_groups']} → {after['multi_groups']})")

    if args.apply:
        outp = run / "claims_normalized.jsonl"
        write_jsonl(outp, normalized)
        print(f"\nwrote {outp}")

    if args.redetect:
        from aigraph.graph import build_graph
        from aigraph.anomalies import detect_anomalies
        from aigraph.models import Paper
        papers = read_jsonl(run / "papers.jsonl", Paper) if (run / "papers.jsonl").exists() else []
        g0 = build_graph(claims, papers=papers)
        a0 = detect_anomalies(g0, claims)
        g1 = build_graph(normalized, papers=papers)
        a1 = detect_anomalies(g1, normalized)
        print(f"\n=== RE-DETECT anomalies: {len(a0)} (original) → {len(a1)} (normalized) ===")
        bt = Counter(a.type for a in a1)
        print(f"  normalized anomaly types: {dict(bt)}")
        if args.apply:
            write_jsonl(run / "anomalies_normalized.jsonl", a1)
            print(f"  wrote {run / 'anomalies_normalized.jsonl'}")


if __name__ == "__main__":
    raise SystemExit(main())
