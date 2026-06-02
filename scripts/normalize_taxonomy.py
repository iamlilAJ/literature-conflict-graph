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


def _parse_json(raw):
    for op, cl in (("{", "}"), ("[", "]")):
        s, e = raw.find(op), raw.rfind(cl)
        if s >= 0 and e > s:
            try:
                return json.loads(raw[s:e + 1])
            except Exception:
                pass
    return None


def _propose_buckets(client, model, topic, labels, axis, n_buckets):
    """Stage 1: a SHORT call returning ~n_buckets mid-level category names.
    Weak models can do this (small output) where one-shot full clustering fails.
    """
    listed = ", ".join(lab for lab, _ in labels[:200])
    system = (
        f"You define a reviewer-level taxonomy. Given many raw {axis} names from "
        f"papers on a topic (most are one-off names for the same underlying "
        f"family), output a JSON list of AT MOST {n_buckets} broad, stable, "
        f"mid-level {axis} categories (kebab-case) that together cover them. "
        f"Categories must be general enough that several raw names map to each "
        f"(e.g. 'memory-poisoning', 'experience-replay', 'self-evolution'). "
        f"Output ONLY a JSON list of strings, no prose."
    )
    try:
        raw = call_llm_text(client, model=model, system=system,
                            user=f"Topic: {topic}\n\n{axis} names:\n{listed}",
                            max_tokens=1200)
    except Exception as exc:
        print(f"  [warn] propose-buckets failed: {exc}", file=sys.stderr)
        return []
    arr = _parse_json(raw)
    if not isinstance(arr, list):
        return []
    return [str(b).strip().lower() for b in arr if str(b).strip()][:n_buckets]


def _assign_to_buckets(client, model, topic, labels, buckets, axis, chunk=50):
    """Stage 2: force-assign each raw label to ONE of the fixed buckets. The
    constrained output space (only `buckets` allowed) is what forces merging.
    Chunked so the JSON never truncates. Values snapped to the bucket set."""
    bucket_set = set(buckets)
    mapping = {}
    names = [lab for lab, _ in labels]
    blist = ", ".join(buckets)
    system = (
        f"Assign each {axis} label to EXACTLY ONE category from this fixed list:\n"
        f"[{blist}]\n"
        f"Use ONLY these categories — never invent new ones. Pick the closest. "
        f'Output STRICT JSON {{"<label>": "<category>", ...}} for EVERY label.'
    )
    for i in range(0, len(names), chunk):
        part = names[i:i + chunk]
        try:
            raw = call_llm_text(client, model=model, system=system,
                                user="Labels:\n" + "\n".join(f"- {n}" for n in part),
                                max_tokens=4000)
        except Exception as exc:
            print(f"  [warn] assign chunk failed: {exc}", file=sys.stderr)
            continue
        m = _parse_json(raw)
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if not isinstance(v, str):
                continue
            vv = v.strip().lower()
            if vv in bucket_set:
                mapping[str(k).strip().lower()] = vv
            else:
                # snap an invented value to a bucket by substring overlap, else 'other'
                hit = next((b for b in buckets if b in vv or vv in b), None)
                mapping[str(k).strip().lower()] = hit or "other"
    return mapping


def _build_mapping(client, model, topic, labels, axis, n_buckets=12):
    """Two-stage: propose ~n_buckets categories, then force-assign every label."""
    if not labels:
        return {}
    buckets = _propose_buckets(client, model, topic, labels, axis, n_buckets)
    if not buckets:
        return {}
    return _assign_to_buckets(client, model, topic, labels, buckets, axis)


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
    print("clustering raw labels into mid-level taxonomy (two-stage)...", file=sys.stderr)
    mmap = _build_mapping(client, model, topic, methods_raw.most_common(args.top_labels), "method", n_buckets=12)
    tmap = _build_mapping(client, model, topic, tasks_raw.most_common(args.top_labels), "task", n_buckets=14)
    print(f"  method: {len(mmap)} labels → {len(set(mmap.values()))} buckets | "
          f"task: {len(tmap)} labels → {len(set(tmap.values()))} buckets")

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
