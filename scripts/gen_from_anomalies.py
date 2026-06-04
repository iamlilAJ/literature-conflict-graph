"""Generate forward-framed, anti-template ideas from (normalized) anomalies,
with an inline critic pass. End-to-end test of: normalize taxonomy (P1) +
forward generation (#4) + critic filter (#6) — does it produce the friend's
"good directions" instead of vacuous bridges / moderator-variable templates?

Reads <run>/anomalies_normalized.jsonl + claims_normalized.jsonl (fallback to
the originals). Writes <run>/anomaly_ideas.jsonl. Run on the LLM host.

  python scripts/gen_from_anomalies.py --run artifacts/runs/<id> --max-anomalies 12
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
from aigraph.io import read_jsonl  # noqa: E402
from aigraph.models import Anomaly, Claim  # noqa: E402
from aigraph.llm_client import build_openai_client, call_llm_text, configured_model  # noqa: E402

_GEN_SYS = (
    "You are a research scientist. You are given a CONFLICT between papers (an "
    "anomaly): a shared method/task where claims disagree. Propose ONE forward-"
    "looking, testable research hypothesis that would RESOLVE the conflict.\n"
    "HARD REQUIREMENTS:\n"
    "- Name a concrete causal MECHANISM. BANNED phrasings (auto-reject): "
    "'unreported moderator variable', 'prompt/decoding/preprocessing confound', "
    "'X and Y may share an evaluation protocol', any generic 'a confound "
    "correlates with outcome'.\n"
    "- Give a SINGLE-VARIABLE minimal test naming a dataset + metric.\n"
    "- State WHY existing methods can't already do this.\n"
    "- Give one FALSIFIABLE prediction.\n"
    "Output STRICT JSON, no prose:\n"
    '{"title":"<=10 words","statement":"2-3 sentences naming the mechanism",'
    '"mechanism":"the causal mechanism","minimal_test":"single-variable exp w/ '
    'dataset+metric","why_existing_fails":"...","prediction":"falsifiable"}'
)

_CRITIC_SYS = (
    "You are a strict research-proposal reviewer. Judge ONE idea. Reject if it "
    "is: (a) just a literature summary, (b) has no concrete new mechanism, (c) a "
    "vacuous cross-domain bridge whose only shared concept is 'evaluation "
    "protocol'/'text', or (d) uses banned template phrasing ('moderator "
    "variable', 'prompt/decoding confound'). Keep only if it has a concrete "
    "mechanism AND a single-variable testable experiment.\n"
    'Output STRICT JSON: {"keep": true|false, "reason":"one sentence", '
    '"specificity_1_5": <int>, "testability_1_5": <int>}'
)


def _json(raw):
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0:
        return None
    try:
        return json.loads(raw[s:e + 1])
    except Exception:
        return None


def _anom_block(a, claims_by_id):
    cl = []
    for cid in (a.claim_ids or [])[:8]:
        c = claims_by_id.get(cid)
        if c:
            cl.append(f"  - [{c.direction}] {(c.method or '?')}: {(c.claim_text or '')[:140]}")
    se = a.shared_entities or {}
    return (f"Conflict type: {a.type}\nShared method: {se.get('method')}  "
            f"task: {se.get('task')}\nCentral question: {a.central_question}\n"
            f"Conflicting claims:\n" + "\n".join(cl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--max-anomalies", type=int, default=12)
    ap.add_argument("--topic", default=None)
    args = ap.parse_args()
    run = args.run

    apath = run / "anomalies_normalized.jsonl"
    cpath = run / "claims_normalized.jsonl"
    if not apath.exists():
        apath = run / "anomalies.jsonl"
    if not cpath.exists():
        cpath = run / "claims.jsonl"
    anomalies = read_jsonl(apath, Anomaly)
    claims = read_jsonl(cpath, Claim)
    claims_by_id = {c.claim_id: c for c in claims}
    # rank anomalies by evidence_impact + topology so we spend the budget well
    anomalies.sort(key=lambda a: (a.evidence_impact or 0) + (a.topology_score or 0), reverse=True)
    anomalies = anomalies[:args.max_anomalies]
    print(f"run={run.name}  anomalies(from {apath.name})={len(anomalies)}  claims={len(claims)}",
          file=sys.stderr)

    client, model = build_openai_client(), configured_model()
    kept, dropped = [], []
    for i, a in enumerate(anomalies):
        block = _anom_block(a, claims_by_id)
        try:
            raw = call_llm_text(client, model=model, system=_GEN_SYS, user=block, max_tokens=2500)
        except Exception as exc:
            print(f"  [{i}] gen failed: {exc}", file=sys.stderr); continue
        idea = _json(raw)
        if not idea or not idea.get("statement"):
            continue
        # critic pass
        try:
            craw = call_llm_text(client, model=model, system=_CRITIC_SYS,
                                 user=json.dumps(idea, ensure_ascii=False), max_tokens=1500)
            verdict = _json(craw) or {}
        except Exception:
            verdict = {"keep": True, "reason": "critic-failed-default-keep"}
        rec = {
            "anomaly_id": a.anomaly_id, "type": a.type,
            "shared": a.shared_entities,
            **idea,
            "critic": verdict,
        }
        if verdict.get("keep"):
            kept.append(rec)
        else:
            dropped.append(rec)
        print(f"  [{i}] {'KEEP' if verdict.get('keep') else 'DROP'}  {idea.get('title','')[:60]}  "
              f"(spec={verdict.get('specificity_1_5')},test={verdict.get('testability_1_5')})",
              file=sys.stderr)

    out = run / "anomaly_ideas.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nKEPT {len(kept)} / dropped {len(dropped)} → {out}")
    # also emit a readable summary to stdout
    for r in kept:
        print(f"\n### {r['title']}  [{r['type']}]")
        print(f"  {r['statement']}")
        print(f"  mechanism: {r.get('mechanism','')[:160]}")
        print(f"  test: {r.get('minimal_test','')[:160]}")


if __name__ == "__main__":
    raise SystemExit(main())
