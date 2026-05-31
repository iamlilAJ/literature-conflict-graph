"""Method 2 — Atlas-seeded prompting A/B.

Question: when we prompt the generator to EXPLICITLY USE the Atlas bottleneck
signals (later paper's identification of the original paper's limitation,
with severity + dimension + quote), does the resulting hypothesis improve
on a generator that has the same context attached but isn't told to use it?

The existing 485 joint_hypotheses.jsonl were generated with the
back-explanation framing — they ignore the bottleneck context (see
docs/quality-next-levers-research-brief.md). This experiment isolates
whether the Atlas bottleneck signal is actionable.

Arms (both use Workflow C — forward-framed single-shot + reflect, the
current production winner):
  Arm P (PLAIN)  — anomaly context = central_question + claims only;
                   bottleneck_signals NOT shown to the model.
  Arm A (ATLAS)  — anomaly context = central_question + claims + the
                   bottleneck_signals quote/severity/dimension, plus an
                   explicit instruction to "propose a mechanism that
                   addresses BOTH the original limitation AND the later
                   paper's bottleneck".

Then rubric (binary 4-axis as before, for continuity with prior runs)
+ blind pairwise A vs P.

Sample: 16 joint anomalies, picked to span dimensions evenly so the
result generalizes across bottleneck types.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

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
random.seed(43)


def _chat(system: str, user: str, mt: int = 5000, temp: float = 0.3) -> str:
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


# --- prompts ---------------------------------------------------------------

SYS_GEN_PLAIN = (
    "You are a research scientist. Given an anomaly in the literature, propose "
    "ONE specific, FORWARD-LOOKING, testable research hypothesis — a new "
    "mechanism or direction to investigate that would advance the field, NOT "
    "merely an explanation of why existing papers differ. Name a concrete "
    "causal mechanism, imply a single-variable minimal test, and a specific "
    "scope (method/task/condition). 2–3 sentences, output only the hypothesis."
)
SYS_GEN_ATLAS = (
    "You are a research scientist. The user payload contains an anomaly in "
    "AI research literature AND a list of bottleneck signals — later papers "
    "that explicitly identified this paper's limitation as a fundamental "
    "bottleneck in some dimension (generalization, accuracy, training "
    "stability, etc.). Propose ONE specific, FORWARD-LOOKING, testable "
    "research hypothesis that addresses BOTH the original limitation AND the "
    "later-identified bottleneck. Your hypothesis MUST:\n"
    "  - name a concrete causal mechanism (not 'moderator variable')\n"
    "  - reference the named bottleneck dimension explicitly\n"
    "  - imply a single-variable minimal test\n"
    "  - have specific scope (method/task)\n"
    "2–3 sentences, output only the hypothesis."
)
SYS_REFLECT = (
    "Critique the draft research hypothesis on 4 axes: (1) forward-looking "
    "(new direction to test, not explaining a disagreement)? (2) named "
    "concrete causal mechanism? (3) single-variable testable? (4) specific "
    "scope (named method/task)? Then output an IMPROVED version fixing any "
    "weakness. Output ONLY the improved hypothesis, 2–3 sentences."
)
SYS_RUBRIC = (
    'Score this research hypothesis. For each criterion output 1 (yes) or 0 (no):\n'
    '- forward_looking: proposes a NEW direction/mechanism to test, not just explaining why '
    'existing papers disagree\n'
    '- named_mechanism: states a concrete causal mechanism, not vague\n'
    '- single_variable_test: implies a specific experiment varying ONE thing\n'
    '- specific_scope: names a concrete method/task/dataset/condition, not generic\n'
    'Output STRICT JSON: {"forward_looking":0,"named_mechanism":0,"single_variable_test":0,"specific_scope":0}'
)
SYS_PAIR = (
    'Two research hypotheses (X and Y) for the same literature anomaly. Pick the one that is '
    'the more useful, specific, testable, FORWARD-LOOKING research idea (not just a restatement '
    'of the disagreement). STRICT JSON: {"winner":"X|Y|tie","why":"one sentence"}'
)


def _parse_json(raw: str) -> dict:
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0:
        return {}
    try:
        return json.loads(raw[s:e + 1])
    except Exception:
        return {}


def rubric(h: str) -> dict:
    o = _parse_json(_chat(SYS_RUBRIC, f"Hypothesis:\n{h}", mt=5000, temp=0))
    keys = ("forward_looking", "named_mechanism", "single_variable_test", "specific_scope")
    sc = {k: (1 if o.get(k) in (1, "1", True) else 0) for k in keys}
    sc["composite"] = sum(sc[k] for k in keys)
    sc["_judge_ok"] = bool(o)
    return sc


def _ctx_plain(a: dict) -> str:
    """Anomaly context WITHOUT atlas bottleneck signals."""
    # Strip Atlas-leakage from the joint central_question: keep just the
    # "This paper reports a limitation: '<quote>'." segment.
    cq = a["central_question"]
    if 'reports a limitation: "' in cq:
        cq = cq.split('reports a limitation: "', 1)[1]
        cq = cq.split('".', 1)[0]
        cq = f'What concrete mechanism would address the reported limitation: "{cq}"?'
    claims = a.get("negative_claims", []) + a.get("positive_claims", [])
    return f"Central question: {cq}\nClaim ids involved: {claims}"


def _ctx_atlas(a: dict) -> str:
    """Anomaly context WITH atlas bottleneck signals."""
    sig = a.get("bottleneck_signals", [])
    sig_block = "\n".join(
        f"  - {s.get('source_title','?')} [{s.get('dimension')}, "
        f"{s.get('severity')}, relation={s.get('relation')}]: "
        f"\"{s.get('quote','')[:300]}\""
        for s in sig[:3]
    )
    claims = a.get("negative_claims", []) + a.get("positive_claims", [])
    return (
        f"Central question: {a['central_question']}\n"
        f"Bottleneck dimension: {a['shared_entities'].get('bottleneck_dimension')}\n"
        f"Claim ids involved: {claims}\n"
        f"Bottleneck signals (later papers identifying this as a known bottleneck):\n{sig_block}"
    )


def run_one(a: dict) -> dict:
    try:
        ctxP, ctxA = _ctx_plain(a), _ctx_atlas(a)
        # Arm P (plain): single-shot + reflect
        draftP = _chat(SYS_GEN_PLAIN, ctxP, mt=4000, temp=0.3)
        if not draftP:
            return {"anomaly_id": a["anomaly_id"], "error": "P draft empty"}
        hP = _chat(SYS_REFLECT, f"Anomaly context:\n{ctxP}\n\nDraft hypothesis:\n{draftP}",
                   mt=4000, temp=0.3) or draftP
        # Arm A (atlas): single-shot + reflect with bottleneck context
        draftA = _chat(SYS_GEN_ATLAS, ctxA, mt=4000, temp=0.3)
        if not draftA:
            return {"anomaly_id": a["anomaly_id"], "error": "A draft empty"}
        hA = _chat(SYS_REFLECT, f"Anomaly context:\n{ctxA}\n\nDraft hypothesis:\n{draftA}",
                   mt=4000, temp=0.3) or draftA
        rP = rubric(hP)
        rA = rubric(hA)
        # Blind pairwise A vs P
        flip = random.random() < 0.5
        X, Y = (hA, hP) if flip else (hP, hA)
        v = _parse_json(_chat(SYS_PAIR, f"X:\n{X}\n\nY:\n{Y}", mt=5000))
        w = v.get("winner", "?")
        if w == "tie":
            winner = "tie"
        elif w in ("X", "Y"):
            winner = ("A" if (w == "X") == flip else "P")
        else:
            winner = w
        return {
            "anomaly_id": a["anomaly_id"],
            "bottleneck_dim": a["shared_entities"].get("bottleneck_dimension"),
            "severity": (a.get("bottleneck_signals") or [{}])[0].get("severity"),
            "rubric": {"P": rP, "A": rA},
            "pair_PvA": winner,
            "hP": hP,
            "hA": hA,
            "pair_why": v.get("why", "")[:200],
        }
    except Exception as ex:
        return {"anomaly_id": a.get("anomaly_id"),
                "error": f"{type(ex).__name__}: {str(ex)[:120]}"}


def main() -> None:
    inp, outp = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    anoms = json.load(open(inp))
    print(f"Method 2 (Atlas A/B) on {len(anoms)} anomalies, {_MODEL}, "
          f"{workers} workers", file=sys.stderr)
    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_one, a) for a in anoms]
        for i, f in enumerate(as_completed(futures)):
            try:
                out.append(f.result())
            except Exception as ex_:
                out.append({"error": f"{type(ex_).__name__}"})
            if i % 2 == 0:
                print(f"  {i}/{len(anoms)} ({time.time() - t0:.0f}s)", file=sys.stderr)
    with open(outp, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in out if "error" not in r]
    if ok:
        def m(arm: str, key: str) -> float:
            return round(sum(r["rubric"][arm][key] for r in ok) / len(ok), 2)
        keys = ("forward_looking", "named_mechanism", "single_variable_test",
                "specific_scope", "composite")
        print(json.dumps({
            "n_ok": len(ok), "n_err": len(out) - len(ok),
            "rubric_mean": {arm: {k: m(arm, k) for k in keys} for arm in ("P", "A")},
            "pair_PvA": dict(Counter(r["pair_PvA"] for r in ok)),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
