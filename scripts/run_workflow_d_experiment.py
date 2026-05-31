"""Workflow D vs Workflow C experiment — runs on the LLM server.

Reads:
  - genexp16.json            (16 anomalies)
  - genexp16_out.jsonl       (prior A/B/C outputs; we reuse hC)

For each anomaly: runs Workflow D (4-stage decomposition), then
rubric-judges hD on the same 4 binary axes as the prior experiment, and a
blind pairwise C-vs-D. Writes one JSON object per anomaly.

scp this + the two inputs to the server, then:
  python3 run_workflow_d_experiment.py genexp16.json genexp16_out.jsonl wfd_out.jsonl 4
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


# --- env / client ----------------------------------------------------------
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


def _chat(system: str, user: str, mt: int = 4000, temp: float = 0.3) -> str:
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


def _ctx(a: dict) -> str:
    cl = "\n".join(f"  - [{c.get('dir')}] {c['text']}" for c in a["claims"])
    return (f"Anomaly type: {a['type']}\nCentral question: {a['central_question']}\n"
            f"Shared entities: {a.get('shared_entities')}\nConflicting claims:\n{cl}")


# --- Workflow D prompts (mirror gen_workflow_d.py, but D produces a flat
# text hypothesis for symmetry with hA/hB/hC in the prior experiment, not
# the full structured JSON Hypothesis schema; this keeps the rubric judge
# comparable across arms) ---------------------------------------------------

SYS_BRAINSTORM = (
    "You are a research scientist surveying possible causal mechanisms. Given "
    "an anomaly in AI research literature, list 4 to 5 DISTINCT candidate "
    "causal mechanisms that could explain when the method helps vs hurts. "
    "Each candidate must name a specific mechanism (not 'moderator' or "
    "'factor'), be testable by varying ONE thing, and be mechanistically "
    "distinct from the others. Output one mechanism per line:\n"
    "  M1: <one-sentence mechanism>\n"
    "  M2: <one-sentence mechanism>\n  ...\n"
    "No preamble, no JSON, no markdown."
)
SYS_SHARPEN = (
    "You are a research scientist picking the strongest candidate. From the "
    "candidate mechanisms below, choose the ONE most likely to be a real, "
    "single-variable causal story, and sharpen it into one precise sentence "
    "naming the mechanism. Output exactly two lines:\n"
    "  PICK: M<n>\n  MECHANISM: <one sentence>"
)
SYS_TEST = (
    "You are designing the minimal-cost discriminating experiment for a "
    "given mechanism. Output exactly two lines, no preamble:\n"
    "  TEST: <one sentence: the experiment, with named method, task, metric, "
    "varying ONE thing>\n"
    "  PREDICTS: <one sentence: the observation that confirms the mechanism>"
)
SYS_DRAFT = (
    "You are writing the final research hypothesis. Given a sharpened "
    "mechanism and a discriminating test, write a 2–3 sentence forward-"
    "looking hypothesis that names the mechanism, the scope (method + task), "
    "and the single-variable test that would confirm it. Output only the "
    "hypothesis text, no preamble, no JSON."
)


# --- judge prompts (identical to the prior experiment for comparability) --

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
    # Generous budget: Kimi's reasoning floor + tiny JSON output. At mt=2000
    # roughly 1 in 7 calls returned empty (→ defaulted to all-0 = false negative).
    o = _parse_json(_chat(SYS_RUBRIC, f"Hypothesis:\n{h}", mt=5000, temp=0))
    keys = ("forward_looking", "named_mechanism", "single_variable_test", "specific_scope")
    sc = {k: (1 if o.get(k) in (1, "1", True) else 0) for k in keys}
    sc["composite"] = sum(sc[k] for k in keys)
    # Surface judge crashes: a 0/0/0/0 may be a real bad hyp OR a judge crash;
    # callers can distinguish by checking _judge_raw.
    sc["_judge_raw"] = bool(o)
    return sc


def run_d(a: dict, hC: str) -> dict:
    """Workflow D + rubric + pairwise C-vs-D for one anomaly."""
    try:
        ctx = _ctx(a)
        # Kimi reasoning floor is ~1500 tok; output budget on top. Sharpen sees
        # ctx + brainstorm (~2K input) so its reasoning grows — 2500 was 8/16
        # empty. Bumping aggressively across all stages.
        brain = _chat(SYS_BRAINSTORM, ctx, mt=5000, temp=0.7)
        if not brain or "M1" not in brain:
            return {"anomaly_id": a["anomaly_id"], "error": f"brainstorm bad (len={len(brain)})"}
        pick = _chat(SYS_SHARPEN, f"{ctx}\n\nCandidate mechanisms:\n{brain}", mt=6000, temp=0.3)
        mech_line = next((l for l in pick.splitlines() if l.strip().startswith("MECHANISM:")), "")
        sharpened = mech_line.split("MECHANISM:", 1)[-1].strip()
        if not sharpened:
            return {"anomaly_id": a["anomaly_id"], "error": f"sharpen bad (len={len(pick)})"}
        test = _chat(SYS_TEST, f"{ctx}\n\nSharpened mechanism: {sharpened}", mt=5000, temp=0.3)
        test_line = next((l for l in test.splitlines() if l.strip().startswith("TEST:")), "")
        pred_line = next((l for l in test.splitlines() if l.strip().startswith("PREDICTS:")), "")
        test_str = test_line.split("TEST:", 1)[-1].strip()
        pred_str = pred_line.split("PREDICTS:", 1)[-1].strip()
        if not test_str:
            return {"anomaly_id": a["anomaly_id"], "error": f"test bad (len={len(test)})"}
        hD = _chat(
            SYS_DRAFT,
            f"Anomaly context:\n{ctx}\n\nSharpened mechanism: {sharpened}\n"
            f"Minimal test: {test_str}\nPrediction: {pred_str}",
            mt=5000, temp=0.2,
        )
        if not hD:
            return {"anomaly_id": a["anomaly_id"], "error": "draft empty"}
        rD = rubric(hD)
        # Blind pairwise C vs D
        flip = random.random() < 0.5
        X, Y = (hD, hC) if flip else (hC, hD)
        # Pair judge: input is two full hypotheses + comparison reasoning.
        # mt=600 returned empty 7/7 (pair_CvD="?" universal). Generous budget.
        v = _parse_json(_chat(SYS_PAIR, f"X:\n{X}\n\nY:\n{Y}", mt=5000))
        w = v.get("winner", "?")
        if w == "tie":
            winner = "tie"
        elif w in ("X", "Y"):
            winner = ("D" if (w == "X") == flip else "C")
        else:
            winner = w
        return {
            "anomaly_id": a["anomaly_id"],
            "type": a["type"],
            "rubric_D": rD,
            "pair_CvD": winner,
            "hD": hD,
            "hC": hC,
            "brainstorm": brain,
            "sharpened_mechanism": sharpened,
            "minimal_test": test_str,
            "prediction": pred_str,
            "pair_why": v.get("why", "")[:200],
        }
    except Exception as ex:
        return {"anomaly_id": a.get("anomaly_id"), "error": f"{type(ex).__name__}: {str(ex)[:120]}"}


def main() -> None:
    anoms_path, prior_path, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    anoms = json.load(open(anoms_path))
    prior = {json.loads(l)["anomaly_id"]: json.loads(l) for l in open(prior_path) if l.strip()}
    pairs = []
    for a in anoms:
        rec = prior.get(a["anomaly_id"])
        if rec and rec.get("hC"):
            pairs.append((a, rec["hC"]))
    print(f"Workflow D vs C on {len(pairs)} anomalies (of {len(anoms)} requested), "
          f"{_MODEL}, {workers} workers", file=sys.stderr)

    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_d, a, hC) for a, hC in pairs]
        for i, f in enumerate(as_completed(futures)):
            try:
                out.append(f.result())
            except Exception as ex_:
                out.append({"error": f"{type(ex_).__name__}"})
            if i % 2 == 0:
                print(f"  {i}/{len(pairs)} ({time.time() - t0:.0f}s)", file=sys.stderr)
    with open(outp, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in out if "error" not in r]
    if ok:
        def mean_(key: str) -> float:
            return round(sum(r["rubric_D"][key] for r in ok) / len(ok), 2)

        # Pull prior C rubric mean from prior records aligned to ok
        c_mean_composite = round(
            sum(prior[r["anomaly_id"]]["rubric"]["C"]["composite"] for r in ok) / len(ok), 2)
        keys = ("forward_looking", "named_mechanism", "single_variable_test",
                "specific_scope", "composite")
        print(json.dumps({
            "n_ok": len(ok),
            "n_err": len(out) - len(ok),
            "rubric_mean_D": {k: mean_(k) for k in keys},
            "rubric_mean_C_for_same_anoms": c_mean_composite,
            "pair_CvD": dict(Counter(r["pair_CvD"] for r in ok)),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
