"""Workflow D — DECOMPOSITION-based hypothesis generation.

Inspired by arxiv 2601.09714 (decomposition workflows ~2× novelty over
reflection). Four-stage pipeline per anomaly:

  Stage 1 — MECHANISM BRAINSTORM    (temp 0.7, free-form, 4-5 candidates)
  Stage 2 — PICK + SHARPEN          (temp 0.3, single-variable mechanism)
  Stage 3 — TEST DESIGN             (temp 0.3, single-variable experiment)
  Stage 4 — ASSEMBLE JSON           (temp 0.1, schema-clean output)

Sibling of /tmp/forward_gen_json.py (Workflow C single-shot + reflect).
Same I/O contract: input = JSON list of anomaly dicts; output = JSONL of
structured Hypothesis records compatible with `score_forward_vs_frozen.py`
and `select_mmr`.

This script is designed to run on the LLM server (8.208.118.99) where the
Kimi-K2.6 endpoint is reachable. Sync via scp before running.

Usage:
  python gen_workflow_d.py <anomalies.json> <out.jsonl> [workers]
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI  # type: ignore


# --- env / client (mirror forward_gen_json.py exactly so it runs on the
# server with the same config path) -----------------------------------------
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


def _chat(messages: list[dict], temperature: float, max_tokens: int = 4000) -> str:
    """Single Kimi call with backoff. Returns '' on persistent failure."""
    for _ in range(3):
        try:
            r = _CLI.chat.completions.create(
                model=_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            txt = (r.choices[0].message.content or "").strip()
            if txt:
                return txt
        except Exception:
            time.sleep(8)
            continue
    return ""


# --- prompts ---------------------------------------------------------------

_SYS_BRAINSTORM = (
    "You are a research scientist surveying possible causal mechanisms. Given "
    "an anomaly in AI research literature (two or more papers contradicting "
    "each other), list 4 to 5 DISTINCT candidate causal mechanisms that could "
    "explain when the method helps vs hurts. Each candidate must:\n"
    "  - Name a specific mechanism (NOT 'moderator', 'factor', 'dataset variance')\n"
    "  - Be testable in principle (a researcher could vary ONE thing to check it)\n"
    "  - Be mechanistically distinct from the others (not a re-phrasing)\n"
    "Output one mechanism per line, in the form:\n"
    "  M1: <one-sentence mechanism>\n"
    "  M2: <one-sentence mechanism>\n"
    "  ...\n"
    "No preamble, no JSON, no markdown."
)

_SYS_SHARPEN = (
    "You are a research scientist picking the strongest candidate. Given 4–5 "
    "candidate causal mechanisms for an anomaly, choose the ONE most likely "
    "to be a real, single-variable causal story, and sharpen it into a single "
    "precise sentence. Criteria for picking:\n"
    "  - Predicts a direction (when method helps vs hurts)\n"
    "  - Has a NAMED mechanism (not 'dataset properties')\n"
    "  - Single-variable testable\n"
    "  - Has specific scope (method + task)\n"
    "Output exactly two lines, no preamble:\n"
    "  PICK: M<n>\n"
    "  MECHANISM: <one sharpened sentence naming the mechanism>"
)

_SYS_TEST = (
    "You are a research scientist designing a minimal-cost discriminating "
    "experiment. Given a sharpened causal mechanism, design the single "
    "experiment that varies ONE variable and would discriminate this "
    "mechanism from plausible alternatives. The test must:\n"
    "  - Vary exactly ONE thing\n"
    "  - Specify the method, the task/benchmark, the metric\n"
    "  - Be runnable by a researcher tomorrow\n"
    "Output exactly two lines, no preamble:\n"
    "  TEST: <one sentence: the experiment, with named method, task, metric>\n"
    "  PREDICTS: <one sentence: the prediction that confirms the mechanism>"
)

_SYS_ASSEMBLE = (
    "You are assembling a final structured research hypothesis from three "
    "pieces: a sharpened mechanism, a minimal discriminating test, and the "
    "original anomaly (with its conflicting claim_ids). Output STRICT JSON "
    "matching this schema, no fences, no preamble:\n"
    '{"hypothesis":"<2-3 sentence forward-looking statement>",'
    '"mechanism":"<the sharpened mechanism, verbatim>",'
    '"explains_claims":["<real claim_id>", ...],'
    '"predictions":["<short prediction 1>","<short prediction 2>"],'
    '"minimal_test":"<the test, verbatim>",'
    '"scope_conditions":{"method":"<specific method>","task":"<specific task>"},'
    '"evidence_gap":"<what evidence is still missing>",'
    '"graph_bridge":{"from":"<source concept>","to":"<target concept>"}}\n'
    "Constraints:\n"
    "- Output JSON only.\n"
    "- explains_claims MUST be real claim_ids from the original anomaly.\n"
    "- scope_conditions MUST name a concrete method AND task (not 'the method').\n"
    "- predictions are short and concrete, not boilerplate."
)


def _anom_block(a: dict) -> str:
    return (
        f"Anomaly type: {a['type']}\n"
        f"Central question: {a['central_question']}\n"
        f"Shared entities: {a.get('shared_entities')}\n"
        "Conflicting claims (use these claim_ids):\n"
        + "\n".join(f"  - {c['id']} [{c.get('dir')}] {c['text']}" for c in a["claims"])
    )


# --- per-anomaly pipeline --------------------------------------------------


def run_d(a: dict) -> dict | None:
    """Returns a structured Hypothesis dict, or None on failure."""
    block = _anom_block(a)

    # Stage 1 — brainstorm. Kimi is a reasoning model: it eats ~1.5K tokens
    # on internal reasoning before emitting any content. Per-stage budgets
    # must clear that floor plus the actual output (≤1000 tok) → 4000 is safe.
    brain = _chat(
        [{"role": "system", "content": _SYS_BRAINSTORM},
         {"role": "user", "content": block}],
        temperature=0.7,
        max_tokens=4000,
    )
    if not brain or "M1" not in brain:
        return None

    # Stage 2 — pick + sharpen
    pick = _chat(
        [{"role": "system", "content": _SYS_SHARPEN},
         {"role": "user", "content": f"{block}\n\nCandidate mechanisms:\n{brain}"}],
        temperature=0.3,
        max_tokens=2500,
    )
    if not pick or "MECHANISM:" not in pick:
        return None
    mech_line = next((l for l in pick.splitlines() if l.strip().startswith("MECHANISM:")), "")
    sharpened = mech_line.split("MECHANISM:", 1)[-1].strip()
    if not sharpened:
        return None

    # Stage 3 — test design
    test = _chat(
        [{"role": "system", "content": _SYS_TEST},
         {"role": "user", "content": f"{block}\n\nSharpened mechanism: {sharpened}"}],
        temperature=0.3,
        max_tokens=2500,
    )
    if not test or "TEST:" not in test:
        return None
    test_line = next((l for l in test.splitlines() if l.strip().startswith("TEST:")), "")
    pred_line = next((l for l in test.splitlines() if l.strip().startswith("PREDICTS:")), "")
    test_str = test_line.split("TEST:", 1)[-1].strip()
    pred_str = pred_line.split("PREDICTS:", 1)[-1].strip()
    if not test_str:
        return None

    # Stage 4 — assemble
    assembly_payload = (
        f"{block}\n\n"
        f"Sharpened mechanism: {sharpened}\n"
        f"Minimal test: {test_str}\n"
        f"Predicted observation: {pred_str}"
    )
    raw = _chat(
        [{"role": "system", "content": _SYS_ASSEMBLE},
         {"role": "user", "content": assembly_payload}],
        temperature=0.1,
        max_tokens=2500,
    )
    if not raw:
        return None
    s, e = raw.find("{"), raw.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(raw[s:e + 1])
    except Exception:
        return None

    # Defaults + identity fields, mirroring forward_gen_json.py
    obj.setdefault("predictions", [])
    obj.setdefault("scope_conditions", {})
    obj.setdefault("graph_bridge", {"from": "", "to": ""})
    obj["anomaly_id"] = a["anomaly_id"]
    obj["hypothesis_id"] = f"wfd_{a['anomaly_id']}"
    # Stash the intermediate stages for ablation / debugging
    obj["_workflow_d"] = {
        "brainstorm": brain,
        "sharpened_mechanism": sharpened,
        "minimal_test": test_str,
        "prediction": pred_str,
    }
    return obj


def main() -> None:
    inp, outp = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6  # 4× more LLM calls
    anoms = json.load(open(inp))
    print(f"Workflow D on {len(anoms)} anomalies, {_MODEL}, {workers} workers",
          file=sys.stderr)
    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_d, a) for a in anoms]
        for i, f in enumerate(as_completed(futures)):
            try:
                h = f.result()
            except Exception as ex_:
                print(f"  worker err: {type(ex_).__name__}", file=sys.stderr)
                h = None
            if h is not None:
                out.append(h)
            if i % 2 == 0:
                print(f"  {i}/{len(anoms)} ({time.time() - t0:.0f}s)", file=sys.stderr)
    with open(outp, "w") as fh:
        for h in out:
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(f"DONE n_ok={len(out)}/{len(anoms)}")


if __name__ == "__main__":
    main()
