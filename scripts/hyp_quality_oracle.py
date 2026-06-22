"""Hypothesis quality oracle — a blind, calibrated LLM judge over a run's
hypotheses (NON-FROZEN tooling, scripts/* is freely editable per the v0.7
freeze §5).

Why this exists
---------------
The v0.7 freeze locks the generator, scorer, and detectors so the *citation-ρ*
predictor stays comparable across runs. But ρ measures influence prediction,
NOT delivered-hypothesis quality. The user's complaint ("都是模版化的东西") is a
quality/diversity problem the ρ metric is blind to. To rebuild the generator
responsibly we need an independent quality yardstick that:

  1. scores the axes the complaint is actually about, on a 1-5 Likert scale
     (binary 0/1 saturates — see memory binary-rubric-ceilings), and
  2. is calibrated against hand-written GOOD/BAD control items so a
     mis-scaled judge is caught before its numbers are trusted
     (the "mismeasured twice" lesson from the novelty-boundary probe), and
  3. is blind by construction — the judge never sees which generator (arm)
     produced a hypothesis, so the same script scores the control (frozen)
     and treatment (forward-design) arms identically.

Axes (1 = worst, 5 = best)
  grounding              anchored in THIS anomaly's specific claims / numbers,
                         not generic boilerplate that would fit any conflict
  falsifiability         minimal_test is concrete and could actually come out
                         negative
  mechanism_specificity  names a specific causal mechanism vs a vague
                         "unreported moderator / some confound"
  forward_design         proposes a NEW thing to build/try going forward vs a
                         retrospective explanation of why past papers disagreed
                         (THE axis the frozen back-explanation generator is
                         expected to score low on)
  novelty                not something the cited papers already did

Set-level
  shape_diversity        distinct structural shapes / n  (homogeneity = low)

Usage
  python scripts/hyp_quality_oracle.py --run-dir artifacts/runs/<run> \
      [--limit 24] [--label control] [--out artifacts/oracle/<run>.json]

Requires the same LLM env as the rest of aigraph (OPENAI_API_KEY +
AIGRAPH_BASE_URL/OPENAI_BASE_URL + AIGRAPH_MODEL). Fail-loud: unlike the
fail-open delivery wraps, the oracle RAISES if it cannot reach the model —
a silent fallback would corrupt the measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# allow running from a source checkout without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aigraph.llm_client import (  # noqa: E402
    build_openai_client,
    call_llm_text,
    configured_api_key,
    configured_model,
)

AXES = ["grounding", "falsifiability", "mechanism_specificity", "forward_design", "novelty"]

SHAPES = [
    "conflict_attribution",  # back-explains why papers disagree (confound/moderator/protocol)
    "interior_optimum",      # X helps up to N then hurts; has an internal sweet spot
    "mechanism",             # specific causal mechanism + differential predictions
    "transfer",              # apply A's method to B's task / cross-pollination
    "scaling",               # effect scales with some measurable property
    "evidence_request",      # what missing evidence/control to collect
    "other",
]

_SYSTEM = (
    "You are a strict, calibrated reviewer of machine-learning research hypotheses. "
    "You score one hypothesis at a time on five axes, each an INTEGER 1-5 (1=worst, "
    "5=best). Be harsh: a generic statement that would fit almost any literature "
    "conflict deserves low grounding and low mechanism_specificity. Reserve 5 for "
    "hypotheses that are specific, grounded in the given claims, and genuinely "
    "testable.\n\n"
    "AXES:\n"
    "- grounding: is it anchored in THESE specific claims (methods, datasets, "
    "numbers) rather than boilerplate that would fit any conflict? Boilerplate like "
    "'an unreported moderator variable drives the conflict' = 1.\n"
    "- falsifiability: is minimal_test concrete enough to actually come out "
    "negative? Vague 'replay in a common harness' = 2; a named dataset+metric+"
    "comparison = 5.\n"
    "- mechanism_specificity: does it name a SPECIFIC causal mechanism, or hand-wave "
    "('some confound', 'a moderator')? Hand-wave = 1.\n"
    "- forward_design: does it propose a NEW thing to build or try going forward "
    "(a method, a regime, a design with a sweet spot), or merely retrospectively "
    "explain why past papers disagreed? Pure retrospective conflict-explanation = 1; "
    "a concrete forward research design = 5.\n"
    "- novelty: is this something the cited papers did NOT already do? If the cited "
    "work already does exactly this = 1.\n\n"
    "Also assign ONE structural shape label from this exact set: "
    + ", ".join(SHAPES) + ".\n\n"
    "Return STRICT JSON ONLY, no prose, no markdown:\n"
    '{"grounding": int, "falsifiability": int, "mechanism_specificity": int, '
    '"forward_design": int, "novelty": int, "shape": "<one label>", '
    '"rationale": "one short sentence"}'
)

# --- calibration controls: hand-written anchors the judge MUST separate ------ #
_CONTROL_BAD = {
    "hypothesis": "An unreported moderator variable drives the conflicting results around the method on the task.",
    "mechanism": "A confound in data preprocessing, prompt formatting, or decoding parameters correlates with outcome direction and is not held constant across the claims.",
    "predictions": ["Holding prompt template and decoding fixed shrinks the between-claim variance by >50%."],
    "minimal_test": "Replay all claims in a common harness with identical prompts and decoding settings; recompute deltas.",
    "evidence_gap": "Prompt and decoding configurations are inconsistently reported.",
    "_claims": [{"claim_text": "Method M helps on task T (+3%).", "method": "M", "dataset": "T", "direction": "positive"}],
    "_anomaly_type": "impact_conflict",
    "_central_question": "Why do papers disagree about M on T?",
}
_CONTROL_GOOD = {
    "hypothesis": "Self-consistency improves GSM8K accuracy up to ~16 sampled chains, beyond which added chains hurt because majority-voting amplifies a systematic arithmetic-carry error mode shared across chains.",
    "mechanism": "Sampled chains are not independent: they share the base model's carry-error prior, so past the point where voting denoises random mistakes, additional correlated chains reinforce the shared bias.",
    "predictions": ["Accuracy vs chain-count is unimodal with a peak near 16 on GSM8K.", "Decorrelating chains via diverse prompts shifts the peak rightward."],
    "minimal_test": "Sweep self-consistency chain count {1,2,4,8,16,32,64} on GSM8K with a fixed 7B model; plot accuracy and measure inter-chain error correlation at each count.",
    "evidence_gap": "Papers report self-consistency at a single chain count and never sweep past the reported optimum.",
    "_claims": [{"claim_text": "Self-consistency raises GSM8K accuracy with more samples.", "method": "self-consistency", "dataset": "GSM8K", "direction": "positive"}],
    "_anomaly_type": "benchmark_inconsistency",
    "_central_question": "When does self-consistency help vs hurt on GSM8K?",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line, strict=False))
        except (ValueError, TypeError):
            continue
    return out


def _normalize_creator(h: dict) -> dict:
    """Map a creator hypothesis (proposed_method / inspired_by schema) onto the
    critic field names the judge payload expects, so the SAME blind judge scores
    both generators on the same axes. Non-destructive (returns a shallow copy)."""
    if "proposed_method" not in h or h.get("hypothesis"):
        return h
    out = dict(h)
    out["hypothesis"] = h.get("proposed_method") or ""
    # inspired_by mixes claim_ids and open-question ids (e.g. "p1#oq01"); the
    # judge payload filters to ids that resolve against claims, so passing the
    # raw list is safe.
    out.setdefault("explains_claims", h.get("inspired_by") or [])
    # creator has no evidence_gap; distinguishes_from is the closest analogue.
    out.setdefault("evidence_gap", h.get("distinguishes_from") or "")
    return out


def load_run(run_dir: Path, hyp_file: str = "hypotheses.jsonl") -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    hyps = [_normalize_creator(h) for h in _read_jsonl(run_dir / hyp_file)]
    claims = {c.get("claim_id"): c for c in _read_jsonl(run_dir / "claims.jsonl")}
    anoms = {a.get("anomaly_id"): a for a in _read_jsonl(run_dir / "anomalies.jsonl")}
    return hyps, claims, anoms


def _judge_payload(hyp: dict, claims: dict[str, dict], anoms: dict[str, dict]) -> dict:
    """Compact, generator-agnostic view of a hypothesis + its grounding, so the
    judge sees what the hyp claims AND what real claims it is supposed to explain.
    Carries NO arm/generator identity — blind by construction."""
    if "_claims" in hyp:  # a control item carries its own evidence
        evidence = hyp["_claims"]
        atype = hyp.get("_anomaly_type", "")
        cq = hyp.get("_central_question", "")
    else:
        cids = hyp.get("explains_claims") or []
        evidence = []
        for cid in cids[:6]:
            c = claims.get(cid)
            if not c:
                continue
            evidence.append({
                "claim_text": c.get("claim_text"),
                "method": c.get("canonical_method") or c.get("method"),
                "dataset": c.get("dataset"),
                "metric": c.get("metric"),
                "direction": c.get("direction"),
            })
        anom = anoms.get(hyp.get("anomaly_id")) or {}
        atype = anom.get("type", "")
        cq = anom.get("central_question", "")
    return {
        "anomaly_type": atype,
        "central_question": cq,
        "hypothesis": hyp.get("hypothesis"),
        "mechanism": hyp.get("mechanism"),
        "predictions": hyp.get("predictions"),
        "minimal_test": hyp.get("minimal_test"),
        "evidence_gap": hyp.get("evidence_gap"),
        "evidence_claims": evidence,
    }


def _parse_scores(reply: str) -> dict[str, Any] | None:
    text = (reply or "").strip()
    if not text:
        return None
    if not text.lstrip().startswith("{"):
        lo = text.find("{")
        if lo == -1:
            return None
        text = text[lo:]
        hi = text.rfind("}")
        if hi != -1:
            text = text[: hi + 1]
    try:
        data = json.loads(text, strict=False)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, Any] = {}
    for ax in AXES:
        v = data.get(ax)
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        out[ax] = max(1, min(5, iv))
    shape = str(data.get("shape", "other")).strip().lower()
    out["shape"] = shape if shape in SHAPES else "other"
    out["rationale"] = str(data.get("rationale", ""))[:240]
    return out


def judge_one(client: Any, model: str, payload: dict, *, max_tokens: int) -> dict[str, Any] | None:
    reply = call_llm_text(
        client,
        model=model,
        system=_SYSTEM,
        user=json.dumps(payload, ensure_ascii=False),
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return _parse_scores(reply)


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def run_oracle(
    run_dir: Path,
    *,
    limit: int | None,
    label: str,
    model: str,
    max_tokens: int,
    hyp_file: str = "hypotheses.jsonl",
) -> dict[str, Any]:
    if not configured_api_key():
        raise SystemExit("oracle requires an API key (OPENAI_API_KEY); refusing to fake scores")
    client = build_openai_client()
    hyps, claims, anoms = load_run(run_dir, hyp_file=hyp_file)
    if not hyps:
        raise SystemExit(f"no hypotheses.jsonl in {run_dir}")
    if limit:
        hyps = hyps[:limit]

    # 1) calibration: the judge must rank GOOD above BAD on the two axes that
    #    encode the complaint, else its numbers are not trustworthy.
    cal = {}
    for name, item in (("bad", _CONTROL_BAD), ("good", _CONTROL_GOOD)):
        s = judge_one(client, model, _judge_payload(item, claims, anoms), max_tokens=max_tokens)
        if s is None:
            raise SystemExit(f"calibration judge returned unparseable output for control '{name}'")
        cal[name] = s
    cal_pass = (
        cal["good"]["forward_design"] > cal["bad"]["forward_design"]
        and cal["good"]["mechanism_specificity"] > cal["bad"]["mechanism_specificity"]
    )

    # 2) score every (sampled) hypothesis, blind.
    scored: list[dict[str, Any]] = []
    for i, h in enumerate(hyps):
        s = judge_one(client, model, _judge_payload(h, claims, anoms), max_tokens=max_tokens)
        if s is None:
            print(f"  [{i+1}/{len(hyps)}] unparseable — skipped", file=sys.stderr)
            continue
        s["hypothesis_id"] = h.get("hypothesis_id")
        s["anomaly_id"] = h.get("anomaly_id")
        scored.append(s)
        print(f"  [{i+1}/{len(hyps)}] {h.get('hypothesis_id')}: "
              + " ".join(f"{ax[:4]}={s[ax]}" for ax in AXES)
              + f" shape={s['shape']}", file=sys.stderr)

    axis_means = {ax: _mean([s[ax] for s in scored]) for ax in AXES}
    shape_counts = Counter(s["shape"] for s in scored)
    n = len(scored)
    diversity = {
        "distinct_shapes": len(shape_counts),
        "shape_entropy_frac": round(len(shape_counts) / len(SHAPES), 3),
        "dominant_shape": shape_counts.most_common(1)[0][0] if shape_counts else None,
        "dominant_frac": round(shape_counts.most_common(1)[0][1] / n, 3) if n else 0.0,
        "shape_histogram": dict(shape_counts),
    }
    return {
        "label": label,
        "run_dir": str(run_dir),
        "model": model,
        "n_scored": n,
        "axis_means": axis_means,
        "diversity": diversity,
        "calibration": {"pass": cal_pass, "good": cal["good"], "bad": cal["bad"]},
        "scored": scored,
    }


def _print_report(res: dict[str, Any]) -> None:
    print(f"\n=== oracle: {res['label']}  (n={res['n_scored']}, model={res['model']}) ===")
    cal = res["calibration"]
    print(f"calibration: {'PASS' if cal['pass'] else 'FAIL — scores NOT trustworthy'} "
          f"(good.forward={cal['good']['forward_design']} vs bad.forward={cal['bad']['forward_design']}; "
          f"good.mech={cal['good']['mechanism_specificity']} vs bad.mech={cal['bad']['mechanism_specificity']})")
    print("axis means (1-5):")
    for ax in AXES:
        print(f"  {ax:22s} {res['axis_means'][ax]}")
    d = res["diversity"]
    print(f"shape diversity: distinct={d['distinct_shapes']}/{len(SHAPES)}  "
          f"dominant={d['dominant_shape']} ({d['dominant_frac']*100:.0f}% of set)")
    print(f"  histogram: {d['shape_histogram']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--hyp-file", default="hypotheses.jsonl",
                    help="hypotheses file under the run dir (e.g. creator_hypotheses.jsonl)")
    ap.add_argument("--limit", type=int, default=None, help="cap #hypotheses judged (cost control)")
    ap.add_argument("--label", default="control", help="arm label for reporting (e.g. control/forward)")
    ap.add_argument("--model", default=None, help="judge model (default: configured AIGRAPH_MODEL)")
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("AIGRAPH_ORACLE_MAX_TOKENS", "2500")))
    ap.add_argument("--out", default=None, help="write full result JSON here")
    args = ap.parse_args(argv)

    model = configured_model(args.model)
    res = run_oracle(
        Path(args.run_dir),
        limit=args.limit,
        label=args.label,
        model=model,
        max_tokens=args.max_tokens,
        hyp_file=args.hyp_file,
    )
    _print_report(res)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
