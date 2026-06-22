"""Matched A/B for the creator (§7 Thaw #5): frozen creator prompt vs the
forward shape-diversity + hard-grounding rewrite. NON-FROZEN tooling.

Same anomalies / claims / open-questions / model; the ONLY variable is the
`creator.CREATOR_SYSTEM_PROMPT` text. The control arm monkeypatches the
pre-thaw prompt (captured below verbatim from the v0.7-frozen tag); the forward
arm uses the module's current (edited) prompt. Both arms' outputs are scored by
the same blind `hyp_quality_oracle` judge.

Usage
  python scripts/creator_ab.py --run-dir artifacts/runs/<run> --anomalies 12 \
      [--gen-model DeepSeek-V4-Flash] [--out artifacts/oracle/creator_ab.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parent / "src"))

import hyp_quality_oracle as oracle  # noqa: E402

import aigraph.creator as creator  # noqa: E402
from aigraph.creator import generate_creator_hypotheses  # noqa: E402
from aigraph.llm_client import configured_api_key, configured_model  # noqa: E402
from aigraph.models import Anomaly, Claim, OpenQuestion  # noqa: E402

# Pre-thaw CREATOR_SYSTEM_PROMPT, verbatim from the v0.7-frozen tag — the
# control arm. (Kept here so the A/B is reproducible in one process without a
# git checkout; the live module text is the forward arm.)
FROZEN_CREATOR_PROMPT = (
    "You are a research assistant proposing concrete *new* methods that resolve a "
    "cluster of conflicting findings, grounded in the open questions and limitations "
    "the original authors stated themselves.\n\n"
    "You will be given:\n"
    "- One Anomaly (papers disagree about a (method, task) pair)\n"
    "- The structured Claims that compose the anomaly\n"
    "- OpenQuestion records (limitations and future work) from those papers\n\n"
    "Propose 1 to 3 NEW methods. Each method must:\n"
    "- Be a concrete combination or extension of techniques mentioned in the cluster, "
    "not a vague slogan.\n"
    "- Reference at least 2 OpenQuestion or Claim ids as the grounding for why it is "
    "needed.\n"
    "- Differ from any method already named in the cluster (if you propose something "
    "that exists in the claims, drop it).\n"
    "- Include a falsifiable minimal_test using benchmarks/metrics from the claims.\n\n"
    'Return JSON {"creator_hypotheses": [...]} where each item has:\n'
    "- proposed_method: 1-line name + 1-line description\n"
    "- mechanism: 2-3 sentences of how it works\n"
    "- predictions: list of 2 specific predictions (each ties to a metric/benchmark)\n"
    "- minimal_test: the simplest experiment to validate; reference dataset + metric\n"
    "- inspired_by: list of open_question_id and/or claim_id strings\n"
    "- distinguishes_from: 1 sentence — how it differs from existing methods in the cluster\n"
    "- anomaly_resolution: 1 sentence — how it resolves the anomaly\n"
    "Do not explain your reasoning. Output only the JSON object.\n"
)


def _load(run_dir: Path):
    anoms, claims = [], []
    oqs = []
    for r in oracle._read_jsonl(run_dir / "anomalies.jsonl"):
        try:
            anoms.append(Anomaly.model_validate(r))
        except Exception:  # noqa: BLE001
            pass
    for r in oracle._read_jsonl(run_dir / "claims.jsonl"):
        try:
            claims.append(Claim.model_validate(r))
        except Exception:  # noqa: BLE001
            pass
    for r in oracle._read_jsonl(run_dir / "open_questions.jsonl"):
        try:
            oqs.append(OpenQuestion.model_validate(r))
        except Exception:  # noqa: BLE001
            pass
    return anoms, claims, oqs


def _write_scratch(scratch: Path, hyps, src: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "hypotheses.jsonl").write_text(
        "\n".join(h.model_dump_json() for h in hyps), encoding="utf-8")
    for name in ("claims.jsonl", "anomalies.jsonl"):
        (scratch / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--anomalies", type=int, default=12)
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--scratch", default="artifacts/oracle/creator_ab_scratch")
    ap.add_argument("--judge-max-tokens", type=int, default=2500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if not configured_api_key():
        raise SystemExit("creator A/B requires an API key")
    run_dir = Path(args.run_dir)
    gen_model = configured_model(args.gen_model)
    judge_model = configured_model(args.judge_model)
    anoms, claims, oqs = _load(run_dir)
    anoms = anoms[: args.anomalies]
    print(f"creator A/B on {len(anoms)} anomalies | gen={gen_model} judge={judge_model}",
          file=sys.stderr)

    forward_prompt = creator.CREATOR_SYSTEM_PROMPT  # current (edited) module text
    arms = {"creator-frozen": FROZEN_CREATOR_PROMPT, "creator-forward": forward_prompt}
    results = {}
    for label, prompt in arms.items():
        creator.CREATOR_SYSTEM_PROMPT = prompt  # monkeypatch the single-pass prompt
        print(f"\n[gen] {label} ...", file=sys.stderr)
        hyps = generate_creator_hypotheses(anoms, claims, oqs, model=gen_model,
                                           max_anomalies=args.anomalies)
        scratch = Path(args.scratch) / label
        _write_scratch(scratch, hyps, run_dir)
        print(f"[gen] {label}: {len(hyps)} hyps", file=sys.stderr)
        res = oracle.run_oracle(scratch, limit=None, label=label,
                                model=judge_model, max_tokens=args.judge_max_tokens)
        oracle._print_report(res)
        res["n_hyps"] = len(hyps)
        results[label] = res
    creator.CREATOR_SYSTEM_PROMPT = forward_prompt  # leave module in forward state

    c, f = results["creator-frozen"], results["creator-forward"]
    print("\n=== CREATOR A/B DELTA (forward − frozen) ===")
    print(f"{'axis':24s} {'frozen':>9s} {'forward':>9s} {'Δ':>7s}")
    for ax in oracle.AXES:
        cv, fv = c["axis_means"][ax], f["axis_means"][ax]
        print(f"{ax:24s} {cv:9.3f} {fv:9.3f} {fv-cv:+7.3f}")
    print(f"{'dominant-shape %':24s} {c['diversity']['dominant_frac']*100:8.0f}% "
          f"{f['diversity']['dominant_frac']*100:8.0f}%")
    print(f"{'distinct shapes':24s} {c['diversity']['distinct_shapes']:9d} "
          f"{f['diversity']['distinct_shapes']:9d}")
    print(f"frozen  hist={c['diversity']['shape_histogram']}")
    print(f"forward hist={f['diversity']['shape_histogram']}")
    print(f"calibration both: {'PASS' if c['calibration']['pass'] and f['calibration']['pass'] else 'FAIL'}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
