"""Blind A/B: frozen back-explanation generator vs forward-design generator
(NON-FROZEN tooling).

Isolates the GENERATION CONTRACT as the only variable: both arms run on the
SAME anomaly seeds, SAME claims, SAME generation model, through the SAME
`hypotheses.generate_hypotheses` orchestrator. The only difference is which
`HypothesisGenerator` is plugged in:

    control = llm_hypotheses.LLMHypothesisGenerator   (frozen, v0.7-frozen tag)
    forward = forward_hypotheses.ForwardDesignGenerator (the rewrite)

Each arm's freshly generated hypotheses are written to a scratch run dir (with
the source run's claims/anomalies copied in) and scored by the SAME blind
`hyp_quality_oracle` judge. The judge never sees arm identity → blind by
construction. The script prints a per-axis + diversity delta table.

Usage
  python scripts/hyp_ab_generators.py --run-dir artifacts/runs/<run> \
      --anomalies 20 [--gen-model DeepSeek-V4-Flash] [--judge-model ...] \
      [--out artifacts/oracle/ab_<run>.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_SCRIPTS.parent / "src"))

import hyp_quality_oracle as oracle  # noqa: E402

from aigraph.hypotheses import generate_hypotheses  # noqa: E402
from aigraph.llm_hypotheses import LLMHypothesisGenerator  # noqa: E402
from aigraph.forward_hypotheses import ForwardDesignGenerator  # noqa: E402
from aigraph.llm_client import configured_api_key, configured_model  # noqa: E402
from aigraph.models import Anomaly, Claim  # noqa: E402


def _load_models(run_dir: Path) -> tuple[list[Anomaly], list[Claim]]:
    anoms: list[Anomaly] = []
    for row in oracle._read_jsonl(run_dir / "anomalies.jsonl"):
        try:
            anoms.append(Anomaly.model_validate(row))
        except Exception as e:  # noqa: BLE001
            print(f"  skip anomaly {row.get('anomaly_id')}: {e}", file=sys.stderr)
    claims: list[Claim] = []
    for row in oracle._read_jsonl(run_dir / "claims.jsonl"):
        try:
            claims.append(Claim.model_validate(row))
        except Exception as e:  # noqa: BLE001
            print(f"  skip claim {row.get('claim_id')}: {e}", file=sys.stderr)
    return anoms, claims


def _stratify(anoms: list[Anomaly]) -> list[Anomaly]:
    """Round-robin anomalies by type so a head-slice spans all types present
    (the file is grouped by type, so a naive [:N] would only see conflicts)."""
    buckets: dict[str, list[Anomaly]] = {}
    for a in anoms:
        buckets.setdefault(a.type, []).append(a)
    order = sorted(buckets, key=lambda t: -len(buckets[t]))  # biggest types first
    out: list[Anomaly] = []
    while any(buckets[t] for t in order):
        for t in order:
            if buckets[t]:
                out.append(buckets[t].pop(0))
    return out


def _write_scratch(scratch: Path, hyps: list, src: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "hypotheses.jsonl").write_text(
        "\n".join(h.model_dump_json() for h in hyps), encoding="utf-8")
    for name in ("claims.jsonl", "anomalies.jsonl"):
        (scratch / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--anomalies", type=int, default=20, help="#anomaly seeds (same for both arms)")
    ap.add_argument("--gen-model", default=None, help="generation model for BOTH arms")
    ap.add_argument("--judge-model", default=None, help="oracle judge model")
    ap.add_argument("--scratch", default="artifacts/oracle/ab_scratch")
    ap.add_argument("--out", default=None)
    ap.add_argument("--judge-max-tokens", type=int, default=2500)
    ap.add_argument("--no-stratify", dest="stratify", action="store_false",
                    help="disable round-robin-by-type sampling (default: stratified)")
    args = ap.parse_args(argv)

    if not configured_api_key():
        raise SystemExit("A/B requires an API key (OPENAI_API_KEY)")

    run_dir = Path(args.run_dir)
    gen_model = configured_model(args.gen_model)
    judge_model = configured_model(args.judge_model)
    anoms, claims = _load_models(run_dir)
    if not anoms:
        raise SystemExit(f"no anomalies in {run_dir}")
    if args.stratify:
        anoms = _stratify(anoms)
    anoms = anoms[: args.anomalies]
    print(f"A/B on {len(anoms)} anomaly seeds | gen_model={gen_model} | judge_model={judge_model}",
          file=sys.stderr)

    arms = {
        "control-frozen": LLMHypothesisGenerator(model=gen_model),
        "forward-design": ForwardDesignGenerator(model=gen_model),
    }
    results: dict[str, Any] = {}
    for label, gen in arms.items():
        print(f"\n[gen] {label} ...", file=sys.stderr)
        hyps = generate_hypotheses(anoms, claims, generator=gen)
        scratch = Path(args.scratch) / label
        _write_scratch(scratch, hyps, run_dir)
        print(f"[gen] {label}: {len(hyps)} hypotheses from {len(anoms)} anomalies "
              f"({len(hyps)/max(1,len(anoms)):.2f}/anomaly)", file=sys.stderr)
        print(f"[judge] {label} ...", file=sys.stderr)
        res = oracle.run_oracle(scratch, limit=None, label=label,
                                model=judge_model, max_tokens=args.judge_max_tokens)
        oracle._print_report(res)
        res["n_hyps"] = len(hyps)
        res["n_anomalies"] = len(anoms)
        results[label] = res

    # delta table
    c, f = results["control-frozen"], results["forward-design"]
    print("\n=== A/B DELTA (forward − control) ===")
    print(f"{'axis':24s} {'control':>9s} {'forward':>9s} {'Δ':>7s}")
    for ax in oracle.AXES:
        cv, fv = c["axis_means"][ax], f["axis_means"][ax]
        print(f"{ax:24s} {cv:9.3f} {fv:9.3f} {fv-cv:+7.3f}")
    print(f"{'hyps/anomaly':24s} {c['n_hyps']/c['n_anomalies']:9.2f} "
          f"{f['n_hyps']/f['n_anomalies']:9.2f}")
    print(f"{'dominant-shape %':24s} {c['diversity']['dominant_frac']*100:8.0f}% "
          f"{f['diversity']['dominant_frac']*100:8.0f}%")
    print(f"{'distinct shapes':24s} {c['diversity']['distinct_shapes']:9d} "
          f"{f['diversity']['distinct_shapes']:9d}")
    print(f"control dominant={c['diversity']['dominant_shape']} hist={c['diversity']['shape_histogram']}")
    print(f"forward dominant={f['diversity']['dominant_shape']} hist={f['diversity']['shape_histogram']}")
    cal_ok = c["calibration"]["pass"] and f["calibration"]["pass"]
    print(f"\ncalibration both arms: {'PASS' if cal_ok else 'FAIL — deltas suspect'}")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
