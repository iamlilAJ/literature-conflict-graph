"""Backtest Phase 1 influence prediction on existing creator hypotheses.

Reads the existing 1000-paper run artifacts (creator_hypotheses.jsonl,
claims.jsonl, papers.jsonl, hierarchy.json) from a run directory,
filters to hypotheses grounded in venue papers (ICML/NeurIPS/ICLR or,
on arxiv-only corpora, papers with the highest cited_by_count), runs
``predict_influence_batch``, and reports:

  - Spearman ρ between predicted total and L1 actual impact (max
    cited_by_count of inspired papers).
  - Top-10 overlap (# of top-10-predicted that also appear in
    top-10-actual).
  - Per-dimension correlations (which dim drives or anti-correlates).
  - Top-5 predicted + top-5 actual for qualitative inspection.

The validation report stub is appended to
``docs/influence-prediction-validation.md`` at the end so the
empirical numbers live in the repo.

Usage:
    python scripts/validate_influence_backtest.py /tmp/fullrun_v2

scipy is intentionally not used; correlation is computed via
``statistics.correlation`` over rank-converted inputs (Pearson on
ranks = Spearman).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import correlation
from typing import Any

# Make `aigraph` importable when this script runs from the repo root.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from aigraph.influence import (  # noqa: E402  (post-sys.path)
    InfluenceScore,
    _load_hierarchy_dict,
    predict_influence_batch,
)
from aigraph.io import read_jsonl  # noqa: E402
from aigraph.models import Claim, Hypothesis, Paper  # noqa: E402


_VENUE_NEEDLES = ("icml", "neurips", "iclr", "advances in neural")


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman ρ via Pearson on rank-converted inputs.

    Returns None on degenerate inputs (size < 2 or constant arrays).
    Uses ``statistics.correlation`` from Python 3.10+.
    """
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None

    def _rank(arr: list[float]) -> list[float]:
        order = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        for r, idx in enumerate(order, start=1):
            ranks[idx] = float(r)
        return ranks

    return correlation(_rank(xs), _rank(ys))


def _l1_actual_impact(
    h: Hypothesis,
    claims_by_id: dict[str, Claim],
    papers_by_id: dict[str, Paper],
) -> int:
    """Max cited_by_count over papers cited via explains_claims."""
    max_cite = 0
    for c_id in h.explains_claims or []:
        c = claims_by_id.get(c_id)
        if c is None:
            continue
        p = papers_by_id.get(c.paper_id)
        if p is None:
            continue
        max_cite = max(max_cite, p.cited_by_count or 0)
    return max_cite


def _is_venue_paper(p: Paper) -> bool:
    venue = (p.venue or "").lower()
    return any(needle in venue for needle in _VENUE_NEEDLES)


def main(run_dir: Path) -> dict[str, Any]:
    print(f"Loading data from {run_dir}...")
    hyps = read_jsonl(run_dir / "creator_hypotheses.jsonl", Hypothesis)
    claims = read_jsonl(run_dir / "claims.jsonl", Claim)
    papers = read_jsonl(run_dir / "papers.jsonl", Paper)
    hierarchy = _load_hierarchy_dict(run_dir / "hierarchy.json")
    claims_by_id = {c.claim_id: c for c in claims}
    papers_by_id = {p.paper_id: p for p in papers}

    print(f"  hypotheses: {len(hyps)}")
    print(f"  claims:     {len(claims)}")
    print(f"  papers:     {len(papers)}")
    print(f"  hierarchy:  {len(hierarchy.get('domains') or {})} domains, "
          f"{len(hierarchy.get('communities') or {})} communities, "
          f"{len(hierarchy.get('clusters') or {})} clusters")

    # Filter venue papers. On arxiv-only corpora the venue field is
    # often "arXiv" — fall back to top-100-by-cited_by_count as a
    # proxy (the design assumes high-citation papers are venue-quality).
    venue_papers = [p for p in papers if _is_venue_paper(p)]
    if len(venue_papers) < 20:
        print(f"  venue-string match weak ({len(venue_papers)}); "
              f"falling back to top-100 cited papers as proxy.")
        venue_papers = sorted(
            (p for p in papers if (p.cited_by_count or 0) > 0),
            key=lambda p: -(p.cited_by_count or 0),
        )[:100]
    venue_paper_ids = {p.paper_id for p in venue_papers}
    print(f"  venue/proxy papers: {len(venue_papers)}")

    venue_hyps = [
        h for h in hyps
        if any(
            (claims_by_id.get(c_id) and claims_by_id[c_id].paper_id in venue_paper_ids)
            for c_id in (h.explains_claims or [])
        )
    ]
    print(f"  venue-grounded hypotheses: {len(venue_hyps)}\n")

    if not venue_hyps:
        print("No venue-grounded hypotheses; backtest aborted.")
        return {"status": "no_data"}

    print("Computing predictions + actuals...")
    scores = predict_influence_batch(venue_hyps, hierarchy, claims_by_id)
    actuals = [_l1_actual_impact(h, claims_by_id, papers_by_id) for h in venue_hyps]
    predicted = [s.total for s in scores]

    rho = _spearman(predicted, actuals)
    rho_str = f"{rho:.3f}" if rho is not None else "N/A (degenerate input)"
    print(f"\n=== RESULT ===")
    print(f"Spearman correlation (total vs max-cite): rho = {rho_str}")
    print(f"  rho > 0.4  -> Phase 1 design validated, proceed to Phase 2")
    print(f"  0.2 < rho  -> partial validation, tune weights")
    print(f"  rho < 0.2  -> re-examine proxy choices")

    # Per-dimension correlations.
    dim_rhos: dict[str, float | None] = {}
    print(f"\n=== PER-DIMENSION ===")
    for dim in ("community_reach", "novelty", "grounding_depth"):
        dim_vals = [getattr(s, dim) for s in scores]
        dim_rhos[dim] = _spearman(dim_vals, actuals)
        rho_d = dim_rhos[dim]
        print(f"  {dim:<22}: rho = "
              f"{f'{rho_d:.3f}' if rho_d is not None else 'N/A (constant)'}")
    # Risk dimension (inverse interpretation).
    risk_vals = [s.scope_overreach_risk for s in scores]
    dim_rhos["scope_overreach_risk"] = _spearman(risk_vals, actuals)
    rho_d = dim_rhos["scope_overreach_risk"]
    print(f"  {'scope_overreach_risk':<22}: rho = "
          f"{f'{rho_d:.3f}' if rho_d is not None else 'N/A (constant)'}  "
          f"(NB: negative ρ here is the desired direction — high risk should hurt impact)")

    # Top-10 overlap.
    n_top = min(10, len(venue_hyps))
    pred_idx = sorted(range(len(scores)), key=lambda i: -predicted[i])[:n_top]
    actual_idx = sorted(range(len(actuals)), key=lambda i: -actuals[i])[:n_top]
    overlap = len(set(pred_idx) & set(actual_idx))
    print(f"\n=== TOP-{n_top} OVERLAP ===")
    print(f"  {overlap}/{n_top} ({100 * overlap / max(n_top, 1):.0f}%)")

    print(f"\n=== TOP-5 PREDICTED ===")
    for i in pred_idx[:5]:
        h = venue_hyps[i]
        print(f"  {h.hypothesis_id}: predicted={predicted[i]:.3f}, "
              f"actual_cite={actuals[i]}")
        print(f"    {h.hypothesis[:120]}")

    print(f"\n=== TOP-5 ACTUAL ===")
    for i in actual_idx[:5]:
        h = venue_hyps[i]
        print(f"  {h.hypothesis_id}: actual_cite={actuals[i]}, "
              f"predicted={predicted[i]:.3f}")
        print(f"    {h.hypothesis[:120]}")

    return {
        "status": "ok",
        "n_hyps": len(venue_hyps),
        "rho_total": rho,
        "rho_per_dim": dim_rhos,
        "top_n_overlap": overlap,
        "top_n": n_top,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: validate_influence_backtest.py <run_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]))
