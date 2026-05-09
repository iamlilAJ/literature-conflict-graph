"""Mine cross-paper conflicts from Intern-Atlas's own paper_evolution_edges.

This is the empirical answer to the differentiation claim in
`docs/intern-atlas-pivot.md` §4.1: Atlas's 7 edge types are all causal
(extends / improves / replaces / adapts / uses_component / compares /
background) — none of them encode "paper A and paper B disagree about
the same effect on the same target". But Atlas's own `impact_json`
field per edge records `improvement_dimensions`, `sacrifice_dimensions`,
and `tradeoff` text, which means we can mine conflicts entirely from
their public data, no LLM required.

A conflict here is: two edges with the same `paper_b_id` (the cited /
extended target) on the same dimension, but one edge has the dimension
in `improvement_dimensions` and the other has it in
`sacrifice_dimensions`. That is, two follow-on papers disagree about
whether a given axis (accuracy / latency / robustness / ...) moves up
or down relative to the same predecessor.

This script outputs three artifacts:

    <out>/atlas_conflicts.parquet         all candidate conflicts
    <out>/atlas_conflicts_top.csv         top-N for human review w/ quotes
    <out>/atlas_conflicts_summary.json    counts per dim / year / venue

Usage:
    python3 scripts/atlas_conflict_finder.py \
        --intern-atlas data/intern_atlas \
        --out artifacts/atlas_conflicts \
        --min-confidence 0.6 \
        --top-n 100

Optionally restrict to edges touching our 1790-paper primary cohort:
    --cohort-parquet artifacts/validation_v1/cohorts/primary_2018_2020.parquet

Determinism: same input parquet + same args + same seed → same output.
No LLM, no network, all polars + json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


# Atlas typology — paper §3.2.
# The strong-causal subset is the union {extends, improves, replaces, adapts}.
STRONG_CAUSAL = {"extends", "improves", "replaces", "adapts"}


def _load_edges(intern_atlas_dir: Path) -> pl.DataFrame:
    single = intern_atlas_dir / "paper_evolution_edges.parquet"
    sharded_dir = intern_atlas_dir / "data" / "paper_evolution_edges"
    if single.exists():
        paths = [single]
    elif sharded_dir.is_dir():
        paths = sorted(sharded_dir.glob("*.parquet"))
        if not paths:
            sys.exit(
                f"No parquet shards in {sharded_dir}/ — "
                "did the HF download step finish?"
            )
    else:
        sys.exit(
            f"Expected {single} or {sharded_dir}/*.parquet — "
            "did the HF download step finish?"
        )
    return pl.read_parquet([str(p) for p in paths])


def _load_cohort_ids(cohort_path: Path | None) -> set[str] | None:
    if cohort_path is None:
        return None
    df = pl.read_parquet(cohort_path)
    # Accept either paper_id or paper_a_id-style columns.
    for col in ("paper_id", "id"):
        if col in df.columns:
            return set(df[col].to_list())
    sys.exit(f"Could not find paper id column in {cohort_path}")


def _parse_impact(raw: str | None) -> dict[str, Any]:
    """Parse impact_json string. Returns dict with normalized arrays."""
    if not raw or raw in ("", "{}", "null"):
        return {"improve": [], "sacrifice": [], "tradeoff": ""}
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"improve": [], "sacrifice": [], "tradeoff": ""}
    return {
        "improve": list(d.get("improvement_dimensions", []) or []),
        "sacrifice": list(d.get("sacrifice_dimensions", []) or []),
        "tradeoff": (d.get("tradeoff") or d.get("tradeoffs") or ""),
    }


def _explode_directions(edges: pl.DataFrame) -> pl.DataFrame:
    """One row per (edge, dim, direction) where direction ∈ {improve, sacrifice}."""
    rows: list[dict[str, Any]] = []
    for r in edges.iter_rows(named=True):
        parsed = _parse_impact(r.get("impact_json"))
        for dim in parsed["improve"]:
            rows.append(
                {
                    "paper_a_id": r["paper_a_id"],
                    "paper_a_title": r.get("paper_a_title"),
                    "paper_a_year": r.get("paper_a_year"),
                    "paper_a_venue": r.get("paper_a_venue_canonical"),
                    "paper_b_id": r["paper_b_id"],
                    "paper_b_title": r.get("paper_b_title"),
                    "paper_b_year": r.get("paper_b_year"),
                    "paper_b_venue": r.get("paper_b_venue_canonical"),
                    "evolution_relation": r["evolution_relation"],
                    "relation_confidence": r.get("relation_confidence"),
                    "dim": str(dim).strip().lower(),
                    "direction": "improve",
                    "tradeoff_quote": parsed["tradeoff"],
                }
            )
        for dim in parsed["sacrifice"]:
            rows.append(
                {
                    "paper_a_id": r["paper_a_id"],
                    "paper_a_title": r.get("paper_a_title"),
                    "paper_a_year": r.get("paper_a_year"),
                    "paper_a_venue": r.get("paper_a_venue_canonical"),
                    "paper_b_id": r["paper_b_id"],
                    "paper_b_title": r.get("paper_b_title"),
                    "paper_b_year": r.get("paper_b_year"),
                    "paper_b_venue": r.get("paper_b_venue_canonical"),
                    "evolution_relation": r["evolution_relation"],
                    "relation_confidence": r.get("relation_confidence"),
                    "dim": str(dim).strip().lower(),
                    "direction": "sacrifice",
                    "tradeoff_quote": parsed["tradeoff"],
                }
            )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def _find_conflicts(direction_rows: pl.DataFrame) -> pl.DataFrame:
    """For each (paper_b_id, dim) collect both directions when both exist."""
    if len(direction_rows) == 0:
        return pl.DataFrame()
    improvers = (
        direction_rows.filter(pl.col("direction") == "improve")
        .rename(
            {
                "paper_a_id": "improver_id",
                "paper_a_title": "improver_title",
                "paper_a_year": "improver_year",
                "paper_a_venue": "improver_venue",
                "evolution_relation": "improver_relation",
                "relation_confidence": "improver_conf",
                "tradeoff_quote": "improver_quote",
            }
        )
        .drop("direction")
    )
    sacrificers = (
        direction_rows.filter(pl.col("direction") == "sacrifice")
        .rename(
            {
                "paper_a_id": "sacrificer_id",
                "paper_a_title": "sacrificer_title",
                "paper_a_year": "sacrificer_year",
                "paper_a_venue": "sacrificer_venue",
                "evolution_relation": "sacrificer_relation",
                "relation_confidence": "sacrificer_conf",
                "tradeoff_quote": "sacrificer_quote",
            }
        )
        .drop("direction")
    )
    # Join on (paper_b_id, dim) — same target, same dimension.
    # Drop self-conflicts (improver_id == sacrificer_id).
    conflicts = improvers.join(
        sacrificers,
        on=["paper_b_id", "dim"],
        how="inner",
        suffix="_drop",
    ).filter(pl.col("improver_id") != pl.col("sacrificer_id"))
    # Drop the duplicated paper_b_* and shared cols brought in by join suffix.
    drop_cols = [c for c in conflicts.columns if c.endswith("_drop")]
    if drop_cols:
        conflicts = conflicts.drop(drop_cols)
    # Joint confidence (geometric mean works well in [0,1]).
    conflicts = conflicts.with_columns(
        (
            (pl.col("improver_conf").fill_null(0.0))
            * (pl.col("sacrificer_conf").fill_null(0.0))
        ).sqrt().alias("joint_conf")
    )
    return conflicts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intern-atlas", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--cohort-parquet",
        type=Path,
        default=None,
        help="Optional: restrict to edges where paper_a or paper_b is in this cohort.",
    )
    ap.add_argument(
        "--strong-causal-only",
        action="store_true",
        help="Restrict to extends/improves/replaces/adapts edges (skip compares/uses_component).",
    )
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"-- loading edges from {args.intern_atlas}")
    edges = _load_edges(args.intern_atlas)
    pre = len(edges)
    print(f"   loaded {pre:,} edges")

    # 1. Filter to edges with non-null impact_json.
    edges = edges.filter(
        pl.col("impact_json").is_not_null() & (pl.col("impact_json") != "")
    )
    print(f"   {len(edges):,} after impact_json filter")

    # 2. Optional strong-causal filter.
    if args.strong_causal_only:
        edges = edges.filter(pl.col("evolution_relation").is_in(STRONG_CAUSAL))
        print(f"   {len(edges):,} after strong-causal filter")

    # 3. Optional confidence filter.
    if args.min_confidence > 0:
        edges = edges.filter(pl.col("relation_confidence") >= args.min_confidence)
        print(f"   {len(edges):,} after confidence ≥ {args.min_confidence} filter")

    # 4. Optional cohort restriction.
    cohort_ids = _load_cohort_ids(args.cohort_parquet)
    if cohort_ids is not None:
        edges = edges.filter(
            pl.col("paper_a_id").is_in(cohort_ids)
            | pl.col("paper_b_id").is_in(cohort_ids)
        )
        print(f"   {len(edges):,} after cohort restriction ({len(cohort_ids):,} ids)")

    # 5. Explode impact_json into per-(edge, dim, direction) rows.
    print(f"-- exploding impact dimensions ...")
    direction_rows = _explode_directions(edges)
    print(f"   {len(direction_rows):,} (edge, dim, direction) rows")

    # 6. Find conflicts.
    print(f"-- finding conflicts ...")
    conflicts = _find_conflicts(direction_rows)
    print(f"   {len(conflicts):,} candidate conflicts")
    if len(conflicts) == 0:
        sys.exit(
            "No conflicts found — try lowering --min-confidence "
            "or removing --strong-causal-only"
        )

    # 7. Persist all conflicts.
    full_path = args.out / "atlas_conflicts.parquet"
    conflicts.write_parquet(full_path)
    print(f"-- wrote {full_path} ({len(conflicts):,} rows)")

    # 8. Top-N for human review (CSV with quotes inline).
    top = conflicts.sort("joint_conf", descending=True).head(args.top_n)
    top_path = args.out / "atlas_conflicts_top.csv"
    top.write_csv(top_path)
    print(f"-- wrote {top_path} (top {len(top):,})")

    # 9. Summary stats.
    by_dim = (
        conflicts.group_by("dim")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(30)
    )
    by_year_pair = (
        conflicts.with_columns(
            pl.min_horizontal("improver_year", "sacrificer_year").alias("year_min"),
            pl.max_horizontal("improver_year", "sacrificer_year").alias("year_max"),
        )
        .group_by(["year_min", "year_max"])
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(30)
    )
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": {
            "intern_atlas": str(args.intern_atlas),
            "cohort_parquet": str(args.cohort_parquet) if args.cohort_parquet else None,
            "strong_causal_only": args.strong_causal_only,
            "min_confidence": args.min_confidence,
            "top_n": args.top_n,
        },
        "edges_in": pre,
        "edges_after_filters": len(edges),
        "direction_rows": len(direction_rows),
        "conflicts": len(conflicts),
        "by_dim_top30": by_dim.to_dicts(),
        "by_year_pair_top30": by_year_pair.to_dicts(),
        "joint_conf_p50": float(conflicts["joint_conf"].quantile(0.5) or 0.0),
        "joint_conf_p90": float(conflicts["joint_conf"].quantile(0.9) or 0.0),
        "joint_conf_max": float(conflicts["joint_conf"].max() or 0.0),
    }
    summary_path = args.out / "atlas_conflicts_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"-- wrote {summary_path}")
    print()
    print("=" * 60)
    print(f"DONE. {len(conflicts):,} candidate conflicts mined.")
    print(f"      top-{args.top_n} CSV: {top_path}")
    print(f"      summary:           {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
