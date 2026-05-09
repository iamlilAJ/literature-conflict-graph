"""Build the controlled-validation cohort by filtering Intern-Atlas papers.

Replaces the OpenAlex/S2 cohort-fetch step in
``docs/controlled-validation-design.md §3.1``. Reads the local
HuggingFace ``OpenRaiser/Intern-Atlas`` parquet for ``papers`` and emits
a per-(venue, year) stratified sample to ``artifacts/validation_v1/``.

Usage::

    python scripts/cohort_from_intern_atlas.py \
        --intern-atlas data/intern_atlas \
        --out artifacts/validation_v1/cohorts \
        --per-cell 200 \
        --seed 17

Inputs (read-only):
    <intern-atlas>/papers.parquet
    <intern-atlas>/paper_evolution_edges.parquet  (optional, only for
                                                   reachability stats)

Outputs:
    <out>/primary_2018_2020.parquet   (1800 papers; 3 venues × 3 years × N)
    <out>/peri_boom_2020_2022.parquet
    <out>/post_boom_2022_2023.parquet (smaller — fewer venue-years)
    <out>/_summary.json               (counts per cell, filter losses,
                                       seed, schema_version)

Quality filters (addresses Q6 of the recon: ~10-20% paper_b URLs etc):
    - title not null, len >= 8 chars
    - abstract not null, len >= 80 chars
    - year in target range
    - venue_canonical in target set (case-sensitive, matches HF strings)
    - has at least 2 IDs out of (arxiv_id, doi, openalex_id, s2_id) -- so
      a downstream lookup has a fallback
    - cited_by_count is present (will be the L1 outcome)

Determinism:
    Sampling uses a fixed seed; same input parquet + same seed = same
    cohort byte-identical. Also writes the input file mtime/size to
    _summary.json so we can detect a silent dataset bump from the
    upstream HF release.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


COHORT_SCHEMA_VERSION = 1

TARGET_VENUES = ["NeurIPS", "ICML", "ICLR"]

COHORTS = {
    "primary_2018_2020":   {"years": [2018, 2019, 2020], "outcome_window": (2020, 2023)},
    "peri_boom_2020_2022": {"years": [2020, 2021, 2022], "outcome_window": (2022, 2024)},
    "post_boom_2022_2023": {"years": [2022, 2023],       "outcome_window": (2024, 2025)},
}

ID_COLS = ["arxiv_id", "doi", "openalex_id", "s2_id"]


def _has_min_ids(df: pl.DataFrame, min_count: int = 1) -> pl.Series:
    """True for rows with >= ``min_count`` non-null ID fields.

    Relaxed from min=2 to min=1 in v1 because Intern-Atlas paper rows
    typically carry only one ID system (usually arxiv_id or s2_id).
    A single ID is enough to look up cite outcomes downstream.
    """
    counts = sum(
        df[col].is_not_null().cast(pl.Int8) for col in ID_COLS if col in df.columns
    )
    return counts >= min_count


def _quality_filter(papers: pl.DataFrame) -> pl.DataFrame:
    """Drop rows that fail the recon §Q6 quality bar."""
    df = papers.filter(
        pl.col("title").is_not_null()
        & (pl.col("title").str.len_chars() >= 8)
        & pl.col("abstract").is_not_null()
        & (pl.col("abstract").str.len_chars() >= 80)
        & pl.col("year").is_not_null()
        & pl.col("venue_canonical").is_in(TARGET_VENUES)
        & pl.col("citation_count").is_not_null()
    )
    df = df.filter(_has_min_ids(df, min_count=1))
    return df


def _stratified_sample(
    df: pl.DataFrame,
    venues: list[str],
    years: list[int],
    per_cell: int,
    seed: int,
) -> tuple[pl.DataFrame, dict]:
    """Sample ``per_cell`` rows per (venue, year) cell, deterministic in seed."""
    parts: list[pl.DataFrame] = []
    cell_counts: dict[str, dict[int, dict]] = {}
    for v in venues:
        cell_counts[v] = {}
        for y in years:
            cell = df.filter((pl.col("venue_canonical") == v) & (pl.col("year") == y))
            available = len(cell)
            n = min(per_cell, available)
            sampled = cell.sample(n=n, seed=seed) if n > 0 else cell.head(0)
            parts.append(sampled)
            cell_counts[v][y] = {"available": available, "sampled": n}
    out = pl.concat(parts) if parts else df.head(0)
    return out, cell_counts


def build_cohorts(
    intern_atlas_dir: Path,
    out_dir: Path,
    per_cell: int = 200,
    seed: int = 17,
) -> dict:
    intern_atlas_dir = Path(intern_atlas_dir)
    # Accept either a single packed file or the HF sharded layout.
    single = intern_atlas_dir / "papers.parquet"
    sharded_dir = intern_atlas_dir / "data" / "papers"
    if single.exists():
        papers_path: Path = single
        shard_paths = [single]
    elif sharded_dir.is_dir():
        shard_paths = sorted(sharded_dir.glob("*.parquet"))
        if not shard_paths:
            raise FileNotFoundError(
                f"Expected parquet shards under {sharded_dir}/ -- run the HF download step first"
            )
        papers_path = sharded_dir
    else:
        raise FileNotFoundError(
            f"Expected {single} or {sharded_dir}/*.parquet -- run the HF download step first"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_size = sum(p.stat().st_size for p in shard_paths)
    latest_mtime = max(p.stat().st_mtime for p in shard_paths)
    src_provenance = {
        "papers_parquet_path": str(papers_path),
        "papers_parquet_size": total_size,
        "papers_parquet_mtime": datetime.fromtimestamp(
            latest_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
        "n_shards": len(shard_paths),
    }

    raw = pl.read_parquet([str(p) for p in shard_paths])
    pre = len(raw)
    filtered = _quality_filter(raw)
    post = len(filtered)

    summary: dict = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "per_cell_target": per_cell,
        "target_venues": TARGET_VENUES,
        "source": src_provenance,
        "filter_losses": {
            "rows_before_filter": pre,
            "rows_after_filter": post,
            "drop_ratio": round(1 - post / pre, 4) if pre else 0.0,
        },
        "cohorts": {},
    }

    for cohort_name, spec in COHORTS.items():
        sampled, cell_counts = _stratified_sample(
            filtered,
            venues=TARGET_VENUES,
            years=spec["years"],
            per_cell=per_cell,
            seed=seed,
        )
        out_path = out_dir / f"{cohort_name}.parquet"
        sampled.write_parquet(out_path)
        summary["cohorts"][cohort_name] = {
            "outcome_window": spec["outcome_window"],
            "years": spec["years"],
            "total_sampled": len(sampled),
            "per_cell": cell_counts,
            "out_path": str(out_path),
        }

    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intern-atlas", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-cell", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    summary = build_cohorts(
        intern_atlas_dir=args.intern_atlas,
        out_dir=args.out,
        per_cell=args.per_cell,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
