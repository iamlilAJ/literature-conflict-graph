"""Intern-Atlas (J0) loader for aigraph.

Reads the local Intern-Atlas HF parquet mirror and exposes:

  load_papers_from_intern_atlas(...)  -> list[Paper]    (corpus source + enrichment)
  load_atlas_inbound_bottlenecks(...) -> dict[arxiv_id, list[bottleneck]]
  load_atlas_method_gazetteer(...)    -> dict[name_lower, method_id]

This is the J0 integration the recon (`docs/atlas-aigraph-fit-recon.md`)
recommended as a clean win — Atlas's `papers` parquet is a field-superset
of aigraph's ``Paper`` and 94% of a typical run's papers are already in
Atlas (joined by base arxiv id). The inbound-bottleneck loader is the
signal source for the empirically-validated `bottleneck_open_q_alignment`
joint anomaly (see ``joint_anomalies.py`` and
`docs/atlas-value-test-findings.md`).

polars is imported lazily so aigraph runs without it unless Atlas is used.
Default dir: $AIGRAPH_INTERN_ATLAS_DIR or data/intern_atlas.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Optional

from .models import Paper

_VER_RE = re.compile(r"v\d+$")


def _default_dir() -> Path:
    return Path(os.environ.get("AIGRAPH_INTERN_ATLAS_DIR", "data/intern_atlas"))


def _data_root(atlas_dir: Path | str | None) -> Path:
    d = Path(atlas_dir) if atlas_dir else _default_dir()
    # accept either the mirror root (has data/) or the data/ dir itself
    return d / "data" if (d / "data").is_dir() else d


def _glob(root: Path, table: str) -> list[str]:
    files = sorted(str(p) for p in (root / table).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards for {table!r} under {root}")
    return files


def _base_arxiv(v: str | None) -> str:
    return _VER_RE.sub("", str(v)) if v else ""


def is_available(atlas_dir: Path | str | None = None) -> bool:
    """True if the Atlas mirror + polars are usable."""
    try:
        import polars  # noqa: F401
    except ImportError:
        return False
    try:
        return (_data_root(atlas_dir) / "papers").is_dir()
    except Exception:
        return False


def load_papers_from_intern_atlas(
    atlas_dir: Path | str | None = None,
    *,
    arxiv_ids: Optional[Iterable[str]] = None,
    venues: Optional[Iterable[str]] = None,
    years: Optional[tuple[int, int]] = None,
    require_abstract: bool = False,
    limit: Optional[int] = None,
) -> list[Paper]:
    """Load Atlas papers as aigraph ``Paper`` objects.

    Filters (all optional, AND-combined): ``arxiv_ids`` (base form, version
    stripped), ``venues`` (matched against venue / venue_canonical),
    ``years`` (inclusive (lo, hi)). Atlas's paper_id is kept verbatim as
    ``Paper.paper_id``; arxiv/openalex/doi/s2 IDs populate the join fields.
    """
    import polars as pl

    root = _data_root(atlas_dir)
    lf = pl.scan_parquet(_glob(root, "papers"))

    if arxiv_ids is not None:
        wanted = {_base_arxiv(a) for a in arxiv_ids if a}
        # Atlas arxiv_id is already base form; compare directly.
        lf = lf.filter(pl.col("arxiv_id").is_in(list(wanted)))
    if years is not None:
        lo, hi = years
        lf = lf.filter(pl.col("year").is_not_null() & (pl.col("year") >= lo) & (pl.col("year") <= hi))
    if venues is not None:
        vs = list(venues)
        lf = lf.filter(pl.col("venue").is_in(vs) | pl.col("venue_canonical").is_in(vs))
    if require_abstract:
        lf = lf.filter(pl.col("abstract").is_not_null() & (pl.col("abstract") != ""))
    if limit:
        lf = lf.limit(limit)

    cols = ["paper_id", "title", "abstract", "year", "venue", "venue_canonical",
            "venue_tier", "citation_count", "influential_citation_count",
            "reference_count", "arxiv_id", "doi", "openalex_id", "s2_id", "pdf_url"]
    have = set(pl.scan_parquet(_glob(root, "papers")).collect_schema().names())
    df = lf.select([c for c in cols if c in have]).collect()

    out: list[Paper] = []
    for r in df.to_dicts():
        arxiv_base = _base_arxiv(r.get("arxiv_id"))
        out.append(Paper(
            paper_id=r["paper_id"],
            title=r.get("title") or "",
            year=int(r["year"]) if r.get("year") is not None else 0,
            venue=r.get("venue") or r.get("venue_canonical") or "",
            abstract=r.get("abstract") or "",
            doi=r.get("doi"),
            cited_by_count=int(r.get("citation_count") or 0),
            openalex_id=r.get("openalex_id"),
            arxiv_id_base=arxiv_base or None,
            arxiv_id_full=r.get("arxiv_id") or None,
            url=r.get("pdf_url"),
            s2_id=r.get("s2_id"),
            venue_canonical=r.get("venue_canonical"),
            venue_tier=r.get("venue_tier"),
            influential_citation_count=int(r.get("influential_citation_count") or 0),
            corpus_tag="intern_atlas",
            retrieval_channel="intern_atlas",
        ))
    return out


def load_atlas_inbound_bottlenecks(
    arxiv_ids: Iterable[str],
    atlas_dir: Path | str | None = None,
    *,
    min_confidence: float = 0.0,
) -> dict[str, list[dict]]:
    """For each base arxiv id in ``arxiv_ids``, return the THIRD-PARTY
    bottlenecks asserted about it — i.e. Atlas evolution edges where the
    paper is ``paper_b`` (the one a newer paper improves/replaces/extends),
    carrying a parsed ``bottleneck_json``.

    Returns ``{arxiv_base: [{source_paper, source_title, relation,
    confidence, dimension, severity, description, quote}, ...]}``.
    """
    import polars as pl

    root = _data_root(atlas_dir)
    wanted = [_base_arxiv(a) for a in arxiv_ids if a]
    if not wanted:
        return {}
    lf = (pl.scan_parquet(_glob(root, "paper_evolution_edges"))
          .filter(pl.col("paper_b_arxiv_id").is_in(wanted))
          .filter(pl.col("bottleneck_json").is_not_null() & (pl.col("bottleneck_json") != "")))
    if min_confidence > 0:
        lf = lf.filter(pl.col("relation_confidence") >= min_confidence)
    df = lf.select("paper_b_arxiv_id", "paper_a_arxiv_id", "paper_a_title",
                   "evolution_relation", "relation_confidence", "bottleneck_json").collect()

    out: dict[str, list[dict]] = {}
    for r in df.to_dicts():
        b = _parse_bottleneck(r.get("bottleneck_json"))
        out.setdefault(r["paper_b_arxiv_id"], []).append({
            "source_paper": r.get("paper_a_arxiv_id") or "",
            "source_title": r.get("paper_a_title") or "",
            "relation": r.get("evolution_relation") or "",
            "confidence": float(r.get("relation_confidence") or 0.0),
            **b,
        })
    return out


def _parse_bottleneck(raw: str | None) -> dict:
    try:
        o = json.loads(raw) if raw else {}
        if isinstance(o, dict):
            return {"dimension": o.get("dimension") or "", "severity": o.get("severity") or "",
                    "description": o.get("description") or "", "quote": (o.get("quote") or "")[:400]}
    except Exception:
        pass
    return {"dimension": "", "severity": "", "description": (raw or "")[:300], "quote": ""}


def load_atlas_method_gazetteer(atlas_dir: Path | str | None = None) -> dict[str, str]:
    """Lowercased method_name -> method_id, from `paper_methods` (V_M).
    Soft gazetteer for canonical-method resolution (J0.5)."""
    import polars as pl

    root = _data_root(atlas_dir)
    df = (pl.scan_parquet(_glob(root, "paper_methods"))
          .select("method_name", "method_id").unique().collect())
    out: dict[str, str] = {}
    for r in df.to_dicts():
        name = (r.get("method_name") or "").strip().lower()
        if name:
            out.setdefault(name, r["method_id"])
    return out
