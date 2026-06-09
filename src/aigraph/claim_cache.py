"""Cross-run claim-extraction cache (non-frozen).

Claim extraction is the per-paper LLM cost in the pipeline. The same arXiv
paper is re-extracted across overlapping runs — several reasoning corpora share
papers, so the LLM is paid two, three, four times for identical work. This
module is a content-addressed, write-once cache: keyed by (paper identity +
paper content + extractor model + reader config), it stores the frozen
extractor's output so any later run reuses it for free.

Design constraints (this is a FREEZE-SAFE optimisation, not a behaviour change):
- Wraps the FROZEN extractor purely from the non-frozen orchestration seam
  (``server.extract_claims_with_status``). It never imports or edits
  ``extract.py`` / ``llm_extract.py``.
- Behaviour-preserving: extraction runs at ``temperature=0.0``, so a cache hit
  returns the same claims a fresh extraction would. Only reader telemetry
  differs (a hit skips the reader, logged with ``reader_mode="cache"``).
- Fail-open: any cache error (missing/corrupt/permission) degrades silently to
  a normal uncached extraction.
- Only NON-EMPTY extractions are cached. An empty result is the frozen
  extractor's failure signal (the orchestrator retries it, #49) — caching it
  would freeze a transient failure forever.

If a frozen-extractor thaw ever changes the prompt or claim schema, bump
``CACHE_SCHEMA_VERSION`` so old entries are ignored.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from .models import Claim, Paper

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = "1"
_DISABLE_VALUES = {"0", "false", "no", "off", ""}


def cache_enabled() -> bool:
    """On by default; disable with ``AIGRAPH_CLAIM_CACHE=0``."""
    return os.environ.get("AIGRAPH_CLAIM_CACHE", "1").strip().lower() not in _DISABLE_VALUES


def cache_dir_for(claims_output: Path) -> Optional[Path]:
    """Resolve the shared cache directory, or ``None`` when caching is off.

    Defaults to ``<runs_root>/_claim_cache`` (a sibling of every run; the
    dashboard skips ``_``-prefixed dirs, so it never shows up as a request).
    Override the location with ``AIGRAPH_CLAIM_CACHE_DIR``.

    ``claims_output`` is the run's ``claims.jsonl`` path
    (``<runs_root>/<run_id>/claims.jsonl``).
    """
    if not cache_enabled():
        return None
    override = os.environ.get("AIGRAPH_CLAIM_CACHE_DIR")
    if override:
        return Path(override)
    return claims_output.parent.parent / "_claim_cache"


def _content_fingerprint(paper: Paper) -> str:
    """Hash the paper text that actually feeds extraction. A changed abstract or
    body produces a different key, so stale claims are never served."""
    parts = [paper.title or "", paper.abstract or "", paper.text or ""]
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def compute_key(
    paper: Paper,
    *,
    model: Optional[str],
    reader_mode: Optional[str] = None,
    reader_max_candidates: Optional[int] = None,
) -> str:
    """Stable cache key for one paper under one extraction configuration."""
    ident = paper.arxiv_id_base or paper.paper_id
    raw = "|".join(
        [
            CACHE_SCHEMA_VERSION,
            str(ident),
            str(model or ""),
            str(reader_mode or ""),
            "" if reader_max_candidates is None else str(reader_max_candidates),
            _content_fingerprint(paper),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _path_for(cache_dir: Path, key: str) -> Path:
    # shard by the first two hex chars to keep directory sizes sane
    return cache_dir / key[:2] / f"{key}.json"


def load(cache_dir: Optional[Path], key: Optional[str]) -> Optional[list[Claim]]:
    """Return the cached claims for ``key``, or ``None`` on miss / any error."""
    if cache_dir is None or not key:
        return None
    path = _path_for(cache_dir, key)
    try:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("claims")
        if not isinstance(rows, list):
            return None
        return [Claim.model_validate(row) for row in rows]
    except Exception as exc:  # pragma: no cover - defensive, fail-open
        logger.debug("claim cache load failed for %s: %s", key, exc)
        return None


def store(
    cache_dir: Optional[Path],
    key: Optional[str],
    claims: list[Claim],
    *,
    paper: Paper,
    model: Optional[str],
    reader_mode: Optional[str] = None,
) -> bool:
    """Persist a NON-EMPTY extraction (write-once, atomic, fail-open).

    Returns ``True`` if a new entry was written, ``False`` otherwise (caching
    off, empty claims, entry already present, or any error).
    """
    if cache_dir is None or not key or not claims:
        return False
    path = _path_for(cache_dir, key)
    try:
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "key": key,
            "paper_id": paper.paper_id,
            "arxiv_id_base": paper.arxiv_id_base,
            "model": model,
            "reader_mode": reader_mode,
            "fingerprint": _content_fingerprint(paper),
            "claim_count": len(claims),
            "claims": [claim.model_dump() for claim in claims],
        }
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception as exc:  # pragma: no cover - defensive, fail-open
        logger.debug("claim cache store failed for %s: %s", key, exc)
        return False


def stats(cache_dir: Optional[Path]) -> dict[str, Any]:
    """Lightweight footprint for reporting/dashboards."""
    if cache_dir is None:
        return {"enabled": cache_enabled(), "entries": 0, "dir": None}
    if not cache_dir.exists():
        return {"enabled": True, "entries": 0, "dir": str(cache_dir)}
    entries = sum(1 for _ in cache_dir.glob("*/*.json"))
    return {"enabled": True, "entries": entries, "dir": str(cache_dir)}
