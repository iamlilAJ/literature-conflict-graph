"""Run-local controlled taxonomy post-pass (#50, NON-FROZEN).

The frozen extractor (``llm_extract.py``) canonicalizes each claim's method/task
onto a GLOBAL fixed vocabulary (``CANONICAL_METHODS`` / ``CANONICAL_TASKS``) and
falls back to ``"other"`` when nothing fits. On a niche topic that buries most
claims in ``canonical_method``/``canonical_task == "other"`` (a diagnostic run
had 28/39 method=other, 36/39 task=other). Anomaly detection clusters on
``(canonical_method, canonical_task)``, so an "other"-heavy run produces no
clusters and therefore no anomalies/hypotheses.

This module inserts a NON-FROZEN post-pass between claim extraction and graph
build: it derives a small RUN-LOCAL controlled vocabulary from the *raw* method
/ task strings of the "other" claims (one LLM call per run) and re-maps ONLY
those "other" claims onto it, so claims about the same actual method/task
cluster together. Claims the global vocab already canonicalized well are left
untouched.

Conservative + fail-open by design:
* never overwrites a claim whose canonical value is already a real (non-"other")
  bucket;
* only acts when there are at least ``_MIN_OTHER`` "other" claims (below that the
  global vocab is doing fine and an LLM call isn't worth it);
* any error / missing API key / unparseable response => claims returned
  unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .llm_client import (
    build_openai_client,
    call_llm_text,
    configured_api_key,
    configured_model,
)
from .models import Claim

__all__ = ["canonicalize_claims", "derive_local_taxonomy", "apply_local_canonicalization"]

_MIN_OTHER = 3
_MAX_RAW_TERMS = 80  # bound the LLM payload


def _is_other(value: Any) -> bool:
    return str(value or "").strip().lower() in {"", "other"}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _enabled() -> bool:
    if os.environ.get("AIGRAPH_LOCAL_TAXONOMY", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(configured_api_key())


_SYSTEM = (
    "You normalize free-text research method and task names into a SMALL "
    "run-local controlled vocabulary, so that claims about the same method (or "
    "the same task) share one label. You are given the raw method names and raw "
    "task names that an upstream global taxonomy could not categorize (it "
    "labeled them 'other'). Cluster near-synonyms together and map each raw name "
    "to a SHORT canonical bucket label (2-4 words, lowercase, hyphen/space "
    "separated). The bucket label MUST be derived from the provided names — do "
    "not invent unrelated categories. If a raw name is too vague to bucket "
    "meaningfully, map it to 'other'. Return STRICT JSON ONLY: "
    '{"method_remap": {"<raw>": "<bucket>", ...}, "task_remap": {"<raw>": "<bucket>", ...}}. '
    "No prose, no markdown."
)


def _distinct_other_raw(claims: list[Claim], raw_attr: str, canon_attr: str) -> list[str]:
    seen: dict[str, str] = {}
    for c in claims:
        if not _is_other(getattr(c, canon_attr, None)):
            continue
        raw = getattr(c, raw_attr, None)
        key = _norm(raw)
        if key and key not in seen:
            seen[key] = str(raw).strip()
    return list(seen.values())[:_MAX_RAW_TERMS]


def _parse_remaps(raw: str) -> tuple[dict[str, str], dict[str, str]]:
    text = (raw or "").strip()
    if not text.startswith("{"):
        lo, hi = text.find("{"), text.rfind("}")
        if lo == -1 or hi == -1:
            return {}, {}
        text = text[lo : hi + 1]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {}, {}
    if not isinstance(data, dict):
        return {}, {}

    def _clean(d: Any) -> dict[str, str]:
        out: dict[str, str] = {}
        if isinstance(d, dict):
            for k, v in d.items():
                bucket = _norm(v)
                if k and bucket and bucket != "other":
                    out[_norm(k)] = bucket  # key normalized for matching
        return out

    return _clean(data.get("method_remap")), _clean(data.get("task_remap"))


def derive_local_taxonomy(
    claims: list[Claim],
    *,
    client: Any | None = None,
    model: str | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(method_remap, task_remap)`` keyed by NORMALIZED raw strings.

    Empty dicts mean "no remapping" (disabled, too few "other" claims, no raw
    strings to cluster, or an LLM failure).
    """
    if not claims or not _enabled():
        return {}, {}
    n_other = sum(
        1 for c in claims
        if _is_other(getattr(c, "canonical_method", None)) or _is_other(getattr(c, "canonical_task", None))
    )
    if n_other < _MIN_OTHER:
        return {}, {}
    methods = _distinct_other_raw(claims, "method", "canonical_method")
    tasks = _distinct_other_raw(claims, "task", "canonical_task")
    if not methods and not tasks:
        return {}, {}
    try:
        client = client or build_openai_client()
        raw = call_llm_text(
            client,
            model=model or configured_model(),
            system=_SYSTEM,
            user=json.dumps({"methods": methods, "tasks": tasks}, ensure_ascii=False),
            temperature=0.0,
            max_tokens=900,
        )
    except Exception:
        return {}, {}
    return _parse_remaps(raw)


def apply_local_canonicalization(
    claims: list[Claim],
    method_remap: dict[str, str],
    task_remap: dict[str, str],
) -> list[Claim]:
    """Re-map ONLY "other" claims onto the run-local buckets. Never overwrites a
    claim that already has a real (non-"other") canonical value. Returns a new
    list (changed claims are copies)."""
    if not method_remap and not task_remap:
        return claims
    out: list[Claim] = []
    for c in claims:
        update: dict[str, str] = {}
        if method_remap and _is_other(getattr(c, "canonical_method", None)):
            bucket = method_remap.get(_norm(getattr(c, "method", None)))
            if bucket:
                update["canonical_method"] = bucket
        if task_remap and _is_other(getattr(c, "canonical_task", None)):
            bucket = task_remap.get(_norm(getattr(c, "task", None)))
            if bucket:
                update["canonical_task"] = bucket
        out.append(c.model_copy(update=update) if update else c)
    return out


def canonicalize_claims(
    claims: list[Claim],
    *,
    client: Any | None = None,
    model: str | None = None,
) -> list[Claim]:
    """Convenience wrapper: derive the run-local taxonomy and apply it. Fully
    fail-open — returns the input claims unchanged on any problem."""
    try:
        method_remap, task_remap = derive_local_taxonomy(claims, client=client, model=model)
        return apply_local_canonicalization(claims, method_remap, task_remap)
    except Exception:
        return claims
