"""Numeric-grounding gate (NON-FROZEN, delivery-time).

The §7-thawed forward generators — and the enricher rewrite on top of them —
sometimes assert specific numeric results about PAST methods/baselines that are
not in the cited claims (an adversarial audit measured ~7% of critic and ~40% of
creator items pre-fix; the anti-fabrication prompt fix cut creator to ~16%). The
residual is a *verification* problem: catch the invented/misattributed numbers
that prompting alone cannot.

This module is the verification layer. Same architecture as
``hypothesis_enricher``: one LLM call per hypothesis, cached to a sidecar
(``numeric_flags.jsonl``), a 0-LLM overlay (``apply_flags``) at delivery that
stamps ``Hypothesis.numeric_flags``, and fail-open everywhere (no key / error /
unparseable → no flags, deliver as-is). It checks the FINAL delivered text — the
enriched statement/mechanism/predictions/minimal_test when an enrichment sidecar
exists, else the raw hypothesis — against the hypothesis's cited claims, and
lists every past-method/baseline number that is absent or misattributed.

Targets/predictions about the NEW proposed method and swept design values are
NOT flagged — only fabricated PAST results.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from .io import read_jsonl
from .llm_client import (
    build_openai_client,
    call_llm_text,
    configured_api_key,
    configured_model,
)
from .models import Anomaly, Claim, Hypothesis

__all__ = [
    "gate_enabled",
    "gate_run",
    "load_flags",
    "apply_flags",
    "FLAGS_FILENAME",
]

FLAGS_FILENAME = "numeric_flags.jsonl"
ENRICH_FILENAME = "hypotheses_enriched.jsonl"
_MAX_EVIDENCE = 10
_CLAIM_CHARS = 320
_DEFAULT_LIMIT = 48
_DEFAULT_MAX_TOKENS = 3000


def _max_tokens() -> int:
    try:
        return max(700, int(os.environ.get("AIGRAPH_GATE_NUMERIC_MAX_TOKENS", _DEFAULT_MAX_TOKENS)))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_TOKENS


_SYSTEM = (
    "You verify the NUMERIC grounding of a research hypothesis. You are given the "
    "hypothesis text and the real paper claims it cites (with their exact text). "
    "List every numeric statement the hypothesis makes about a PAST or EXISTING "
    "method, baseline, or result (e.g. \"SPARC-RAG's 52.2% cost reduction\", "
    "\"BlendFilter's 0.420 EM\", \"closes the 13.79% gap\") that is NOT supported "
    "by the cited claims — the number is either ABSENT from every cited claim, or "
    "MISATTRIBUTED (the value exists in a claim but for a different method, "
    "setting, or metric).\n"
    "ONLY flag a number presented as an ACHIEVED RESULT of a named PAST method — "
    "an accuracy, F1, EM, recall, %-improvement, or cost-reduction attributed to "
    "an existing method/baseline. Do NOT flag, even when specific: (a) "
    "TARGETS/predictions about the NEW proposed method (\"will exceed 0.45 EM\", "
    "\"expected +5%\"); (b) hyperparameters, thresholds, or design settings "
    "(\"confidence threshold 0.3\", \"lambda=0.5\", \"top-k from 1 to 30\", "
    "\"16 chains\"); (c) numbers that genuinely appear in a cited claim for that "
    "method/setting. Be precise and conservative — when unsure whether a number is "
    "a past RESULT or a design SETTING, do NOT flag it.\n"
    "Return STRICT JSON ONLY, no prose, no markdown:\n"
    '{"unverified": [{"number": "the value as written", "context": "the method/'
    'baseline it is attached to", "issue": "absent" or "misattributed"}]}\n'
    "Return an empty list if every past-result number checks out."
)


def gate_enabled() -> bool:
    """On by default when an API key exists; AIGRAPH_NUMERIC_GATE=0 disables.
    Mirrors enricher_enabled — without a key (tests/offline) it is a strict
    no-op so deterministic callers are unchanged."""
    if os.environ.get("AIGRAPH_NUMERIC_GATE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(configured_api_key())


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _final_text(hyp: Hypothesis, enriched: Optional[dict]) -> dict[str, Any]:
    """The text actually delivered: the enriched rewrite when present, else the
    raw hypothesis. This is what must be number-checked."""
    if enriched:
        return {
            "statement": _clean(enriched.get("statement") or hyp.hypothesis),
            "mechanism": _clean(enriched.get("mechanism") or hyp.mechanism),
            "predictions": [_clean(p) for p in (enriched.get("predictions") or hyp.predictions or [])],
            "minimal_test": _clean(enriched.get("minimal_test") or hyp.minimal_test),
        }
    return {
        "statement": _clean(hyp.hypothesis),
        "mechanism": _clean(hyp.mechanism),
        "predictions": [_clean(p) for p in (hyp.predictions or [])],
        "minimal_test": _clean(hyp.minimal_test),
    }


def _evidence(hyp: Hypothesis, anomaly: Optional[Anomaly], claims_by_id: dict[str, Claim]) -> list[dict]:
    ids: list[str] = []
    for cid in list(hyp.explains_claims) + (list(anomaly.claim_ids) if anomaly else []):
        if cid and cid not in ids:
            ids.append(cid)
    out = []
    for cid in ids:
        c = claims_by_id.get(cid)
        if not c:
            continue
        out.append({
            "claim_id": cid,
            "method": _clean(c.method or c.canonical_method or ""),
            "dataset": _clean(c.dataset or ""),
            "text": _clean(c.claim_text)[:_CLAIM_CHARS],
        })
        if len(out) >= _MAX_EVIDENCE:
            break
    return out


def _parse(raw: str) -> Optional[list[dict]]:
    text = (raw or "").strip()
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
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    items = data.get("unverified")
    if not isinstance(items, list):
        return []
    flags: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        num = _clean(it.get("number"))
        if not num:
            continue
        flags.append({
            "number": num[:60],
            "context": _clean(it.get("context"))[:120],
            "issue": (it.get("issue") if it.get("issue") in {"absent", "misattributed"} else "absent"),
        })
    return flags


def gate_one(hyp: Hypothesis, anomaly: Optional[Anomaly], enriched: Optional[dict],
             evidence: list[dict], *, client: Any, model: str) -> Optional[dict]:
    """Verify one hypothesis. Returns a record {hypothesis_id, flags:[...]} or
    None on failure (fail-open). A clean hyp returns {..., flags: []}."""
    if not evidence:
        return None
    payload = {"hypothesis": _final_text(hyp, enriched), "cited_claims": evidence}
    try:
        raw = call_llm_text(client, model=model, system=_SYSTEM,
                            user=json.dumps(payload, ensure_ascii=False),
                            temperature=0.0, max_tokens=_max_tokens())
    except Exception:
        return None
    flags = _parse(raw)
    if flags is None:
        return None
    return {"hypothesis_id": hyp.hypothesis_id, "flags": flags}


def gate_run(
    run_dir: Path | str,
    *,
    hyp_file: str = "hypotheses.jsonl",
    force: bool = False,
    only_types: Optional[Iterable[str]] = None,
    limit: int = _DEFAULT_LIMIT,
    client: Any | None = None,
    model: str | None = None,
) -> dict[str, dict]:
    """Number-check a run's hypotheses against their cited claims; persist the
    flags sidecar. Checks the ENRICHED text when an enrichment sidecar exists.
    Returns ``{hypothesis_id: {flags: [...]}}`` for everything cached. No-op
    returning the existing cache when disabled / no key."""
    run_dir = Path(run_dir)
    existing = load_flags(run_dir)
    if not gate_enabled():
        return existing
    hpath = run_dir / hyp_file
    if not hpath.exists():
        return existing
    hyps: list[Hypothesis] = read_jsonl(hpath, Hypothesis)
    claims = read_jsonl(run_dir / "claims.jsonl", Claim) if (run_dir / "claims.jsonl").exists() else []
    anoms = read_jsonl(run_dir / "anomalies.jsonl", Anomaly) if (run_dir / "anomalies.jsonl").exists() else []
    claims_by_id = {c.claim_id: c for c in claims}
    anom_by_id = {a.anomaly_id: a for a in anoms}
    enriched_side = _load_enriched(run_dir)
    type_filter = set(only_types) if only_types else None

    try:
        client = client or build_openai_client()
    except Exception:
        return existing
    model = model or configured_model()

    out = dict(existing)
    made = 0
    for hyp in hyps:
        if made >= max(0, int(limit)):
            break
        if not force and hyp.hypothesis_id in out:
            continue
        anomaly = anom_by_id.get(hyp.anomaly_id)
        if type_filter is not None and getattr(anomaly, "type", None) not in type_filter:
            continue
        evidence = _evidence(hyp, anomaly, claims_by_id)
        rec = gate_one(hyp, anomaly, enriched_side.get(hyp.hypothesis_id),
                       evidence, client=client, model=model)
        if rec is not None:
            out[hyp.hypothesis_id] = {"flags": rec["flags"]}
            made += 1

    if made:
        _write_sidecar(run_dir, out)
    return out


def _load_enriched(run_dir: Path) -> dict[str, dict]:
    path = run_dir / ENRICH_FILENAME
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            hid = rec.get("hypothesis_id")
            if hid:
                out[hid] = rec
    except Exception:
        return out
    return out


def _write_sidecar(run_dir: Path, flags: dict[str, dict]) -> None:
    try:
        tmp = run_dir / (FLAGS_FILENAME + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for hid, rec in flags.items():
                f.write(json.dumps({"hypothesis_id": hid, **rec}, ensure_ascii=False) + "\n")
        tmp.replace(run_dir / FLAGS_FILENAME)
    except Exception:
        pass


def load_flags(run_dir: Path | str) -> dict[str, dict]:
    """Read the flags sidecar (0-LLM). Returns ``{hypothesis_id: {flags:[...]}}``."""
    path = Path(run_dir) / FLAGS_FILENAME
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            hid = rec.get("hypothesis_id")
            if hid:
                out[hid] = {"flags": rec.get("flags", [])}
    except Exception:
        return out
    return out


def apply_flags(selected: list[Hypothesis], run_dir: Path | str) -> int:
    """Overlay cached numeric flags onto the selected Hypothesis objects in place
    (0-LLM). Stamps ``hyp.numeric_flags`` (list of short strings) on any
    hypothesis with at least one unverified number. Returns the count flagged.
    No-op when the sidecar is absent (fail-open)."""
    flags = load_flags(run_dir)
    if not flags:
        return 0
    n = 0
    for hyp in selected:
        rec = flags.get(hyp.hypothesis_id)
        items = (rec or {}).get("flags") or []
        if not items:
            continue
        labels = []
        for it in items:
            num = it.get("number", "")
            ctx = it.get("context", "")
            issue = it.get("issue", "absent")
            labels.append(f"{num} ({ctx}) — {issue}" if ctx else f"{num} — {issue}")
        try:
            hyp.numeric_flags = labels
            n += 1
        except Exception:
            pass
    return n
