"""LLM hypothesis enricher — grounds the frozen templated hypotheses in their
real evidence claims (NON-FROZEN).

Why
---
The frozen ``hypotheses.py`` back-explanation contract emits a *generic
template* for every anomaly — the recurring "An unreported moderator variable
drives the conflicting results … a confound in data preprocessing, prompt
formatting, or decoding parameters" boilerplate. That text is identical across
anomalies and throws away the real signal the pipeline already attached: the
actual evidence claims (paper, finding, stance, method, dataset). Measured on a
real ``impact_conflict`` anomaly (SCoT vs CoT on code generation), the template
said "a confound …"; an LLM reading the *real claims* instead produced a
specific, grounded, falsifiable hypothesis ("SCoT beats CoT by 13.79% Pass@1 but
the advantage is HumanEval-specific; on MBPP/CodeContests the rigid structure may
not help …" with concrete predictions and a minimal test).

This module wraps that rigid "anomaly → templated hypothesis" seam with ONE
LLM call per hypothesis that reads the anomaly's real evidence claims and
rewrites the ``mechanism`` / ``predictions`` / ``minimal_test`` / statement into
something specific to *those* papers. Same pattern as ``query_planner`` /
``semantic_gate``: a mandatory LLM step at a frozen joint, **without touching any
frozen module** — the templated hypothesis stays in ``hypotheses.jsonl``; the
enriched view is a cached ``hypotheses_enriched.jsonl`` sidecar that the renderer
overlays at delivery time (0-LLM).

Design guarantees
-----------------
* **Fail-open.** No key / gate off / LLM error / unparseable reply → no
  enrichment for that hypothesis; the caller keeps the original template. The
  enricher can only *improve* a hypothesis, never break one.
* **Cached.** Enrichment runs once per run and is persisted to the sidecar;
  ``apply_enrichment`` (the query-time overlay) is 0-LLM and reads the sidecar.
* **Frozen-safe.** Reads ``hypotheses.jsonl`` + ``claims.jsonl`` +
  ``anomalies.jsonl`` (frozen-pipeline outputs) and writes only the non-frozen
  sidecar; the frozen hypothesis contract is untouched.
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
from .models import Anomaly, Claim, Hypothesis, Paper

__all__ = [
    "enricher_enabled",
    "enrich_run",
    "load_enriched",
    "apply_enrichment",
    "ENRICH_FILENAME",
]

ENRICH_FILENAME = "hypotheses_enriched.jsonl"
_MAX_EVIDENCE = 8
_CLAIM_CHARS = 280
_DEFAULT_LIMIT = 24
# Output-token budget. Thinking models (e.g. Kimi-K2.6) spend most of the budget
# on reasoning_content before emitting the JSON, so a small cap returns EMPTY
# content (finish_reason=length). 4000 leaves room for ~1-2k reasoning tokens +
# the JSON; env-tunable for slower/cheaper models. Harmless on non-thinking
# models (they stop early).
_DEFAULT_MAX_TOKENS = 4000


def _max_tokens() -> int:
    try:
        return max(700, int(os.environ.get("AIGRAPH_ENRICH_MAX_TOKENS", _DEFAULT_MAX_TOKENS)))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_TOKENS

_SYSTEM = (
    "You are a research scientist turning a detected cross-paper anomaly and its "
    "REAL evidence claims (each: paper, finding, stance, method, dataset) into a "
    "FORWARD, ACTIONABLE research item. You are told which REGISTER to write in. "
    "Do NOT write an interrogative critique of the literature ('why do papers "
    "disagree', 'is there an unreported moderator variable', 'could X and Y be "
    "connected') — that framing is the MOTIVATION, not the deliverable. Write a "
    "DECLARATIVE research item in the assigned register, grounded in the actual "
    "methods/datasets/findings shown (name them explicitly, do not be generic). "
    "Return STRICT JSON ONLY, no prose, no markdown:\n"
    '{"statement": "the declarative research item written IN THE ASSIGNED '
    'REGISTER, naming the real methods/datasets — NOT a question", '
    '"motivation": "one sentence: the conflict/gap that makes this non-obvious, '
    'grounded in the evidence", '
    '"mechanism": "how it works / why it would resolve or exploit the gap, in '
    'the named methods/conditions", '
    '"predictions": ["a falsifiable prediction with a benchmark/number", "..."], '
    '"minimal_test": "a concrete experiment using the actual benchmarks/methods '
    'named in the claims"}'
)

# Forward, declarative registers — the framing each enriched hypothesis is cast
# in. Rotating these across a run kills the monotone "why do these papers
# disagree / is there a moderator variable" critic signature (external feedback).
_REGISTERS = {
    "proposal":   "PROPOSAL — propose a concrete NEW METHOD or approach; lead with what you build and how it works.",
    "benchmark":  "BENCHMARK — propose a controlled evaluation/benchmark that isolates the disputed variable and would settle the question.",
    "refutation": "REFUTATION — propose a falsification study: directly test whether the claimed effect actually holds under controlled conditions, or a counter-example probe.",
    "mechanism":  "MECHANISM PROBE — propose an intervention/ablation that isolates WHY the effect occurs or fails.",
    "synthesis":  "SYNTHESIS — propose a unifying method/framework connecting the two lines of work into one approach, and a task where the unification pays off.",
    "transfer":   "TRANSFER — propose transferring a technique from one setting to the other, with the concrete adaptation and where it should win.",
}

# Per anomaly-type pool of FITTING registers, rotated within type so even a
# single-type corpus (e.g. all community_disconnect) gets varied framings.
_TYPE_POOLS = {
    "impact_conflict":              ["refutation", "benchmark", "mechanism"],
    "benchmark_inconsistency":      ["benchmark", "refutation"],
    "metric_mismatch":              ["benchmark", "mechanism"],
    "setting_mismatch":             ["benchmark", "mechanism"],
    "replication_conflict":         ["refutation", "benchmark"],
    "evidence_gap":                 ["proposal", "benchmark", "mechanism"],
    "community_disconnect":         ["synthesis", "transfer", "proposal"],
    "bridge_opportunity":           ["synthesis", "transfer"],
    "bottleneck_open_q_alignment":  ["proposal", "mechanism"],
}
_DEFAULT_POOL = ["proposal", "benchmark", "mechanism"]


def _register_for(anomaly_type: str | None, counter: dict[str, int]) -> str:
    """Assign a register by anomaly type, rotating within the type's pool so the
    same type yields varied framings across a run."""
    pool = _TYPE_POOLS.get(anomaly_type or "", _DEFAULT_POOL)
    i = counter.get(anomaly_type or "", 0)
    counter[anomaly_type or ""] = i + 1
    return pool[i % len(pool)]


def enricher_enabled() -> bool:
    """Enricher runs only when enabled (default on) AND an API key exists.

    Mirrors :func:`query_planner.planner_enabled` / :func:`semantic_gate.gate_enabled`
    — without a key (tests, offline) it is a strict no-op so deterministic
    callers are unchanged.
    """
    if os.environ.get("AIGRAPH_HYP_ENRICH", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(configured_api_key())


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _claim_brief(claim: Claim, paper_lookup: dict[str, Paper]) -> dict[str, Any]:
    paper = paper_lookup.get(claim.paper_id)
    title = (paper.title if paper else "") or claim.paper_id
    return {
        "paper": _clean(title)[:90],
        "finding": _clean(claim.claim_text)[:_CLAIM_CHARS],
        "stance": claim.direction or "?",
        "method": _clean(claim.method or claim.subject_canonical or ""),
        "dataset": _clean(claim.dataset or claim.dataset_canonical or ""),
    }


def _evidence_for(hyp: Hypothesis, anomaly: Optional[Anomaly],
                  claims_by_id: dict[str, Claim], paper_lookup: dict[str, Paper]) -> list[dict]:
    """Collect the real evidence claims behind a hypothesis (its own
    explains_claims, plus the anomaly's claim_ids), deduped and capped."""
    ids: list[str] = []
    for cid in list(hyp.explains_claims) + (list(anomaly.claim_ids) if anomaly else []):
        if cid and cid not in ids:
            ids.append(cid)
    briefs = [_claim_brief(claims_by_id[c], paper_lookup) for c in ids if c in claims_by_id]
    return briefs[:_MAX_EVIDENCE]


def _parse(raw: str) -> Optional[dict[str, Any]]:
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
    statement = _clean(data.get("statement"))
    mechanism = _clean(data.get("mechanism"))
    if not statement and not mechanism:
        return None  # nothing usable -> treat as fail-open
    preds_raw = data.get("predictions") or []
    if isinstance(preds_raw, str):
        preds_raw = [preds_raw]
    predictions = [_clean(p) for p in preds_raw if _clean(p)][:4]
    return {
        "statement": statement,
        "motivation": _clean(data.get("motivation")),
        "mechanism": mechanism,
        "predictions": predictions,
        "minimal_test": _clean(data.get("minimal_test")),
    }


def enrich_one(
    hyp: Hypothesis,
    anomaly: Optional[Anomaly],
    evidence: list[dict],
    *,
    register: str,
    client: Any,
    model: str,
) -> Optional[dict[str, Any]]:
    """Enrich a single hypothesis from its real evidence, cast in ``register``.
    Returns the enriched fields, or ``None`` on any failure (fail-open)."""
    if not evidence:
        return None
    payload = {
        "register": register,
        "register_instruction": _REGISTERS.get(register, ""),
        "anomaly_type": getattr(anomaly, "type", None),
        "central_question": _clean(getattr(anomaly, "central_question", "")),
        "evidence_claims": evidence,
    }
    try:
        raw = call_llm_text(
            client, model=model, system=_SYSTEM,
            user=json.dumps(payload, ensure_ascii=False),
            temperature=0.3, max_tokens=_max_tokens(),
        )
    except Exception:
        return None
    parsed = _parse(raw)
    if parsed is None:
        return None
    parsed["hypothesis_id"] = hyp.hypothesis_id
    parsed["anomaly_type"] = getattr(anomaly, "type", None)
    parsed["register"] = register
    return parsed


def enrich_run(
    run_dir: Path | str,
    *,
    force: bool = False,
    only_types: Optional[Iterable[str]] = None,
    limit: int = _DEFAULT_LIMIT,
    client: Any | None = None,
    model: str | None = None,
) -> dict[str, dict]:
    """Enrich a run's hypotheses and persist to the sidecar.

    Reads ``hypotheses.jsonl`` + ``claims.jsonl`` + ``anomalies.jsonl``, enriches
    each hypothesis not already cached (unless ``force``), merges into
    ``hypotheses_enriched.jsonl``, and returns ``{hypothesis_id: enriched}`` for
    everything in the sidecar. No-op returning the existing cache when the
    enricher is disabled / no key. ``only_types`` restricts to certain anomaly
    types; ``limit`` caps the number of *new* LLM calls this pass."""
    run_dir = Path(run_dir)
    existing = load_enriched(run_dir)
    if not enricher_enabled():
        return existing

    hpath = run_dir / "hypotheses.jsonl"
    if not hpath.exists():
        return existing
    hyps: list[Hypothesis] = read_jsonl(hpath, Hypothesis)
    claims = read_jsonl(run_dir / "claims.jsonl", Claim) if (run_dir / "claims.jsonl").exists() else []
    anoms = read_jsonl(run_dir / "anomalies.jsonl", Anomaly) if (run_dir / "anomalies.jsonl").exists() else []
    papers = read_jsonl(run_dir / "papers.jsonl", Paper) if (run_dir / "papers.jsonl").exists() else []
    claims_by_id = {c.claim_id: c for c in claims}
    anom_by_id = {a.anomaly_id: a for a in anoms}
    paper_lookup = {p.paper_id: p for p in papers}
    type_filter = set(only_types) if only_types else None

    try:
        client = client or build_openai_client()
    except Exception:
        return existing
    model = model or configured_model()

    out = dict(existing)
    made = 0
    reg_counter: dict[str, int] = {}
    for hyp in hyps:
        if made >= max(0, int(limit)):
            break
        if not force and hyp.hypothesis_id in out:
            continue
        anomaly = anom_by_id.get(hyp.anomaly_id)
        atype = getattr(anomaly, "type", None)
        if type_filter is not None and atype not in type_filter:
            continue
        evidence = _evidence_for(hyp, anomaly, claims_by_id, paper_lookup)
        # Assign a forward register by type, rotated, so the run shows varied
        # framings (proposal / benchmark / refutation / …) not a monotone critique.
        register = _register_for(atype, reg_counter)
        enriched = enrich_one(hyp, anomaly, evidence, register=register,
                              client=client, model=model)
        if enriched is not None:
            out[hyp.hypothesis_id] = enriched
            made += 1

    if made:
        _write_sidecar(run_dir, out)
    return out


def _write_sidecar(run_dir: Path, enriched: dict[str, dict]) -> None:
    try:
        tmp = run_dir / (ENRICH_FILENAME + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in enriched.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(run_dir / ENRICH_FILENAME)
    except Exception:
        pass


def load_enriched(run_dir: Path | str) -> dict[str, dict]:
    """Read the enrichment sidecar (0-LLM). Returns ``{hypothesis_id: fields}``
    or ``{}`` if absent / unreadable."""
    path = Path(run_dir) / ENRICH_FILENAME
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


def apply_enrichment(selected: list[Hypothesis], run_dir: Path | str) -> int:
    """Overlay cached enrichment onto the selected Hypothesis objects in place
    (0-LLM). Replaces the templated ``hypothesis`` / ``mechanism`` /
    ``predictions`` / ``minimal_test`` with the grounded versions when present,
    stamping ``enriched`` for provenance. Returns the count overlaid. No-op (0)
    when the sidecar is absent — the templated text is kept (fail-open)."""
    enriched = load_enriched(run_dir)
    if not enriched:
        return 0
    n = 0
    for hyp in selected:
        rec = enriched.get(hyp.hypothesis_id)
        if not rec:
            continue
        if rec.get("statement"):
            hyp.hypothesis = rec["statement"]
        if rec.get("mechanism"):
            hyp.mechanism = rec["mechanism"]
        if rec.get("predictions"):
            hyp.predictions = list(rec["predictions"])
        if rec.get("minimal_test"):
            hyp.minimal_test = rec["minimal_test"]
        try:
            hyp.enriched = {"applied": True, "anomaly_type": rec.get("anomaly_type"),
                            "register": rec.get("register"),
                            "motivation": rec.get("motivation")}
        except Exception:
            pass
        n += 1
    return n
