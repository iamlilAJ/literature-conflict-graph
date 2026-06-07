"""LLM-backed semantic relevance gate for retrieved corpora (NON-FROZEN).

aigraph's frozen per-source selection (``paper_select.py``) ranks and filters
papers by a *lexical* token-overlap relevance signal (``title_relevance_score``
+ ``_has_core_overlap`` + ``_second_stage_topic_cleanup``). On broad / ``union``
retrieval that lets off-topic papers which merely share generic tokens with the
query (e.g. "policy", "distillation", "memory") leak into the corpus — the
precision failure observed on topics like "memory based on-policy distillation",
where the corpus bridged medicine / energy / chemical-engineering / HAR.

This module adds an optional *semantic* pass that scores each paper's
on-topic-ness against the **human topic** using the same OpenAI-compatible
endpoint the rest of the pipeline already uses (``llm_client``). The server
layer (``server.py`` ``_retrieve_corpus`` — non-frozen) calls
:func:`apply_semantic_gate` after union+dedup to (a) drop the off-topic tail and
(b) re-order survivors by semantic relevance — delivering both the "semantic
gate" and the "semantic ranking" upgrade **without touching any frozen module**.

Design guarantees:

* **Fail-open.** If the LLM is unavailable, no API key is configured, the gate
  is disabled, or the model returns garbage, scoring yields no verdict and the
  caller keeps the corpus unchanged. The gate can only *raise* precision, never
  drop recall below what lexical retrieval already produced.
* **Per-paper conservative.** A paper is removed only if the model
  *affirmatively* judged it off-topic. Papers the model failed to score are
  kept.
* **Recall floor preserved.** If gating would leave fewer than ``keep_floor``
  papers, the highest-scored dropped papers are topped back up to the floor.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterable

from .llm_client import (
    build_openai_client,
    call_llm_text,
    configured_api_key,
    configured_model,
)
from .models import Paper

__all__ = ["apply_semantic_gate", "score_relevance", "gate_enabled"]

# 0 = off-topic, 1 = tangential, 2 = clearly related, 3 = directly on topic.
_MAX_SCORE = 3
_DEFAULT_MIN_SCORE = 2
_DEFAULT_BATCH = 20
_ABSTRACT_CHARS = 500

_SYSTEM = (
    "You are a precise research-librarian relevance grader. You are given a "
    "research TOPIC and a list of candidate papers (id, title, abstract). Rate "
    "how directly each paper is ABOUT the topic on this 0-3 scale:\n"
    "  3 = directly about the topic (same problem AND method/task family).\n"
    "  2 = clearly related (shares the core problem or the core method, applies "
    "it, or studies a close variant in the same area).\n"
    "  1 = tangential (shares only a generic technique or a buzzword but tackles "
    "a different problem or a different domain).\n"
    "  0 = off-topic / unrelated.\n"
    "Be STRICT: a paper that merely reuses a generic ML technique (distillation, "
    "attention, RL, embeddings) but addresses an unrelated problem or a different "
    "domain (medicine, energy, materials, remote sensing, vision-only, ...) is 0 "
    "or 1, NOT 2. Judge by the actual research problem, not shared vocabulary.\n"
    "Return STRICT JSON ONLY: an object mapping each paper id (string) to its "
    "integer score, e.g. {\"0\": 3, \"1\": 0, \"2\": 2}. No prose, no markdown."
)


def gate_enabled() -> bool:
    """The gate runs only when explicitly enabled (default on) AND a key exists.

    Without an API key (tests, mock/offline runs) the gate is a strict no-op so
    deterministic corpora are unchanged.
    """
    if os.environ.get("AIGRAPH_SEMANTIC_GATE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(configured_api_key())


def _min_score() -> int:
    try:
        return max(1, min(_MAX_SCORE, int(os.environ.get("AIGRAPH_SEMANTIC_GATE_MIN", _DEFAULT_MIN_SCORE))))
    except (TypeError, ValueError):
        return _DEFAULT_MIN_SCORE


def _batch_size() -> int:
    try:
        return max(1, int(os.environ.get("AIGRAPH_SEMANTIC_GATE_BATCH", _DEFAULT_BATCH)))
    except (TypeError, ValueError):
        return _DEFAULT_BATCH


def _coerce_score(value: Any) -> int | None:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(_MAX_SCORE, score))


def _parse_scores(raw: str, valid_ids: set[str]) -> dict[str, int]:
    """Defensively parse an id->score mapping out of the model's reply.

    Accepts ``{"0": 3}``, ``{"scores": {...}}``, ``{"papers": [{"id","score"}]}``
    or a bare list ``[{"id","score"}]``. Unknown ids are ignored.
    """
    text = (raw or "").strip()
    if not text:
        return {}
    # tolerate ```json fences / leading prose by slicing to the outermost braces
    if not text.lstrip().startswith(("{", "[")):
        lo = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
        if lo == -1:
            return {}
        text = text[lo:]
        hi = max(text.rfind("}"), text.rfind("]"))
        if hi != -1:
            text = text[: hi + 1]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {}

    out: dict[str, int] = {}

    def _ingest_pair(key: Any, value: Any) -> None:
        pid = str(key)
        if pid not in valid_ids:
            return
        score = _coerce_score(value)
        if score is not None:
            out[pid] = score

    if isinstance(data, dict):
        inner = data.get("scores") if isinstance(data.get("scores"), dict) else None
        if inner is not None:
            for k, v in inner.items():
                _ingest_pair(k, v)
        elif isinstance(data.get("papers"), list):
            for item in data["papers"]:
                if isinstance(item, dict):
                    _ingest_pair(item.get("id"), item.get("score"))
        else:
            for k, v in data.items():
                if isinstance(v, dict):  # {"0": {"score": 3}}
                    _ingest_pair(k, v.get("score"))
                else:
                    _ingest_pair(k, v)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                _ingest_pair(item.get("id"), item.get("score"))
    return out


def _paper_payload(idx: int, paper: Paper) -> dict[str, Any]:
    return {
        "id": str(idx),
        "title": (paper.title or "").strip()[:300],
        "abstract": (paper.abstract or "").strip()[:_ABSTRACT_CHARS],
    }


def score_relevance(
    topic: str,
    papers: list[Paper],
    *,
    client: Any | None = None,
    model: str | None = None,
    batch_size: int | None = None,
) -> dict[str, int]:
    """Return ``{paper_id: 0..3}`` semantic-relevance scores for *papers*.

    Papers the model declines to score (or whole batches that error) are simply
    absent from the returned dict — the caller treats absence as "keep".
    Returns ``{}`` when the gate is disabled or no API key is configured.
    """
    if not papers or not gate_enabled():
        return {}
    topic = " ".join(str(topic or "").split())
    if not topic:
        return {}
    try:
        client = client or build_openai_client()
    except Exception:
        return {}
    model = model or configured_model()
    batch = batch_size or _batch_size()

    scores: dict[str, int] = {}
    for start in range(0, len(papers), batch):
        chunk = papers[start : start + batch]
        # local idx (0..len-1 within the chunk) -> real paper_id, keeps the
        # prompt compact and avoids putting arbitrary id strings in JSON keys.
        idx_to_pid = {str(i): chunk[i].paper_id for i in range(len(chunk))}
        valid = set(idx_to_pid)
        payload = {"topic": topic, "papers": [_paper_payload(i, p) for i, p in enumerate(chunk)]}
        try:
            raw = call_llm_text(
                client,
                model=model,
                system=_SYSTEM,
                user=json.dumps(payload, ensure_ascii=False),
                temperature=0.0,
                max_tokens=400,
            )
        except Exception:
            continue  # fail-open for this batch
        for local_id, score in _parse_scores(raw, valid).items():
            pid = idx_to_pid.get(local_id)
            if pid is not None:
                scores[pid] = score
    return scores


def apply_semantic_gate(
    topic: str,
    papers: list[Paper],
    *,
    keep_floor: int = 1,
    min_score: int | None = None,
    score_fn: Callable[..., dict[str, int]] | None = None,
    client: Any | None = None,
    model: str | None = None,
) -> tuple[list[Paper], dict[str, Any]]:
    """Drop off-topic papers and re-order survivors by semantic relevance.

    Parameters
    ----------
    topic:
        The human topic string (richest description of intent).
    papers:
        The deduped union corpus.
    keep_floor:
        Never return fewer than this many papers (recall guard); if gating would
        undershoot, the highest-scored dropped papers are added back.
    min_score:
        Minimum score (default :data:`_DEFAULT_MIN_SCORE`, env-overridable) a
        *scored* paper must reach to survive. Unscored papers always survive.
    score_fn:
        Injection point for tests; defaults to :func:`score_relevance`.

    Returns ``(kept_papers, audit)``. ``audit["applied"]`` is ``False`` when the
    gate was a no-op (disabled / empty / no scores) and *papers* is returned
    unchanged.
    """
    if not papers:
        return papers, {"applied": False, "reason": "empty"}
    scorer = score_fn or (score_relevance if gate_enabled() else None)
    if scorer is None:
        return papers, {"applied": False, "reason": "disabled"}

    threshold = _min_score() if min_score is None else max(1, min(_MAX_SCORE, int(min_score)))
    try:
        scores = scorer(topic, papers, client=client, model=model)
    except Exception:
        return papers, {"applied": False, "reason": "scorer_error"}
    if not scores:
        return papers, {"applied": False, "reason": "no_scores"}

    kept: list[tuple[float, int, Paper]] = []
    dropped: list[tuple[int, int, Paper]] = []  # (raw_score, original_idx, paper)
    histogram = {str(s): 0 for s in range(_MAX_SCORE + 1)}
    unscored = 0
    for idx, paper in enumerate(papers):
        raw = scores.get(paper.paper_id)
        if raw is None:
            unscored += 1
            # keep, but order just below explicitly on-topic papers
            kept.append((threshold - 0.5, idx, paper))
            continue
        histogram[str(raw)] += 1
        if raw >= threshold:
            kept.append((float(raw), idx, paper))
        else:
            dropped.append((raw, idx, paper))

    topped_up = 0
    if len(kept) < keep_floor and dropped:
        # add back the strongest dropped papers to honor the recall floor
        dropped.sort(key=lambda t: (-t[0], t[1]))
        need = keep_floor - len(kept)
        for raw, idx, paper in dropped[:need]:
            kept.append((float(raw), idx, paper))
            topped_up += 1

    # strongest-first, stable on original order for ties
    kept.sort(key=lambda t: (-t[0], t[1]))
    ordered = [paper for _score, _idx, paper in kept]

    audit = {
        "applied": True,
        "min_score": threshold,
        "before": len(papers),
        "kept": len(ordered),
        "dropped": len(papers) - len(ordered),
        "unscored_kept": unscored,
        "topped_up": topped_up,
        "score_histogram": histogram,
    }
    return ordered, audit
