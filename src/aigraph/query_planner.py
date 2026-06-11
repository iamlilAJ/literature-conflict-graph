"""LLM query planner — the mandatory normalize step in front of the frozen,
rigid LEXICAL query layer (NON-FROZEN).

Why
---
aigraph's topic→hypothesis match is a *lexical* bag-of-words signal
(``aigraph_query._topic_relevance`` over ``_tokenize``). It is fast and free but
*brittle*: a typo, a full natural-language sentence, or a niche phrase whose
tokens are rare in the corpus all under-match and return a "weak"/"none"
coverage banner — even when the corpus DOES cover the topic under its canonical
vocabulary. Measured on run ``arxiv-reasoning-v0.7-540p-thaw1``:

    "graph of thoguhts"                         -> weak    6/78  (typo)
    "graph of thoughts"                         -> weak    6/78  (spelling fixed, still rare tokens)
    "graph-of-thoughts prompting LLM reasoning" -> strong 44/78  (canonical term + domain anchors)

So fixing spelling alone does nothing; adding the *canonical field term* plus
2-3 *domain-anchor words the literature actually uses* is the lever (6 -> 44).

This module wraps that rigid joint with one **mandatory LLM call** that rewrites
whatever the caller typed into a lexical query optimised for the matcher:
canonical hyphenated method names + domain anchors + distinctive content words,
filler dropped. It also emits *broader* fallback queries to try if the primary
under-matches, and an ``is_niche`` flag signalling the generic corpus probably
will not cover the topic (so the caller can escalate to a topic-specific build).

Design guarantees
-----------------
* **Fail-open.** No API key / gate off / model error / unparseable reply → a
  pass-through plan (``applied=False``) carrying a whitespace-collapsed copy of
  the raw topic, so the caller behaves exactly as it would have unplanned. The
  planner can only *improve* a query, never break it.
* **Never worse than baseline.** The raw topic is always retained as a final
  fallback candidate, so a bad rewrite cannot lower coverage below the
  unplanned query.
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

__all__ = ["plan_query", "planner_enabled", "candidate_queries"]

_SYSTEM = (
    "You rewrite a user's research topic into a compact LEXICAL search query for "
    "a bag-of-words matcher that scores hypotheses by literal word overlap (no "
    "embeddings, no synonyms). The corpus indexes ML/AI paper claims. Your job: "
    "maximise overlap with how the literature phrases this topic.\n"
    "Rules for the primary `query`:\n"
    "  1. Fix spelling and expand the CANONICAL field term, hyphenated as the "
    "literature writes it: graph-of-thoughts, tree-of-thoughts, chain-of-thought, "
    "mixture-of-experts, retrieval-augmented generation, MCTS, RLHF, in-context "
    "learning. Spelling fixes ALONE are not enough.\n"
    "  2. Add 2-3 DOMAIN-ANCHOR words the corpus actually uses (reasoning, LLM, "
    "prompting, math, retrieval, evaluation, benchmark, agent, fine-tuning, "
    "alignment). Anchors pull in the relevant cluster; the canonical term focuses "
    "it.\n"
    "  3. Keep 3-6 distinctive content words, space-separated. Drop filler / "
    "stopwords / verbs (a, the, for, of, how, using, based, towards, improve).\n"
    "  4. Output keywords, NOT a sentence or a question.\n"
    "Also return `alternates`: 1-2 BROADER fallback queries (more domain anchors, "
    "drop the single most niche term) to try if the primary under-matches.\n"
    "Set `is_niche`=true when the topic is narrow/emerging and a broad pre-built "
    "corpus likely will not cover it (so the caller should build a topic-specific "
    "corpus instead of querying a generic one).\n"
    "Return STRICT JSON ONLY, no prose, no markdown:\n"
    '{"query": "...", "canonical_terms": ["..."], "domain_anchors": ["..."], '
    '"alternates": ["...", "..."], "is_niche": false}'
)


def planner_enabled() -> bool:
    """Planner runs only when enabled (default on) AND an API key exists.

    Mirrors :func:`semantic_gate.gate_enabled` — without a key (tests, offline)
    the planner is a strict no-op pass-through so deterministic callers are
    unchanged.
    """
    if os.environ.get("AIGRAPH_QUERY_PLANNER", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool(configured_api_key())


def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def _passthrough(raw: str, reason: str) -> dict[str, Any]:
    return {
        "applied": False,
        "reason": reason,
        "raw": _clean(raw),
        "query": _clean(raw),
        "canonical_terms": [],
        "domain_anchors": [],
        "alternates": [],
        "is_niche": False,
    }


def _as_str_list(value: Any, *, cap: int = 4) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        s = _clean(item if isinstance(item, str) else "")
        if s and s.lower() not in {x.lower() for x in out}:
            out.append(s)
        if len(out) >= cap:
            break
    return out


def _parse_plan(raw_reply: str, raw_topic: str) -> dict[str, Any] | None:
    text = (raw_reply or "").strip()
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
    query = _clean(data.get("query") or "")
    if not query:
        return None
    return {
        "applied": True,
        "reason": "planned",
        "raw": _clean(raw_topic),
        "query": query,
        "canonical_terms": _as_str_list(data.get("canonical_terms")),
        "domain_anchors": _as_str_list(data.get("domain_anchors")),
        "alternates": _as_str_list(data.get("alternates"), cap=2),
        "is_niche": bool(data.get("is_niche")),
    }


def plan_query(
    raw_topic: str,
    *,
    client: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Rewrite *raw_topic* into a lexical-matcher-optimised query plan.

    Returns a dict with keys: ``applied`` (bool — False = fail-open pass-through),
    ``query`` (primary), ``alternates`` (broader fallbacks), ``canonical_terms``,
    ``domain_anchors``, ``is_niche``, ``raw``. Never raises — on any failure it
    returns a pass-through plan carrying the cleaned raw topic.
    """
    raw_topic = str(raw_topic or "")
    if not _clean(raw_topic):
        return _passthrough(raw_topic, "empty")
    if not planner_enabled():
        return _passthrough(raw_topic, "disabled")
    try:
        client = client or build_openai_client()
    except Exception:
        return _passthrough(raw_topic, "no_client")
    model = model or configured_model()
    try:
        reply = call_llm_text(
            client,
            model=model,
            system=_SYSTEM,
            user=json.dumps({"topic": _clean(raw_topic)}, ensure_ascii=False),
            temperature=0.0,
            max_tokens=300,
        )
    except Exception:
        return _passthrough(raw_topic, "llm_error")
    plan = _parse_plan(reply, raw_topic)
    if plan is None:
        return _passthrough(raw_topic, "unparseable")
    return plan


def candidate_queries(plan: dict[str, Any]) -> list[str]:
    """Ordered, de-duplicated query candidates to probe: primary, then broader
    alternates, then the raw topic as a guaranteed never-worse-than-baseline
    backstop. Case-insensitive de-dup, order preserved.
    """
    ordered = [plan.get("query", "")]
    ordered.extend(plan.get("alternates", []) or [])
    ordered.append(plan.get("raw", ""))
    seen: set[str] = set()
    out: list[str] = []
    for q in ordered:
        q = _clean(q)
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            out.append(q)
    return out
