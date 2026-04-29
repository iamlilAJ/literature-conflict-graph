"""Citation stance classification for Paper→Paper cites edges.

For each (B cites A) edge in the graph, look up B's full text, find the
sentence(s) that mention A's title (substring match against the citing
paper's full text), and ask an LLM to label B's stance toward A as one of:

  extends      — B explicitly builds on A's framework / dataset / method
  contradicts  — B reports a finding opposite to A's claim
  contrasts    — B compares its approach to A's without directly contradicting
  builds_on    — B uses A as foundation (cites for setup) without extending
  mentions     — B references A in passing (related work boilerplate)

If B has no full text, or A's title isn't found in B's text outside the
References section, the edge is left unchanged (no stance attribute). The
system degrades gracefully as the offline corpus grows.

Cost note: a 100-paper corpus typically has 200-500 internal cites edges.
At gpt-5.4-mini-class pricing for a ~200-token call, that's $0.02-$0.05
per run. The ``max_edges`` and ``dry_run`` knobs on
``classify_cites_edges`` are the budget guards.

Idempotency: re-running on a graph that already has stance attributes
skips classified edges by default (``skip_classified=True``). Useful for
incremental corpus growth — only the new cites edges pay for an LLM call.

Persistence note: the ``stance`` / ``stance_confidence`` /
``stance_rationale`` attributes live on the cites edge attributes only.
``realign.py`` was updated to preserve these across rebuilds when
``SCHEMA_VERSION`` is bumped (otherwise a rebuild via
``build_graph(classify_stance=False)`` would silently erase stance work).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from typing import Any

import networkx as nx

from .llm_client import (
    DEFAULT_MAX_TOKENS,
    build_openai_client,
    call_llm_text,
    configured_api_key,
    configured_base_url,
    configured_model,
)
from .llm_extract import _load_json
from .models import Paper


logger = logging.getLogger(__name__)


STANCE_LABELS = frozenset({"extends", "contradicts", "contrasts", "builds_on", "mentions"})


SYSTEM_PROMPT = """You classify how paper B engages with paper A based on the citation context.

You receive: A's title, and the sentence(s) in B that cite A.

Output STRICT JSON, no fences, no prose:
  {"stance": "<one_of: extends | contradicts | contrasts | builds_on | mentions>",
   "confidence": <float 0.0-1.0>,
   "rationale": "<one short sentence>"}

Definitions (pick the strongest applicable label):
- extends:     B explicitly builds on A's framework/method/dataset to do something new
- contradicts: B reports a finding opposite to A's claim
- contrasts:   B compares its approach to A's without directly contradicting
- builds_on:   B uses A as foundation (cites for setup) without extending
- mentions:    B references A in passing (related work boilerplate, no engagement)

If the context is too thin to tell, return "mentions" with confidence <= 0.4.
"""


# A sentence splitter that's good enough for academic prose. Splits on
# `[.!?]` followed by whitespace. Preserves the trailing punctuation on
# the previous sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Section headers that mark the start of a paper's reference list. Matched
# at line boundaries (case-insensitive). The first match terminates the
# scannable text — anything after it is bibliography boilerplate.
_BIBLIOGRAPHY_HEADER_RE = re.compile(
    r"^\s*(references|bibliography|works\s+cited)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_LEADING_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
_WHITESPACE_RUN_RE = re.compile(r"\s+")
_PUNCT_STRIP_RE = re.compile(r"[^\w\s\-]")


def _normalize_title(title: str) -> str:
    """Lowercase, collapse whitespace, strip leading articles, strip non-word
    punctuation. Used both at index time and match time so the comparison
    is symmetric."""
    if not title:
        return ""
    norm = title.strip().lower()
    norm = _LEADING_ARTICLE_RE.sub("", norm)
    norm = _PUNCT_STRIP_RE.sub(" ", norm)
    norm = _WHITESPACE_RUN_RE.sub(" ", norm).strip()
    return norm


def _truncate_for_match(normalized_title: str) -> str:
    """If the normalized title has more than 10 words, return the first 5.
    Long titles tend to include a colon-introduced subtitle that the citing
    paper omits, so the prefix is a more robust match key."""
    words = normalized_title.split()
    if len(words) > 10:
        return " ".join(words[:5])
    return normalized_title


def _strip_bibliography(text: str) -> str:
    """Truncate at the first References / Bibliography / Works Cited section
    header. If no header is found, return the text unchanged.

    Mitigates the most common stance false positive: a title appearing only
    in B's bibliography list is not a stance signal — it's just B citing A
    structurally. By stripping the bibliography section we make sure our
    sentence match only fires when the title appears in B's actual prose.
    """
    if not text:
        return text
    match = _BIBLIOGRAPHY_HEADER_RE.search(text)
    if match is None:
        return text
    return text[: match.start()]


def _extract_citation_context(b_text: str, a_title: str) -> list[str]:
    """Return up to 3 distinct citation contexts where A's title appears in
    B's prose. Each context is a window of ±1 sentences around the match.
    Returns an empty list if no match is found.

    The match is done on normalized text (lowercased, articles stripped,
    punctuation collapsed) using either the full normalized title or its
    first-5-word prefix when the title is long. The output context strings
    are returned with their original casing.
    """
    if not b_text or not a_title:
        return []
    normalized_title = _normalize_title(a_title)
    if not normalized_title:
        return []
    match_key = _truncate_for_match(normalized_title)
    if len(match_key) < 4:
        # Single-word, super-short titles (e.g. "GPT") would over-match.
        # Skip — caller treats this as no_match.
        return []

    # Sentence-split once on the original text so we can return original casing.
    sentences = _SENTENCE_SPLIT_RE.split(b_text)
    # Build a normalized parallel array for matching.
    normalized_sentences = [_normalize_title(s) for s in sentences]

    contexts: list[str] = []
    seen: set[str] = set()
    for i, ns in enumerate(normalized_sentences):
        if match_key not in ns:
            continue
        lo = max(0, i - 1)
        hi = min(len(sentences), i + 2)
        window = " ".join(s.strip() for s in sentences[lo:hi] if s.strip())
        if not window or window in seen:
            continue
        seen.add(window)
        contexts.append(window)
        if len(contexts) >= 3:
            break
    return contexts


def _validate_llm_payload(payload: Any) -> dict[str, Any] | None:
    """Validate the LLM response shape. Returns the cleaned dict on success,
    or None if any required field is malformed. We deliberately do NOT
    fall back to a default stance label — silently writing low-quality
    stance is worse than no stance, and the next idempotent re-run can
    retry the edge."""
    if not isinstance(payload, dict):
        return None
    stance = payload.get("stance")
    if not isinstance(stance, str) or stance.strip().lower() not in STANCE_LABELS:
        return None
    stance = stance.strip().lower()
    confidence_raw = payload.get("confidence")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    rationale = payload.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""
    rationale = rationale.strip()
    return {"stance": stance, "confidence": round(confidence, 3), "rationale": rationale}


def _call_llm_for_stance(
    client: Any,
    model: str,
    contexts: list[str],
    a_title: str,
) -> dict[str, Any] | None:
    """Make one LLM call, parse the JSON, validate, return None on any
    failure. Failures are caller-visible via the llm_failed counter."""
    user = json.dumps(
        {
            "cited_paper_title": a_title,
            "citation_contexts": contexts,
        },
        ensure_ascii=False,
        indent=2,
    )
    try:
        raw = call_llm_text(
            client,
            model=model,
            system=SYSTEM_PROMPT,
            user=user,
            temperature=float(os.environ.get("AIGRAPH_STANCE_TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get("AIGRAPH_STANCE_MAX_TOKENS", "200")),
        )
    except Exception as exc:  # pragma: no cover - defensive for network/model errors
        logger.warning("stance LLM call failed: %s", exc)
        return None
    parsed = _load_json(raw)
    return _validate_llm_payload(parsed)


def _build_default_client(model: str | None) -> tuple[Any, str]:
    """Lazy-construct a real OpenAI client when the caller didn't supply one.
    Mirrors the LLMHypothesisGenerator pattern."""
    resolved_model = configured_model(model)
    client = build_openai_client(
        api_key=configured_api_key(None),
        base_url=configured_base_url(None),
    )
    return client, resolved_model


def classify_cites_edges(
    g: nx.MultiDiGraph,
    papers: list[Paper],
    *,
    client: Any = None,
    model: str | None = None,
    skip_classified: bool = True,
    max_edges: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Classify all cites edges in g. Mutates g in place by adding stance,
    stance_confidence, and stance_rationale attributes to each cites edge.

    Returns a counter dict with keys:
      total_seen, classified, skipped_already, skipped_self_citation,
      skipped_no_text, skipped_no_match, llm_failed, would_classify
        (only when dry_run=True).

    Args:
      skip_classified: if True (default), edges with a non-None stance
        attribute are not re-classified. Makes incremental runs cheap.
      max_edges: optional budget cap. After deduping and skipping, only
        the first ``max_edges`` will be sent to the LLM. The remainder is
        unprocessed and visible in subsequent runs.
      dry_run: if True, return the counters without making any LLM calls.
        ``would_classify`` reports how many edges would have been
        processed.
    """
    papers_by_id = {p.paper_id: p for p in papers}
    # Pre-seed every counter key to 0 so the result dict has a stable schema
    # regardless of which code paths fired. Callers (tests, monitoring) can
    # rely on `counters["classified"]` always being present.
    counters: Counter[str] = Counter(
        {
            "total_seen": 0,
            "classified": 0,
            "skipped_already": 0,
            "skipped_self_citation": 0,
            "skipped_no_text": 0,
            "skipped_no_match": 0,
            "llm_failed": 0,
        }
    )

    # Phase 1: walk edges, dedupe (u, v), apply skip rules.
    edges_to_process: list[tuple[str, str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for u, v, key, data in g.edges(keys=True, data=True):
        counters["total_seen"] += 1
        if data.get("edge_type") != "cites":
            continue
        if u == v:
            counters["skipped_self_citation"] += 1
            continue
        if (u, v) in seen_pairs:
            # Parallel edges between the same pair — already queued.
            continue
        seen_pairs.add((u, v))
        if skip_classified and data.get("stance") is not None:
            counters["skipped_already"] += 1
            continue
        edges_to_process.append((u, v, key))

    if max_edges is not None:
        edges_to_process = edges_to_process[:max_edges]

    if dry_run:
        counters["would_classify"] = len(edges_to_process)
        return dict(counters)

    if not edges_to_process:
        logger.info("classify_cites_edges summary: %s", dict(counters))
        return dict(counters)

    # Lazily build the LLM client once the first edge is ready to fire.
    resolved_model = model
    actual_client = client
    if actual_client is None:
        actual_client, resolved_model = _build_default_client(model)
    elif resolved_model is None:
        resolved_model = configured_model(None)

    # Phase 2: classify.
    for u, v, key in edges_to_process:
        pid_b = u.removeprefix("Paper:")
        pid_a = v.removeprefix("Paper:")
        paper_b = papers_by_id.get(pid_b)
        paper_a = papers_by_id.get(pid_a)
        if paper_b is None or paper_a is None:
            counters["skipped_no_text"] += 1
            continue
        if not (paper_b.text or "").strip():
            counters["skipped_no_text"] += 1
            continue
        contexts = _extract_citation_context(
            _strip_bibliography(paper_b.text),
            paper_a.title or "",
        )
        if not contexts:
            counters["skipped_no_match"] += 1
            continue
        result = _call_llm_for_stance(actual_client, resolved_model, contexts, paper_a.title or "")
        if result is None:
            counters["llm_failed"] += 1
            continue
        g[u][v][key]["stance"] = result["stance"]
        g[u][v][key]["stance_confidence"] = result["confidence"]
        g[u][v][key]["stance_rationale"] = result["rationale"]
        counters["classified"] += 1

    logger.info("classify_cites_edges summary: %s", dict(counters))
    return dict(counters)
