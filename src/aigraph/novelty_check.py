"""Post-hypothesis arxiv-based novelty check.

For each generated :class:`Hypothesis`, query arXiv for the closest
related papers (using simple keyword extraction over
``hypothesis + mechanism``) and ask the LLM whether the proposal is
substantively novel relative to the retrieved candidates.

Output is ``list[dict]`` (not ``list[Hypothesis]``) because
:class:`~aigraph.models.Hypothesis` extends :class:`LooseModel` with
``extra="ignore"`` — pydantic would silently drop the
``novelty_check`` extras key on serialization. Production callers
either dump these dicts straight to JSONL with ``json.dumps`` or pass
them through ``Hypothesis.model_validate`` to get the typed
hypothesis (with ``novelty_check`` silently dropped). This mirrors the
v0.4 multi-grain creator's ``multi_grain`` extras pattern.

Failures are non-fatal: if arXiv is unreachable, the LLM call errors,
or JSON parsing fails, ``check_hypothesis_novelty`` returns
``{"is_novel": None, "similar_papers": [], "rationale": "<reason>"}``.
The caller never sees an exception bubble up from this module.
"""

from __future__ import annotations

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Optional

from .llm_client import (
    build_openai_client,
    call_llm_text,
    configured_api_key,
    configured_base_url,
    configured_model,
)
from .models import Hypothesis

logger = logging.getLogger(__name__)


ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_RATE_LIMIT_SECONDS = 3.0

# Retry settings for transient arxiv failures (timeouts / 429 / 5xx).
# Backoff list = delay before each retry; len = max retry count.
# 17% of records lost is_novel signal to one-shot timeouts on a 90-hyp
# run; with these retries that should drop below 5%.
_ARXIV_RETRY_BACKOFF_SECONDS = (0.5, 1.5, 4.5)
_ARXIV_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

# Lightweight English stopword list — we intentionally avoid pulling in
# nltk or spaCy. The set covers the most common low-signal tokens that
# would otherwise dominate keyword frequency counts.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "so", "of", "on", "in", "at",
        "by", "for", "with", "to", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "this", "that", "these", "those", "it", "its", "we", "our",
        "they", "their", "i", "you", "he", "she", "his", "her", "them", "us", "any",
        "all", "more", "most", "less", "least", "some", "such", "no", "not", "only",
        "than", "then", "when", "where", "while", "which", "who", "whom", "whose",
        "what", "how", "why", "can", "could", "would", "should", "may", "might",
        "must", "do", "does", "did", "have", "has", "had", "having", "will", "shall",
        "into", "out", "up", "down", "over", "under", "about", "between", "through",
        "during", "before", "after", "across", "via", "per", "also", "however",
        "therefore", "hence", "thus", "moreover", "furthermore", "additionally",
        "yet", "still", "very", "much", "many", "few", "several", "each", "every",
        "another", "same", "different", "new", "old", "good", "bad", "high", "low",
        "use", "uses", "used", "using", "make", "makes", "made", "show", "shows",
        "shown", "find", "finds", "found", "based", "given", "include", "includes",
        "including", "see", "seen", "result", "results", "method", "methods",
        "approach", "approaches", "model", "models", "paper", "papers", "work",
        "works", "study", "studies", "task", "tasks", "data", "datasets",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
# Conservative proper-cased multi-word phrase: 2-4 capitalized tokens.
_PROPER_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9\-]+)(?:\s+[A-Z][a-zA-Z0-9\-]+){1,3}\b"
)


def _extract_keywords(text: str, *, max_keywords: int = 6) -> list[str]:
    """Pick keywords from a hypothesis blob: keep proper-cased multi-word
    phrases verbatim, plus single tokens that appear 2+ times. Returns
    keywords in descending priority order, lowercased single tokens but
    preserving case for multi-word phrases (which arXiv search treats as
    a phrase when quoted)."""
    if not text:
        return []
    keywords: list[str] = []
    seen_lower: set[str] = set()

    for match in _PROPER_PHRASE_RE.findall(text):
        squashed = " ".join(match.split())
        key = squashed.lower()
        if key in seen_lower:
            continue
        # Strip phrases that are just stopwords with capital first letters.
        tokens = [t for t in re.split(r"\s+", squashed) if t]
        if all(t.lower() in _STOPWORDS for t in tokens):
            continue
        seen_lower.add(key)
        keywords.append(squashed)
        if len(keywords) >= max_keywords:
            return keywords

    counts: dict[str, int] = {}
    for token in _TOKEN_RE.findall(text):
        lo = token.lower()
        if lo in _STOPWORDS or len(lo) < 3:
            continue
        counts[lo] = counts.get(lo, 0) + 1
    repeated = sorted(
        (tok for tok, n in counts.items() if n >= 2),
        key=lambda t: (-counts[t], t),
    )
    for tok in repeated:
        if tok in seen_lower:
            continue
        seen_lower.add(tok)
        keywords.append(tok)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _build_arxiv_query(keywords: list[str]) -> str:
    """Build a ``ti:KW OR abs:KW`` arXiv search expression. Multi-word
    phrases are quoted so the search engine treats them as a phrase."""
    if not keywords:
        return ""
    parts: list[str] = []
    for kw in keywords:
        if " " in kw:
            term = f'"{kw}"'
        else:
            term = kw
        parts.append(f"ti:{term}")
        parts.append(f"abs:{term}")
    return " OR ".join(parts)


def _parse_atom_feed(text: str) -> list[dict]:
    """Parse the arXiv Atom response into ``{arxiv_id, title, abstract}``
    records. Tolerant of malformed XML — returns ``[]`` on parse error."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        logger.warning("arxiv atom parse failed: %s", exc)
        return []
    out: list[dict] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        url = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
        arxiv_id = url.rstrip("/").rsplit("/", 1)[-1] if url else ""
        title = " ".join(
            (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").split()
        )
        abstract = " ".join(
            (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").split()
        )
        out.append({"arxiv_id": arxiv_id, "title": title, "abstract": abstract})
    return out


def _is_transient_arxiv_error(exc: BaseException) -> bool:
    """True if exc is a network timeout / connection error / 5xx / 429,
    i.e. worth retrying. Anything else (4xx other than 429, parse
    errors) is non-transient and propagates.

    Resolves httpx exception types via getattr to stay robust against
    test setups that replace ``sys.modules['httpx']`` with a partial stub.
    """
    try:
        import httpx
    except ImportError:
        httpx = None  # type: ignore[assignment]
    if httpx is not None:
        for attr in ("TimeoutException", "ConnectError", "RemoteProtocolError"):
            cls = getattr(httpx, attr, None)
            if isinstance(cls, type) and isinstance(exc, cls):
                return True
        status_cls = getattr(httpx, "HTTPStatusError", None)
        if isinstance(status_cls, type) and isinstance(exc, status_cls):
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status in _ARXIV_RETRY_STATUS_CODES:
                return True
    import socket
    import urllib.error
    if isinstance(exc, (socket.timeout, TimeoutError, urllib.error.URLError)):
        return True
    return False


def _http_get_text(url: str, params: dict, *, http_client: Any | None = None) -> str:
    """Fetch a URL and return the response body as text. Uses ``httpx``
    when available; falls back to ``urllib.request`` so the module stays
    usable without the ``[real]`` extra installed.

    A caller-supplied ``http_client`` short-circuits both branches and is
    used as-is (must expose ``client.get(url, params=...)`` returning an
    object with ``.text`` and ``.raise_for_status()``).

    Transient failures (timeouts, connection errors, 5xx, 429) are
    retried with exponential backoff per ``_ARXIV_RETRY_BACKOFF_SECONDS``.
    Non-transient errors raise immediately. The caller (``query_arxiv``)
    catches anything that escapes."""

    def _do_get() -> str:
        if http_client is not None:
            response = http_client.get(url, params=params)
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            return response.text
        try:
            import httpx
        except ImportError:
            from urllib.parse import urlencode
            from urllib.request import urlopen

            full_url = f"{url}?{urlencode(params)}"
            with urlopen(full_url, timeout=45.0) as resp:
                return resp.read().decode("utf-8", errors="replace")
        response = httpx.get(url, params=params, timeout=45.0, follow_redirects=True)
        response.raise_for_status()
        return response.text

    backoffs = _ARXIV_RETRY_BACKOFF_SECONDS
    for attempt in range(len(backoffs) + 1):  # initial + N retries
        try:
            return _do_get()
        except Exception as exc:
            is_last = attempt == len(backoffs)
            if is_last or not _is_transient_arxiv_error(exc):
                raise
            delay = backoffs[attempt]
            logger.info(
                "arxiv transient error (attempt %d/%d): %s; retrying in %.1fs",
                attempt + 1, len(backoffs) + 1, exc, delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def query_arxiv(
    query: str,
    *,
    max_results: int = 10,
    http_client: Any | None = None,
) -> list[dict]:
    """Run a search against the arXiv Atom API and return parsed
    candidates. Returns ``[]`` on empty query or transport/parse errors —
    never raises."""
    if not query:
        return []
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max(1, int(max_results)),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        text = _http_get_text(ARXIV_QUERY_URL, params, http_client=http_client)
    except Exception as exc:  # pragma: no cover - network-dependent
        logger.warning("arxiv query failed: %s", exc)
        return []
    return _parse_atom_feed(text)


def _load_json(raw: str) -> Optional[dict]:
    """Tolerant JSON loader: strip Markdown fences, fall back to brace
    extraction. Returns ``None`` if no JSON object can be recovered."""
    if not raw:
        return None
    text = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


NOVELTY_SYSTEM_PROMPT = (
    "You are a research novelty reviewer. Given a proposed hypothesis "
    "(claim plus mechanism) and a short list of related papers retrieved "
    "from arXiv, decide whether the proposal is substantively new versus "
    "the closest 3 retrieved papers.\n"
    "Output STRICT JSON with exactly these keys:\n"
    "- is_novel: boolean — true only if the core mechanism or empirical "
    "claim differs in a material way from all 3 closest related papers.\n"
    "- similar: array (length up to 3) of objects with keys arxiv_id and "
    "title for the closest related papers.\n"
    "- rationale: one or two sentences justifying the decision, citing "
    "specific differences or overlaps.\n"
    "Return only the JSON object — no prose, no Markdown fences."
)


def _format_candidates_for_prompt(candidates: list[dict]) -> str:
    lines: list[str] = []
    for i, cand in enumerate(candidates, start=1):
        title = cand.get("title", "").strip() or "(no title)"
        abstract = (cand.get("abstract", "") or "").strip()
        arxiv_id = cand.get("arxiv_id", "") or ""
        snippet = abstract[:600] + ("..." if len(abstract) > 600 else "")
        lines.append(f"[{i}] arxiv_id={arxiv_id}\n    title: {title}\n    abstract: {snippet}")
    return "\n".join(lines) if lines else "(no related papers found)"


def _build_novelty_prompt(hypothesis: Hypothesis, candidates: list[dict]) -> str:
    parts = [
        "Proposed hypothesis:",
        f"  {hypothesis.hypothesis.strip()}",
    ]
    mechanism = (hypothesis.mechanism or "").strip()
    if mechanism:
        parts.extend(["", "Proposed mechanism:", f"  {mechanism}"])
    parts.extend(
        [
            "",
            "Related papers (from arXiv search):",
            _format_candidates_for_prompt(candidates),
        ]
    )
    return "\n".join(parts)


def _normalize_similar_papers(value: Any) -> list[dict]:
    """Coerce the LLM's ``similar`` field into a list of
    ``{arxiv_id, title}`` dicts, dropping malformed entries."""
    if not isinstance(value, list):
        return []
    out: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        arxiv_id = str(item.get("arxiv_id", "") or "")
        title = str(item.get("title", "") or "")
        if not arxiv_id and not title:
            continue
        out.append({"arxiv_id": arxiv_id, "title": title})
    return out


def check_hypothesis_novelty(
    hypothesis: Hypothesis,
    *,
    http_client: Any | None = None,
    llm_client: Any | None = None,
    model: str | None = None,
    max_candidates: int = 5,
) -> dict[str, Any]:
    """Decide whether ``hypothesis`` is substantively novel relative to
    the top arXiv hits for keywords mined from its text + mechanism.

    Always returns a dict with keys ``is_novel`` (``bool | None``),
    ``similar_papers`` (``list[dict]``), ``rationale`` (``str``). On any
    failure (no keywords, arXiv error, LLM error, JSON parse failure)
    ``is_novel`` is ``None`` and ``rationale`` describes the failure. The
    function never raises."""
    blob = f"{hypothesis.hypothesis or ''} {hypothesis.mechanism or ''}".strip()
    keywords = _extract_keywords(blob)
    if not keywords:
        return {
            "is_novel": None,
            "similar_papers": [],
            "rationale": "no keywords could be extracted from hypothesis text",
        }
    query = _build_arxiv_query(keywords)
    try:
        candidates = query_arxiv(
            query,
            max_results=max(1, int(max_candidates)),
            http_client=http_client,
        )
    except Exception as exc:  # defensive — query_arxiv should not raise.
        return {
            "is_novel": None,
            "similar_papers": [],
            "rationale": f"arxiv query failed: {exc}",
        }
    if not candidates:
        return {
            "is_novel": None,
            "similar_papers": [],
            "rationale": "arxiv returned no candidates (possibly unavailable or empty match)",
        }

    resolved_model = configured_model(model)
    if llm_client is None:
        try:
            llm_client = build_openai_client(
                api_key=configured_api_key(None),
                base_url=configured_base_url(None),
            )
        except Exception as exc:
            return {
                "is_novel": None,
                "similar_papers": [],
                "rationale": f"LLM client init failed: {exc}",
            }

    user_prompt = _build_novelty_prompt(hypothesis, candidates[:3])
    try:
        raw = call_llm_text(
            llm_client,
            model=resolved_model,
            system=NOVELTY_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.0,
        )
    except Exception as exc:
        return {
            "is_novel": None,
            "similar_papers": [],
            "rationale": f"LLM call failed: {exc}",
        }

    payload = _load_json(raw)
    if not isinstance(payload, dict):
        return {
            "is_novel": None,
            "similar_papers": [],
            "rationale": "LLM output was not valid JSON",
        }

    is_novel_raw = payload.get("is_novel")
    if isinstance(is_novel_raw, bool):
        is_novel: Optional[bool] = is_novel_raw
    elif isinstance(is_novel_raw, str) and is_novel_raw.lower() in {"true", "false"}:
        is_novel = is_novel_raw.lower() == "true"
    else:
        is_novel = None

    similar = _normalize_similar_papers(payload.get("similar"))
    rationale = str(payload.get("rationale", "") or "").strip()
    if not rationale:
        rationale = "(LLM did not provide a rationale)"
    return {
        "is_novel": is_novel,
        "similar_papers": similar,
        "rationale": rationale,
    }


def annotate_hypotheses_with_novelty(
    hypotheses: list[Hypothesis],
    *,
    http_client: Any | None = None,
    llm_client: Any | None = None,
    model: str | None = None,
    max_candidates: int = 5,
    sleep_seconds: float = ARXIV_RATE_LIMIT_SECONDS,
) -> list[dict]:
    """Run :func:`check_hypothesis_novelty` over a batch of hypotheses
    and yield ``hyp.model_dump(by_alias=True)`` enriched with a
    ``novelty_check`` field. Sleeps ``sleep_seconds`` between iterations
    to respect arXiv's polite 1-request-per-3-seconds rate limit."""
    out: list[dict] = []
    total = len(hypotheses)
    for i, hyp in enumerate(hypotheses):
        if i > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        result = check_hypothesis_novelty(
            hyp,
            http_client=http_client,
            llm_client=llm_client,
            model=model,
            max_candidates=max_candidates,
        )
        record = hyp.model_dump(by_alias=True)
        record["novelty_check"] = result
        out.append(record)
        logger.info(
            "novelty_check %d/%d hyp=%s is_novel=%s",
            i + 1,
            total,
            getattr(hyp, "hypothesis_id", "?"),
            result.get("is_novel"),
        )
    return out
