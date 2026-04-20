"""Fetch AI-paper records from OpenAlex at the abstract level (no PDFs).

OpenAlex serves abstracts as an inverted index (``abstract_inverted_index``) to
sidestep redistribution rules; we reconstruct running text locally.
"""

from __future__ import annotations

import os
from math import ceil
from typing import Any, Iterator, Optional

from .models import Paper
from .paper_select import normalize_topic_query, select_representative_papers


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: Optional[dict[str, list[int]]]) -> str:
    """Invert OpenAlex's ``abstract_inverted_index`` back into running text."""
    if not inverted_index:
        return ""
    max_pos = -1
    positions: list[tuple[int, str]] = []
    for word, locs in inverted_index.items():
        for pos in locs:
            positions.append((pos, word))
            if pos > max_pos:
                max_pos = pos
    if max_pos < 0:
        return ""
    slots: list[str] = [""] * (max_pos + 1)
    for pos, word in positions:
        slots[pos] = word
    return " ".join(s for s in slots if s).strip()


def _openalex_id_suffix(work_id: str) -> str:
    """Strip the URL prefix off an OpenAlex Work id."""
    if not work_id:
        return ""
    return work_id.rsplit("/", 1)[-1]


def normalize_openalex_work_id(work_id: str | None) -> str | None:
    """Normalize an OpenAlex Work id/URL to the local ``openalex:W...`` id."""
    if not work_id:
        return None
    suffix = _openalex_id_suffix(str(work_id).strip())
    if not suffix:
        return None
    return f"openalex:{suffix}"


def _venue_name(work: dict[str, Any]) -> str:
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    name = source.get("display_name")
    if name:
        return name
    host = work.get("host_venue") or {}
    if host.get("display_name"):
        return host["display_name"]
    return "unknown"


def work_to_paper(work: dict[str, Any], retrieval_channel: str | None = None) -> Paper:
    work_id = _openalex_id_suffix(work.get("id", ""))
    title = work.get("title") or work.get("display_name") or ""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    year = int(work.get("publication_year") or 0)
    openalex_url = work.get("id") if isinstance(work.get("id"), str) else None
    doi = work.get("doi") if isinstance(work.get("doi"), str) else None
    referenced = [
        rid
        for rid in (normalize_openalex_work_id(w) for w in (work.get("referenced_works") or []))
        if rid
    ]
    return Paper(
        paper_id=f"openalex:{work_id}" if work_id else f"openalex:unknown-{year}",
        title=title,
        year=year,
        venue=_venue_name(work),
        url=openalex_url,
        doi=doi,
        cited_by_count=int(work.get("cited_by_count") or 0),
        referenced_works=referenced,
        counts_by_year=list(work.get("counts_by_year") or []),
        retrieval_channel=retrieval_channel,
        abstract=abstract,
        text=(f"{title}\n\n{abstract}" if abstract else title).strip(),
        structured_hint=None,
    )


def fetch_openalex_papers(
    query: str,
    from_year: int,
    to_year: int,
    limit: int = 50,
    mailto: str | None = None,
    client: Any | None = None,
    strategy: str = "balanced",
    candidate_multiplier: int = 4,
    citation_weight: float | None = None,
    min_relevance: float | None = None,
    query_variants: list[str] | None = None,
) -> list[Paper]:
    """Fetch representative works from OpenAlex, mapped to :class:`Paper`.

    ``client`` is an injected httpx-like client (for tests). If ``None`` we lazy-import
    :mod:`httpx` so that the synthetic demo stays importable without httpx installed.
    """
    if limit <= 0:
        return []
    limit = min(limit, 200)
    strategy = strategy if strategy in {"balanced", "high-impact", "recent"} else "balanced"
    normalized_query = normalize_topic_query(query) or query

    mailto = mailto or os.environ.get("AIGRAPH_MAILTO")
    base_filter = f"from_publication_date:{from_year}-01-01,to_publication_date:{to_year}-12-31"

    if client is None:
        import httpx  # lazy import

        client = httpx.Client(timeout=30.0)
        owns_client = True
    else:
        owns_client = False

    papers: list[Paper] = []
    try:
        channels = _channel_specs(normalized_query, strategy, query_variants=query_variants or [])
        total_candidates = max(limit, min(200, limit * max(1, candidate_multiplier)))
        channel_limit = max(limit, 5, ceil(total_candidates / max(1, len(channels))))
        for channel_name, channel_query, sort in channels:
            params = {
                "search": channel_query,
                "filter": base_filter,
                "per_page": min(channel_limit, 50),
                "cursor": "*",
            }
            if sort:
                params["sort"] = sort
            if mailto:
                params["mailto"] = mailto
            for work in _iter_works(client, params, channel_limit):
                papers.append(work_to_paper(work, retrieval_channel=channel_name))
    finally:
        if owns_client:
            client.close()
    return select_representative_papers(
        papers,
        query=normalized_query,
        limit=limit,
        strategy=strategy,
        citation_weight=citation_weight,
        min_relevance=min_relevance,
        require_core_match=True,
    )


def _channel_specs(query: str, strategy: str, *, query_variants: list[str]) -> list[tuple[str, str, str | None]]:
    relevance = ("relevance", query, None)
    impact = ("impact", query, "cited_by_count:desc")
    recent = ("recent", query, "publication_date:desc")
    survey = ("survey", f"{query} survey review benchmark evaluation", None)
    critical = ("critical", f"{query} limitation challenge failure robustness bias", None)
    extra = [
        (f"variant-{index + 1}", variant, None)
        for index, variant in enumerate(query_variants)
        if variant and variant != query
    ][:4]
    if strategy == "high-impact":
        return [impact, relevance, *extra, survey]
    if strategy == "recent":
        return [recent, relevance, *extra, critical]
    return [relevance, impact, recent, *extra, survey, critical]


def _iter_works(client: Any, params: dict[str, Any], limit: int) -> Iterator[dict[str, Any]]:
    fetched = 0
    while True:
        response = client.get(OPENALEX_WORKS_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            return
        for work in results:
            yield work
            fetched += 1
            if fetched >= limit:
                return
        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            return
        params = {**params, "cursor": next_cursor}
