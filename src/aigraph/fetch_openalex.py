"""Fetch AI-paper records from OpenAlex at the abstract level (no PDFs).

OpenAlex serves abstracts as an inverted index (``abstract_inverted_index``) to
sidestep redistribution rules; we reconstruct running text locally.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Optional

from .models import Paper


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


def work_to_paper(work: dict[str, Any]) -> Paper:
    work_id = _openalex_id_suffix(work.get("id", ""))
    title = work.get("title") or work.get("display_name") or ""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    year = int(work.get("publication_year") or 0)
    openalex_url = work.get("id") if isinstance(work.get("id"), str) else None
    doi = work.get("doi") if isinstance(work.get("doi"), str) else None
    return Paper(
        paper_id=f"openalex:{work_id}" if work_id else f"openalex:unknown-{year}",
        title=title,
        year=year,
        venue=_venue_name(work),
        url=openalex_url,
        doi=doi,
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
) -> list[Paper]:
    """Fetch up to ``limit`` works from OpenAlex, mapped to :class:`Paper`.

    ``client`` is an injected httpx-like client (for tests). If ``None`` we lazy-import
    :mod:`httpx` so that the synthetic demo stays importable without httpx installed.
    """
    if limit <= 0:
        return []
    limit = min(limit, 200)

    mailto = mailto or os.environ.get("AIGRAPH_MAILTO")
    params = {
        "search": query,
        "filter": f"from_publication_date:{from_year}-01-01,to_publication_date:{to_year}-12-31",
        "per_page": min(limit, 50),
        "cursor": "*",
    }
    if mailto:
        params["mailto"] = mailto

    if client is None:
        import httpx  # lazy import

        client = httpx.Client(timeout=30.0)
        owns_client = True
    else:
        owns_client = False

    papers: list[Paper] = []
    try:
        for work in _iter_works(client, params, limit):
            paper = work_to_paper(work)
            papers.append(paper)
            if len(papers) >= limit:
                break
    finally:
        if owns_client:
            client.close()
    return papers


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
