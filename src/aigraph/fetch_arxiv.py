"""Fetch abstract-level paper records from the public arXiv Atom API."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import Any, Iterator

from .models import Paper
from .paper_select import select_representative_papers


ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def entry_to_paper(entry: ET.Element) -> Paper:
    arxiv_url = _text(entry, "atom:id")
    arxiv_id = _arxiv_id_from_url(arxiv_url)
    title = _squash_ws(_text(entry, "atom:title"))
    abstract = _squash_ws(_text(entry, "atom:summary"))
    published = _text(entry, "atom:published")
    year = _year_from_timestamp(published)

    return Paper(
        paper_id=f"arxiv:{arxiv_id}" if arxiv_id else f"arxiv:unknown-{year}",
        title=title,
        year=year,
        venue="arXiv",
        url=arxiv_url or None,
        doi=None,
        cited_by_count=0,
        referenced_works=[],
        counts_by_year=[],
        retrieval_channel="arxiv",
        abstract=abstract,
        text=(f"{title}\n\n{abstract}" if abstract else title).strip(),
        structured_hint=None,
    )


def fetch_arxiv_papers(
    query: str,
    from_year: int,
    to_year: int,
    limit: int = 50,
    client: Any | None = None,
    page_size: int = 50,
    sleep_seconds: float = 3.0,
    strategy: str = "balanced",
) -> list[Paper]:
    """Fetch up to ``limit`` arXiv records and map them to :class:`Paper`.

    arXiv does not provide citation counts through the Atom API, so citation
    fields are left as zero/empty. The query is passed through as an arXiv search
    expression, for example ``all:"large language models" AND all:finance``.
    """
    if limit <= 0:
        return []
    limit = min(limit, 200)
    strategy = strategy if strategy in {"balanced", "recent"} else "balanced"
    page_size = max(1, min(page_size, 100))

    if client is None:
        import httpx

        client = httpx.Client(timeout=45.0, follow_redirects=True)
        owns_client = True
    else:
        owns_client = False

    papers: list[Paper] = []
    seen: set[str] = set()
    try:
        max_results = min(200, limit * 4)
        sort_by = "submittedDate" if strategy == "recent" else "relevance"
        for entry in _iter_entries(
            client,
            query,
            max_results=max_results,
            page_size=page_size,
            sleep_seconds=sleep_seconds,
            sort_by=sort_by,
        ):
            paper = entry_to_paper(entry)
            paper = paper.model_copy(update={"retrieval_channel": f"arxiv-{strategy}"})
            if paper.paper_id in seen:
                continue
            seen.add(paper.paper_id)
            if paper.year and not (from_year <= paper.year <= to_year):
                continue
            papers.append(paper)
    finally:
        if owns_client:
            client.close()
    return select_representative_papers(papers, query=query, limit=limit, strategy=strategy)


def _iter_entries(
    client: Any,
    query: str,
    max_results: int,
    page_size: int,
    sleep_seconds: float,
    sort_by: str = "relevance",
) -> Iterator[ET.Element]:
    fetched = 0
    start = 0
    while fetched < max_results:
        batch_size = min(page_size, max_results - fetched)
        response = client.get(
            ARXIV_QUERY_URL,
            params={
                "search_query": query,
                "start": start,
                "max_results": batch_size,
                "sortBy": sort_by,
                "sortOrder": "descending",
            },
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        entries = root.findall("atom:entry", ATOM_NS)
        if not entries:
            return
        for entry in entries:
            yield entry
        fetched += len(entries)
        start += len(entries)
        if len(entries) < batch_size:
            return
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def _text(entry: ET.Element, path: str) -> str:
    value = entry.findtext(path, default="", namespaces=ATOM_NS)
    return value or ""


def _squash_ws(text: str) -> str:
    return " ".join(text.split())


def _arxiv_id_from_url(url: str) -> str:
    if not url:
        return ""
    return url.rstrip("/").rsplit("/", 1)[-1]


def _year_from_timestamp(value: str) -> int:
    if not value or len(value) < 4:
        return 0
    try:
        return int(value[:4])
    except ValueError:
        return 0
