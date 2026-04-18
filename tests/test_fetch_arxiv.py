from aigraph.fetch_arxiv import entry_to_paper, fetch_arxiv_papers
from aigraph.fetch_arxiv import ATOM_NS
import xml.etree.ElementTree as ET


ATOM = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2401.12345v1</id>
    <title>  Large Language Models for Finance  </title>
    <summary>
      LLMs improve financial forecasting under regime shift.
    </summary>
    <published>2024-02-03T00:00:00Z</published>
  </entry>
  <entry>
    <id>https://arxiv.org/abs/1901.00001v1</id>
    <title>Old Paper</title>
    <summary>Too old.</summary>
    <published>2019-01-01T00:00:00Z</published>
  </entry>
</feed>
"""


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        pass


class _FakeClient:
    def __init__(self, pages: list[str]):
        self.pages = pages
        self.calls: list[dict] = []

    def get(self, url: str, params: dict) -> _FakeResponse:
        self.calls.append(params)
        idx = len(self.calls) - 1
        return _FakeResponse(self.pages[min(idx, len(self.pages) - 1)])


def test_entry_to_paper_maps_atom_entry():
    root = ET.fromstring(ATOM)
    entry = root.find("atom:entry", ATOM_NS)
    assert entry is not None
    paper = entry_to_paper(entry)
    assert paper.paper_id == "arxiv:2401.12345v1"
    assert paper.title == "Large Language Models for Finance"
    assert paper.year == 2024
    assert paper.venue == "arXiv"
    assert paper.url == "https://arxiv.org/abs/2401.12345v1"
    assert paper.cited_by_count == 0
    assert paper.referenced_works == []
    assert paper.retrieval_channel == "arxiv"
    assert "financial forecasting" in paper.abstract


def test_fetch_arxiv_filters_year_and_respects_limit():
    fake = _FakeClient([ATOM])
    papers = fetch_arxiv_papers(
        query='all:"large language models" AND all:finance',
        from_year=2020,
        to_year=2026,
        limit=1,
        client=fake,
        page_size=2,
        sleep_seconds=0,
    )
    assert [p.paper_id for p in papers] == ["arxiv:2401.12345v1"]
    assert fake.calls[0]["search_query"] == 'all:"large language models" AND all:finance'
    assert fake.calls[0]["max_results"] == 2
    assert fake.calls[0]["sortBy"] == "relevance"
    assert "arXiv has no citation metadata" in (papers[0].selection_reason or "")


def test_fetch_arxiv_recent_strategy_uses_submitted_date_sort():
    fake = _FakeClient([ATOM])
    papers = fetch_arxiv_papers(
        query="all:forecasting",
        from_year=2020,
        to_year=2026,
        limit=1,
        client=fake,
        page_size=2,
        sleep_seconds=0,
        strategy="recent",
    )
    assert fake.calls[0]["sortBy"] == "submittedDate"
    assert papers[0].retrieval_channel == "arxiv-recent"
