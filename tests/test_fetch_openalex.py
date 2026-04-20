from aigraph.fetch_openalex import _iter_works, fetch_openalex_papers, reconstruct_abstract, work_to_paper


def test_reconstruct_abstract_inverts_index():
    inverted = {
        "RAG": [0],
        "improves": [1],
        "factual": [2],
        "QA": [3],
        "by": [4],
        "+8.2": [5],
        "EM": [6],
    }
    assert reconstruct_abstract(inverted) == "RAG improves factual QA by +8.2 EM"


def test_reconstruct_abstract_handles_duplicates_and_gaps():
    inverted = {"the": [0, 3], "model": [1], "is": [2], "good": [4]}
    assert reconstruct_abstract(inverted) == "the model is the good"


def test_reconstruct_abstract_none_returns_empty():
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_work_to_paper_basic_mapping():
    work = {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.123/example",
        "title": "RAG: a study",
        "publication_year": 2023,
        "primary_location": {"source": {"display_name": "NeurIPS"}},
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
        "cited_by_count": 42,
        "referenced_works": ["https://openalex.org/W999"],
        "counts_by_year": [{"year": 2025, "cited_by_count": 12}],
    }
    paper = work_to_paper(work)
    assert paper.paper_id == "openalex:W123"
    assert paper.title == "RAG: a study"
    assert paper.year == 2023
    assert paper.venue == "NeurIPS"
    assert paper.url == "https://openalex.org/W123"
    assert paper.doi == "https://doi.org/10.123/example"
    assert paper.cited_by_count == 42
    assert paper.referenced_works == ["openalex:W999"]
    assert paper.counts_by_year == [{"year": 2025, "cited_by_count": 12}]
    assert paper.abstract == "Hello world"
    assert paper.text.startswith("RAG: a study\n\nHello world")


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        pass

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, pages: list[dict]):
        self._pages = pages
        self.calls: list[dict] = []

    def get(self, url: str, params: dict) -> _FakeResponse:
        self.calls.append(params)
        idx = len(self.calls) - 1
        page = self._pages[min(idx, len(self._pages) - 1)]
        return _FakeResponse(page)


def test_iter_works_respects_limit_and_paginates():
    pages = [
        {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "title": "A",
                    "publication_year": 2022,
                    "primary_location": {"source": {"display_name": "ACL"}},
                    "abstract_inverted_index": {"x": [0]},
                },
                {
                    "id": "https://openalex.org/W2",
                    "title": "B",
                    "publication_year": 2023,
                    "primary_location": {"source": {"display_name": "ACL"}},
                    "abstract_inverted_index": {"y": [0]},
                },
            ],
            "meta": {"next_cursor": "CURSOR_2"},
        },
        {
            "results": [
                {
                    "id": "https://openalex.org/W3",
                    "title": "C",
                    "publication_year": 2024,
                    "primary_location": {"source": {"display_name": "ACL"}},
                    "abstract_inverted_index": {"z": [0]},
                }
            ],
            "meta": {"next_cursor": None},
        },
    ]
    fake = _FakeClient(pages)
    works = list(
        _iter_works(
            fake,
            {"search": "rag", "filter": "x", "per_page": 2, "cursor": "*"},
            limit=3,
        )
    )
    assert [w["id"].rsplit("/", 1)[-1] for w in works] == ["W1", "W2", "W3"]
    # Second call should have advanced the cursor.
    assert fake.calls[1]["cursor"] == "CURSOR_2"


class _ChannelFakeClient:
    def __init__(self):
        self.calls: list[dict] = []

    def get(self, url: str, params: dict) -> _FakeResponse:
        self.calls.append(dict(params))
        search = params.get("search", "")
        sort = params.get("sort")
        if sort == "cited_by_count:desc":
            work = _work("W-impact", "High Impact RAG Survey", 2022, cited_by_count=500)
        elif sort == "publication_date:desc":
            work = _work("W-recent", "Recent RAG Failure Analysis", 2026, cited_by_count=5)
        elif "survey" in search:
            work = _work("W-survey", "Survey of RAG Evaluation Benchmarks", 2024, cited_by_count=60)
        elif "limitation" in search:
            work = _work("W-critical", "Limitations and Robustness of RAG", 2025, cited_by_count=12)
        else:
            work = _work("W-relevance", "Retrieval Augmented Generation for QA", 2023, cited_by_count=40)
        return _FakeResponse({"results": [work], "meta": {"next_cursor": None}})


class _RelevanceGateFakeClient:
    def __init__(self):
        self.calls: list[dict] = []

    def get(self, url: str, params: dict) -> _FakeResponse:
        self.calls.append(dict(params))
        return _FakeResponse(
            {
                "results": [
                    _work("W-off", "Global Disease Burden Healthcare Study", 2024, cited_by_count=10000),
                    _work("W-on", "Large Language Models for Finance Time Series Forecasting", 2025, cited_by_count=10),
                ],
                "meta": {"next_cursor": None},
            }
        )


def _work(work_id: str, title: str, year: int, cited_by_count: int = 0) -> dict:
    return {
        "id": f"https://openalex.org/{work_id}",
        "title": title,
        "publication_year": year,
        "primary_location": {"source": {"display_name": "ACL"}},
        "abstract_inverted_index": {"retrieval": [0], "augmented": [1], "generation": [2]},
        "cited_by_count": cited_by_count,
    }


def test_fetch_openalex_balanced_uses_multi_channel_pool_and_reranks():
    fake = _ChannelFakeClient()
    papers = fetch_openalex_papers(
        query="retrieval augmented generation",
        from_year=2020,
        to_year=2026,
        limit=3,
        client=fake,
        strategy="balanced",
    )
    assert any(call.get("sort") == "cited_by_count:desc" for call in fake.calls)
    assert any(call.get("sort") == "publication_date:desc" for call in fake.calls)
    assert any("survey review benchmark evaluation" in call.get("search", "") for call in fake.calls)
    assert len(papers) == 3
    assert all(p.selection_score > 0 for p in papers)
    assert {p.retrieval_channel for p in papers} <= {"relevance", "impact", "recent", "survey", "critical"}
    assert any(p.paper_role in {"survey", "benchmark"} for p in papers)
    assert any(p.paper_role == "failure" for p in papers)


def test_fetch_openalex_accepts_custom_citation_weight():
    fake = _ChannelFakeClient()
    papers = fetch_openalex_papers(
        query="retrieval augmented generation",
        from_year=2020,
        to_year=2026,
        limit=1,
        client=fake,
        strategy="balanced",
        citation_weight=0.75,
    )
    assert papers[0].retrieval_channel == "impact"


def test_fetch_openalex_filters_off_topic_high_citation_candidates():
    fake = _RelevanceGateFakeClient()
    papers = fetch_openalex_papers(
        query="large language models finance time series forecasting",
        from_year=2020,
        to_year=2026,
        limit=1,
        client=fake,
        strategy="balanced",
        citation_weight=0.80,
        min_relevance=0.30,
    )
    assert papers[0].paper_id == "openalex:W-on"
