from aigraph.fetch_openalex import fetch_openalex_papers, reconstruct_abstract, work_to_paper


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
    }
    paper = work_to_paper(work)
    assert paper.paper_id == "openalex:W123"
    assert paper.title == "RAG: a study"
    assert paper.year == 2023
    assert paper.venue == "NeurIPS"
    assert paper.url == "https://openalex.org/W123"
    assert paper.doi == "https://doi.org/10.123/example"
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


def test_fetch_openalex_respects_limit_and_paginates():
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
    papers = fetch_openalex_papers(
        query="rag", from_year=2020, to_year=2025, limit=3, client=fake,
    )
    assert [p.paper_id for p in papers] == ["openalex:W1", "openalex:W2", "openalex:W3"]
    # Second call should have advanced the cursor.
    assert fake.calls[1]["cursor"] == "CURSOR_2"
