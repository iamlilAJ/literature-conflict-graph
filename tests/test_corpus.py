import io
import tarfile

from aigraph.corpus import (
    export_corpus_paper,
    seed_reasoning_corpus,
    sync_arxiv_corpus,
    validate_corpus,
)
from aigraph.extract import RuleBasedExtractor, extract_claims
from aigraph.io import read_jsonl, write_json, write_jsonl
from aigraph.models import Paper
from aigraph.paper_reader import read_paper_candidates


class _FakeResponse:
    def __init__(self, content: bytes = b"", status_code: int = 200):
        self.content = content
        self.status_code = status_code
        self.text = content.decode("utf-8", errors="ignore")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _FakeClient:
    def __init__(self, routes: dict[str, _FakeResponse]):
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str):
        self.calls.append(url)
        return self.routes.get(url, _FakeResponse(status_code=404))


def _paper(paper_id: str = "arxiv:2401.12345v1", *, title: str = "Reasoning Paper") -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        year=2024,
        venue="arXiv",
        url=f"https://arxiv.org/abs/{paper_id.split(':', 1)[1]}",
        abstract="Short abstract.",
    )


def _source_tar_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        files = {
            "main.tex": r"""
                \documentclass{article}
                \begin{document}
                \begin{abstract}
                We study reasoning in large language models.
                \end{abstract}
                \section{Introduction}
                Intro text.
                \input{sections/results}
                \appendix
                \section{Extra}
                Extra appendix text.
                \end{document}
            """,
            "sections/results.tex": r"""
                \subsection{Experimental Setup}
                We improve accuracy by +12 on LogicQA.
            """,
        }
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_seed_reasoning_corpus_dedupes_and_merges_seed_reasons(tmp_path):
    p1 = _paper("openalex:W1", title="CoT").model_copy(
        update={"arxiv_id_full": "2401.00001v1", "arxiv_id_base": "2401.00001", "openalex_id": "openalex:W1", "cited_by_count": 30}
    )
    p2 = _paper("openalex:W2", title="Logic").model_copy(
        update={"arxiv_id_full": "2401.00001v2", "arxiv_id_base": "2401.00001", "openalex_id": "openalex:W2", "cited_by_count": 80}
    )
    p3 = _paper("openalex:W3", title="Verifier").model_copy(
        update={"arxiv_id_full": "2401.00002v1", "arxiv_id_base": "2401.00002", "openalex_id": "openalex:W3", "cited_by_count": 20}
    )

    call_count = {"value": 0}

    def fake_fetcher(*, query, from_year, to_year, limit, strategy, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return [p1, p2]
        if call_count["value"] == 2:
            return [p2, p3]
        return []

    papers = seed_reasoning_corpus(tmp_path, fetcher=fake_fetcher, per_query_limit=2)
    stored = {paper.paper_id: paper for paper in read_jsonl(tmp_path / "papers.jsonl", Paper)}
    assert len(papers) == 2
    assert stored["arxiv:2401.00001v2"].corpus_tag == "reasoning-core"
    assert ";" in str(stored["arxiv:2401.00001v2"].seed_reason or "")
    assert stored["arxiv:2401.00001v2"].sync_status == "queued"
    assert stored["arxiv:2401.00001v2"].priority_score > 0


def test_sync_arxiv_corpus_source_builds_text_sections_and_sentences(tmp_path):
    paper = _paper()
    write_jsonl(tmp_path / "papers.jsonl", [paper])
    client = _FakeClient(
        {
            "https://export.arxiv.org/e-print/2401.12345v1": _FakeResponse(_source_tar_bytes()),
        }
    )

    statuses = sync_arxiv_corpus(tmp_path, client=client)

    assert statuses[0].canonical_source == "tex"
    exported = export_corpus_paper(tmp_path, paper.paper_id)
    sections_by_title = {item["title"]: item for item in exported["sections"]}
    section_titles = list(sections_by_title)
    assert "Abstract" in section_titles
    assert "Introduction" in section_titles
    assert "Experimental Setup" in section_titles
    assert sections_by_title["Abstract"]["canonical_type"] == "abstract"
    assert sections_by_title["Introduction"]["canonical_type"] == "introduction"
    assert sections_by_title["Experimental Setup"]["canonical_type"] == "results"
    assert "LogicQA" in exported["text"]["text"]
    assert exported["sentences"]


def test_sync_arxiv_corpus_falls_back_to_html(tmp_path):
    paper = _paper("arxiv:2402.00001v1", title="HTML Reasoning")
    write_jsonl(tmp_path / "papers.jsonl", [paper])
    client = _FakeClient(
        {
            "https://arxiv.org/html/2402.00001v1": _FakeResponse(
                b"<html><body><h1>Introduction</h1><p>Reasoning works in practice.</p><h2>Limitations</h2><p>Open problems remain.</p></body></html>"
            ),
        }
    )

    statuses = sync_arxiv_corpus(tmp_path, client=client)
    exported = export_corpus_paper(tmp_path, paper.paper_id)

    assert statuses[0].canonical_source == "html"
    assert [item["title"] for item in exported["sections"]] == ["Introduction", "Limitations"]
    assert exported["sections"][0]["canonical_type"] == "introduction"
    assert exported["sections"][1]["canonical_type"] == "limitations"
    assert "Reasoning works in practice." in exported["text"]["text"]


def test_sync_arxiv_corpus_falls_back_to_pdf(tmp_path):
    paper = _paper("arxiv:2403.00001v1", title="PDF Reasoning")
    write_jsonl(tmp_path / "papers.jsonl", [paper])
    pdf_blob = b"BT (1 Introduction) Tj (Reasoning improves accuracy.) Tj (2 Limits) Tj (It fails under shift.) Tj ET"
    client = _FakeClient(
        {
            "https://arxiv.org/pdf/2403.00001v1.pdf": _FakeResponse(pdf_blob),
        }
    )

    statuses = sync_arxiv_corpus(tmp_path, client=client)
    exported = export_corpus_paper(tmp_path, paper.paper_id)

    assert statuses[0].canonical_source == "pdf"
    assert exported["sections"][0]["title"] == "Introduction"
    assert exported["sections"][0]["canonical_type"] == "introduction"
    assert "It fails under shift." in exported["text"]["text"]


def test_sync_arxiv_corpus_adds_canonical_section_types_for_alias_titles(tmp_path):
    paper = _paper("arxiv:2405.00001v1", title="Alias Reasoning")
    write_jsonl(tmp_path / "papers.jsonl", [paper])
    html = b"""
        <html><body>
          <h1>Results and Analysis</h1><p>We improve performance.</p>
          <h2>Limitations &amp; Future Work</h2><p>Open issues remain.</p>
          <h2>Conclusion and Future Work</h2><p>We summarize findings.</p>
        </body></html>
    """
    client = _FakeClient({"https://arxiv.org/html/2405.00001v1": _FakeResponse(html)})

    sync_arxiv_corpus(tmp_path, client=client)
    exported = export_corpus_paper(tmp_path, paper.paper_id)
    sections = {item["title"]: item for item in exported["sections"]}

    assert sections["Results and Analysis"]["canonical_type"] == "results"
    assert sections["Results and Analysis"]["canonical_matched_by"] in {"title_exact", "title_substring"}
    assert sections["Limitations & Future Work"]["canonical_type"] == "limitations"
    assert sections["Conclusion and Future Work"]["canonical_type"] == "conclusion"


def test_sync_arxiv_corpus_is_resumable(tmp_path):
    paper = _paper()
    write_jsonl(tmp_path / "papers.jsonl", [paper])
    client = _FakeClient(
        {
            "https://export.arxiv.org/e-print/2401.12345v1": _FakeResponse(_source_tar_bytes()),
        }
    )

    sync_arxiv_corpus(tmp_path, client=client)
    first_calls = list(client.calls)
    sync_arxiv_corpus(tmp_path, client=client)

    assert client.calls == first_calls


def test_sync_arxiv_corpus_selects_high_priority_unfinished_papers(tmp_path):
    high = _paper("arxiv:2406.00002v1", title="High").model_copy(
        update={
            "arxiv_id_full": "2406.00002v1",
            "arxiv_id_base": "2406.00002",
            "priority_score": 0.95,
            "sync_status": "queued",
        }
    )
    low = _paper("arxiv:2406.00001v1", title="Low").model_copy(
        update={
            "arxiv_id_full": "2406.00001v1",
            "arxiv_id_base": "2406.00001",
            "priority_score": 0.10,
            "sync_status": "queued",
        }
    )
    complete = _paper("arxiv:2406.00003v1", title="Done").model_copy(
        update={
            "arxiv_id_full": "2406.00003v1",
            "arxiv_id_base": "2406.00003",
            "priority_score": 0.99,
            "sync_status": "complete",
        }
    )
    write_jsonl(tmp_path / "papers.jsonl", [low, complete, high])
    client = _FakeClient(
        {
            "https://export.arxiv.org/e-print/2406.00002v1": _FakeResponse(_source_tar_bytes()),
        }
    )

    statuses = sync_arxiv_corpus(tmp_path, client=client, limit=1)
    manifest = {paper.paper_id: paper for paper in read_jsonl(tmp_path / "papers.jsonl", Paper)}

    assert [status.paper_id for status in statuses] == ["arxiv:2406.00002v1"]
    assert manifest["arxiv:2406.00002v1"].sync_status == "complete"
    assert manifest["arxiv:2406.00001v1"].sync_status == "queued"
    assert manifest["arxiv:2406.00003v1"].sync_status == "complete"


def test_extract_claims_and_reader_can_use_offline_corpus(tmp_path, monkeypatch):
    paper = _paper("arxiv:2404.00001v1", title="Offline Reasoning")
    paper_root = tmp_path / "artifacts" / "arxiv__2404.00001v1"
    write_json(
        paper_root / "text.json",
        {
            "paper_id": paper.paper_id,
            "text": "RAG improves factual QA on NaturalQuestions by +4.0 EM.",
            "canonical_source": "tex",
            "parser_version": "corpus-v1",
        },
    )
    write_json(
        paper_root / "sections.json",
        [
            {
                "section_id": "sec001",
                "title": "Results",
                "kind": "section",
                "level": 1,
                "parent_id": None,
                "char_start": 0,
                "char_end": 55,
                "source": "tex",
            }
        ],
    )
    write_json(
        paper_root / "sentences.json",
        [
            {
                "sentence_id": "s0001",
                "section_id": "sec001",
                "text": "RAG improves factual QA on NaturalQuestions by +4.0 EM.",
                "char_start": 0,
                "char_end": 55,
                "sentence_index": 0,
                "section_sentence_index": 0,
            }
        ],
    )
    monkeypatch.setenv("AIGRAPH_CORPUS_ROOT", str(tmp_path))

    candidates = read_paper_candidates(paper, mode="heuristic")
    claims = extract_claims([paper], extractor=RuleBasedExtractor(), reader_mode="heuristic")
    summary = validate_corpus(tmp_path)

    assert candidates.candidates[0].section_title == "Results"
    assert claims[0].claim_text.startswith("RAG improves factual QA")
    assert summary["pct_with_sections"] == 1.0
