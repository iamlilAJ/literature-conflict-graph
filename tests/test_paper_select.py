from aigraph.models import Paper
from aigraph.paper_select import (
    dedupe_papers,
    infer_paper_role,
    normalize_topic_query,
    normalized_title,
    select_representative_papers,
)


def _paper(paper_id: str, title: str, year: int, cited_by_count: int = 0, abstract: str = "") -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        year=year,
        venue="test",
        cited_by_count=cited_by_count,
        abstract=abstract,
        text=f"{title}\n\n{abstract}",
    )


def test_normalized_title_and_dedupe_by_id_doi_title():
    assert normalized_title("LLMs for Finance: A Survey!") == "llms for finance a survey"
    p1 = _paper("openalex:W1", "LLMs for Finance", 2022, 5)
    p2 = _paper("openalex:W1", "LLMs for Finance", 2024, 10)
    p3 = _paper("openalex:W3", "Different Title", 2023, 1)
    p4 = _paper("openalex:W4", "Another Title", 2021, 0)
    p3.doi = "https://doi.org/10.1/x"
    p4.doi = "doi:10.1/x"
    deduped = dedupe_papers([p1, p2, p3, p4])
    assert [p.paper_id for p in deduped] == ["openalex:W1", "openalex:W3"]
    assert deduped[0].cited_by_count == 10


def test_balanced_selection_keeps_recent_relevant_and_impact_mix():
    papers = [
        _paper("p-old", "Foundational Neural Networks", 2016, 2000, "general methods"),
        _paper("p-new", "Large Language Models for Finance Forecasting", 2026, 5, "time series"),
        _paper("p-survey", "Survey of LLM Finance Time Series Benchmarks", 2024, 30, "review benchmark"),
    ]
    selected = select_representative_papers(
        papers,
        "llm finance time series forecasting",
        limit=2,
        strategy="balanced",
        current_year=2026,
    )
    ids = {p.paper_id for p in selected}
    assert "p-old" not in ids
    assert {"p-new", "p-survey"} <= ids
    assert all(p.selection_score > 0 for p in selected)


def test_high_impact_and_recent_strategies_sort_differently():
    old_cited = _paper("old", "Classic Forecasting", 2018, 3000)
    fresh = _paper("fresh", "Fresh Forecasting", 2026, 1)
    medium = _paper("medium", "Medium Forecasting", 2024, 50)
    high_impact = select_representative_papers(
        [fresh, medium, old_cited],
        "forecasting",
        limit=1,
        strategy="high-impact",
        current_year=2026,
    )
    recent = select_representative_papers(
        [old_cited, medium, fresh],
        "forecasting",
        limit=1,
        strategy="recent",
        current_year=2026,
    )
    assert high_impact[0].paper_id == "old"
    assert recent[0].paper_id == "fresh"


def test_balanced_selection_can_make_citations_primary():
    old_cited = _paper("old", "Classic Finance Time Series Models", 2018, 3000, "forecasting")
    fresh_relevant = _paper("fresh", "LLM Finance Time Series Forecasting", 2026, 2, "forecasting")
    selected_default = select_representative_papers(
        [old_cited, fresh_relevant],
        "llm finance time series forecasting",
        limit=1,
        strategy="balanced",
        current_year=2026,
    )
    selected_citation = select_representative_papers(
        [old_cited, fresh_relevant],
        "llm finance time series forecasting",
        limit=1,
        strategy="balanced",
        current_year=2026,
        citation_weight=0.75,
    )
    assert selected_default[0].paper_id == "fresh"
    assert selected_citation[0].paper_id == "old"


def test_high_citation_off_topic_paper_must_pass_relevance_gate():
    off_topic = _paper(
        "off",
        "Global Burden of Disease in Healthcare",
        2024,
        10000,
        "clinical practice health policy epidemiology",
    )
    on_topic = _paper(
        "on",
        "Large Language Models for Finance Time Series Forecasting",
        2025,
        20,
        "forecast horizon non-stationarity financial forecasting",
    )
    selected = select_representative_papers(
        [off_topic, on_topic],
        "large language models finance time series forecasting",
        limit=1,
        strategy="balanced",
        current_year=2026,
        citation_weight=0.80,
        min_relevance=0.30,
    )
    assert selected[0].paper_id == "on"


def test_normalize_topic_query_expands_broad_research_phrases():
    normalized = normalize_topic_query("llm rag finance time-series")
    assert "large language models" in normalized
    assert "retrieval augmented generation" in normalized
    assert "finance" in normalized
    assert "time series" in normalized


def test_infer_paper_role_handles_survey_benchmark_dataset_failure_and_industry():
    assert infer_paper_role("A Survey of Finance LLMs", "")["role"] == "survey"
    assert infer_paper_role("Benchmarking Medical RAG Faithfulness", "")["role"] == "benchmark"
    assert infer_paper_role("MedQA-Shift: A Dataset for Clinical QA", "")["role"] == "dataset"
    assert infer_paper_role("Revisiting RAG Hallucination in Medical QA", "")["role"] == "failure"
    assert infer_paper_role("Production Deployment of Retrieval-Augmented Agents", "")["role"] == "industry"


def test_infer_paper_role_prefers_benchmark_for_rethinking_benchmark_titles():
    info = infer_paper_role("Rethinking Benchmarking for Scientific Claim Verification", "")
    assert info["role"] == "benchmark"


def test_balanced_selection_prefers_role_diversity_when_candidates_exist():
    papers = [
        _paper("survey", "A Survey of Medical RAG Evaluation", 2024, 80, "survey review benchmark"),
        _paper("benchmark", "Benchmarking Faithfulness in Medical RAG", 2025, 30, "evaluation benchmark"),
        _paper("failure", "Revisiting RAG Hallucination in Medical QA", 2026, 18, "limitations robustness"),
        _paper("dataset", "MedFaith: A Dataset for Medical Faithfulness", 2025, 10, "dataset resource"),
        _paper("method", "A New Medical RAG Framework", 2026, 8, "framework improves retrieval"),
    ]
    selected = select_representative_papers(
        papers,
        "hallucination detection and faithfulness evaluation in rag for medical qa",
        limit=4,
        strategy="balanced",
        current_year=2026,
    )
    roles = {p.paper_role for p in selected}
    assert "method" in roles
    assert "failure" in roles
    assert roles & {"survey", "benchmark"}


def test_decompose_query_adds_role_seeking_variants_for_harder_topics():
    from aigraph.paper_select import decompose_topic_query

    plan = decompose_topic_query("hallucination detection and faithfulness evaluation in rag for medical qa")
    variants = " | ".join(plan["retrieval_variants"])
    assert "survey review" in variants
    assert "benchmark evaluation" in variants
    assert "limitation failure robustness" in variants
    assert "dataset corpus resource" in variants


def test_second_stage_cleanup_filters_broad_but_off_topic_candidates():
    weak = _paper("weak", "Large Language Models in Healthcare", 2025, 1200, "medical diagnosis patient outcomes")
    strong = _paper("strong", "Large Language Models for Finance Time Series Forecasting", 2025, 50, "non-stationarity forecast horizon temporal leakage")
    selected = select_representative_papers(
        [weak, strong],
        "large language models finance time series forecasting",
        limit=1,
        strategy="balanced",
        current_year=2026,
        citation_weight=0.65,
        min_relevance=0.25,
    )
    assert selected[0].paper_id == "strong"
