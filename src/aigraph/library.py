"""Curated homepage demos and quick-start seed topics."""

from __future__ import annotations

from urllib.parse import quote


def search_prefill_href(
    topic: str,
    *,
    source: str | None = None,
    strategy: str | None = None,
    citation_weight: float | None = None,
    min_relevance: float | None = None,
) -> str:
    params = [("topic", topic)]
    if source:
        params.append(("source", source))
    if strategy:
        params.append(("strategy", strategy))
    if citation_weight is not None:
        params.append(("citation_weight", f"{citation_weight:.2f}"))
    if min_relevance is not None:
        params.append(("min_relevance", f"{min_relevance:.2f}"))
    query = "&".join(f"{key}={quote(str(value))}" for key, value in params)
    return f"/?{query}"


def curated_demos() -> list[dict[str, str | float]]:
    return [
        {
            "slug": "finance-timeseries",
            "title": "Finance + Stock Prediction",
            "topic": "llm finance stock movement prediction time series forecasting",
            "blurb": "Stock movement prediction, financial forecasting, and when setup changes flip the reported gain.",
            "source": "arxiv",
            "strategy": "balanced",
            "citation_weight": 0.45,
            "min_relevance": 0.45,
            "preferred_run_id": "20260417-170322-29ff93",
            "pinned_line": "Financial prediction gains can flip from win to failure once the setup changes.",
        },
        {
            "slug": "alignment-rlhf",
            "title": "Alignment + RLHF",
            "topic": "reinforcement learning llm",
            "blurb": "Policy optimization, safety alignment, and where reward signals flip from help to harm.",
            "source": "arxiv",
            "strategy": "balanced",
            "citation_weight": 0.45,
            "min_relevance": 0.30,
            "preferred_run_id": "20260417-103728-44ad0f",
            "pinned_line": "Safety alignment can flip from win to failure depending on the setup.",
        },
        {
            "slug": "bayesian-design",
            "title": "Bayesian Experimental Design",
            "topic": "bayesian experimental design language model",
            "blurb": "Adaptive experiment planning, evaluation setup, and where apparent gains depend on how the study is framed.",
            "source": "arxiv",
            "strategy": "balanced",
            "citation_weight": 0.45,
            "min_relevance": 0.30,
            "preferred_run_id": "20260417-142447-57408c",
            "pinned_line": "Evaluation setup can flip an apparent experimental win into a fragile result.",
        },
    ]


def specialty_libraries() -> list[dict[str, str | float]]:
    return [
        {
            "slug": "finance-timeseries-seed",
            "title": "Finance + Time Series",
            "topic": "large language models finance time series forecasting",
            "blurb": "Forecasting, regime shift, temporal leakage, factor extraction.",
            "source": "openalex",
            "strategy": "balanced",
            "citation_weight": 0.55,
            "min_relevance": 0.35,
        },
        {
            "slug": "rag-science-seed",
            "title": "RAG + Scientific QA",
            "topic": "multimodal rag scientific literature",
            "blurb": "Retrieval, evidence grounding, scientific claim verification.",
            "source": "openalex",
            "strategy": "balanced",
            "citation_weight": 0.45,
            "min_relevance": 0.35,
        },
        {
            "slug": "alignment-rlhf-seed",
            "title": "Alignment + RLHF",
            "topic": "dpo ppo rlhf alignment safety",
            "blurb": "Preference learning, policy optimization, reward hacking.",
            "source": "arxiv",
            "strategy": "balanced",
            "citation_weight": 0.45,
            "min_relevance": 0.30,
        },
        {
            "slug": "medical-rag-seed",
            "title": "Medical RAG",
            "topic": "rag hallucination medical qa",
            "blurb": "Domain QA, hallucination reduction, grounding in high-risk settings.",
            "source": "openalex",
            "strategy": "balanced",
            "citation_weight": 0.55,
            "min_relevance": 0.35,
        },
    ]


def library_href(topic: str) -> str:
    return search_prefill_href(topic)
