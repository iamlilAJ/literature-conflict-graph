from aigraph.models import Anomaly, Claim, Hypothesis, Insight, Paper, ScoreBreakdown
from aigraph.overview import (
    _normalize_candidate_text,
    _rewrite_explanation_line,
    build_search_overview,
)


def test_build_search_overview_surfaces_conflicts_bridges_papers_and_reading_path():
    papers = [
        Paper(
            paper_id="p1",
            title="Survey of LLM Finance Forecasting Benchmarks",
            year=2024,
            venue="ACL",
            cited_by_count=50,
            selection_score=0.8,
            selection_reason="survey/benchmark signal",
            retrieval_channel="survey",
            abstract="review benchmark evaluation",
        ),
        Paper(
            paper_id="p2",
            title="Classic Time Series Transformer",
            year=2021,
            venue="AAAI",
            cited_by_count=3000,
            selection_score=0.7,
            selection_reason="strong age-normalized citation impact",
            retrieval_channel="impact",
        ),
    ]
    claims = [
        Claim(claim_id="c001", paper_id="p1", claim_text="LLMs help forecasting.", direction="positive"),
        Claim(claim_id="c002", paper_id="p2", claim_text="Transformers fail under shift.", direction="negative"),
    ]
    anomalies = [
        Anomaly(
            anomaly_id="a001",
            type="impact_conflict",
            central_question="When do LLMs help time-series forecasting?",
            claim_ids=["c001", "c002"],
            positive_claims=["c001"],
            negative_claims=["c002"],
            evidence_impact=2.0,
            topology_score=0.9,
        )
    ]
    insights = [
        Insight(
            insight_id="i001",
            type="unifying_theory",
            title="Finance and time series share regime shift",
            insight="Both communities share non-stationarity.",
            communities=["finance", "time series"],
            shared_concepts=["non-stationarity", "temporal leakage"],
            evidence_papers=["p1"],
            confidence_score=0.8,
            topology_score=0.7,
        )
    ]

    overview = build_search_overview(
        "llm finance time series",
        papers,
        claims,
        anomalies,
        insights,
        selected=[],
        scores={},
    )

    assert "llm finance time series" in overview["headline"]
    assert overview["hero_line"]["line"]
    assert overview["why_this_matters"]["line"]
    assert overview["best_conflict_lines"]
    assert overview["best_bridge_lines"]
    assert overview["top_conflicts"][0]["anomaly_id"] == "a001"
    assert overview["hidden_bridges"][0]["insight_id"] == "i001"
    assert overview["top_papers"][0]["paper_id"] == "p1"
    assert overview["reading_path"][0]["step"] == "Start with the landscape"
    assert overview["why_this_matters"]["next_step"]


def test_build_search_overview_backfills_legacy_arxiv_selection_scores():
    papers = [
        Paper(
            paper_id="arxiv:1",
            title="Large Language Models for Finance Forecasting",
            year=2026,
            venue="arXiv",
            cited_by_count=0,
            abstract="time series forecasting",
        )
    ]
    overview = build_search_overview(
        "llm finance time series forecasting",
        papers,
        claims=[],
        anomalies=[],
        insights=[],
        selected=[],
        scores={},
    )
    card = overview["top_papers"][0]
    assert card["selection_score"] > 0
    assert card["citation_available"] is False
    assert "arXiv has no citation metadata" in card["selection_reason"]


def test_overview_hides_low_score_community_disconnect_and_weak_insight():
    anomalies = [
        Anomaly(
            anomaly_id="a001",
            type="community_disconnect",
            central_question="Could broad communities connect?",
            topology_score=0.1,
        )
    ]
    insights = [
        Insight(
            insight_id="i001",
            type="community_disconnect",
            title="Weak bridge",
            insight="Weak signal.",
            confidence_score=0.1,
            topology_score=0.1,
        )
    ]
    overview = build_search_overview(
        "scientific literature",
        papers=[],
        claims=[],
        anomalies=anomalies,
        insights=insights,
        selected=[],
        scores={},
    )
    assert overview["top_conflicts"] == []
    assert overview["hidden_bridges"] == []


def test_overview_curated_lines_clean_up_baseline_inflation_and_drop_other():
    anomaly = Anomaly(
        anomaly_id="a001",
        type="benchmark_inconsistency",
        central_question="When does RAG help on scientific QA, and when does it fail?",
        claim_ids=["c001"],
        shared_entities={"method": "RAG", "task": "scientific QA"},
        positive_claims=["c001"],
        negative_claims=["c002"],
        topology_score=0.8,
    )
    hypotheses = [
        Hypothesis(
            hypothesis_id="h001",
            anomaly_id="a001",
            hypothesis="Gains attributed to the method are inflated when compared against weak baselines; the sign of the effect depends primarily on baseline choice.",
            mechanism="Weak baselines leave more headroom.",
        ),
        Hypothesis(
            hypothesis_id="h002",
            anomaly_id="a001",
            hypothesis="Gains attributed to other are inflated when compared against weak baselines; the sign of the effect depends primarily on baseline choice.",
            mechanism="Weak baselines leave more headroom.",
        ),
    ]
    overview = build_search_overview(
        "rag scientific qa",
        papers=[],
        claims=[],
        anomalies=[anomaly],
        insights=[],
        selected=hypotheses,
        scores={
            "h001": ScoreBreakdown(hypothesis_id="h001", utility=0.9),
            "h002": ScoreBreakdown(hypothesis_id="h002", utility=0.9),
        },
    )
    lines = [item["line"] for item in overview["best_explanation_lines"]]
    assert "Weak baselines can make RAG look stronger than it is." in lines
    assert not any("other" in line.lower() for line in lines)


def test_normalize_candidate_text_drops_unresolved_other_and_repairs_placeholders():
    anomaly = Anomaly(
        anomaly_id="a001",
        type="benchmark_inconsistency",
        central_question="When does RAG help?",
        shared_entities={"method": "RAG", "task": "domain QA"},
    )
    assert _normalize_candidate_text("Gains attributed to other are inflated.") == ""
    repaired = _normalize_candidate_text("Gains attributed to the method are inflated on the task.", anomaly=anomaly)
    assert repaired == "Gains attributed to RAG are inflated on domain QA."


def test_rewrite_explanation_line_preserves_core_claim():
    anomaly = Anomaly(
        anomaly_id="a001",
        type="benchmark_inconsistency",
        central_question="When does RAG help?",
        shared_entities={"method": "RAG", "task": "multi-hop QA"},
    )
    hypothesis = Hypothesis(
        hypothesis_id="h001",
        anomaly_id="a001",
        hypothesis="High top-k retrieval injects distractor passages that overwhelm the generator on multi-hop-QA, flipping the effect of RAG from positive to negative.",
        mechanism="Distractors swamp evidence use.",
    )
    line = _rewrite_explanation_line(hypothesis, anomaly)
    assert "distractor" in line.lower()
    assert "retrieval" in line.lower() or "multi-hop" in line.lower()


def test_overview_marks_hypothesis_only_runs_as_exploratory_and_hides_bridge_conflicts():
    papers = [
        Paper(
            paper_id="p1",
            title="VBMO for Path Planning",
            year=2024,
            venue="arXiv",
            selection_score=0.7,
            selection_reason="topical",
            retrieval_channel="arxiv-balanced",
        )
    ]
    anomalies = [
        Anomaly(
            anomaly_id="a001",
            type="bridge_opportunity",
            central_question="Could the effect on UAV path planning with BPMO-UAVPP transfer to multi-objective path planning with VBMO?",
            claim_ids=["c001", "c002"],
            positive_claims=["c001"],
            negative_claims=["c002"],
            shared_entities={"method_from": "BPMO-UAVPP", "method_to": "VBMO"},
            topology_score=0.8,
        )
    ]
    selected = [
        Hypothesis(
            hypothesis_id="h001",
            anomaly_id="a001",
            hypothesis="An unreported moderator variable drives the conflicting results around BPMO-UAVPP on path planning.",
            minimal_test="Replay both path planning methods in a common harness.",
        )
    ]
    overview = build_search_overview(
        "path planning",
        papers,
        claims=[],
        anomalies=anomalies,
        insights=[],
        selected=selected,
        scores={"h001": ScoreBreakdown(hypothesis_id="h001", utility=0.8)},
    )
    assert overview["hero_line"] is None
    assert overview["top_conflicts"] == []
    assert "exploratory" in overview["why_this_matters"]["line"].lower()
    assert overview["best_explanation_lines"][0]["supporting_text"].startswith("2 claims")
