from aigraph.models import Anomaly, Claim, Hypothesis, Paper, ScoreBreakdown
from aigraph.report import render_report


def test_render_report_separates_bridge_opportunities_and_marks_exploratory():
    claims = [
        Claim(claim_id="c001", paper_id="p1", claim_text="A conflict claim.", direction="positive"),
        Claim(claim_id="c002", paper_id="p2", claim_text="A bridge claim.", direction="negative"),
    ]
    anomalies = [
        Anomaly(
            anomaly_id="a001",
            type="impact_conflict",
            central_question="Why do the papers disagree?",
            claim_ids=["c001"],
            positive_claims=["c001"],
            negative_claims=["c002"],
        ),
        Anomaly(
            anomaly_id="a002",
            type="bridge_opportunity",
            central_question="Could the effect transfer across path planning settings?",
            claim_ids=["c002"],
            positive_claims=["c002"],
            negative_claims=[],
        ),
    ]
    selected = [
        Hypothesis(hypothesis_id="h001", anomaly_id="a001", hypothesis="Conflict explanation.", explains_claims=["c001"]),
        Hypothesis(hypothesis_id="h002", anomaly_id="a002", hypothesis="Bridge explanation.", explains_claims=["c002"]),
    ]
    scores = {
        "h001": ScoreBreakdown(hypothesis_id="h001", utility=0.8),
        "h002": ScoreBreakdown(hypothesis_id="h002", utility=0.7),
    }
    papers = {
        "p1": Paper(paper_id="p1", title="Conflict Paper", year=2024, venue="ACL"),
        "p2": Paper(paper_id="p2", title="Bridge Paper", year=2024, venue="ACL"),
    }

    report = render_report(selected, anomalies, claims, scores, paper_lookup=papers, insights=[])

    assert "Exploratory report" in report
    assert "## Conflict Hypotheses" in report
    assert "## Bridge Opportunities" in report
    assert "**Transfer question:** Could the effect transfer across path planning settings?" in report
