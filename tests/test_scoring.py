from aigraph.models import Anomaly, Claim, GraphBridge, Hypothesis, Setting
from aigraph.scoring import (
    cost_penalty as _cost_penalty,
    score_hypothesis as _score_hypothesis,
    select_mmr as _select_mmr,
    score_all as _score_all,
    testability_score as _testability_score,
)


def _claim(cid: str, direction: str, text: str) -> Claim:
    return Claim(
        claim_id=cid,
        paper_id=f"p{cid[1:]}",
        claim_text=text,
        method="RAG",
        task="factual QA",
        dataset="NaturalQuestions",
        metric="EM",
        direction=direction,
        setting=Setting(top_k="5"),
    )


def _anomaly(cids: list[str], pos: list[str], neg: list[str], anomaly_id: str = "a001") -> Anomaly:
    return Anomaly(
        anomaly_id=anomaly_id,
        type="benchmark_inconsistency",
        central_question="Q?",
        claim_ids=cids,
        positive_claims=pos,
        negative_claims=neg,
        shared_entities={"method": "RAG", "task": "factual QA"},
    )


def _hyp(
    hid: str,
    text: str,
    predictions: list[str],
    mechanism: str = "",
    anomaly_id: str = "a001",
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hid,
        anomaly_id=anomaly_id,
        hypothesis=text,
        mechanism=mechanism,
        explains_claims=["c001", "c002"],
        predictions=predictions,
        minimal_test="Run experiment X on dataset Y.",
        scope_conditions={"top_k": "5"},
        evidence_gap="Open question.",
        graph_bridge=GraphBridge(**{"from": "RAG", "to": "factual QA"}),
    )


def test_testability_rewards_predictions_and_minimal_test():
    h = _hyp("h001", "x", ["p1", "p2"])
    assert _testability_score(h) == 1.0
    h_no_test = Hypothesis(
        hypothesis_id="h002",
        anomaly_id="a001",
        hypothesis="x",
        predictions=["p1"],
        minimal_test="",
    )
    assert _testability_score(h_no_test) == 0.0


def test_cost_penalty_flags_vague_phrases():
    h = _hyp("h001", "The method may possibly help under complex interactions.", ["p1", "p2"])
    assert _cost_penalty(h) > 0


def test_utility_rewards_pos_neg_explain_balance():
    claims = [_claim("c001", "positive", "RAG improves NQ."), _claim("c002", "negative", "RAG hurts NQ.")]
    a = _anomaly(["c001", "c002"], ["c001"], ["c002"])
    peers = [_hyp("h001", "Concrete predictive hypothesis about retrieval noise.", ["a", "b"])]
    score = _score_hypothesis(peers[0], a, {c.claim_id: c for c in claims}, peers)
    assert score.explain > 0
    assert score.utility > 0


def test_select_mmr_diversifies():
    claims = [_claim("c001", "positive", "RAG improves NQ."), _claim("c002", "negative", "RAG hurts NQ.")]
    a = _anomaly(["c001", "c002"], ["c001"], ["c002"])
    h1 = _hyp("h001", "Retrieval noise drives the flip.", ["p1 retrieval", "p2 retrieval"])
    h2 = _hyp("h002", "Retrieval noise drives the flip.", ["p1 retrieval", "p2 retrieval"])  # duplicate
    h3 = _hyp("h003", "Metric sensitivity drives the apparent reversal.", ["rescore with evidence F1", "compare rankings"])
    hyps = [h1, h2, h3]
    scores = _score_all(hyps, [a], claims)
    selected = _select_mmr(hyps, scores, k=2, lambda_=0.7)
    ids = {h.hypothesis_id for h in selected}
    # MMR should avoid picking both duplicates.
    assert "h003" in ids
    assert not ({"h001", "h002"}.issubset(ids))


def test_select_mmr_can_enforce_anomaly_coverage():
    claims = [_claim("c001", "positive", "RAG improves NQ."), _claim("c002", "negative", "RAG hurts NQ.")]
    a1 = _anomaly(["c001", "c002"], ["c001"], ["c002"], anomaly_id="a001")
    a2 = _anomaly(["c001", "c002"], ["c001"], ["c002"], anomaly_id="a002")
    h1 = _hyp("h001", "Highest utility retrieval noise hypothesis.", ["p1 retrieval", "p2 retrieval"], anomaly_id="a001")
    h2 = _hyp("h002", "Similar high utility retrieval noise hypothesis.", ["p1 retrieval", "p2 retrieval"], anomaly_id="a001")
    h3 = _hyp("h003", "Different anomaly metric hypothesis.", ["rescore with evidence F1", "compare rankings"], anomaly_id="a002")
    hyps = [h1, h2, h3]
    scores = _score_all(hyps, [a1, a2], claims)
    selected = _select_mmr(hyps, scores, k=2, lambda_=0.7, min_anomalies=2)
    assert {h.anomaly_id for h in selected} == {"a001", "a002"}
