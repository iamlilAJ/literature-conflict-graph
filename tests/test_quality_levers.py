"""Tests for the default-quality levers (non-frozen):

A. resolve_best_run richness-weighting — a thin leftover corpus must not beat a
   rich same-topic corpus on token overlap alone.
C. _select demote_weak_anomalies — weak community_disconnect bridges rank after
   substantive conflicts (kept, not dropped).

(B, the coverage absolute-floor, is covered in test_run_dashboard.test_coverage_*.)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import idea_cascade as ic  # noqa: E402
from aigraph_query import _is_weak_anomaly, query_records  # noqa: E402
from aigraph.models import Anomaly, Claim, Hypothesis, Paper  # noqa: E402


# --------------------------------------------------------------------------- #
# A. resolve_best_run richness
# --------------------------------------------------------------------------- #

def _mk_run(runs_root: Path, name: str, topic: str, n_hyps: int) -> Path:
    d = runs_root / name
    d.mkdir()
    (d / "query.txt").write_text(topic, encoding="utf-8")
    (d / "claims.jsonl").write_text('{"claim_id":"c1"}\n', encoding="utf-8")  # non-empty
    (d / "hypotheses.jsonl").write_text("".join("{}\n" for _ in range(n_hyps)), encoding="utf-8")
    (d / "status.json").write_text('{"status":"done"}', encoding="utf-8")
    return d


def test_resolve_prefers_rich_over_thin_same_topic(tmp_path):
    # thin: exact topic overlap (1.0) but only 2 hypotheses
    _mk_run(tmp_path, "thin", "graph of thoughts reasoning prompting llm", 2)
    # rich: partial overlap (0.4) but 40 hypotheses
    _mk_run(tmp_path, "rich", "graph reasoning", 40)
    best = ic.resolve_best_run(tmp_path, "graph of thoughts reasoning prompting llm")
    assert best == "rich"  # richness-weighted overlap beats thin exact-match


def test_resolve_falls_back_to_thin_when_nothing_richer(tmp_path):
    _mk_run(tmp_path, "thin", "graph of thoughts reasoning", 2)
    best = ic.resolve_best_run(tmp_path, "graph of thoughts reasoning")
    assert best == "thin"  # still returns the only match (build decision is the caller's)


def test_resolve_none_when_below_overlap(tmp_path):
    _mk_run(tmp_path, "other", "quantum chemistry catalysis", 50)
    assert ic.resolve_best_run(tmp_path, "graph of thoughts reasoning") is None


# --------------------------------------------------------------------------- #
# C. _select demote_weak_anomalies
# --------------------------------------------------------------------------- #

def _write_run(tmp_path, hyps, anoms, claims, papers):
    run = tmp_path / "run"
    run.mkdir()
    (run / "hypotheses_scored.jsonl").write_text(
        "\n".join(h.model_dump_json() for h in hyps), encoding="utf-8")
    (run / "anomalies.jsonl").write_text(
        "\n".join(a.model_dump_json() for a in anoms), encoding="utf-8")
    (run / "claims.jsonl").write_text(
        "\n".join(c.model_dump_json() for c in claims), encoding="utf-8")
    (run / "papers.jsonl").write_text(
        "\n".join(p.model_dump_json() for p in papers), encoding="utf-8")
    return run


def _weak_then_strong_run(tmp_path):
    # weak (community_disconnect) listed FIRST, substantive (impact_conflict)
    # second; equal topic relevance so only the demotion flag breaks the tie.
    hyps = [
        Hypothesis(hypothesis_id="hw", anomaly_id="aw", hypothesis="reasoning method study",
                   predictions=["p"], minimal_test="t", explains_claims=["c1"]),
        Hypothesis(hypothesis_id="hs", anomaly_id="as", hypothesis="reasoning method study",
                   predictions=["p"], minimal_test="t", explains_claims=["c2"]),
    ]
    anoms = [
        Anomaly(anomaly_id="aw", type="community_disconnect",
                central_question="reasoning conflict", claim_ids=["c1"]),
        Anomaly(anomaly_id="as", type="impact_conflict",
                central_question="reasoning conflict", claim_ids=["c2"]),
    ]
    claims = [Claim(claim_id="c1", paper_id="p1", claim_text="reasoning result"),
              Claim(claim_id="c2", paper_id="p2", claim_text="reasoning result")]
    papers = [Paper(paper_id="p1", title="A", year=2024, venue="x"),
              Paper(paper_id="p2", title="B", year=2024, venue="x")]
    return _write_run(tmp_path, hyps, anoms, claims, papers)


def test_is_weak_anomaly_helper(tmp_path):
    aw = Anomaly(anomaly_id="aw", type="community_disconnect", central_question="q")
    as_ = Anomaly(anomaly_id="as", type="impact_conflict", central_question="q")
    lut = {"aw": aw, "as": as_}
    hw = Hypothesis(hypothesis_id="h", anomaly_id="aw", hypothesis="x")
    hs = Hypothesis(hypothesis_id="h", anomaly_id="as", hypothesis="x")
    assert _is_weak_anomaly(hw, lut) is True
    assert _is_weak_anomaly(hs, lut) is False


def test_demotion_surfaces_substantive_conflict_first(tmp_path):
    run = _weak_then_strong_run(tmp_path)
    # max_hypotheses=1 truncates the candidate pool to the top-ranked match.
    recs, _ = query_records(run, "reasoning", k=1, max_hypotheses=1, min_anomalies=1,
                            drop_self_conflict=False, demote_weak_anomalies=True)
    assert len(recs) == 1
    assert recs[0]["anomaly_type"] == "impact_conflict"  # weak bridge demoted out


def test_no_demotion_keeps_input_order(tmp_path):
    run = _weak_then_strong_run(tmp_path)
    recs, _ = query_records(run, "reasoning", k=1, max_hypotheses=1, min_anomalies=1,
                            drop_self_conflict=False, demote_weak_anomalies=False)
    assert len(recs) == 1
    assert recs[0]["anomaly_type"] == "community_disconnect"  # default order unchanged
