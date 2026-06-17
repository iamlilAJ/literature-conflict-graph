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


def _run_with(tmp_path, n_sub, n_weak):
    """A run with n_sub substantive (impact_conflict) + n_weak weak
    (community_disconnect) hypotheses, all equal topic relevance."""
    hyps, anoms, claims, papers = [], [], [], []
    for i in range(n_sub):
        hyps.append(Hypothesis(hypothesis_id=f"hs{i}", anomaly_id=f"as{i}",
                    hypothesis="reasoning method study", predictions=["p"],
                    minimal_test="t", explains_claims=[f"cs{i}"]))
        anoms.append(Anomaly(anomaly_id=f"as{i}", type="impact_conflict",
                     central_question="reasoning conflict", claim_ids=[f"cs{i}"]))
        claims.append(Claim(claim_id=f"cs{i}", paper_id=f"ps{i}", claim_text="reasoning result"))
        papers.append(Paper(paper_id=f"ps{i}", title="S", year=2024, venue="x"))
    for i in range(n_weak):
        hyps.append(Hypothesis(hypothesis_id=f"hw{i}", anomaly_id=f"aw{i}",
                    hypothesis="reasoning method study", predictions=["p"],
                    minimal_test="t", explains_claims=[f"cw{i}"]))
        anoms.append(Anomaly(anomaly_id=f"aw{i}", type="community_disconnect",
                     central_question="reasoning conflict", claim_ids=[f"cw{i}"]))
        claims.append(Claim(claim_id=f"cw{i}", paper_id=f"pw{i}", claim_text="reasoning result"))
        papers.append(Paper(paper_id=f"pw{i}", title="W", year=2024, venue="x"))
    return _write_run(tmp_path, hyps, anoms, claims, papers)


def test_is_weak_anomaly_helper():
    aw = Anomaly(anomaly_id="aw", type="community_disconnect", central_question="q")
    as_ = Anomaly(anomaly_id="as", type="impact_conflict", central_question="q")
    lut = {"aw": aw, "as": as_}
    assert _is_weak_anomaly(Hypothesis(hypothesis_id="h", anomaly_id="aw", hypothesis="x"), lut) is True
    assert _is_weak_anomaly(Hypothesis(hypothesis_id="h", anomaly_id="as", hypothesis="x"), lut) is False


def test_demotion_excludes_weak_when_enough_substantive(tmp_path):
    # 3 substantive >= k=3 → pool restricted to substantive; select_mmr (which
    # would otherwise re-rank weak bridges back in by utility) never sees them.
    run = _run_with(tmp_path, n_sub=3, n_weak=3)
    recs, _ = query_records(run, "reasoning", k=3, max_hypotheses=10, min_anomalies=1,
                            drop_self_conflict=False, demote_weak_anomalies=True)
    types = [r["anomaly_type"] for r in recs]
    assert len(recs) == 3
    assert "community_disconnect" not in types  # all substantive


def test_demotion_keeps_weak_when_substantive_scarce(tmp_path):
    # only 1 substantive < k=3 → pool keeps the weak bridges so top-k fills.
    run = _run_with(tmp_path, n_sub=1, n_weak=3)
    recs, _ = query_records(run, "reasoning", k=3, max_hypotheses=10, min_anomalies=1,
                            drop_self_conflict=False, demote_weak_anomalies=True)
    types = [r["anomaly_type"] for r in recs]
    assert len(recs) == 3
    assert "community_disconnect" in types  # weak needed to reach k


def test_no_demotion_does_not_restrict(tmp_path):
    # with the flag off, the pool is not restricted (weak bridges remain eligible).
    run = _run_with(tmp_path, n_sub=3, n_weak=3)
    recs_off, _ = query_records(run, "reasoning", k=6, max_hypotheses=10, min_anomalies=1,
                                drop_self_conflict=False, demote_weak_anomalies=False)
    # all 6 (3 sub + 3 weak) are eligible and selected when k=6
    assert len(recs_off) == 6
    assert any(r["anomaly_type"] == "community_disconnect" for r in recs_off)
