"""Unit tests for the method-novelty gate (src/aigraph/novelty_gate.py).

The gate is non-frozen and flags method proposals that are confidently judged
NOT novel against arXiv prior art. These tests pin the opt-in/fail-open
contract, the cached sidecar round-trip, the is_novel three-state handling
(False=flag, True/None=no-flag), and the 0-LLM overlay — without arXiv or an LLM
(the prior-art check is injected).
"""
import json

import aigraph.novelty_gate as ng
from aigraph.models import Hypothesis


def _run(tmp_path, hyps, fname="creator_hypotheses.jsonl"):
    (tmp_path / fname).write_text("\n".join(json.dumps(r) for r in hyps), encoding="utf-8")
    return tmp_path


def _hyp(hid="h1", hypothesis="Confidence-Gated DPO"):
    return {"hypothesis_id": hid, "anomaly_id": "a1", "hypothesis": hypothesis,
            "mechanism": "gate DPO updates by confidence", "explains_claims": ["c1"],
            "predictions": ["p1"], "minimal_test": "t"}


def _enable(monkeypatch):
    monkeypatch.setenv("AIGRAPH_NOVELTY_GATE", "1")
    monkeypatch.setattr(ng, "configured_api_key", lambda *a, **k: "sk-x")
    monkeypatch.setattr(ng, "configured_model", lambda *a, **k: "m")
    monkeypatch.setattr(ng, "build_openai_client", lambda *a, **k: object())
    monkeypatch.setattr(ng, "ARXIV_RATE_LIMIT_SECONDS", 0)  # no sleeps in tests


def _verdict(is_novel, papers=(), rationale="r"):
    return lambda *a, **k: {"is_novel": is_novel,
                            "similar_papers": [{"title": t} for t in papers],
                            "rationale": rationale}


# --- enabled / fail-open ---------------------------------------------------- #

def test_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AIGRAPH_NOVELTY_GATE", raising=False)
    monkeypatch.setattr(ng, "configured_api_key", lambda *a, **k: "sk-x")
    assert ng.novelty_gate_enabled() is False
    _run(tmp_path, [_hyp()])
    assert ng.novelty_run(tmp_path) == {}


def test_no_key_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("AIGRAPH_NOVELTY_GATE", "1")
    monkeypatch.setattr(ng, "configured_api_key", lambda *a, **k: None)
    assert ng.novelty_gate_enabled() is False
    _run(tmp_path, [_hyp()])
    assert ng.novelty_run(tmp_path) == {}


# --- happy path + three-state is_novel -------------------------------------- #

def test_flags_non_novel_method(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ng, "check_hypothesis_novelty",
                        _verdict(False, papers=["Gated DPO (2024)"], rationale="already exists"))
    _run(tmp_path, [_hyp()])
    out = ng.novelty_run(tmp_path)
    assert out["h1"]["is_novel"] is False
    assert out["h1"]["similar_papers"][0]["title"] == "Gated DPO (2024)"
    assert (tmp_path / ng.NOVELTY_FILENAME).exists()


def test_novel_method_cached_not_flagged(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ng, "check_hypothesis_novelty", _verdict(True))
    _run(tmp_path, [_hyp()])
    out = ng.novelty_run(tmp_path)
    assert out["h1"]["is_novel"] is True
    hyp = Hypothesis(hypothesis_id="h1", anomaly_id="a1", hypothesis="x")
    assert ng.apply_novelty([hyp], tmp_path) == 0  # True -> not flagged
    assert hyp.novelty_flag is None


def test_unknown_is_not_flagged(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ng, "check_hypothesis_novelty", _verdict(None, rationale="arxiv down"))
    _run(tmp_path, [_hyp()])
    ng.novelty_run(tmp_path)
    hyp = Hypothesis(hypothesis_id="h1", anomaly_id="a1", hypothesis="x")
    assert ng.apply_novelty([hyp], tmp_path) == 0  # None -> unknown, not flagged
    assert hyp.novelty_flag is None


def test_cache_skips_second_pass(tmp_path, monkeypatch):
    _enable(monkeypatch)
    calls = {"n": 0}

    def _count(*a, **k):
        calls["n"] += 1
        return {"is_novel": False, "similar_papers": [], "rationale": "r"}

    monkeypatch.setattr(ng, "check_hypothesis_novelty", _count)
    _run(tmp_path, [_hyp()])
    ng.novelty_run(tmp_path)
    ng.novelty_run(tmp_path)
    assert calls["n"] == 1


def test_limit_caps_new_calls(tmp_path, monkeypatch):
    _enable(monkeypatch)
    calls = {"n": 0}
    monkeypatch.setattr(ng, "check_hypothesis_novelty",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                         {"is_novel": True, "similar_papers": [], "rationale": "r"})[1])
    _run(tmp_path, [_hyp(hid=f"h{i}") for i in range(5)])
    ng.novelty_run(tmp_path, limit=2)
    assert calls["n"] == 2


def test_apply_novelty_overlays_collision(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ng, "check_hypothesis_novelty",
                        _verdict(False, papers=["Confidence-gated preference opt (2023)"],
                                 rationale="same idea"))
    _run(tmp_path, [_hyp()])
    ng.novelty_run(tmp_path)
    hyp = Hypothesis(hypothesis_id="h1", anomaly_id="a1", hypothesis="x")
    n = ng.apply_novelty([hyp], tmp_path)
    assert n == 1
    assert hyp.novelty_flag["is_novel"] is False
    assert "Confidence-gated" in hyp.novelty_flag["similar_papers"][0]["title"]


def test_apply_novelty_noop_without_sidecar(tmp_path):
    hyp = Hypothesis(hypothesis_id="h1", anomaly_id="a1", hypothesis="x")
    assert ng.apply_novelty([hyp], tmp_path) == 0
    assert hyp.novelty_flag is None


def test_missing_creator_file_is_noop(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ng, "check_hypothesis_novelty", _verdict(False))
    # no creator_hypotheses.jsonl present
    assert ng.novelty_run(tmp_path) == {}
