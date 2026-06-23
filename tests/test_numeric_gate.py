"""Unit tests for the numeric-grounding gate (src/aigraph/numeric_gate.py).

The gate is non-frozen and flags past-result numbers a delivered hypothesis
asserts that are not supported by its cited claims. These tests pin the
fail-open contract, JSON parsing, the cached sidecar round-trip, the
enriched-text-preference, and the 0-LLM overlay — all without a real LLM.
"""
import json

import aigraph.numeric_gate as ng
from aigraph.models import Anomaly, Claim, Hypothesis


def _run(tmp_path, hyps, claims, anoms, enriched=None):
    for name, rows in (("hypotheses", hyps), ("claims", claims), ("anomalies", anoms)):
        (tmp_path / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    if enriched is not None:
        (tmp_path / "hypotheses_enriched.jsonl").write_text(
            "\n".join(json.dumps(r) for r in enriched), encoding="utf-8")
    return tmp_path


def _hyp(hid="h1", aid="a1", claims=("c1",), hypothesis="SPARC-RAG cuts cost 52.2%"):
    return {"hypothesis_id": hid, "anomaly_id": aid, "hypothesis": hypothesis,
            "mechanism": "m", "explains_claims": list(claims),
            "predictions": ["p1", "p2"], "minimal_test": "test on HotpotQA"}


def _claim(cid="c1", pid="p1"):
    return {"claim_id": cid, "paper_id": pid,
            "claim_text": "SPARC-RAG uses less token cost than the baseline.",
            "direction": "positive", "method": "SPARC-RAG", "dataset": "HotpotQA"}


def _anom(aid="a1", typ="impact_conflict"):
    return {"anomaly_id": aid, "type": typ, "central_question": "q", "claim_ids": ["c1"]}


_FLAGGED = json.dumps({"unverified": [
    {"number": "52.2%", "context": "SPARC-RAG cost reduction", "issue": "absent"}]})
_CLEAN = json.dumps({"unverified": []})


def _enable(monkeypatch):
    monkeypatch.setenv("AIGRAPH_NUMERIC_GATE", "1")
    monkeypatch.setattr(ng, "configured_api_key", lambda *a, **k: "sk-x")
    monkeypatch.setattr(ng, "configured_model", lambda *a, **k: "m")
    monkeypatch.setattr(ng, "build_openai_client", lambda *a, **k: object())


# --- enabled / fail-open ---------------------------------------------------- #

def test_disabled_env_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("AIGRAPH_NUMERIC_GATE", "0")
    monkeypatch.setattr(ng, "configured_api_key", lambda *a, **k: "sk-x")
    assert ng.gate_enabled() is False
    _run(tmp_path, [_hyp()], [_claim()], [_anom()])
    assert ng.gate_run(tmp_path) == {}


def test_no_key_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("AIGRAPH_NUMERIC_GATE", "1")
    monkeypatch.setattr(ng, "configured_api_key", lambda *a, **k: None)
    assert ng.gate_enabled() is False
    _run(tmp_path, [_hyp()], [_claim()], [_anom()])
    assert ng.gate_run(tmp_path) == {}


def test_llm_error_leaves_no_sidecar(tmp_path, monkeypatch):
    _enable(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("gateway 500")

    monkeypatch.setattr(ng, "call_llm_text", _boom)
    _run(tmp_path, [_hyp()], [_claim()], [_anom()])
    assert ng.gate_run(tmp_path) == {}
    assert not (tmp_path / ng.FLAGS_FILENAME).exists()


def test_unparseable_reply_is_skipped(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ng, "call_llm_text", lambda *a, **k: "I cannot help")
    _run(tmp_path, [_hyp()], [_claim()], [_anom()])
    assert ng.gate_run(tmp_path) == {}


# --- happy path + cache + overlay ------------------------------------------- #

def test_gate_flags_unverified_number(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ng, "call_llm_text", lambda *a, **k: _FLAGGED)
    _run(tmp_path, [_hyp()], [_claim()], [_anom()])
    out = ng.gate_run(tmp_path)
    assert set(out) == {"h1"}
    assert out["h1"]["flags"][0]["number"] == "52.2%"
    assert out["h1"]["flags"][0]["issue"] == "absent"
    assert (tmp_path / ng.FLAGS_FILENAME).exists()


def test_clean_hypothesis_caches_empty_flags(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ng, "call_llm_text", lambda *a, **k: _CLEAN)
    _run(tmp_path, [_hyp()], [_claim()], [_anom()])
    out = ng.gate_run(tmp_path)
    assert out["h1"]["flags"] == []


def test_cache_skips_second_pass(tmp_path, monkeypatch):
    _enable(monkeypatch)
    calls = {"n": 0}

    def _count(*a, **k):
        calls["n"] += 1
        return _FLAGGED

    monkeypatch.setattr(ng, "call_llm_text", _count)
    _run(tmp_path, [_hyp()], [_claim()], [_anom()])
    ng.gate_run(tmp_path)
    ng.gate_run(tmp_path)
    assert calls["n"] == 1


def test_limit_caps_new_calls(tmp_path, monkeypatch):
    _enable(monkeypatch)
    calls = {"n": 0}
    monkeypatch.setattr(ng, "call_llm_text",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), _CLEAN)[1])
    hyps = [_hyp(hid=f"h{i}") for i in range(5)]
    _run(tmp_path, hyps, [_claim()], [_anom()])
    ng.gate_run(tmp_path, limit=2)
    assert calls["n"] == 2


def test_checks_enriched_text_when_present(tmp_path, monkeypatch):
    _enable(monkeypatch)
    seen = {}
    monkeypatch.setattr(ng, "call_llm_text",
                        lambda *a, **k: (seen.update(json.loads(k["user"])), _CLEAN)[1])
    enriched = [{"hypothesis_id": "h1", "statement": "ENRICHED statement 81.28%",
                 "mechanism": "em", "predictions": ["ep"], "minimal_test": "et"}]
    _run(tmp_path, [_hyp()], [_claim()], [_anom()], enriched=enriched)
    ng.gate_run(tmp_path)
    # the gate must number-check the ENRICHED text, not the raw hypothesis
    assert "81.28%" in seen["hypothesis"]["statement"]


def test_apply_flags_overlays_in_place(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ng, "call_llm_text", lambda *a, **k: _FLAGGED)
    _run(tmp_path, [_hyp()], [_claim()], [_anom()])
    ng.gate_run(tmp_path)
    hyp = Hypothesis(hypothesis_id="h1", anomaly_id="a1", hypothesis="x")
    n = ng.apply_flags([hyp], tmp_path)
    assert n == 1
    assert hyp.numeric_flags and "52.2%" in hyp.numeric_flags[0]


def test_apply_flags_noop_without_sidecar(tmp_path):
    hyp = Hypothesis(hypothesis_id="h1", anomaly_id="a1", hypothesis="x")
    assert ng.apply_flags([hyp], tmp_path) == 0
    assert hyp.numeric_flags is None


def test_apply_flags_skips_clean(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(ng, "call_llm_text", lambda *a, **k: _CLEAN)
    _run(tmp_path, [_hyp()], [_claim()], [_anom()])
    ng.gate_run(tmp_path)
    hyp = Hypothesis(hypothesis_id="h1", anomaly_id="a1", hypothesis="x")
    assert ng.apply_flags([hyp], tmp_path) == 0  # empty flags -> not stamped


def test_parse_tolerates_prose_and_fences():
    rec = ng._parse('Sure!\n{"unverified": [{"number": "0.420", "context": "BlendFilter EM"}]}\nok')
    assert rec and rec[0]["number"] == "0.420"
    assert rec[0]["issue"] == "absent"  # default when omitted


def test_parse_empty_list_is_clean():
    assert ng._parse('{"unverified": []}') == []
