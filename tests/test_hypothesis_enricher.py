"""Unit tests for the LLM hypothesis enricher (src/aigraph/hypothesis_enricher.py).

The enricher is non-frozen and grounds the frozen templated hypotheses in their
real evidence claims at delivery time. These tests pin the fail-open contract,
defensive JSON parsing, the cached sidecar round-trip, and the 0-LLM overlay onto
Hypothesis objects — all without a real LLM (the gateway call is injected).
"""
import json

import aigraph.hypothesis_enricher as he
from aigraph.models import Anomaly, Claim, Hypothesis, Paper


def _run(tmp_path, hyps, claims, anoms, papers=()):
    for name, rows in (("hypotheses", hyps), ("claims", claims),
                       ("anomalies", anoms), ("papers", papers)):
        (tmp_path / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


def _hyp(hid="h1", aid="a1", claims=("c1",)):
    return {"hypothesis_id": hid, "anomaly_id": aid,
            "hypothesis": "TEMPLATE: a confound drives the conflict",
            "mechanism": "TEMPLATE mechanism", "explains_claims": list(claims),
            "predictions": ["template pred"], "minimal_test": "template test"}


def _claim(cid="c1", pid="p1"):
    return {"claim_id": cid, "paper_id": pid,
            "claim_text": "SCoT beats CoT by 13.79% Pass@1", "direction": "positive",
            "method": "SCoT", "dataset": "HumanEval"}


def _anom(aid="a1", typ="impact_conflict"):
    return {"anomaly_id": aid, "type": typ, "central_question": "why disagree?",
            "claim_ids": ["c1"]}


def _paper(pid="p1"):
    return {"paper_id": pid, "title": "Structured CoT", "year": 2023, "venue": "arXiv"}


_GOOD = json.dumps({
    "statement": "SCoT beats CoT by 13.79% on HumanEval but may not generalize",
    "mechanism": "SCoT mirrors source-code control flow, aligning with HumanEval",
    "predictions": ["On MBPP the gap shrinks to <5%", "On CodeContests CoT may win"],
    "minimal_test": "Run SCoT vs CoT on HumanEval and MBPP, same LLM, compare Pass@1",
})


# --------------------------------------------------------------------------- #
# enabled / fail-open
# --------------------------------------------------------------------------- #

def test_disabled_env_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("AIGRAPH_HYP_ENRICH", "0")
    monkeypatch.setattr(he, "configured_api_key", lambda *_a, **_k: "sk-x")
    assert he.enricher_enabled() is False
    _run(tmp_path, [_hyp()], [_claim()], [_anom()], [_paper()])
    assert he.enrich_run(tmp_path) == {}  # no sidecar written


def test_no_key_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("AIGRAPH_HYP_ENRICH", "1")
    monkeypatch.setattr(he, "configured_api_key", lambda *_a, **_k: None)
    assert he.enricher_enabled() is False
    _run(tmp_path, [_hyp()], [_claim()], [_anom()], [_paper()])
    assert he.enrich_run(tmp_path) == {}


def test_llm_error_leaves_no_sidecar(tmp_path, monkeypatch):
    _enable(monkeypatch)

    def _boom(*_a, **_k):
        raise RuntimeError("gateway 500")

    monkeypatch.setattr(he, "call_llm_text", _boom)
    _run(tmp_path, [_hyp()], [_claim()], [_anom()], [_paper()])
    assert he.enrich_run(tmp_path) == {}
    assert not (tmp_path / he.ENRICH_FILENAME).exists()


def test_unparseable_reply_is_skipped(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(he, "call_llm_text", lambda *_a, **_k: "I cannot help")
    _run(tmp_path, [_hyp()], [_claim()], [_anom()], [_paper()])
    assert he.enrich_run(tmp_path) == {}


def test_no_evidence_is_skipped(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(he, "call_llm_text", lambda *_a, **_k: _GOOD)
    # hypothesis + anomaly reference no resolvable claims -> no evidence -> skip
    anom = {"anomaly_id": "a1", "type": "evidence_gap", "central_question": "q", "claim_ids": []}
    _run(tmp_path, [_hyp(claims=("missing",))], [], [anom], [])
    assert he.enrich_run(tmp_path) == {}


# --------------------------------------------------------------------------- #
# happy path + cache + overlay
# --------------------------------------------------------------------------- #

def _enable(monkeypatch):
    monkeypatch.setenv("AIGRAPH_HYP_ENRICH", "1")
    monkeypatch.setattr(he, "configured_api_key", lambda *_a, **_k: "sk-x")
    monkeypatch.setattr(he, "configured_model", lambda *_a, **_k: "m")
    monkeypatch.setattr(he, "build_openai_client", lambda *_a, **_k: object())


def test_enrich_writes_sidecar_and_parses(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(he, "call_llm_text", lambda *_a, **_k: _GOOD)
    _run(tmp_path, [_hyp()], [_claim()], [_anom()], [_paper()])
    out = he.enrich_run(tmp_path)
    assert set(out) == {"h1"}
    rec = out["h1"]
    assert "13.79%" in rec["statement"]
    assert len(rec["predictions"]) == 2
    assert rec["anomaly_type"] == "impact_conflict"
    assert (tmp_path / he.ENRICH_FILENAME).exists()


def test_cache_skips_second_pass(tmp_path, monkeypatch):
    _enable(monkeypatch)
    calls = {"n": 0}

    def _count(*_a, **_k):
        calls["n"] += 1
        return _GOOD

    monkeypatch.setattr(he, "call_llm_text", _count)
    _run(tmp_path, [_hyp()], [_claim()], [_anom()], [_paper()])
    he.enrich_run(tmp_path)
    he.enrich_run(tmp_path)  # cached — must not call the LLM again
    assert calls["n"] == 1


def test_limit_caps_new_calls(tmp_path, monkeypatch):
    _enable(monkeypatch)
    calls = {"n": 0}
    monkeypatch.setattr(he, "call_llm_text",
                        lambda *_a, **_k: (calls.__setitem__("n", calls["n"] + 1), _GOOD)[1])
    hyps = [_hyp(hid=f"h{i}") for i in range(5)]
    _run(tmp_path, hyps, [_claim()], [_anom()], [_paper()])
    he.enrich_run(tmp_path, limit=2)
    assert calls["n"] == 2


def test_only_types_filter(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(he, "call_llm_text", lambda *_a, **_k: _GOOD)
    _run(tmp_path, [_hyp()], [_claim()], [_anom(typ="community_disconnect")], [_paper()])
    assert he.enrich_run(tmp_path, only_types={"impact_conflict"}) == {}  # filtered out


def test_apply_enrichment_overlays_in_place(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(he, "call_llm_text", lambda *_a, **_k: _GOOD)
    _run(tmp_path, [_hyp()], [_claim()], [_anom()], [_paper()])
    he.enrich_run(tmp_path)
    hyp = Hypothesis(hypothesis_id="h1", anomaly_id="a1",
                     hypothesis="TEMPLATE", mechanism="TEMPLATE", explains_claims=["c1"],
                     predictions=["t"], minimal_test="t")
    n = he.apply_enrichment([hyp], tmp_path)
    assert n == 1
    assert "13.79%" in hyp.hypothesis
    assert "source-code" in hyp.mechanism
    assert hyp.enriched == {"applied": True, "anomaly_type": "impact_conflict"}


def test_apply_enrichment_noop_without_sidecar(tmp_path):
    hyp = Hypothesis(hypothesis_id="h1", anomaly_id="a1", hypothesis="TEMPLATE")
    assert he.apply_enrichment([hyp], tmp_path) == 0
    assert hyp.hypothesis == "TEMPLATE"  # untouched (fail-open)


def test_load_enriched_round_trip(tmp_path, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(he, "call_llm_text", lambda *_a, **_k: _GOOD)
    _run(tmp_path, [_hyp()], [_claim()], [_anom()], [_paper()])
    he.enrich_run(tmp_path)
    loaded = he.load_enriched(tmp_path)
    assert "h1" in loaded and "13.79%" in loaded["h1"]["statement"]


def test_parse_tolerates_prose_and_fences():
    rec = he._parse('Sure!```json\n' + _GOOD + '\n``` done')
    assert rec is not None and "13.79%" in rec["statement"]


def test_parse_rejects_empty_payload():
    assert he._parse('{"predictions": []}') is None  # no statement/mechanism
