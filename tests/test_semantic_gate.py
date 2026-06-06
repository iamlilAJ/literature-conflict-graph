"""Unit tests for the LLM semantic relevance gate (src/aigraph/semantic_gate.py).

The gate is non-frozen and sits at the server seam; these tests pin its
fail-open contract, off-topic drop + re-order behavior, recall-floor top-up, and
defensive score parsing — all without a real LLM (scoring is injected).
"""
import aigraph.semantic_gate as sg
from aigraph.models import Paper


def _paper(pid: str, title: str = "", abstract: str = "") -> Paper:
    return Paper(paper_id=pid, title=title or pid, year=2024, venue="arXiv", abstract=abstract)


def _fixed_scores(mapping):
    """Build a score_fn that returns a fixed paper_id->score mapping."""
    def _fn(topic, papers, **_kw):
        return {p.paper_id: mapping[p.paper_id] for p in papers if p.paper_id in mapping}
    return _fn


def test_disabled_without_key_is_noop(monkeypatch):
    monkeypatch.delenv("AIGRAPH_SEMANTIC_GATE", raising=False)
    monkeypatch.setattr(sg, "configured_api_key", lambda *_a, **_k: None)
    papers = [_paper("a"), _paper("b")]
    out, audit = sg.apply_semantic_gate("topic", papers, keep_floor=1)
    assert out == papers
    assert audit["applied"] is False and audit["reason"] == "disabled"


def test_drops_offtopic_and_reorders():
    papers = [_paper("lo"), _paper("hi"), _paper("mid"), _paper("off")]
    score_fn = _fixed_scores({"lo": 2, "hi": 3, "mid": 2, "off": 0})
    out, audit = sg.apply_semantic_gate("t", papers, keep_floor=1, min_score=2, score_fn=score_fn)
    ids = [p.paper_id for p in out]
    assert "off" not in ids  # score 0 < min_score 2 -> dropped
    assert ids[0] == "hi"  # strongest first
    assert set(ids) == {"hi", "lo", "mid"}
    assert audit["applied"] is True
    assert audit["kept"] == 3 and audit["dropped"] == 1
    assert audit["score_histogram"]["0"] == 1 and audit["score_histogram"]["3"] == 1


def test_unscored_papers_are_kept():
    papers = [_paper("a"), _paper("b"), _paper("c")]
    # model only scored 'a' (off-topic); b, c get no verdict -> keep
    out, audit = sg.apply_semantic_gate(
        "t", papers, keep_floor=1, min_score=2, score_fn=_fixed_scores({"a": 0})
    )
    ids = {p.paper_id for p in out}
    assert ids == {"b", "c"}  # 'a' dropped, unscored kept
    assert audit["unscored_kept"] == 2


def test_recall_floor_tops_up_dropped():
    # everything is off-topic, but keep_floor=2 must be honored from the best dropped
    papers = [_paper("x"), _paper("y"), _paper("z")]
    score_fn = _fixed_scores({"x": 0, "y": 1, "z": 0})
    out, audit = sg.apply_semantic_gate("t", papers, keep_floor=2, min_score=2, score_fn=score_fn)
    assert len(out) == 2
    assert out[0].paper_id == "y"  # highest dropped score added back first
    assert audit["topped_up"] == 2


def test_failopen_when_scorer_raises():
    def boom(topic, papers, **_kw):
        raise RuntimeError("llm down")

    papers = [_paper("a"), _paper("b")]
    out, audit = sg.apply_semantic_gate("t", papers, keep_floor=1, score_fn=boom)
    assert out == papers and audit["applied"] is False and audit["reason"] == "scorer_error"


def test_failopen_when_no_scores():
    papers = [_paper("a"), _paper("b")]
    out, audit = sg.apply_semantic_gate("t", papers, keep_floor=1, score_fn=lambda *a, **k: {})
    assert out == papers and audit["applied"] is False and audit["reason"] == "no_scores"


def test_empty_corpus():
    out, audit = sg.apply_semantic_gate("t", [], keep_floor=5, score_fn=lambda *a, **k: {"x": 3})
    assert out == [] and audit["applied"] is False


# ---- _parse_scores defensive parsing ----

def test_parse_plain_object():
    assert sg._parse_scores('{"0": 3, "1": 0, "2": 2}', {"0", "1", "2"}) == {"0": 3, "1": 0, "2": 2}


def test_parse_scores_wrapper_and_fences():
    raw = '```json\n{"scores": {"0": 2, "9": 1}}\n```'
    assert sg._parse_scores(raw, {"0", "1"}) == {"0": 2}  # id "9" not valid -> dropped


def test_parse_list_form_and_clamp():
    raw = '[{"id": "0", "score": 5}, {"id": "1", "score": -2}, {"id": "2", "score": "3"}]'
    # 5 clamps to 3, -2 clamps to 0, "3" coerces to 3
    assert sg._parse_scores(raw, {"0", "1", "2"}) == {"0": 3, "1": 0, "2": 3}


def test_parse_garbage_returns_empty():
    assert sg._parse_scores("the model refused to answer", {"0"}) == {}
    assert sg._parse_scores("", {"0"}) == {}


def test_score_relevance_batches_and_translates(monkeypatch):
    # gate enabled, fake client, capture the batches and return index->score JSON
    monkeypatch.setattr(sg, "configured_api_key", lambda *_a, **_k: "k")
    monkeypatch.setattr(sg, "build_openai_client", lambda *a, **k: object())
    monkeypatch.setattr(sg, "configured_model", lambda *a, **k: "m")
    seen_batches = []

    def fake_call(client, *, model, system, user, temperature, max_tokens):
        import json as _j
        payload = _j.loads(user)
        seen_batches.append(len(payload["papers"]))
        # score each in-batch id by its index parity
        return _j.dumps({p["id"]: (3 if int(p["id"]) % 2 == 0 else 0) for p in payload["papers"]})

    monkeypatch.setattr(sg, "call_llm_text", fake_call)
    papers = [_paper(f"p{i}") for i in range(5)]
    scores = sg.score_relevance("topic", papers, batch_size=2)
    # 5 papers, batch 2 -> batches of [2, 2, 1]
    assert seen_batches == [2, 2, 1]
    # within each batch local idx 0 -> score 3, idx 1 -> score 0
    assert scores == {"p0": 3, "p1": 0, "p2": 3, "p3": 0, "p4": 3}
