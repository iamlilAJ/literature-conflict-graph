"""Tests for #52 multi-layer novelty audit (idea_cascade._novelty_audit) +
its model field + report rendering. The honest core: 'unknown' must NOT be
dressed up as 'novel' when no strong layer could actually verify novelty."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import idea_cascade as ic  # noqa: E402

from aigraph.models import Hypothesis  # noqa: E402
from aigraph.report import _render_hypothesis  # noqa: E402


def _idea():
    return {"title": "X", "statement": "Y", "mechanism": "Z"}


# ---- _strong_layer_verdict (3-state) -------------------------------------

def test_layer_covered(monkeypatch):
    monkeypatch.setattr(ic, "call_llm_text", lambda *a, **k: '{"novel": false}')
    assert ic._strong_layer_verdict(None, "m", _idea(), lambda q, k=6: ["(24) p"], "existing_titles") == "covered"


def test_layer_novel(monkeypatch):
    monkeypatch.setattr(ic, "call_llm_text", lambda *a, **k: '{"novel": true}')
    assert ic._strong_layer_verdict(None, "m", _idea(), lambda q, k=6: ["(24) p"], "existing_titles") == "novel"


def test_layer_unknown_when_no_prior_art():
    assert ic._strong_layer_verdict(None, "m", _idea(), lambda q, k=6: [], "existing_titles") == "unknown"


def test_layer_unknown_on_fetch_error():
    def boom(q, k=6):
        raise RuntimeError("down")
    assert ic._strong_layer_verdict(None, "m", _idea(), boom, "existing_titles") == "unknown"


def test_layer_unknown_on_llm_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(ic, "call_llm_text", boom)
    assert ic._strong_layer_verdict(None, "m", _idea(), lambda q, k=6: ["p"], "existing_titles") == "unknown"


# ---- _novelty_audit (gate decision + honest state) -----------------------

def test_audit_verified_novel(monkeypatch):
    monkeypatch.setattr(ic, "_novelty_gate", lambda *a, **k: True)
    monkeypatch.setattr(ic, "atlas_prior_art_strong", lambda q, k=6: ["prior"])
    monkeypatch.setattr(ic, "call_llm_text", lambda *a, **k: '{"novel": true}')
    monkeypatch.delenv("AIGRAPH_WEB_NOVELTY", raising=False)
    passed, audit = ic._novelty_audit(None, "m", "abs", _idea())
    assert passed is True
    assert audit["state"] == "novel" and audit["corpus_verdict"] == "novel"
    assert audit["web_verdict"] == "skipped"


def test_audit_covered_by_corpus(monkeypatch):
    monkeypatch.setattr(ic, "_novelty_gate", lambda *a, **k: True)
    monkeypatch.setattr(ic, "atlas_prior_art_strong", lambda q, k=6: ["prior"])
    monkeypatch.setattr(ic, "call_llm_text", lambda *a, **k: '{"novel": false}')
    passed, audit = ic._novelty_audit(None, "m", "abs", _idea())
    assert passed is False and audit["state"] == "covered"


def test_audit_unknown_not_faked_novel(monkeypatch):
    # lexical passes, but atlas has no prior art to compare (can't verify) and
    # web is off -> state must be 'unknown' (honest), while the gate stays open.
    monkeypatch.setattr(ic, "_novelty_gate", lambda *a, **k: True)
    monkeypatch.setattr(ic, "atlas_prior_art_strong", lambda q, k=6: [])
    monkeypatch.delenv("AIGRAPH_WEB_NOVELTY", raising=False)
    passed, audit = ic._novelty_audit(None, "m", "abs", _idea())
    assert passed is True                  # fail-open gate preserved
    assert audit["state"] == "unknown"     # but NOT dressed up as novel
    assert audit["corpus_verdict"] == "unknown"
    assert "no_strong_layer_could_confirm_novelty" in audit["reasons"]


def test_audit_lexical_fail_is_covered(monkeypatch):
    monkeypatch.setattr(ic, "_novelty_gate", lambda *a, **k: False)
    monkeypatch.setattr(ic, "atlas_prior_art_strong", lambda q, k=6: [])
    passed, audit = ic._novelty_audit(None, "m", "abs", _idea())
    assert passed is False and audit["state"] == "covered" and audit["lexical_novelty"] == "fail"


# ---- model field + render ------------------------------------------------

def test_hypothesis_carries_audit_and_renders():
    h = Hypothesis(
        hypothesis_id="h1", anomaly_id="a1", hypothesis="H",
        novelty_audit={"state": "unknown", "corpus_verdict": "unknown",
                       "web_verdict": "skipped", "lexical_novelty": "pass",
                       "reasons": ["no_strong_layer_could_confirm_novelty"]},
    )
    assert h.novelty_audit["state"] == "unknown"
    md = "\n".join(_render_hypothesis(h, {}))
    assert "Novelty audit" in md and "unknown" in md


def test_render_skips_when_no_audit():
    h = Hypothesis(hypothesis_id="h2", anomaly_id="a2", hypothesis="H")
    md = "\n".join(_render_hypothesis(h, {}))
    assert "Novelty audit" not in md
