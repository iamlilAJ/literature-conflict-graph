"""Unit tests for the P5 novelty gate in idea_cascade.tier_d.

No network: monkeypatch the module-level call_llm_text.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

import idea_cascade as ic  # noqa: E402


def _patch(monkeypatch, reply: str):
    monkeypatch.setattr(ic, "call_llm_text", lambda *a, **k: reply)


def test_gate_drops_when_already_in_paper(monkeypatch):
    _patch(monkeypatch, '{"already_in_paper": true, "reason": "abstract already adds memory"}')
    keep = ic._novelty_gate(object(), "m", "MobEvolve already stores trial memory.", {"title": "add memory"})
    assert keep is False


def test_gate_keeps_when_novel(monkeypatch):
    _patch(monkeypatch, '{"already_in_paper": false, "reason": "genuinely new"}')
    assert ic._novelty_gate(object(), "m", "some abstract", {"title": "x"}) is True


def test_gate_fails_open_on_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("endpoint down")
    monkeypatch.setattr(ic, "call_llm_text", boom)
    assert ic._novelty_gate(object(), "m", "abstract", {"title": "x"}) is True


def test_gate_keeps_when_no_abstract(monkeypatch):
    # no abstract -> cannot judge -> keep (fail open), no LLM call
    called = {"n": 0}
    def counter(*a, **k):
        called["n"] += 1
        return '{"already_in_paper": true}'
    monkeypatch.setattr(ic, "call_llm_text", counter)
    assert ic._novelty_gate(object(), "m", "", {"title": "x"}) is True
    assert called["n"] == 0


def test_gate_keeps_on_unparseable(monkeypatch):
    _patch(monkeypatch, "not json at all")
    assert ic._novelty_gate(object(), "m", "abstract", {"title": "x"}) is True
