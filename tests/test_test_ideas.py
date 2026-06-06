"""Unit tests for the test_ideas.py CLI async poll loop (#53)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import test_ideas as ti  # noqa: E402


def test_poll_until_done_returns_on_done(monkeypatch):
    seq = [
        {"status": "running", "stage": "fetching", "papers": 0},
        {"status": "running", "stage": "extracting", "papers": 10},
        {"status": "done", "stage": "complete", "papers": 10},
    ]
    calls = {"i": 0}

    def fake_call(name, args, timeout=30):
        i = calls["i"]
        calls["i"] += 1
        return seq[min(i, len(seq) - 1)]

    monkeypatch.setattr(ti, "tool_call", fake_call)
    monkeypatch.setattr(ti.time, "sleep", lambda *_a, **_k: None)
    out = ti.poll_until_done("r", interval=0, max_wait=100)
    assert out["status"] == "done"
    assert calls["i"] >= 3  # polled through the building stages


def test_poll_until_done_backgrounds_after_max_wait(monkeypatch):
    # A run that never finishes must return the last status (so the CLI can tell
    # the user to --resume), not block forever.
    monkeypatch.setattr(ti, "tool_call",
                        lambda *a, **k: {"status": "running", "stage": "extracting", "papers": 5})
    monkeypatch.setattr(ti.time, "sleep", lambda *_a, **_k: None)
    out = ti.poll_until_done("r", interval=10, max_wait=20)
    assert out["status"] == "running"


def test_poll_until_done_survives_transient_poll_error(monkeypatch):
    # A dropped/timed-out poll request is NOT a run failure — keep polling.
    state = {"n": 0}

    def flaky(name, args, timeout=30):
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("connection reset")
        return {"status": "done", "stage": "complete", "papers": 3}

    monkeypatch.setattr(ti, "tool_call", flaky)
    monkeypatch.setattr(ti.time, "sleep", lambda *_a, **_k: None)
    out = ti.poll_until_done("r", interval=1, max_wait=100)
    assert out["status"] == "done"
