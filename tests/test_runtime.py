"""Smoke tests for aigraph.runtime.RunResult.

Avoid actually booting the FastAPI server in tests by monkey-patching
_ensure_server. Only verify object construction + URL composition.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_run(tmp_path: Path) -> Path:
    """Build a minimal complete run dir under tmp/artifacts/runs/."""
    runs_root = tmp_path / "artifacts" / "runs"
    run_dir = runs_root / "fake_run_abc"
    run_dir.mkdir(parents=True)
    # Both files RunResult requires for `from_run_dir`
    (run_dir / "hypotheses_scored.jsonl").write_text(
        json.dumps({"hypothesis_id": "h1", "anomaly_id": "a1", "hypothesis": "x"}) + "\n"
    )
    (run_dir / "papers.jsonl").write_text("")
    return run_dir


def test_from_run_dir_requires_hypotheses_file(tmp_path):
    from aigraph.runtime import RunResult

    runs_root = tmp_path / "artifacts" / "runs"
    bad = runs_root / "empty"
    bad.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="hypotheses_scored"):
        RunResult.from_run_dir(bad)


def test_from_run_dir_rejects_path_outside_runs_root(tmp_path, monkeypatch):
    """Run dirs under a *different* root should error out clearly."""
    from aigraph import runtime, web as _web

    # Point the module's default at a separate location
    expected_root = tmp_path / "other_root" / "runs"
    expected_root.mkdir(parents=True)
    monkeypatch.setattr(_web, "DEFAULT_RUNS_ROOT", expected_root)

    # Build a run dir somewhere else
    stray = tmp_path / "stray" / "run_x"
    stray.mkdir(parents=True)
    (stray / "hypotheses_scored.jsonl").write_text("{}\n")
    (stray / "papers.jsonl").write_text("")

    with pytest.raises(ValueError, match="outside the server runs_root"):
        runtime.RunResult.from_run_dir(stray)


def test_url_uses_singleton_server(fake_run, monkeypatch):
    """``url`` should call _ensure_server once and format the URL."""
    from aigraph import runtime, web as _web

    monkeypatch.setattr(_web, "DEFAULT_RUNS_ROOT", fake_run.parent)
    monkeypatch.setattr(runtime, "_ensure_server", lambda *a, **kw: ("127.0.0.1", 9999))

    r = runtime.RunResult.from_run_dir(fake_run)
    assert r.url == "http://127.0.0.1:9999/run/fake_run_abc"
    assert "fake_run_abc" in repr(r)


def test_repr_html_renders_card(fake_run, monkeypatch):
    from aigraph import runtime, web as _web

    monkeypatch.setattr(_web, "DEFAULT_RUNS_ROOT", fake_run.parent)
    monkeypatch.setattr(runtime, "_ensure_server", lambda *a, **kw: ("127.0.0.1", 9999))

    r = runtime.RunResult.from_run_dir(fake_run)
    html = r._repr_html_()
    assert "fake_run_abc" in html
    assert "0 LLM at query time" in html
    assert 'href="http://127.0.0.1:9999/run/fake_run_abc"' in html


def test_load_shorthand(fake_run, monkeypatch):
    from aigraph import runtime, web as _web

    monkeypatch.setattr(_web, "DEFAULT_RUNS_ROOT", fake_run.parent)
    r = runtime.load(fake_run)
    assert r.run_id == "fake_run_abc"
    assert r.run_dir == fake_run.resolve()


def test_lazy_aigraph_namespace_exposes_load():
    import aigraph

    assert callable(aigraph.load)
    assert callable(aigraph.RunResult)
    assert callable(aigraph.server_url)
    assert callable(aigraph.start_server)
    with pytest.raises(AttributeError):
        aigraph.does_not_exist
