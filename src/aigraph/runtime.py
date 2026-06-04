"""Python API: ``RunResult`` returned by pipeline functions.

Usage:

    from aigraph.runtime import RunResult, run_corpus

    # Load an existing run produced by run_local_corpus.py / finish_local_run.py
    result = RunResult.from_run_dir("artifacts/runs/arxiv-reasoning-v2")
    print(result.url)        # → http://127.0.0.1:8765/run/arxiv-reasoning-v2
    result.show()             # → opens the URL in a browser tab

    # In Jupyter, ``result`` renders as a clickable preview card.

The first ``RunResult.url`` or ``show()`` call lazily starts a
FastAPI server in a background thread. The server is a singleton —
subsequent ``RunResult`` instances share it.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import web as _web_module

_DEFAULT_PORT_RANGE = (8765, 8775)


# ---- Server-thread singleton ----------------------------------------

_server_lock = threading.Lock()
_server_thread: Optional[threading.Thread] = None
_server_port: Optional[int] = None
_server_host: str = "127.0.0.1"
_server_runs_root: Optional[Path] = None


def _find_free_port(host: str = "127.0.0.1", port_range: tuple[int, int] = _DEFAULT_PORT_RANGE) -> int:
    for p in range(*port_range):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"no free port in {port_range}")


def _wait_for_server(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return
            except OSError:
                time.sleep(0.1)
    raise RuntimeError(f"server on {host}:{port} did not come up in {timeout}s")


def _ensure_server(runs_root: Optional[Path] = None) -> tuple[str, int]:
    """Idempotent: start the FastAPI server in a daemon thread if not
    already running. Returns ``(host, port)``."""
    global _server_thread, _server_port, _server_host, _server_runs_root
    with _server_lock:
        if _server_thread is not None and _server_thread.is_alive():
            return _server_host, _server_port  # type: ignore[return-value]
        host = "127.0.0.1"
        port = _find_free_port(host)
        runs_root = Path(runs_root) if runs_root else _web_module.DEFAULT_RUNS_ROOT
        app = _web_module.create_app(runs_root)

        def _run() -> None:
            import uvicorn
            cfg = uvicorn.Config(app, host=host, port=port, log_level="warning")
            server = uvicorn.Server(cfg)
            server.run()

        t = threading.Thread(target=_run, name="aigraph-web", daemon=True)
        t.start()
        _wait_for_server(host, port, timeout=15.0)
        _server_thread = t
        _server_port = port
        _server_host = host
        _server_runs_root = runs_root
        return host, port


def server_url() -> str:
    """Current server URL (auto-starts if not running)."""
    host, port = _ensure_server()
    return f"http://{host}:{port}"


# ---- RunResult -------------------------------------------------------


@dataclass
class RunResult:
    """A reference to a completed aigraph run, with a self-serving URL.

    Lifecycle: created from a run directory (containing
    ``hypotheses_scored.jsonl``). Calling ``.url`` lazily boots a
    singleton FastAPI server in the background and returns a link
    that, when opened, shows the run.

    Attributes:
        run_dir: Path to the run directory.
        run_id: Last path component of run_dir; used in the URL.
    """

    run_dir: Path
    run_id: str

    @classmethod
    def from_run_dir(cls, run_dir: str | Path) -> "RunResult":
        run_dir = Path(run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")
        scored = run_dir / "hypotheses_scored.jsonl"
        if not scored.exists():
            raise FileNotFoundError(
                f"{run_dir} missing hypotheses_scored.jsonl — not a complete aigraph run"
            )
        # Verify the run is under the server's runs_root, else the URL
        # would 404 because the server scans only its configured root.
        runs_root = _server_runs_root or _web_module.DEFAULT_RUNS_ROOT
        try:
            run_dir.relative_to(runs_root.resolve())
        except ValueError:
            raise ValueError(
                f"{run_dir} is outside the server runs_root ({runs_root}). "
                f"Either move the run, or start the server with a different "
                f"runs_root via aigraph.runtime.start_server(runs_root=...)"
            )
        return cls(run_dir=run_dir, run_id=run_dir.name)

    @property
    def url(self) -> str:
        host, port = _ensure_server()
        return f"http://{host}:{port}/run/{self.run_id}"

    def show(self) -> str:
        """Open the run's URL in the user's browser. Returns the URL."""
        url = self.url
        webbrowser.open(url)
        return url

    def __repr__(self) -> str:
        return f"<RunResult {self.run_id!r} {self.url}>"

    def _repr_html_(self) -> str:
        """IPython/Jupyter: render as a clickable card with stats."""
        # Light counts without forcing full load
        scored = self.run_dir / "hypotheses_scored.jsonl"
        papers = self.run_dir / "papers.jsonl"
        n_hyp = sum(1 for _ in scored.open()) if scored.exists() else 0
        n_papers = sum(1 for _ in papers.open()) if papers.exists() else 0
        url = self.url
        return f"""<div style="font-family:-apple-system,Helvetica Neue,sans-serif;
                            border:1px solid #ddd; border-radius:8px;
                            padding:12px 16px; background:#fafafa;
                            max-width:680px;">
  <div style="font-size:14px; color:#444;">aigraph RunResult</div>
  <div style="font-size:18px; margin:4px 0;">
    <a href="{url}" target="_blank" style="color:#0366d6; text-decoration:none;">
      <code>{self.run_id}</code> ↗
    </a>
  </div>
  <div style="font-size:12px; color:#666;">
    {n_papers} papers · {n_hyp} hypotheses · 0 LLM at query time
  </div>
  <div style="font-size:11px; color:#888; margin-top:6px;">
    <code>{self.run_dir}</code>
  </div>
</div>"""


# ---- Convenience constructors --------------------------------------


def load(run_dir: str | Path) -> RunResult:
    """Shorthand: ``aigraph.runtime.load("artifacts/runs/X")``."""
    return RunResult.from_run_dir(run_dir)


def start_server(
    *,
    host: str = "127.0.0.1",
    runs_root: Optional[Path] = None,
    block: bool = False,
) -> str:
    """Start (or reuse) the singleton server. Returns the base URL.

    If ``block`` is True, runs the server in the foreground (the
    process won't return until you Ctrl-C). If False (default), runs
    the server in a daemon thread and returns immediately.
    """
    if block:
        _web_module.serve(host=host, runs_root=runs_root)
        return ""  # unreachable
    _ensure_server(runs_root=Path(runs_root) if runs_root else None)
    return server_url()
