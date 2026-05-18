"""Thin wrapper for the v0.7-frozen explorer.

Prefer ``aigraph web`` (CLI) or ``aigraph.runtime.start_server()`` (Python).
This script remains so that ``python scripts/aigraph_web.py`` still works.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aigraph.web import serve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--runs-root", type=Path, default=Path("artifacts/runs"))
    args = ap.parse_args()
    serve(host=args.host, port=args.port, runs_root=args.runs_root)


if __name__ == "__main__":
    main()
