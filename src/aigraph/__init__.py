"""aigraph: a graph-based literature conflict explorer for AI paper claims."""

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy re-export so `import aigraph; aigraph.load(...)` works without
    pulling fastapi/uvicorn/markdown at every import."""
    if name in {"RunResult", "load", "start_server", "server_url"}:
        from . import runtime as _rt
        return getattr(_rt, name)
    raise AttributeError(f"module 'aigraph' has no attribute {name!r}")
