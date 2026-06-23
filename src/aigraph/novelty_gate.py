"""Method-novelty gate (NON-FROZEN, delivery-time prior-art check).

The creator path proposes NEW methods, but its biggest failure mode is the
novelty collapse: proposing something the literature already does (an earlier
probe found ~4/5 "new" methods already existed). Analysis ideas don't need this
— a method paper does. This gate is the prior-art layer for the method path.

Same architecture as ``numeric_gate`` / ``hypothesis_enricher``: one cached
prior-art check per hypothesis (reuses ``novelty_check.check_hypothesis_novelty``
— mine keywords from the proposal, query arXiv for the closest papers, ask the
LLM whether it is substantively novel), persisted to a sidecar
(``novelty_flags.jsonl``), a 0-LLM overlay (``apply_novelty``) at delivery that
stamps ``Hypothesis.novelty_flag``, and fail-open everywhere (arXiv unreachable /
no key / error → ``is_novel=None`` → no flag, deliver as-is).

A hypothesis is flagged ONLY when the check is confident it is NOT novel
(``is_novel is False``) — surfaced as "⚠ prior art" with the colliding papers.
``is_novel is None`` (check could not run) is treated as unknown, NOT a flag.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from .io import read_jsonl
from .llm_client import build_openai_client, configured_api_key, configured_model
from .models import Hypothesis
from .novelty_check import ARXIV_RATE_LIMIT_SECONDS, check_hypothesis_novelty

__all__ = [
    "novelty_gate_enabled",
    "novelty_run",
    "load_novelty",
    "apply_novelty",
    "NOVELTY_FILENAME",
]

NOVELTY_FILENAME = "novelty_flags.jsonl"
_DEFAULT_LIMIT = 24


def novelty_gate_enabled() -> bool:
    """Off by default — the prior-art check makes an arXiv request per
    hypothesis (rate-limited to ~1/3s), so it is opt-in rather than run on every
    build. AIGRAPH_NOVELTY_GATE=1 enables it (and a key must exist)."""
    if os.environ.get("AIGRAPH_NOVELTY_GATE", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    return bool(configured_api_key())


def _max_similar() -> int:
    try:
        return max(1, int(os.environ.get("AIGRAPH_NOVELTY_MAX_CANDIDATES", "5")))
    except (TypeError, ValueError):
        return 5


def novelty_run(
    run_dir: Path | str,
    *,
    hyp_file: str = "creator_hypotheses.jsonl",
    force: bool = False,
    only_types: Optional[Iterable[str]] = None,
    limit: int = _DEFAULT_LIMIT,
    llm_client: Any | None = None,
) -> dict[str, dict]:
    """Prior-art-check a run's hypotheses; persist the novelty sidecar. Defaults
    to the creator (method-proposal) file — that is where novelty matters. Returns
    ``{hypothesis_id: {is_novel, similar_papers, rationale}}`` for everything
    cached. No-op returning the existing cache when disabled / no key."""
    run_dir = Path(run_dir)
    existing = load_novelty(run_dir)
    if not novelty_gate_enabled():
        return existing
    hpath = run_dir / hyp_file
    if not hpath.exists():
        return existing
    hyps: list[Hypothesis] = read_jsonl(hpath, Hypothesis)
    type_filter = set(only_types) if only_types else None

    if llm_client is None:
        try:
            llm_client = build_openai_client()
        except Exception:
            return existing
    model = configured_model()
    max_c = _max_similar()

    out = dict(existing)
    made = 0
    for hyp in hyps:
        if made >= max(0, int(limit)):
            break
        if not force and hyp.hypothesis_id in out:
            continue
        # (creator hyps have no anomaly type; only_types only filters when set)
        if type_filter is not None and getattr(hyp, "anomaly_type", None) not in type_filter:
            continue
        if made > 0 and ARXIV_RATE_LIMIT_SECONDS > 0:
            time.sleep(ARXIV_RATE_LIMIT_SECONDS)  # polite arXiv rate limit
        res = check_hypothesis_novelty(hyp, llm_client=llm_client, model=model, max_candidates=max_c)
        out[hyp.hypothesis_id] = {
            "is_novel": res.get("is_novel"),
            "similar_papers": res.get("similar_papers", [])[:3],
            "rationale": str(res.get("rationale", ""))[:300],
        }
        made += 1

    if made:
        _write_sidecar(run_dir, out)
    return out


def _write_sidecar(run_dir: Path, recs: dict[str, dict]) -> None:
    import json
    try:
        tmp = run_dir / (NOVELTY_FILENAME + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for hid, rec in recs.items():
                f.write(json.dumps({"hypothesis_id": hid, **rec}, ensure_ascii=False) + "\n")
        tmp.replace(run_dir / NOVELTY_FILENAME)
    except Exception:
        pass


def load_novelty(run_dir: Path | str) -> dict[str, dict]:
    """Read the novelty sidecar (0-LLM). ``{hypothesis_id: {is_novel, ...}}``."""
    import json
    path = Path(run_dir) / NOVELTY_FILENAME
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            hid = rec.get("hypothesis_id")
            if hid:
                out[hid] = {"is_novel": rec.get("is_novel"),
                            "similar_papers": rec.get("similar_papers", []),
                            "rationale": rec.get("rationale", "")}
    except Exception:
        return out
    return out


def apply_novelty(selected: list[Hypothesis], run_dir: Path | str) -> int:
    """Overlay cached novelty verdicts onto selected Hypothesis objects in place
    (0-LLM). Stamps ``hyp.novelty_flag`` ONLY on hypotheses confidently judged
    NOT novel (``is_novel is False``) — the prior-art collisions a method paper
    must avoid. Returns the count flagged. No-op when the sidecar is absent."""
    recs = load_novelty(run_dir)
    if not recs:
        return 0
    n = 0
    for hyp in selected:
        rec = recs.get(hyp.hypothesis_id)
        if not rec or rec.get("is_novel") is not False:
            continue  # novel or unknown -> no flag
        try:
            hyp.novelty_flag = {
                "is_novel": False,
                "similar_papers": rec.get("similar_papers", [])[:3],
                "rationale": rec.get("rationale", ""),
            }
            n += 1
        except Exception:
            pass
    return n
