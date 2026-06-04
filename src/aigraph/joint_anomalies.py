"""Joint (Atlas-grounded) anomaly detectors — the J2 prototype.

These run OUTSIDE the frozen v0.7 pipeline (anomalies.py is untouched) and
are opt-in: they require Intern-Atlas data. They compose with the 8 frozen
detectors by appending to an existing anomalies list.

`bottleneck_open_q_alignment` is the empirically-validated detector
(`docs/atlas-value-test-findings.md`): on the val1-primary cohort it fired
on 484 papers, 87% of which an LLM judged "complementary" (Atlas's
third-party bottleneck adds a NEW weakness dimension to the paper's own
first-party limitation, not a redundant or conflicting one). The test also
showed the value is in SURFACING these candidates as anomalies — not in
padding the hypothesis prompt — so this is a detector, not a prompt hack.

Signal:
  first-party weakness  = a paper's own negative-direction OR limitation claims
  third-party bottleneck = Atlas evolution edge into the paper carrying a
                           structured bottleneck_json {dimension, severity, quote}
A paper with BOTH is a `bottleneck_open_q_alignment` candidate.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .models import Anomaly, Claim, Paper

_VER_RE = re.compile(r"v\d+$")


def _base_arxiv_of_paper(p: Paper) -> str:
    v = p.arxiv_id_base or p.arxiv_id_full or ""
    if not v and str(p.paper_id).startswith("arxiv:"):
        v = p.paper_id.split(":", 1)[1]
    return _VER_RE.sub("", str(v)) if v else ""


def _is_weakness(c: Claim) -> bool:
    return getattr(c, "direction", None) == "negative" or getattr(c, "claim_type", None) == "limitation"


def _entity(c: Claim, primary: str, fallback: str) -> str:
    return (getattr(c, primary, None) or getattr(c, fallback, None) or "").strip()


def detect_bottleneck_open_q_alignment(
    claims: list[Claim],
    papers: list[Paper],
    atlas_dir: Path | str | None = None,
    *,
    min_confidence: float = 0.0,
    max_bottlenecks_per_paper: int = 4,
    id_prefix: str = "jb",
    inbound_bottlenecks: Optional[dict[str, list[dict]]] = None,
) -> list[Anomaly]:
    """Emit one `bottleneck_open_q_alignment` Anomaly per paper that has BOTH
    a first-party weakness claim AND an Atlas third-party bottleneck.

    ``inbound_bottlenecks`` may be passed in (e.g. for tests) to skip the
    Atlas parquet read; otherwise it is loaded via ``intern_atlas_loader``.
    """
    # first-party weakness claims, grouped by paper
    weakness_by_pid: dict[str, list[Claim]] = {}
    for c in claims:
        if _is_weakness(c):
            weakness_by_pid.setdefault(c.paper_id, []).append(c)
    if not weakness_by_pid:
        return []

    papers_by_pid = {p.paper_id: p for p in papers}
    # only papers that have weakness AND an arxiv id to join on
    pid_to_arxiv = {pid: _base_arxiv_of_paper(papers_by_pid[pid])
                    for pid in weakness_by_pid if pid in papers_by_pid}
    pid_to_arxiv = {pid: a for pid, a in pid_to_arxiv.items() if a}
    if not pid_to_arxiv:
        return []

    if inbound_bottlenecks is None:
        from .intern_atlas_loader import load_atlas_inbound_bottlenecks
        inbound_bottlenecks = load_atlas_inbound_bottlenecks(
            set(pid_to_arxiv.values()), atlas_dir, min_confidence=min_confidence)

    anomalies: list[Anomaly] = []
    i = 0
    for pid, arxiv in pid_to_arxiv.items():
        btls = inbound_bottlenecks.get(arxiv)
        if not btls:
            continue
        i += 1
        wk = weakness_by_pid[pid]
        # rank bottlenecks: severity then confidence
        sev_rank = {"fundamental": 3, "significant": 2, "moderate": 1, "minor": 0}
        btls = sorted(btls, key=lambda b: (sev_rank.get(b.get("severity", ""), 0),
                                           b.get("confidence", 0.0)), reverse=True)[:max_bottlenecks_per_paper]
        lead = wk[0]
        method = _entity(lead, "canonical_method", "method")
        task = _entity(lead, "canonical_task", "task")
        top = btls[0]
        dim = top.get("dimension") or "an unaddressed dimension"
        third = (top.get("quote") or top.get("description") or "a limitation").strip()
        first = (lead.claim_text or "").strip()

        # Build the central question around the CLEAN signals — the concrete
        # first-party limitation quote and the third-party bottleneck
        # (dimension + verbatim quote). Only name method/task when they are
        # real (the extractor often emits "other"/empty, which made earlier
        # CQs read "studies other on …" and dragged hypothesis quality down —
        # see docs/atlas-value-test-findings.md). The clean bottleneck is the
        # spine; method/task is an optional scope tag.
        _bad = {"", "other", "the method", "the task", "unknown", "none"}
        scope = ""
        if method.lower() not in _bad and task.lower() not in _bad:
            scope = f" (on {method} for {task})"
        elif method.lower() not in _bad:
            scope = f" (on {method})"
        cq = (
            f"This paper{scope} reports a limitation: \"{first[:200]}\". "
            f"Independently, later work identifies a {dim} bottleneck in it: "
            f"\"{third[:200]}\". What concrete mechanism would resolve both the "
            f"self-reported limitation and this {dim} bottleneck?"
        )

        anomalies.append(Anomaly(
            anomaly_id=f"{id_prefix}{i:03d}",
            type="bottleneck_open_q_alignment",
            central_question=cq,
            claim_ids=[c.claim_id for c in wk],
            negative_claims=[c.claim_id for c in wk],
            shared_entities={
                "method": method, "task": task,
                "bottleneck_dimension": dim,
                "paper_id": pid, "arxiv_id": arxiv,
            },
            evidence_impact=float(max((b.get("confidence", 0.0) for b in btls), default=0.0)),
            bottleneck_signals=[{
                "source_paper": b.get("source_paper", ""),
                "source_title": b.get("source_title", ""),
                "relation": b.get("relation", ""),
                "dimension": b.get("dimension", ""),
                "severity": b.get("severity", ""),
                "quote": b.get("quote", "") or b.get("description", ""),
            } for b in btls],
        ))
    return anomalies


def merge_joint_anomalies(existing: list[Anomaly], joint: list[Anomaly]) -> list[Anomaly]:
    """Append joint anomalies after the frozen ones, keeping ids unique."""
    seen = {a.anomaly_id for a in existing}
    out = list(existing)
    n = 0
    for a in joint:
        while a.anomaly_id in seen:
            n += 1
            a.anomaly_id = f"{a.anomaly_id}_{n}"
        seen.add(a.anomaly_id)
        out.append(a)
    return out
