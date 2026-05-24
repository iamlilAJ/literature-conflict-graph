"""Per-query layer over a cached aigraph run.

Service-mode POC. Loads a run directory produced by
``run_local_corpus.py`` / ``finish_local_run.py``, filters cached
hypotheses by topic relevance, runs MMR selection with diversity, and
renders the top-K to markdown. Default mode makes 0 LLM calls and
returns sub-second.

The point: corpus pre-processing (extract + graph + anomalies + hyp-gen
+ score) costs ~150 LLM calls one time per corpus. Each subsequent
user query at this layer costs 0 LLM calls (retrieve mode) or up to
``--llm-refine`` LLM calls (refine mode, optional).

Example::

    # 0 LLM calls, ~1 sec
    python3 scripts/aigraph_query.py \
        --run-dir artifacts/runs/arxiv-reasoning-v0.7-100p \
        --topic "agent reasoning" \
        --k 5 \
        --output -
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from aigraph.io import read_jsonl  # noqa: E402
from aigraph.models import Anomaly, Claim, Hypothesis, Paper  # noqa: E402
from aigraph.scoring import score_all, select_mmr  # noqa: E402
from aigraph.report import render_report  # noqa: E402


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "for", "and", "or", "in", "on", "at",
    "to", "by", "with", "as", "is", "are", "be", "this", "that",
    "from", "into", "it", "we", "you", "i",
})


def _tokenize(s: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((s or "").lower()) if t not in _STOPWORDS and len(t) > 1}


# Critic (conflict-explanation) hypotheses frequently open with meta-commentary
# about *why two research communities disagree* ("The disconnect may persist
# because…", "The contradiction arises because…", cross-pollination/venue/
# terminology framing) rather than proposing an actionable method. These read
# poorly as research ideas (measured: LLM-judge 2/10 vs creator 8/10). We don't
# drop them — they can be a last-resort fallback — but we demote them below any
# concrete hypothesis so the actionable bridges surface first. Creator
# hypotheses never match this pattern.
_BOILERPLATE_RE = re.compile(
    r"\b(the disconnect may persist|the disconnect persists|the contradiction "
    r"(?:arises|reflects|is driven)|cross-pollinat|may remain disconnected|"
    r"generational lag|venue separation|terminology drift|methodological prior)",
    re.IGNORECASE,
)


def _is_boilerplate(hyp: Hypothesis) -> bool:
    """True for conflict-explanation meta-commentary (see _BOILERPLATE_RE)."""
    return bool(_BOILERPLATE_RE.search(hyp.hypothesis or ""))


def _topic_relevance(
    hyp: Hypothesis,
    anomaly_lookup: dict[str, Anomaly],
    claims_lookup: dict[str, Claim],
    query_tokens: set[str],
) -> int:
    """Count of query tokens that appear anywhere in the hypothesis,
    its mechanism, its parent anomaly's central_question +
    shared_entities, and its cited claims' text. Cheap bag-of-words."""
    if not query_tokens:
        return 0
    haystack: list[str] = [
        hyp.hypothesis or "",
        hyp.mechanism or "",
        " ".join(hyp.predictions or []),
        hyp.evidence_gap or "",
    ]
    anomaly = anomaly_lookup.get(hyp.anomaly_id)
    if anomaly is not None:
        haystack.append(anomaly.central_question or "")
        haystack.extend(str(v) for v in (anomaly.shared_entities or {}).values())
    for cid in hyp.explains_claims or []:
        c = claims_lookup.get(cid)
        if c is not None:
            haystack.append(c.claim_text or "")
            for field in ("method", "task", "dataset", "metric"):
                v = getattr(c, field, None)
                if v:
                    haystack.append(str(v))
    text_tokens = _tokenize(" ".join(haystack))
    return len(query_tokens & text_tokens)


def _load_run_dir(
    run_dir: Path,
    hyp_kind: str = "critic",
) -> tuple[list[Hypothesis], list[Anomaly], list[Claim], list[Paper]]:
    """Load a run. ``hyp_kind`` selects which hypotheses to return:
      - "critic"  (default): conflict-explanation hypotheses (hypotheses_scored
        .jsonl, falling back to hypotheses.jsonl)
      - "creator": new-method-proposal hypotheses (creator_hypotheses.jsonl);
        these are the forward-looking research ideas. Falls back to critic if
        no creator file exists.
      - "both": creator hypotheses first, then critic.
    """
    critic_path = run_dir / "hypotheses_scored.jsonl"
    if not critic_path.exists():
        critic_path = run_dir / "hypotheses.jsonl"
    creator_path = run_dir / "creator_hypotheses.jsonl"

    hyp_paths: list[Path] = []
    if hyp_kind in ("creator", "both") and creator_path.exists():
        hyp_paths.append(creator_path)
    if hyp_kind in ("critic", "both") or not hyp_paths:
        hyp_paths.append(critic_path)

    needed = {
        "anomalies": run_dir / "anomalies.jsonl",
        "claims": run_dir / "claims.jsonl",
        "papers": run_dir / "papers.jsonl",
    }
    for path in hyp_paths + list(needed.values()):
        if not path.exists():
            raise FileNotFoundError(f"missing required file at {path}")
    hyps: list[Hypothesis] = []
    for p in hyp_paths:
        hyps.extend(read_jsonl(p, Hypothesis))
    anoms = read_jsonl(needed["anomalies"], Anomaly)
    claims = read_jsonl(needed["claims"], Claim)
    papers = read_jsonl(needed["papers"], Paper)
    return hyps, anoms, claims, papers


def _select(
    run_dir: Path,
    topic: str,
    *,
    k: int,
    max_hypotheses: int,
    mmr_lambda: float,
    min_anomalies: int,
    hyp_kind: str = "critic",
):
    """Shared core: topic-filter + MMR-select. Returns
    ``(selected, breakdowns, anoms, claims, papers, stats)`` or
    ``(None, None, anoms, claims, papers, stats)`` on no-match.
    """
    t0 = time.monotonic()
    hyps, anoms, claims, papers = _load_run_dir(run_dir, hyp_kind)

    query_tokens = _tokenize(topic)
    if not query_tokens:
        raise ValueError(f"no usable tokens in topic {topic!r} after stopword strip")

    anom_lookup = {a.anomaly_id: a for a in anoms}
    claim_lookup = {c.claim_id: c for c in claims}

    scored = [
        (h, _topic_relevance(h, anom_lookup, claim_lookup, query_tokens))
        for h in hyps
    ]
    matched = [(h, r) for (h, r) in scored if r > 0]
    # Rank concrete hypotheses ahead of conflict-explanation boilerplate, then
    # by topic relevance. Boilerplate stays in the list as a last-resort filler.
    matched.sort(key=lambda hr: (_is_boilerplate(hr[0]), -hr[1]))

    base_stats = {
        "n_hypotheses_total": len(hyps),
        "n_matched": len(matched),
        "top_relevance": matched[0][1] if matched else 0,
        "topic_tokens": sorted(query_tokens),
        "llm_calls": 0,
    }
    if not matched:
        base_stats.update(n_candidates=0, n_selected=0,
                          wall_seconds=round(time.monotonic() - t0, 3))
        return None, None, anoms, claims, papers, base_stats

    candidates = [h for (h, _) in matched[:max_hypotheses]]
    breakdowns = score_all(candidates, anoms, claims)
    selected = select_mmr(
        candidates, breakdowns,
        k=k, lambda_=mmr_lambda, min_anomalies=min_anomalies,
    )
    base_stats.update(
        n_candidates=len(candidates),
        n_selected=len(selected),
        wall_seconds=round(time.monotonic() - t0, 3),
    )
    return selected, breakdowns, anoms, claims, papers, base_stats


def query(
    run_dir: Path,
    topic: str,
    *,
    k: int = 5,
    max_hypotheses: int = 30,
    mmr_lambda: float = 0.7,
    min_anomalies: int = 2,
    hyp_kind: str = "critic",
) -> tuple[str, dict]:
    """Filter cached hypotheses by topic relevance and MMR-select top-K.

    ``hyp_kind``: "critic" (conflict explanations), "creator" (new-method
    proposals — the forward-looking research ideas), or "both".
    Returns (markdown, stats). Zero LLM calls.
    """
    selected, breakdowns, anoms, claims, papers, stats = _select(
        run_dir, topic, k=k, max_hypotheses=max_hypotheses,
        mmr_lambda=mmr_lambda, min_anomalies=min_anomalies, hyp_kind=hyp_kind,
    )
    if selected is None:
        return f"# Selected Hypotheses\n\n_No matches for topic_ `{topic}`.\n", stats
    md = render_report(
        selected=selected,
        anomalies=anoms,
        claims=claims,
        scores=breakdowns,
        paper_lookup={p.paper_id: p for p in papers},
        topic=topic,
        paper_count=len(papers),
    )
    return md, stats


def query_records(
    run_dir: Path,
    topic: str,
    *,
    k: int = 5,
    max_hypotheses: int = 30,
    mmr_lambda: float = 0.7,
    min_anomalies: int = 2,
) -> tuple[list[dict], dict]:
    """Like ``query()`` but returns structured hypothesis records (for
    MCP / programmatic clients) instead of rendered markdown.

    Each record: hypothesis_id, anomaly_id, anomaly_type,
    central_question, hypothesis, mechanism, predictions, minimal_test,
    scope_conditions, evidence_gap, graph_bridge, evidence_claims
    (list of {claim_id, paper_id, title, year, direction, claim_text}),
    and utility (the score breakdown). Returns (records, stats).
    Zero LLM calls.
    """
    selected, breakdowns, anoms, claims, papers, stats = _select(
        run_dir, topic, k=k, max_hypotheses=max_hypotheses,
        mmr_lambda=mmr_lambda, min_anomalies=min_anomalies,
    )
    if selected is None:
        return [], stats

    anom_lookup = {a.anomaly_id: a for a in anoms}
    claim_lookup = {c.claim_id: c for c in claims}
    paper_lookup = {p.paper_id: p for p in papers}
    # score_all returns dict[hypothesis_id, ScoreBreakdown]
    score_lookup = breakdowns

    records = []
    for h in selected:
        anom = anom_lookup.get(h.anomaly_id)
        ev = []
        for cid in (h.explains_claims or []):
            c = claim_lookup.get(cid)
            if not c:
                continue
            p = paper_lookup.get(c.paper_id)
            ev.append({
                "claim_id": cid,
                "paper_id": c.paper_id,
                "title": p.title if p else None,
                "year": p.year if p else None,
                "direction": c.direction,
                "claim_text": c.claim_text,
            })
        sb = score_lookup.get(h.hypothesis_id)
        records.append({
            "hypothesis_id": h.hypothesis_id,
            "anomaly_id": h.anomaly_id,
            "anomaly_type": anom.type if anom else None,
            "central_question": anom.central_question if anom else None,
            "hypothesis": h.hypothesis,
            "mechanism": h.mechanism,
            "predictions": list(h.predictions or []),
            "minimal_test": h.minimal_test,
            "scope_conditions": dict(h.scope_conditions or {}),
            "evidence_gap": h.evidence_gap,
            "graph_bridge": {"from": h.graph_bridge.from_, "to": h.graph_bridge.to}
                            if h.graph_bridge else None,
            "evidence_claims": ev,
            "utility": (sb.model_dump() if sb else None),
        })
    return records, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--topic", required=True, type=str)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-hypotheses", type=int, default=30,
                    help="Cap on candidates fed to MMR")
    ap.add_argument("--mmr-lambda", type=float, default=0.7)
    ap.add_argument("--min-anomalies", type=int, default=2)
    ap.add_argument("--output", default="-",
                    help="'-' for stdout, else a file path")
    ap.add_argument("--stats-out", default=None,
                    help="Optional path to write stats as JSON")
    args = ap.parse_args()

    md, stats = query(
        run_dir=args.run_dir,
        topic=args.topic,
        k=args.k,
        max_hypotheses=args.max_hypotheses,
        mmr_lambda=args.mmr_lambda,
        min_anomalies=args.min_anomalies,
    )

    if args.output == "-":
        sys.stdout.write(md)
    else:
        Path(args.output).write_text(md)
        print(f"wrote {args.output}", file=sys.stderr)

    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(json.dumps(stats, indent=2, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
