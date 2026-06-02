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
    r"generational lag|venue separation|terminology drift|methodological prior|"
    # P4: the frozen template-generator's generic fallback frame — a vacuous
    # "some hidden variable explains the conflict" non-explanation.
    r"unreported moderator|moderator variable (?:drives|explains|accounts)|"
    r"an? (?:unreported|unmeasured|hidden|confounding) (?:moderator|variable|confound))",
    re.IGNORECASE,
)


def _is_boilerplate(hyp: Hypothesis) -> bool:
    """True for conflict-explanation meta-commentary (see _BOILERPLATE_RE)."""
    return bool(_BOILERPLATE_RE.search(hyp.hypothesis or ""))


def _is_untestable(hyp: Hypothesis) -> bool:
    """True if the hypothesis carries no minimal_test AND no concrete
    predictions — nothing to actually run as an experiment.

    Downstream the chat-mode advisor (talent/literature-conflict-graph
    launch.sh) is instructed to drop any hypothesis it can't turn into
    a (claim, construction, falsifiable-prediction) triple — these are
    exactly the ones it would drop. Filtering them earlier means MMR
    selects from a higher-quality pool instead of spending one of K
    slots on a candidate the LLM will discard.
    """
    has_test = bool((hyp.minimal_test or "").strip())
    has_predictions = any((p or "").strip() for p in (hyp.predictions or []))
    return not (has_test or has_predictions)


def _norm_entity(s) -> str:
    """Normalize an entity label for self-reference comparison."""
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _is_degenerate_anomaly(anom: "Anomaly | None") -> bool:
    """True for self-referential anomalies whose two sides are the same entity
    — e.g. method == task ("when does planning help on planning?") or a bridge
    whose from/to collapse. The frozen detectors emit these unguarded (no
    method != task check); they produce vacuous central questions, so we drop
    them at delivery time (the cached anomaly file is untouched).
    """
    if anom is None:
        return False
    se = anom.shared_entities or {}
    m, t = _norm_entity(se.get("method")), _norm_entity(se.get("task"))
    if m and t and m == t:
        return True
    mf, mt = _norm_entity(se.get("method_from")), _norm_entity(se.get("method_to"))
    tf, tt = _norm_entity(se.get("task_from")), _norm_entity(se.get("task_to"))
    if mf and mt and tf and tt and mf == mt and tf == tt:
        return True
    # community_disconnect / generic two-sided keys collapsing to one entity
    for a, b in (("from", "to"), ("community_from", "community_to"), ("source", "target")):
        va, vb = _norm_entity(se.get(a)), _norm_entity(se.get(b))
        if va and vb and va == vb:
            return True
    return False


# Conflict-type anomalies whose whole premise is a DISAGREEMENT and therefore
# require ≥2 distinct papers. A same-paper "conflict" is a detector artifact —
# the frozen (method,task) grouping in _detect_benchmark_inconsistency et al.
# pairs a positive vs a non-positive claim without requiring distinct papers,
# so a single paper that states a result AND (often a mis-labeled) description
# manufactures a fake conflict (e.g. CoARS a001: c016 'introduces two schemes'
# mis-tagged mixed + c017 'outperforms baselines', both arxiv:2604.10029).
# This is a literature-CONFLICT graph; intra-paper "conflicts" don't belong.
_CROSS_PAPER_CONFLICT_TYPES = frozenset({
    "benchmark_inconsistency", "impact_conflict", "metric_mismatch", "setting_mismatch",
})


def _is_same_paper_conflict(anom: "Anomaly | None", claim_lookup: dict) -> bool:
    """True for a conflict-type anomaly whose evidence claims span <2 distinct
    papers — a fabricated intra-paper conflict. Dropped at delivery (the frozen
    detector + cached anomaly file are untouched). replication_conflict already
    enforces distinct papers in its detector, so it's not included here."""
    if anom is None or anom.type not in _CROSS_PAPER_CONFLICT_TYPES:
        return False
    papers = {
        claim_lookup[c].paper_id
        for c in (anom.claim_ids or [])
        if c in claim_lookup and claim_lookup[c].paper_id
    }
    # Drop only on POSITIVE confirmation of a single paper. If claims don't
    # resolve (0 papers — incomplete claim_lookup), fail OPEN (keep): we can't
    # prove it's an artifact, and dropping unverifiable conflicts would silently
    # eat legit cross-paper ones.
    return len(papers) == 1


def _load_atlas_overlap_sidecar(run_dir: Path) -> dict[str, int]:
    """Load per-hypothesis Atlas-overlap scores from a sidecar produced by
    scripts/precompute_atlas_overlap.py (or the analytic m12_joined.jsonl).

    The sidecar format is one JSON record per line with at least
    {hypothesis_id, atlas_overlap}. Returns hyp_id → atlas_overlap (1-5);
    hypotheses missing from the sidecar are treated as "unscored" (return
    value is 0 for those when consulted).

    Empirical motivation (docs/method12-likert-vs-utility-verdict.md): the
    production scorer leaks ~20% tangential/unanchored hyps into the top-K;
    Likert atlas_overlap is statistically independent of utility (ρ=0.112)
    and identifies them. Filtering overlap ∈ {1, 2} drops 21% of hyps with
    no false-negative cost on the verified 259-hyp run.
    """
    sidecar = run_dir / "atlas_overlap.jsonl"
    if not sidecar.exists():
        return {}
    out: dict[str, int] = {}
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        hid = rec.get("hypothesis_id") or rec.get("hyp_id")
        ov = rec.get("atlas_overlap")
        if hid and isinstance(ov, int):
            out[hid] = ov
    return out


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
    drop_boilerplate: bool = False,
    min_relevance: int = 1,
    drop_self_conflict: bool = True,
    max_per_anomaly: int = 2,
    drop_untestable: bool = False,
    semantic: bool = False,
    semantic_threshold: float = 0.05,
    min_atlas_overlap: int = 0,
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
    # Optional TF-IDF cosine layer: catches "tree search" ↔ "MCTS" and other
    # phrase-level matches the per-token bag-of-words misses. When enabled,
    # a hypothesis enters `matched` if EITHER bag-of-words ≥ min_relevance
    # OR TF-IDF cosine ≥ semantic_threshold. Default off (back-compat).
    sem_by_id: dict[str, float] = {}
    if semantic:
        from aigraph.relevance import hypothesis_haystack, tfidf_topic_scores
        cited_per_h = {
            h.hypothesis_id: [claim_lookup[c] for c in (h.explains_claims or [])
                              if c in claim_lookup]
            for h in hyps
        }
        hyp_texts = [
            hypothesis_haystack(h, anom_lookup.get(h.anomaly_id),
                                cited_per_h.get(h.hypothesis_id))
            for h in hyps
        ]
        sem_scores = tfidf_topic_scores(hyp_texts, topic)
        sem_by_id = {h.hypothesis_id: s for h, s in zip(hyps, sem_scores)}
    # min_relevance gates topic scope-drift: when the corpus only loosely
    # matches the query (low token overlap), returning those hypotheses misleads
    # downstream stages into "off-topic" research ideas. Require >= min_relevance
    # query-token hits; if nothing clears the bar, the caller gets a no-match
    # (honest "insufficient coverage") instead of plausible-but-irrelevant output.
    if semantic:
        matched = [(h, r) for (h, r) in scored
                   if r >= min_relevance
                   or sem_by_id.get(h.hypothesis_id, 0.0) >= semantic_threshold]
    else:
        matched = [(h, r) for (h, r) in scored if r >= min_relevance]
    # Diagnostic: how many candidates the TF-IDF layer rescued (bow < gate
    # but sem ≥ threshold). 0 means the corpus had no semantic-only neighbors
    # for this topic (typical when bag-of-words already catches everything).
    semantic_only_added = 0
    if semantic:
        bow_matched_ids = {h.hypothesis_id for (h, r) in scored if r >= min_relevance}
        semantic_only_added = sum(
            1 for (h, _) in matched if h.hypothesis_id not in bow_matched_ids
        )
    if drop_boilerplate:
        concrete = [(h, r) for (h, r) in matched if not _is_boilerplate(h)]
        if concrete:  # only drop meta-commentary when actionable ones remain
            matched = concrete
    # Drop hypotheses whose parent anomaly is a degenerate self-conflict
    # (method==task etc.) — vacuous "X on X" framings the frozen detectors emit
    # unguarded — OR a same-paper "conflict" (all evidence from one paper), a
    # fabricated intra-paper disagreement. Both are detector ARTIFACTS, not
    # weak-but-real ideas, so the drop is UNCONDITIONAL: a fabricated conflict
    # is worse than an empty conflict section (the idea-generation cascade
    # backfills non-emptiness via community-bridge / forward tiers). This is
    # the only filter here that drops even when nothing remains.
    if drop_self_conflict:
        matched = [
            (h, r) for (h, r) in matched
            if not _is_degenerate_anomaly(anom_lookup.get(h.anomaly_id))
            and not _is_same_paper_conflict(anom_lookup.get(h.anomaly_id), claim_lookup)
        ]
    # Drop hypotheses whose precomputed Atlas-overlap score is below the
    # caller's threshold (default 0 = off). Overlap is a 1-5 Likert from a
    # one-time Kimi judge run; sidecar lives at run_dir/atlas_overlap.jsonl.
    # The Method 12 verdict (docs/method12-likert-vs-utility-verdict.md):
    # production utility leaks 20% tangential/unanchored hyps into top-K;
    # filtering overlap < 3 drops them with 79% survival rate.
    # Defensive: hypotheses not in the sidecar are kept (unscored ≠ bad);
    # the drop is skipped entirely when no candidate clears the bar.
    atlas_overlap_lookup: dict[str, int] = {}
    if min_atlas_overlap and min_atlas_overlap > 0:
        atlas_overlap_lookup = _load_atlas_overlap_sidecar(run_dir)
        if atlas_overlap_lookup:
            anchored = [
                (h, r) for (h, r) in matched
                if atlas_overlap_lookup.get(h.hypothesis_id, 0) == 0  # unscored: keep
                or atlas_overlap_lookup[h.hypothesis_id] >= min_atlas_overlap
            ]
            if anchored:
                matched = anchored
    # Drop hypotheses that lack any concrete test/prediction (the chat-mode
    # advisor would reject them anyway via its claim/construction/prediction
    # rule). Only drop when testable candidates remain.
    if drop_untestable:
        testable = [(h, r) for (h, r) in matched if not _is_untestable(h)]
        if testable:
            matched = testable
    # Rank concrete hypotheses ahead of conflict-explanation boilerplate, then
    # by topic relevance. Boilerplate stays in the list as a last-resort filler.
    # When semantic is enabled, blend bag-of-words count with TF-IDF cosine so a
    # high-semantic / low-token hypothesis isn't demoted (×10 scales cosine into
    # the same ballpark as token-hit counts for typical topics).
    if semantic:
        matched.sort(key=lambda hr: (
            _is_boilerplate(hr[0]),
            -max(hr[1], sem_by_id.get(hr[0].hypothesis_id, 0.0) * 10),
        ))
    else:
        matched.sort(key=lambda hr: (_is_boilerplate(hr[0]), -hr[1]))
    # Cap near-duplicate frames per anomaly: the frozen generator emits EXACTLY
    # 3 near-identical hypotheses per anomaly (differ only by moderator). MMR
    # handles cross-anomaly diversity; this removes the within-anomaly dups it
    # doesn't catch. Applied post-sort so the highest-relevance frames survive.
    if max_per_anomaly and max_per_anomaly > 0:
        per_anom: dict[str, int] = {}
        capped: list = []
        for h, r in matched:
            n = per_anom.get(h.anomaly_id, 0) + 1
            per_anom[h.anomaly_id] = n
            if n <= max_per_anomaly:
                capped.append((h, r))
        matched = capped

    base_stats = {
        "n_hypotheses_total": len(hyps),
        "n_matched": len(matched),
        "top_relevance": matched[0][1] if matched else 0,
        "topic_tokens": sorted(query_tokens),
        "llm_calls": 0,
        "semantic_enabled": semantic,
        "semantic_only_added": semantic_only_added,
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
    drop_boilerplate: bool = False,
    min_relevance: int = 1,
    drop_self_conflict: bool = True,
    max_per_anomaly: int = 2,
    drop_untestable: bool = False,
    semantic: bool = False,
    semantic_threshold: float = 0.05,
    min_atlas_overlap: int = 0,
) -> tuple[str, dict]:
    """Filter cached hypotheses by topic relevance and MMR-select top-K.

    ``hyp_kind``: "critic" (conflict explanations), "creator" (new-method
    proposals — the forward-looking research ideas), or "both".
    Returns (markdown, stats). Zero LLM calls.
    """
    selected, breakdowns, anoms, claims, papers, stats = _select(
        run_dir, topic, k=k, max_hypotheses=max_hypotheses,
        mmr_lambda=mmr_lambda, min_anomalies=min_anomalies, hyp_kind=hyp_kind,
        drop_boilerplate=drop_boilerplate, min_relevance=min_relevance,
        drop_self_conflict=drop_self_conflict, max_per_anomaly=max_per_anomaly,
        drop_untestable=drop_untestable,
        semantic=semantic, semantic_threshold=semantic_threshold,
        min_atlas_overlap=min_atlas_overlap,
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
    hyp_kind: str = "critic",
    drop_boilerplate: bool = False,
    min_relevance: int = 1,
    drop_self_conflict: bool = True,
    max_per_anomaly: int = 2,
    drop_untestable: bool = False,
    semantic: bool = False,
    semantic_threshold: float = 0.05,
    min_atlas_overlap: int = 0,
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
        hyp_kind=hyp_kind, drop_boilerplate=drop_boilerplate,
        min_relevance=min_relevance,
        drop_self_conflict=drop_self_conflict, max_per_anomaly=max_per_anomaly,
        drop_untestable=drop_untestable,
        semantic=semantic, semantic_threshold=semantic_threshold,
        min_atlas_overlap=min_atlas_overlap,
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
    ap.add_argument("--hyp-kind", choices=["critic", "creator", "both"],
                    default="critic",
                    help="Which hypothesis set to surface (creator = new-method "
                         "proposals; falls back to critic if absent)")
    ap.add_argument("--drop-boilerplate", action="store_true",
                    help="Drop conflict-explanation meta-commentary hypotheses "
                         "(LLM-judge ~2/10) when actionable ones also match")
    ap.add_argument("--min-relevance", type=int, default=1,
                    help="Min query-token hits a hypothesis must have to be "
                         "surfaced; raise to 2-3 to gate off-topic scope-drift "
                         "when the corpus only loosely matches the topic")
    ap.add_argument("--keep-self-conflict", action="store_true",
                    help="Keep degenerate self-conflict anomalies (method==task, "
                         "'X on X'); default drops them at delivery")
    ap.add_argument("--max-per-anomaly", type=int, default=2,
                    help="Max near-duplicate hypotheses kept per anomaly "
                         "(frozen generator emits 3 near-identical frames); 0 = no cap")
    ap.add_argument("--drop-untestable", action="store_true",
                    help="Drop hypotheses with no minimal_test AND no concrete "
                         "predictions — the chat-mode advisor rejects these "
                         "anyway, so dropping them earlier means MMR picks "
                         "from a higher-quality pool")
    ap.add_argument("--semantic", action="store_true",
                    help="Enable TF-IDF cosine relevance alongside bag-of-words. "
                         "Catches phrase-level matches the per-token filter "
                         "misses ('tree search' ↔ 'MCTS', 'uncertainty "
                         "quantification' ↔ 'calibration'). Off by default for "
                         "back-compat. Adds ~milliseconds per query.")
    ap.add_argument("--semantic-threshold", type=float, default=0.05,
                    help="Min TF-IDF cosine for a hypothesis to clear the "
                         "topic filter on semantic similarity alone (when its "
                         "bag-of-words hit-count is below --min-relevance). "
                         "Empirically the max cosine on a 4-token query vs the "
                         "540p reasoning corpus is ~0.14; the previous default "
                         "0.15 was a never-triggers threshold. 0.05 keeps the "
                         "top ~2-3 percent of semantically-related hypotheses. Only "
                         "actually changes selection when paired with a "
                         "stricter --min-relevance (≥ 2) — at min_relevance=1 "
                         "bag-of-words usually already catches everything "
                         "semantic would also catch. Only used when --semantic "
                         "is set.")
    ap.add_argument("--min-atlas-overlap", type=int, default=0,
                    help="Drop hypotheses whose precomputed Atlas-overlap "
                         "score (Likert 1-5) is below this threshold. 0 = off "
                         "(default; back-compat). 3 = Method 12 recommended "
                         "(filters tangential overlap=2 and unanchored "
                         "overlap=1 hyps). Requires a "
                         "<run_dir>/atlas_overlap.jsonl sidecar produced by "
                         "scripts/precompute_atlas_overlap.py — hyps not in "
                         "the sidecar are kept (unscored ≠ bad). See "
                         "docs/method12-likert-vs-utility-verdict.md.")
    ap.add_argument("--output", default="-",
                    help="'-' for stdout, else a file path")
    ap.add_argument("--stats-out", default=None,
                    help="Optional path to write stats as JSON")
    ap.add_argument("--records-out", default=None,
                    help="Optional path to write the selected hypotheses as a "
                         "structured JSON list (one record per hypothesis with "
                         "utility score breakdown). Use this when the caller "
                         "wants to consume ranked candidates programmatically "
                         "(e.g. inject LCG scores into a downstream LLM prompt) "
                         "instead of (or in addition to) the rendered markdown.")
    args = ap.parse_args()

    md, stats = query(
        run_dir=args.run_dir,
        topic=args.topic,
        k=args.k,
        max_hypotheses=args.max_hypotheses,
        mmr_lambda=args.mmr_lambda,
        min_anomalies=args.min_anomalies,
        hyp_kind=args.hyp_kind,
        drop_boilerplate=args.drop_boilerplate,
        min_relevance=args.min_relevance,
        drop_self_conflict=not args.keep_self_conflict,
        max_per_anomaly=args.max_per_anomaly,
        drop_untestable=args.drop_untestable,
        semantic=args.semantic,
        semantic_threshold=args.semantic_threshold,
        min_atlas_overlap=args.min_atlas_overlap,
    )

    if args.output == "-":
        # The report contains non-ASCII (em-dash —, arrow →): write UTF-8 bytes
        # directly so output is correct regardless of the process locale
        # (cron/tmux often run under a C/POSIX locale → stdout defaults to ascii,
        # which mangles or crashes on these characters → mojibake downstream).
        sys.stdout.buffer.write(md.encode("utf-8"))
    else:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)

    if args.stats_out:
        Path(args.stats_out).write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.records_out:
        records, _ = query_records(
            run_dir=args.run_dir,
            topic=args.topic,
            k=args.k,
            max_hypotheses=args.max_hypotheses,
            mmr_lambda=args.mmr_lambda,
            min_anomalies=args.min_anomalies,
            hyp_kind=args.hyp_kind,
            drop_boilerplate=args.drop_boilerplate,
            min_relevance=args.min_relevance,
            drop_self_conflict=not args.keep_self_conflict,
            max_per_anomaly=args.max_per_anomaly,
            drop_untestable=args.drop_untestable,
            semantic=args.semantic,
            semantic_threshold=args.semantic_threshold,
            min_atlas_overlap=args.min_atlas_overlap,
        )
        Path(args.records_out).write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(stats, indent=2, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
