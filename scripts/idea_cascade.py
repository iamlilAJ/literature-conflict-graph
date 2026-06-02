"""Cascade idea generator — guarantees a non-empty idea set for any topic+corpus.

Product contract: ideas(topic, run) ALWAYS returns >= min_ideas ideas as long
as the run has >= 1 paper. It does this by cascading through six tiers from
highest-signal to most-permissive, stopping as soon as it has enough:

  Tier A  critic-conflict      cross-paper contradiction hypotheses (needs anomalies)
  Tier B  creator-newmethod    new-method proposals grounded in open questions (needs anomalies)
  Tier C  community-bridge     cross-community unifying-theory insights (needs >=2 communities)
  Tier D  method-extension     per-paper "extend this method" ideas (needs >=1 paper)   [LLM]
  Tier E  limitation-forward   turn self-reported limitations into research directions  [LLM]
  Tier F  paper-seeded         deterministic abstract/limitation-seeded directions       [no LLM]

Tiers A/B/C read already-generated run artifacts (0 LLM). Tiers D/E are LLM
generators that work on a barren corpus (0 anomalies, 0 open_questions). Tier F
is the DETERMINISTIC terminal backstop (no LLM) that makes the non-empty
guarantee structural — it cannot fail given >=1 paper, regardless of LLM
reachability or JSON validity. D/E results are cached to
<run>/forward_ideas.jsonl so subsequent calls are 0-LLM.

Both critic and creator are anomaly-gated upstream, so on a sparse/too-new
corpus (many distinct method names, no cross-paper conflicts) A and B yield
nothing and the cascade falls through to C/D/E.

Usage (standalone):
    python scripts/idea_cascade.py --run artifacts/runs/<id> \
        --topic "agentic memory for self-evolving agent" --min-ideas 5
    # add --json for machine output, --no-llm to forbid D/E (may return < min)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from aigraph.io import read_jsonl  # noqa: E402
from aigraph.models import Paper, Claim, Insight  # noqa: E402
from aigraph.llm_client import (  # noqa: E402
    build_openai_client,
    call_llm_text,
    configured_model,
)
from aigraph_query import query_records, _tokenize  # noqa: E402


# --------------------------------------------------------------------------- #
# Idea record helpers
# --------------------------------------------------------------------------- #

TIER_LABELS = {
    "A": "critic-conflict",
    "B": "creator-newmethod",
    "C": "community-bridge",
    "D": "method-extension",
    "E": "limitation-forward",
    "F": "paper-seeded",
}
# Higher = surfaced first when trimming to k. A/B are most grounded; F is the
# deterministic last-resort backstop. Ties broken by per-idea confidence.
TIER_RANK = {"A": 6, "B": 5, "C": 4, "D": 3, "E": 2, "F": 1}


def _idea_id(tier: str, seed: str) -> str:
    h = hashlib.sha1(f"{tier}:{seed}".encode("utf-8")).hexdigest()[:8]
    return f"idea_{tier.lower()}_{h}"


def _mk_idea(tier: str, *, title: str, statement: str, mechanism: str = "",
             minimal_test: str = "", predictions: list[str] | None = None,
             source_papers: list[str] | None = None, grounding: str = "",
             confidence: float = 0.5, scope: dict | None = None,
             seed: str = "", extra: dict | None = None) -> dict:
    # predictions may arrive straight from LLM JSON — coerce non-list shapes
    # (a bare string would otherwise iterate into single characters).
    if isinstance(predictions, list):
        preds = predictions
    elif predictions in (None, ""):
        preds = []
    else:
        preds = [str(predictions)]
    try:
        conf = round(float(confidence), 3)
    except (TypeError, ValueError):
        conf = 0.5
    return {
        "idea_id": _idea_id(tier, seed or title),
        "tier": tier,
        "tier_label": TIER_LABELS.get(tier, tier),
        "title": (title or "").strip()[:200],
        "statement": (statement or "").strip(),
        "mechanism": (mechanism or "").strip(),
        "minimal_test": (minimal_test or "").strip(),
        "predictions": [str(p).strip() for p in preds if p and str(p).strip()][:3],
        "source_papers": list(dict.fromkeys(source_papers or []))[:6],
        "grounding": (grounding or "").strip()[:300],
        "confidence": conf,
        "scope": scope or {},
        **({"novelty_vs_paper": str(extra["novelty_vs_paper"]).strip()[:300]}
           if extra and extra.get("novelty_vs_paper") else {}),
    }


def _safe_read_jsonl(path: Path, model) -> list:
    """Tolerant JSONL read: skip unparseable/schema-invalid lines instead of
    aborting the whole read (a half-written or stale-schema line must not crash
    the cascade). Mirrors read_jsonl but per-line guarded."""
    if not path.exists():
        return []
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(model.model_validate_json(line))
            except Exception:
                continue
    except Exception:
        return out
    return out


# --------------------------------------------------------------------------- #
# Tier A / B — cached hypotheses (0 LLM)
# --------------------------------------------------------------------------- #

def _records_to_ideas(records: list[dict], tier: str) -> list[dict]:
    ideas = []
    for r in records:
        util = (r.get("utility") or {})
        conf = 0.7
        if isinstance(util, dict):
            try:
                conf = float(util.get("utility", 0.7))
            except (TypeError, ValueError):
                conf = 0.7
        papers = []
        for ev in (r.get("evidence_claims") or []):
            pid = ev.get("paper_id")
            if pid:
                papers.append(pid)
        ideas.append(_mk_idea(
            tier,
            title=(r.get("hypothesis") or "")[:120],
            statement=r.get("hypothesis", ""),
            mechanism=r.get("mechanism", ""),
            minimal_test=r.get("minimal_test", ""),
            predictions=r.get("predictions", []),
            source_papers=papers,
            grounding=r.get("central_question", "") or (r.get("anomaly_type") or ""),
            confidence=conf,
            scope=r.get("scope_conditions") or {},
            seed=r.get("hypothesis_id", "") or r.get("hypothesis", ""),
        ))
    return ideas


def tier_a_critic(run_dir: Path, topic: str, k: int) -> list[dict]:
    try:
        records, _ = query_records(run_dir, topic, k=k, hyp_kind="critic",
                                   min_anomalies=1, min_atlas_overlap=0)
    except Exception:
        return []
    return _records_to_ideas(records, "A")


def tier_b_creator(run_dir: Path, topic: str, k: int) -> list[dict]:
    # creator hyps live in creator_hypotheses.jsonl; query_records falls back to
    # critic when that file is absent, so guard on the file existing to avoid
    # double-counting critic hyps as creator.
    if not (run_dir / "creator_hypotheses.jsonl").exists():
        return []
    try:
        records, _ = query_records(run_dir, topic, k=k, hyp_kind="creator",
                                   min_anomalies=1, min_atlas_overlap=0)
    except Exception:
        return []
    return _records_to_ideas(records, "B")


# --------------------------------------------------------------------------- #
# Tier C — community bridge insights (0 LLM)
# --------------------------------------------------------------------------- #

def tier_c_bridges(run_dir: Path, topic: str, k: int) -> list[dict]:
    path = run_dir / "insights.jsonl"
    if not path.exists():
        return []
    try:
        insights = read_jsonl(path, Insight)
    except Exception:
        return []
    insights = sorted(insights, key=lambda i: -(i.quality_score or 0.0))
    ideas = []
    for ins in insights:
        # Each insight's transfer_suggestions are concrete forward directions.
        suggestions = [s for s in (ins.transfer_suggestions or []) if s.strip()]
        statement = ins.unifying_frame or ins.insight or ins.title
        minimal_test = suggestions[0] if suggestions else ""
        preds = suggestions[1:3]
        ideas.append(_mk_idea(
            "C",
            title=ins.title,
            statement=statement,
            mechanism=ins.insight or "",
            minimal_test=minimal_test,
            predictions=preds,
            source_papers=list(ins.evidence_papers or []),
            grounding=f"cross-community bridge: {', '.join(ins.communities or [])}; "
                      f"{ins.citation_gap or ''}",
            confidence=float(ins.quality_score or ins.confidence_score or 0.4),
            scope={"communities": list(ins.communities or []),
                   "shared_concepts": list(ins.shared_concepts or [])},
            seed=ins.insight_id,
        ))
    return ideas[:k] if k else ideas


# --------------------------------------------------------------------------- #
# Tier D / E — LLM forward generators (anomaly-free; the non-empty guarantee)
# --------------------------------------------------------------------------- #

def _parse_json_block(raw: str) -> Any:
    if not raw:
        return None
    s, e = raw.find("{"), raw.rfind("}")
    a, b = raw.find("["), raw.rfind("]")
    # prefer whichever bracket type encloses the larger span
    cand = None
    if s != -1 and e > s:
        cand = raw[s:e + 1]
    if a != -1 and b > a and (cand is None or (b - a) > (e - s)):
        cand = raw[a:b + 1]
    if cand is None:
        return None
    try:
        return json.loads(cand)
    except Exception:
        return None


def _topic_rank_papers(papers: list[Paper], topic: str) -> list[Paper]:
    qtok = _tokenize(topic)

    def score(p: Paper) -> float:
        text = f"{p.title} {p.abstract}"
        ptok = _tokenize(text)
        overlap = len(qtok & ptok)
        return overlap * 2.0 + float(p.selection_score or 0.0)

    return sorted(papers, key=score, reverse=True)


_D_SYS = (
    "You are a research scientist proposing concrete, testable extensions of an "
    "existing method. Given ONE paper (its method, claims, and any stated "
    "limitations), propose ONE specific NEW method that extends or fixes it.\n"
    "CRITICAL NOVELTY RULE: the abstract describes what the paper ALREADY does. "
    "Your proposal must go strictly BEYOND it — do NOT restate, rename, or "
    "re-operationalize the paper's own contribution. If the paper already adds "
    "memory / dense rewards / diversity / retrieval / self-distillation, "
    "proposing to 'add' that is NOT novel. Name explicitly what is NEW that the "
    "paper does not already do.\n"
    "Output STRICT JSON, no markdown:\n"
    '{"title":"<=10 word name","statement":"2-3 sentence proposal naming a '
    'concrete mechanism","mechanism":"the causal mechanism","minimal_test":'
    '"one single-variable experiment with named dataset+metric","predictions":'
    '["short prediction 1","short prediction 2"],"novelty_vs_paper":"one '
    'sentence: what this adds that the paper does NOT already do"}'
)

# P5 novelty gate: a separate adversarial check that catches the failure mode
# the research_ideas eval found (4/5 ideas restated the cited paper's own
# contribution). Asks a skeptic whether the abstract ALREADY covers the
# proposal; drops ideas that do. Cheap (1 short LLM call per tier-D idea,
# cached with the idea). Fails OPEN (keep) on any error — never blocks the
# non-empty guarantee.
_NOVELTY_GATE_SYS = (
    "You are a strict novelty reviewer. Given a paper ABSTRACT and a PROPOSED "
    "extension, decide whether the abstract ALREADY describes that proposal as "
    "the paper's own contribution (i.e. the proposal is a restatement, not a "
    "novel extension). Be skeptical: 'add memory/diversity/dense-reward to X' "
    "when the abstract says X already has it is NOT novel. "
    'Output STRICT JSON: {"already_in_paper": true|false, "reason":"one sentence"}'
)


def _novelty_gate(client, model, abstract: str, idea: dict) -> bool:
    """True if the idea is genuinely novel vs the paper abstract (keep it);
    False if the abstract already covers it (drop). Fails OPEN (True)."""
    if not abstract:
        return True
    user = json.dumps({
        "abstract": abstract[:1500],
        "proposed_title": idea.get("title", ""),
        "proposed_statement": idea.get("statement", ""),
        "claimed_novelty": idea.get("novelty_vs_paper", ""),
    }, ensure_ascii=False)
    try:
        raw = call_llm_text(client, model=model, system=_NOVELTY_GATE_SYS,
                            user=user, max_tokens=400)
        obj = _parse_json_block(raw)
        if isinstance(obj, list):
            obj = next((x for x in obj if isinstance(x, dict)), None)
        if isinstance(obj, dict) and obj.get("already_in_paper") is True:
            return False
    except Exception:
        return True
    return True

_E_SYS = (
    "You are a research scientist turning papers' self-reported limitations into "
    "forward research directions. Given a list of limitations/failure-modes from "
    "recent papers, propose N distinct, concrete, testable research directions "
    "that would address them. Each must name a mechanism and a single-variable "
    "minimal test. Output STRICT JSON, no markdown:\n"
    '{"ideas":[{"title":"<=10 words","statement":"2-3 sentences","mechanism":'
    '"...","minimal_test":"one experiment with dataset+metric","predictions":'
    '["p1","p2"],"addresses":"which limitation"}]}'
)


def _llm_client_and_model():
    client = build_openai_client()
    model = configured_model()
    return client, model


def tier_d_method_extensions(run_dir: Path, topic: str, papers: list[Paper],
                             claims: list[Claim], need: int,
                             max_papers: int = 8) -> list[dict]:
    if need <= 0:
        return []
    try:
        client, model = _llm_client_and_model()
    except Exception:
        return []  # missing openai / bad client config → fall through to Tier F
    claims_by_paper: dict[str, list[Claim]] = {}
    for c in claims:
        claims_by_paper.setdefault(c.paper_id, []).append(c)
    ranked = _topic_rank_papers(papers, topic)
    ideas: list[dict] = []
    for p in ranked[: max(max_papers, need)]:
        if len(ideas) >= need:
            break
        pclaims = claims_by_paper.get(p.paper_id, [])
        method = next((c.method for c in pclaims if c.method), None) or "the paper's method"
        limits = [c.claim_text for c in pclaims if c.claim_type == "limitation"][:4]
        fails = [c.failure_mode for c in pclaims if (c.failure_mode or "").strip()][:4]
        user = json.dumps({
            "paper_title": p.title,
            "paper_id": p.paper_id,
            "abstract": (p.abstract or "")[:1200],
            "method": method,
            "limitations": limits,
            "failure_modes": fails,
            "topic": topic,
        }, ensure_ascii=False)
        try:
            raw = call_llm_text(client, model=model, system=_D_SYS, user=user,
                                max_tokens=2000)
        except Exception:
            continue
        obj = _parse_json_block(raw)
        if isinstance(obj, list):  # model wrapped the object in an array
            obj = next((x for x in obj if isinstance(x, dict)), None)
        if not isinstance(obj, dict) or not obj.get("statement"):
            continue
        # P5 novelty gate: drop ideas that merely restate the paper's own
        # contribution (the eval's dominant failure). Fails open on error.
        if not _novelty_gate(client, model, p.abstract or "", obj):
            continue
        ideas.append(_mk_idea(
            "D",
            title=obj.get("title", "") or f"Extension of {method}",
            statement=obj.get("statement", ""),
            mechanism=obj.get("mechanism", ""),
            minimal_test=obj.get("minimal_test", ""),
            predictions=obj.get("predictions", []),
            source_papers=[p.paper_id],
            grounding=f"extends {method} ({p.title[:80]})",
            confidence=0.55,
            scope={"method": method},
            seed=p.paper_id,
            extra={"novelty_vs_paper": obj.get("novelty_vs_paper", "")},
        ))
    return ideas


def tier_e_limitation_forward(run_dir: Path, topic: str, papers: list[Paper],
                              claims: list[Claim], need: int) -> list[dict]:
    if need <= 0:
        return []
    try:
        client, model = _llm_client_and_model()
    except Exception:
        return []  # missing openai / bad client config → fall through to Tier F
    # Build the limitation list from claims; fall back to abstracts if none.
    paper_title = {p.paper_id: p.title for p in papers}
    limitations = []
    for c in claims:
        txt = ""
        if c.claim_type == "limitation":
            txt = c.claim_text
        elif (c.failure_mode or "").strip():
            txt = f"{c.method or ''}: {c.failure_mode}"
        if txt:
            limitations.append({
                "paper_id": c.paper_id,
                "paper": paper_title.get(c.paper_id, ""),
                "limitation": txt[:300],
            })
    if not limitations:
        # ultimate fallback: derive pseudo-limitations from abstracts
        for p in _topic_rank_papers(papers, topic)[:10]:
            if (p.abstract or "").strip():
                limitations.append({
                    "paper_id": p.paper_id,
                    "paper": p.title,
                    "limitation": f"(from abstract) {p.abstract[:280]}",
                })
    if not limitations:
        return []
    limitations = limitations[:18]
    user = json.dumps({
        "topic": topic,
        "n_requested": max(need, 3),
        "limitations": limitations,
    }, ensure_ascii=False)
    try:
        raw = call_llm_text(client, model=model, system=_E_SYS, user=user,
                            max_tokens=3500)
    except Exception:
        return []
    obj = _parse_json_block(raw)
    items = (obj or {}).get("ideas") if isinstance(obj, dict) else (obj if isinstance(obj, list) else None)
    if not items:
        return []
    # map each idea back to a source paper when the model named the limitation
    lim_by_text = {l["limitation"][:60]: l for l in limitations}
    ideas = []
    for it in items:
        if not isinstance(it, dict) or not it.get("statement"):
            continue
        addresses = it.get("addresses", "")
        src = []
        for key, l in lim_by_text.items():
            if key[:30] and key[:30] in str(addresses):
                src = [l["paper_id"]]
                break
        ideas.append(_mk_idea(
            "E",
            title=it.get("title", ""),
            statement=it.get("statement", ""),
            mechanism=it.get("mechanism", ""),
            minimal_test=it.get("minimal_test", ""),
            predictions=it.get("predictions", []),
            source_papers=src,
            grounding=f"addresses limitation: {addresses}"[:300],
            confidence=0.45,
            seed=it.get("title", "") + it.get("statement", "")[:40],
        ))
    return ideas


# --------------------------------------------------------------------------- #
# Tier F — deterministic, LLM-free terminal backstop (the structural guarantee)
# --------------------------------------------------------------------------- #

def tier_f_deterministic(topic: str, papers: list[Paper], claims: list[Claim],
                         need: int) -> list[dict]:
    """Last resort with NO LLM round-trip. Turns each top topic-ranked paper's
    abstract (or self-reported limitation, or — as a final fallback — its title)
    into a replicate-and-ablate research direction. This makes the non-empty
    guarantee STRUCTURAL: it cannot fail as long as >=1 paper exists, regardless
    of whether the LLM is reachable or emits parseable JSON.
    """
    if need <= 0:
        return []
    limit_by_paper: dict[str, str] = {}
    for c in claims:
        if c.claim_type == "limitation" and c.claim_text and c.paper_id not in limit_by_paper:
            limit_by_paper[c.paper_id] = c.claim_text
        elif (c.failure_mode or "").strip() and c.paper_id not in limit_by_paper:
            limit_by_paper[c.paper_id] = f"{c.method or ''}: {c.failure_mode}".strip(": ")
    ideas = []
    for p in _topic_rank_papers(papers, topic):
        if len(ideas) >= need:
            break
        lim = limit_by_paper.get(p.paper_id, "")
        if lim:
            statement = (f"Address the limitation reported by '{p.title}': {lim} "
                         f"Propose and test a method that removes this constraint.")
            grounding = f"self-reported limitation of {p.title[:80]}"
        elif (p.abstract or "").strip():
            statement = (f"Build on '{p.title}'. {p.abstract[:240]} "
                         f"Extend this with a concrete new mechanism and test the gain.")
            grounding = f"abstract of {p.title[:80]}"
        else:
            statement = (f"Investigate the approach in '{p.title}' for {topic}: "
                         f"replicate its main result, then ablate its core component.")
            grounding = f"title of {p.title[:80]}"
        ideas.append(_mk_idea(
            "F",
            title=f"Extend: {p.title[:90]}",
            statement=statement,
            minimal_test=("Replicate the paper's headline result, then ablate its "
                          "core component to isolate the contribution."),
            source_papers=[p.paper_id],
            grounding=grounding,
            confidence=0.3,
            seed=p.paper_id,
        ))
    return ideas


# --------------------------------------------------------------------------- #
# Cascade orchestrator
# --------------------------------------------------------------------------- #

def _idea_key(idea: dict) -> str:
    text = (idea.get("statement") or idea.get("title") or "")
    return re.sub(r"\W+", " ", text.lower()).strip()[:80]


def _dedup(ideas: list[dict]) -> list[dict]:
    seen, out = set(), []
    for idea in ideas:
        if not isinstance(idea, dict):
            continue
        key = _idea_key(idea)
        if key and key in seen:
            continue
        seen.add(key)
        out.append(idea)
    return out


def _valid_cached(records: list) -> list[dict]:
    """Keep only cached idea dicts that carry the fields the orchestrator
    indexes (tier/confidence/statement|title). Drops stale-schema/torn lines so
    they can never crash _dedup or the final sort."""
    ok = []
    for r in records:
        if (isinstance(r, dict) and r.get("tier") and (r.get("statement") or r.get("title"))
                and isinstance(r.get("confidence", 0.0), (int, float))):
            ok.append(r)
    return ok


def generate_ideas(run_dir: Path, topic: str, *, min_ideas: int = 5,
                   allow_llm: bool = True, cache: bool = True) -> dict:
    """Cascade A->E until >= min_ideas. Returns {ideas, stats}.

    Guarantee: if run has >= 1 paper with an abstract and allow_llm=True, the
    returned ideas list is non-empty.
    """
    run_dir = Path(run_dir)
    # Tolerant reads: a single torn/stale line must not abort the cascade.
    papers = _safe_read_jsonl(run_dir / "papers.jsonl", Paper)
    claims = _safe_read_jsonl(run_dir / "claims.jsonl", Claim)

    tiers_used: list[str] = []
    collected: list[dict] = []

    def _take(new: list[dict], tier: str):
        if new:
            tiers_used.append(tier)
        collected.extend(new)

    def _have() -> int:
        return len(_dedup(collected))

    # cached forward_ideas from a prior LLM run of D/E — normalized so a stale
    # or partially-written line can never crash _dedup/sort downstream.
    fwd_path = run_dir / "forward_ideas.jsonl"
    cached_fwd: list[dict] = []
    if fwd_path.exists():
        try:
            raw_lines = []
            for ln in fwd_path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    raw_lines.append(json.loads(ln))
                except Exception:
                    continue  # skip torn line, keep the rest
            cached_fwd = _valid_cached(raw_lines)
        except Exception:
            cached_fwd = []

    k = max(min_ideas * 2, 8)
    _take(tier_a_critic(run_dir, topic, k), "A")
    if _have() < min_ideas:
        _take(tier_b_creator(run_dir, topic, k), "B")
    if _have() < min_ideas:
        _take(tier_c_bridges(run_dir, topic, k), "C")

    # cached D/E before re-calling the LLM
    if _have() < min_ideas and cached_fwd:
        _take(cached_fwd, "cache")

    new_forward: list[dict] = []
    if allow_llm and _have() < min_ideas:
        d = tier_d_method_extensions(run_dir, topic, papers, claims, min_ideas - _have())
        _take(d, "D")
        new_forward.extend(d)
    if allow_llm and _have() < min_ideas:
        e = tier_e_limitation_forward(run_dir, topic, papers, claims, min_ideas - _have())
        _take(e, "E")
        new_forward.extend(e)

    # Tier F — deterministic, LLM-free. Always runs when still short, making the
    # non-empty guarantee structural (independent of LLM reachability/JSON).
    if _have() < min_ideas:
        _take(tier_f_deterministic(topic, papers, claims, min_ideas - _have()), "F")

    # persist newly generated forward ideas (D/E only; F is cheap to recompute)
    if cache and new_forward:
        try:
            existing_ids = {i.get("idea_id") for i in cached_fwd}
            with open(fwd_path, "a", encoding="utf-8") as f:
                for idea in new_forward:
                    if idea.get("idea_id") and idea["idea_id"] not in existing_ids:
                        existing_ids.add(idea["idea_id"])  # dedup within this batch too
                        f.write(json.dumps(idea, ensure_ascii=False) + "\n")
        except Exception:
            pass

    ideas = _dedup(collected)
    ideas.sort(key=lambda i: (TIER_RANK.get(i.get("tier"), 0),
                              i.get("confidence", 0.0)), reverse=True)

    return {
        "topic": topic,
        "run": run_dir.name,
        "ideas": ideas,
        "stats": {
            "n_ideas": len(ideas),
            "n_papers": len(papers),
            "n_claims": len(claims),
            "min_ideas": min_ideas,
            "tiers_used": list(dict.fromkeys(tiers_used)),
            "by_tier": {t: sum(1 for i in ideas if i["tier"] == t) for t in TIER_LABELS},
            "allow_llm": allow_llm,
            "guaranteed_nonempty": len(ideas) > 0,
        },
    }


def resolve_best_run(runs_root: Path, topic: str, *, min_overlap: float = 0.34,
                     require_done: bool = True) -> str | None:
    """Find an existing run whose topic best matches `topic`, for corpus reuse.

    Scores each run by the fraction of the query's tokens covered by the run's
    topic (read from query.txt / status.json / run_metadata.json). Returns the
    best run id if it clears `min_overlap` AND (when require_done) the run
    actually finished (has a non-empty papers.jsonl). Returns None otherwise so
    the caller knows to build a fresh corpus.
    """
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return None
    qtok = _tokenize(topic)
    if not qtok:
        return None

    def _completeness(d: Path) -> int:
        # prefer a fully-finished corpus over a papers-only failed run
        score = 0
        cf = d / "claims.jsonl"
        if cf.exists() and cf.stat().st_size > 0:
            score += 2
        sp = d / "status.json"
        if sp.exists():
            try:
                if json.loads(sp.read_text(encoding="utf-8")).get("status") == "done":
                    score += 1
            except Exception:
                pass
        return score

    # rank by (topic overlap, completeness); tie-break favors finished corpora
    best_id, best_key = None, (0.0, -1)
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        # run's own topic
        rtopic = ""
        qf = d / "query.txt"
        if qf.exists():
            try:
                rtopic = qf.read_text(encoding="utf-8").strip()
            except Exception:
                rtopic = ""
        if not rtopic:
            for meta in ("status.json", "run_metadata.json"):
                mf = d / meta
                if mf.exists():
                    try:
                        rtopic = (json.loads(mf.read_text(encoding="utf-8")).get("topic") or "")
                    except Exception:
                        rtopic = ""
                    if rtopic:
                        break
        if not rtopic:
            continue
        rtok = _tokenize(rtopic)
        if not rtok:
            continue
        overlap = len(qtok & rtok) / len(qtok)
        if overlap < min_overlap:
            continue
        if require_done:
            # a usable corpus must have claims (a papers-only failed run is not
            # auto-reused; pass its run_id to generate_ideas explicitly to use it)
            cf = d / "claims.jsonl"
            if not cf.exists() or cf.stat().st_size == 0:
                continue
        key = (round(overlap, 4), _completeness(d))
        if key > best_key:
            best_id, best_key = d.name, key
    return best_id


def render_ideas_markdown(result: dict) -> str:
    topic = result["topic"]
    ideas = result["ideas"]
    st = result["stats"]
    out = [f"# Ideas — {topic}", ""]
    out.append(f"> {len(ideas)} idea(s) from {st['n_papers']} papers / "
               f"{st['n_claims']} claims. Tiers used: "
               f"{', '.join(st['tiers_used']) or 'none'}.")
    out.append("")
    for i, idea in enumerate(ideas, 1):
        out.append(f"## {i}. {idea['title']}  _({idea['tier_label']}, conf={idea['confidence']})_")
        out.append("")
        out.append(idea["statement"])
        out.append("")
        if idea["mechanism"]:
            out.append(f"**Mechanism.** {idea['mechanism']}")
        if idea["minimal_test"]:
            out.append(f"**Minimal test.** {idea['minimal_test']}")
        if idea["predictions"]:
            out.append("**Predictions:**")
            for p in idea["predictions"]:
                out.append(f"- {p}")
        if idea["source_papers"]:
            out.append(f"**Source papers:** {', '.join(idea['source_papers'])}")
        if idea["grounding"]:
            out.append(f"**Grounding.** {idea['grounding']}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--min-ideas", type=int, default=5)
    ap.add_argument("--no-llm", action="store_true", help="forbid Tier D/E LLM fallback")
    ap.add_argument("--no-cache", action="store_true", help="do not persist forward_ideas.jsonl")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = ap.parse_args()

    result = generate_ideas(args.run, args.topic, min_ideas=args.min_ideas,
                            allow_llm=not args.no_llm, cache=not args.no_cache)
    if args.json:
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        sys.stdout.buffer.write(render_ideas_markdown(result).encode("utf-8"))
        sys.stderr.write("\n" + json.dumps(result["stats"], ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
