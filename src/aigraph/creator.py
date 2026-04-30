"""Creator-mode pipeline: extract authors' OpenQuestions and synthesize new method ideas.

Two stages plug into the existing pipeline:

1. ``extract_open_questions`` — per paper, scan ``limitations`` and ``conclusion``
   sections for acknowledged limitations / future-work suggestions / untested
   extensions; produce :class:`OpenQuestion` records with verbatim spans.
2. ``generate_creator_hypotheses`` — per anomaly, combine the conflict structure
   with the OpenQuestions from papers in that cluster and prompt an LLM to
   propose *new* methods (mechanism + test + grounding), not critiques.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from typing import Any, Iterable, Optional

from .corpus import (
    hydrate_paper_from_corpus,
    load_corpus_sections,
    load_corpus_sentences,
)
from .llm_client import (
    DEFAULT_MAX_TOKENS,
    build_openai_client,
    call_llm_text,
    configured_api_key,
    configured_base_url,
    configured_model,
)
from .models import Anomaly, Claim, Hypothesis, OpenQuestion, Paper

logger = logging.getLogger(__name__)


_LIMITATION_KINDS = {"limitations", "conclusion"}
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


OPEN_QUESTION_SYSTEM_PROMPT = (
    "You read the limitations and conclusion sections of an AI research paper. "
    "Extract 0 to 4 OpenQuestion records that capture three kinds of forward-looking "
    "signals the authors put on paper:\n"
    "- acknowledged_limitation: a concrete weakness, failure mode, or unresolved "
    "issue the authors admit.\n"
    "- future_work_suggestion: a specific direction the authors explicitly say is "
    "promising but not done in this paper.\n"
    "- untested_extension: a variant, dataset, or scaled-up setting they mention "
    "but did not evaluate.\n\n"
    "Never invent details. Skip vague boilerplate (\"more research is needed\"). "
    'Return JSON {"open_questions": [...]} where each item has:\n'
    "- text: 1-2 sentence concise restatement\n"
    "- kind: one of acknowledged_limitation | future_work_suggestion | untested_extension\n"
    "- evidence_span: verbatim sentence from the paper\n"
    "- related_method: free text or null\n"
    "- related_task: free text or null\n"
    "- related_dataset: free text or null\n"
)


CREATOR_SYSTEM_PROMPT = (
    "You are a research assistant proposing concrete *new* methods that resolve a "
    "cluster of conflicting findings, grounded in the open questions and limitations "
    "the original authors stated themselves.\n\n"
    "You will be given:\n"
    "- One Anomaly (papers disagree about a (method, task) pair)\n"
    "- The structured Claims that compose the anomaly\n"
    "- OpenQuestion records (limitations and future work) from those papers\n\n"
    "Propose 1 to 3 NEW methods. Each method must:\n"
    "- Be a concrete combination or extension of techniques mentioned in the cluster, "
    "not a vague slogan.\n"
    "- Reference at least 2 OpenQuestion or Claim ids as the grounding for why it is "
    "needed.\n"
    "- Differ from any method already named in the cluster (if you propose something "
    "that exists in the claims, drop it).\n"
    "- Include a falsifiable minimal_test using benchmarks/metrics from the claims.\n\n"
    'Return JSON {"creator_hypotheses": [...]} where each item has:\n'
    "- proposed_method: 1-line name + 1-line description\n"
    "- mechanism: 2-3 sentences of how it works\n"
    "- predictions: list of 2 specific predictions (each ties to a metric/benchmark)\n"
    "- minimal_test: the simplest experiment to validate; reference dataset + metric\n"
    "- inspired_by: list of open_question_id and/or claim_id strings\n"
    "- distinguishes_from: 1 sentence — how it differs from existing methods in the cluster\n"
    "- anomaly_resolution: 1 sentence — how it resolves the anomaly\n"
    "Do not explain your reasoning. Output only the JSON object.\n"
)


# --- Multi-grain prompts (used by generate_creator_hypotheses_multi_grain) ---

SYSTEM_PROMPT_COARSE = (
    "You generate AMBITIOUS research direction proposals using LANDSCAPE "
    "context — domain trends, community sibling clusters, and cross-cluster "
    "patterns. The hypothesis can:\n"
    "- propose a unifying framework across multiple sub-fields\n"
    "- identify a missing piece across communities\n"
    "- name a trend the field has not formalized yet\n\n"
    "It must still:\n"
    "- reference >=3 specific evidence pieces from the provided domain/community summary "
    "(claim_ids, sibling cluster keys, or method/task names that appear in the payload)\n"
    "- be falsifiable: include a minimal_test that distinguishes it from the status quo\n"
    "- name a specific mechanism/protocol/dataset family — no slogans\n\n"
    'Return JSON {"creator_hypotheses": [...]} with the same item schema as '
    "the fine-grain prompt: proposed_method, mechanism, predictions, minimal_test, "
    "inspired_by, distinguishes_from, anomaly_resolution.\n"
    "Do not explain your reasoning. Output only the JSON object.\n"
)


SYSTEM_PROMPT_SYNTHESIZE = (
    "You receive two hypothesis variants for the SAME anomaly:\n"
    "- a FINE variant (rigorous, narrow, grounded in a specific cluster)\n"
    "- a COARSE variant (ambitious, broad, drawing on domain/community patterns)\n\n"
    "Synthesize ONE hypothesis that:\n"
    "- inherits rigorous grounding from FINE (specific methods, real benchmarks, "
    "concrete minimal_test)\n"
    "- inherits ambition from COARSE (cross-community framing, unifying mechanism)\n"
    "- in 1 sentence in `distinguishes_from`, names what is NEW vs existing work in "
    "BOTH the cluster and the wider community\n\n"
    'Return JSON {"creator_hypotheses": [<exactly one>]} with the same item schema '
    "as the fine-grain prompt.\n"
    "Do not explain your reasoning. Output only the JSON object.\n"
)


def _load_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    text = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _open_question_context(paper: Paper) -> Optional[str]:
    """Build a prompt body covering only limitations and conclusion sections."""
    sections = load_corpus_sections(paper)
    if not sections:
        return None
    sentences = load_corpus_sentences(paper)
    section_index = {s.section_id: s for s in sections}
    target_ids = {
        s.section_id
        for s in sections
        if (s.canonical_type or "other") in _LIMITATION_KINDS
    }
    if not target_ids:
        return None
    chosen: list[tuple[str, str]] = []
    for sentence in sentences:
        if sentence.section_id not in target_ids:
            continue
        section = section_index.get(sentence.section_id)
        if section is None:
            continue
        chosen.append((section.title or section.canonical_type, sentence.text.strip()))
        if len(chosen) >= 80:
            break
    if not chosen:
        return None
    grouped: dict[str, list[str]] = {}
    for title, text in chosen:
        grouped.setdefault(title, []).append(text)
    parts = [f"Title: {paper.title}", f"Paper id: {paper.paper_id}", ""]
    for title, lines in grouped.items():
        parts.append(f"## {title}")
        parts.extend(f"- {line}" for line in lines[:30])
        parts.append("")
    return "\n".join(parts).strip()


def extract_open_questions(
    papers: Iterable[Paper],
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_papers: Optional[int] = None,
) -> list[OpenQuestion]:
    """LLM-extract OpenQuestion records for each paper that has limitations/conclusion sections."""
    resolved_model = configured_model(model)
    client = build_openai_client(
        api_key=configured_api_key(api_key),
        base_url=configured_base_url(base_url),
    )
    out: list[OpenQuestion] = []
    counter = 0
    for paper in papers:
        if max_papers is not None and counter >= max_papers:
            break
        counter += 1
        paper = hydrate_paper_from_corpus(paper)
        body = _open_question_context(paper)
        if not body:
            continue
        try:
            raw = call_llm_text(
                client,
                model=resolved_model,
                system=OPEN_QUESTION_SYSTEM_PROMPT,
                user=body,
                temperature=0.0,
                max_tokens=int(os.environ.get("AIGRAPH_OQ_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
            )
        except Exception as exc:  # pragma: no cover - network / model errors
            logger.warning("OpenQuestion LLM failed for %s: %s", paper.paper_id, exc)
            continue
        payload = _load_json(raw)
        items = payload.get("open_questions") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items[:4]):
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            evidence = (item.get("evidence_span") or "").strip()
            if not text or not evidence:
                continue
            kind = (item.get("kind") or "acknowledged_limitation").strip()
            if kind not in {
                "acknowledged_limitation",
                "future_work_suggestion",
                "untested_extension",
            }:
                kind = "acknowledged_limitation"
            oq_id = f"{paper.paper_id}#oq{idx + 1:02d}"
            out.append(
                OpenQuestion(
                    open_question_id=oq_id,
                    paper_id=paper.paper_id,
                    text=text,
                    kind=kind,  # type: ignore[arg-type]
                    section_kind="limitations_or_conclusion",
                    related_method=item.get("related_method") or None,
                    related_task=item.get("related_task") or None,
                    related_dataset=item.get("related_dataset") or None,
                    evidence_span=evidence,
                )
            )
    return out


def generate_creator_hypotheses(
    anomalies: Iterable[Anomaly],
    claims: Iterable[Claim],
    open_questions: Iterable[OpenQuestion],
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_anomalies: Optional[int] = None,
) -> list[Hypothesis]:
    """For each anomaly, ask the LLM to propose new methods grounded in claims + open_questions."""
    claims_by_id = {c.claim_id: c for c in claims}
    oq_by_paper: dict[str, list[OpenQuestion]] = {}
    for oq in open_questions:
        oq_by_paper.setdefault(oq.paper_id, []).append(oq)

    resolved_model = configured_model(model)
    client = build_openai_client(
        api_key=configured_api_key(api_key),
        base_url=configured_base_url(base_url),
    )
    out: list[Hypothesis] = []
    counter = 0
    for anomaly in anomalies:
        if max_anomalies is not None and counter >= max_anomalies:
            break
        counter += 1
        anomaly_claims = [claims_by_id[cid] for cid in anomaly.claim_ids if cid in claims_by_id]
        if not anomaly_claims:
            continue
        paper_ids = {c.paper_id for c in anomaly_claims}
        paper_oqs = [oq for pid in paper_ids for oq in oq_by_paper.get(pid, [])]
        if not paper_oqs:
            continue
        body = _creator_user_prompt(anomaly, anomaly_claims, paper_oqs)
        try:
            raw = call_llm_text(
                client,
                model=resolved_model,
                system=CREATOR_SYSTEM_PROMPT,
                user=body,
                temperature=float(os.environ.get("AIGRAPH_CREATOR_TEMPERATURE", "0.3")),
                max_tokens=int(os.environ.get("AIGRAPH_CREATOR_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Creator LLM failed for %s: %s", anomaly.anomaly_id, exc)
            continue
        payload = _load_json(raw)
        items = payload.get("creator_hypotheses") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items[:3]):
            if not isinstance(item, dict):
                continue
            method = (item.get("proposed_method") or "").strip()
            mechanism = (item.get("mechanism") or "").strip()
            if not method or not mechanism:
                continue
            inspired = item.get("inspired_by")
            if not isinstance(inspired, list):
                inspired = []
            allowed_grounding = {oq.open_question_id for oq in paper_oqs} | {
                c.claim_id for c in anomaly_claims
            }
            inspired = [str(x) for x in inspired if str(x) in allowed_grounding]
            preds = item.get("predictions")
            if not isinstance(preds, list):
                preds = []
            preds = [str(p).strip() for p in preds if str(p).strip()][:5]
            hyp_id = f"{anomaly.anomaly_id}#cr{idx + 1:02d}"
            distinguishes = (item.get("distinguishes_from") or "").strip()
            anomaly_resolution = (item.get("anomaly_resolution") or "").strip()
            scope_dict: dict[str, str] = {}
            if distinguishes:
                scope_dict["distinguishes_from"] = distinguishes
            if inspired:
                scope_dict["inspired_by"] = ", ".join(inspired)
            out.append(
                Hypothesis(
                    hypothesis_id=hyp_id,
                    anomaly_id=anomaly.anomaly_id,
                    hypothesis=method,
                    mechanism=mechanism,
                    explains_claims=[c.claim_id for c in anomaly_claims][:20],
                    predictions=preds,
                    minimal_test=(item.get("minimal_test") or "").strip(),
                    scope_conditions=scope_dict,
                    evidence_gap=anomaly_resolution,
                    graph_bridge={
                        "from": "open_questions",
                        "to": "creator_hypothesis",
                    },
                )
            )
    return out


def generate_creator_hypotheses_multi_grain(
    anomalies: Iterable[Anomaly],
    claims: Iterable[Claim],
    open_questions: Iterable[OpenQuestion],
    hierarchy: dict,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_anomalies: Optional[int] = None,
    client: Any = None,
    existing_hypotheses: Optional[list[Hypothesis]] = None,
) -> list[dict]:
    """Multi-grain creator: for each (top-N by topology_score) anomaly, run
    fine -> coarse -> synthesize and return the SYNTHESIZED hypothesis as a
    record dict that bundles the intermediates as ``multi_grain``.

    Returns a list of dicts (NOT Hypothesis objects). Each dict has the same
    keys as a Hypothesis serialized via ``model_dump(by_alias=True)`` PLUS a
    ``multi_grain`` key carrying the raw fine + coarse LLM outputs and a
    ``fine_source`` audit field:

        {..hypothesis fields..,
         "multi_grain": {"fine": {...}, "coarse": {...}, "fine_source": "..."}}

    ``fine_source`` is one of:
    - ``"existing"`` — reused from ``existing_hypotheses`` (no fine LLM call)
    - ``"generated"`` — fine pass made a fresh LLM call

    Production callers can ``json.dumps`` each record directly to JSONL or
    pass it through ``Hypothesis.model_validate`` — the latter silently
    drops the extra ``multi_grain`` key (LooseModel is ``extra="ignore"``),
    so downstream code reading the same file as plain ``Hypothesis`` records
    works untouched.

    Skips an anomaly when (a) none of its claim_ids resolve, OR (b) no paper-OQ
    overlap (mirrors existing ``generate_creator_hypotheses`` at lines 249-250),
    so the multi-grain output set is comparable apples-to-apples with the
    single-grain one. Sorting + ``max_anomalies`` truncation happen INSIDE
    the function so callers don't need to pre-sort.

    When ``existing_hypotheses`` is supplied, fine pass is ablated for any
    anomaly_id covered by an existing record — the existing hypothesis is
    converted back to creator-payload shape and used as the synthesis fine
    anchor. This drops the per-anomaly LLM cost from 3 calls to 2 (~33%
    cheaper) and anchors fine quality to known-good production output.
    """
    claims_by_id = {c.claim_id: c for c in claims}
    oq_by_paper: dict[str, list[OpenQuestion]] = {}
    for oq in open_questions:
        oq_by_paper.setdefault(oq.paper_id, []).append(oq)

    existing_by_anomaly: dict[str, Hypothesis] = {}
    if existing_hypotheses:
        for h in existing_hypotheses:
            # First record per anomaly wins; the existing creator emits up to
            # 3 hypotheses per anomaly in score order, so the first is the
            # strongest variant.
            existing_by_anomaly.setdefault(h.anomaly_id, h)

    resolved_model = configured_model(model)
    if client is None:
        client = build_openai_client(
            api_key=configured_api_key(api_key),
            base_url=configured_base_url(base_url),
        )

    sorted_anomalies = sorted(
        list(anomalies),
        key=lambda a: float(getattr(a, "topology_score", 0.0) or 0.0),
        reverse=True,
    )
    if max_anomalies is not None:
        sorted_anomalies = sorted_anomalies[:max_anomalies]

    out: list[dict] = []
    for anomaly in sorted_anomalies:
        anomaly_claims = [claims_by_id[cid] for cid in anomaly.claim_ids if cid in claims_by_id]
        if not anomaly_claims:
            continue
        paper_ids = {c.paper_id for c in anomaly_claims}
        paper_oqs = [oq for pid in paper_ids for oq in oq_by_paper.get(pid, [])]
        existing = existing_by_anomaly.get(anomaly.anomaly_id)

        # Skip-rule: needs paper-OQ overlap only when we'd be generating a
        # fresh fine pass. With an existing anchor, fine is reused.
        # community_disconnect anomalies bypass fine entirely (cross-community
        # anomalies have no single-cluster home and no aligned OQs by
        # construction), so they never need paper_oqs either.
        is_cdis = anomaly.type == "community_disconnect"
        if existing is None and not is_cdis and not paper_oqs:
            continue

        if is_cdis:
            # community_disconnect path: no fine, no synthesize. The coarse
            # prompt has its own framing_note that asks for a unifying
            # mechanism across the two communities; its output IS the
            # hypothesis.
            fine_item = None
            fine_source = "skipped_community_disconnect"
            synthesize_source = "coarse_direct"
        elif existing is not None:
            fine_item = _hypothesis_to_creator_dict(existing)
            fine_source = "existing"
            synthesize_source = "synthesized"
        else:
            fine_user = _creator_user_prompt(anomaly, anomaly_claims, paper_oqs)
            fine_item = _call_creator_pass(client, resolved_model, CREATOR_SYSTEM_PROMPT, fine_user, anomaly)
            if fine_item is None:
                continue
            fine_source = "generated"
            synthesize_source = "synthesized"

        coarse_user = _prompt_payload_coarse(anomaly, claims_by_id, hierarchy)
        coarse_item = _call_creator_pass(client, resolved_model, SYSTEM_PROMPT_COARSE, coarse_user, anomaly)
        if coarse_item is None:
            continue

        if is_cdis:
            final_item = coarse_item
        else:
            synth_user = json.dumps(
                {"fine": fine_item, "coarse": coarse_item, "anomaly_id": anomaly.anomaly_id},
                ensure_ascii=False,
                indent=2,
            )
            synth_item = _call_creator_pass(client, resolved_model, SYSTEM_PROMPT_SYNTHESIZE, synth_user, anomaly)
            if synth_item is None:
                continue
            final_item = synth_item

        method_text = (final_item.get("proposed_method") or "").strip()
        mechanism = (final_item.get("mechanism") or "").strip()
        if not method_text or not mechanism:
            continue
        inspired = final_item.get("inspired_by")
        if not isinstance(inspired, list):
            inspired = []
        # The grounding allowlist is the union of paper OpenQuestions (when
        # any exist) and the anomaly's own claim_ids. paper_oqs may be empty
        # for the existing-anchor or community_disconnect paths; that's fine
        # — we still validate against the claim_ids the anomaly is built on.
        allowed_grounding = {oq.open_question_id for oq in paper_oqs} | {
            c.claim_id for c in anomaly_claims
        }
        inspired = [str(x) for x in inspired if str(x) in allowed_grounding]
        preds = final_item.get("predictions")
        if not isinstance(preds, list):
            preds = []
        preds = [str(p).strip() for p in preds if str(p).strip()][:5]
        hyp_id = f"{anomaly.anomaly_id}#mg01"
        distinguishes = (final_item.get("distinguishes_from") or "").strip()
        anomaly_resolution = (final_item.get("anomaly_resolution") or "").strip()
        scope_dict: dict[str, str] = {}
        if distinguishes:
            scope_dict["distinguishes_from"] = distinguishes
        if inspired:
            scope_dict["inspired_by"] = ", ".join(inspired)

        hyp = Hypothesis(
            hypothesis_id=hyp_id,
            anomaly_id=anomaly.anomaly_id,
            hypothesis=method_text,
            mechanism=mechanism,
            explains_claims=[c.claim_id for c in anomaly_claims][:20],
            predictions=preds,
            minimal_test=(final_item.get("minimal_test") or "").strip(),
            scope_conditions=scope_dict,
            evidence_gap=anomaly_resolution,
            graph_bridge={"from": "open_questions", "to": "creator_hypothesis"},
        )
        record = {
            **hyp.model_dump(by_alias=True),
            "multi_grain": {
                "fine": fine_item,
                "coarse": coarse_item,
                "fine_source": fine_source,
                "synthesize_source": synthesize_source,
            },
        }
        out.append(record)
    return out


def _hypothesis_to_creator_dict(h: Hypothesis) -> dict:
    """Convert a single-grain ``Hypothesis`` record (as emitted by
    ``generate_creator_hypotheses``) back to the dict shape that the
    synthesize prompt expects for ``fine``.

    Field-mapping notes — verified against ``creator_hypotheses.jsonl`` from
    the 1000-paper run, NOT a guess:

    - ``proposed_method`` <- ``hypothesis``
    - ``mechanism`` <- ``mechanism``
    - ``predictions`` <- ``predictions``
    - ``minimal_test`` <- ``minimal_test``
    - ``inspired_by`` <- parsed from ``scope_conditions["inspired_by"]``
      which the existing creator stores as a comma-joined string
      (creator.py:293). The new dict carries it as a list to match the
      shape that ``SYSTEM_PROMPT_COARSE`` and ``SYSTEM_PROMPT_SYNTHESIZE``
      already expect.
    - ``distinguishes_from`` <- ``scope_conditions["distinguishes_from"]``
    - ``anomaly_resolution`` <- ``evidence_gap``
    """
    scope = h.scope_conditions or {}
    inspired_raw = scope.get("inspired_by", "")
    if isinstance(inspired_raw, list):
        inspired = [str(x).strip() for x in inspired_raw if str(x).strip()]
    elif isinstance(inspired_raw, str):
        inspired = [x.strip() for x in inspired_raw.split(",") if x.strip()]
    else:
        inspired = []
    return {
        "proposed_method": h.hypothesis or "",
        "mechanism": h.mechanism or "",
        "predictions": list(h.predictions or []),
        "minimal_test": h.minimal_test or "",
        "inspired_by": inspired,
        "distinguishes_from": scope.get("distinguishes_from", ""),
        "anomaly_resolution": h.evidence_gap or "",
    }


def _call_creator_pass(
    client: Any,
    model: str,
    system_prompt: str,
    user_body: str,
    anomaly: Anomaly,
) -> Optional[dict]:
    """One LLM call returning the FIRST creator_hypotheses item or None on
    failure. Used by all three multi-grain passes."""
    try:
        raw = call_llm_text(
            client,
            model=model,
            system=system_prompt,
            user=user_body,
            temperature=float(os.environ.get("AIGRAPH_CREATOR_TEMPERATURE", "0.3")),
            max_tokens=int(os.environ.get("AIGRAPH_CREATOR_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Multi-grain creator LLM failed for %s: %s", anomaly.anomaly_id, exc)
        return None
    payload = _load_json(raw)
    items = payload.get("creator_hypotheses") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        logger.warning("Multi-grain creator: empty/invalid items for %s", anomaly.anomaly_id)
        return None
    item = items[0]
    if not isinstance(item, dict):
        return None
    return item


def _prompt_payload_coarse(
    anomaly: Anomaly,
    claims_by_id: dict[str, Claim],
    hierarchy: dict,
) -> str:
    """Build the coarse-grain user prompt — domain landscape + community
    siblings + a small handful of anchor claims. Truncates lists to keep
    payload around 5K tokens. Returns indented JSON string.

    Falls back gracefully when ``hierarchy`` is empty (all dicts default
    to {} / 0 / [] in the payload). For ``community_disconnect`` anomalies
    a different payload shape is used (community_from/_to + shared_concepts
    + framing_note that asks for a unifying mechanism across the two
    communities) — see ``_prompt_payload_coarse_community_disconnect``.
    The community_disconnect path is the FINAL hypothesis (no synthesize
    pass after) so the framing_note matters more than for the other
    anomaly types.
    """
    if anomaly.type == "community_disconnect":
        return _prompt_payload_coarse_community_disconnect(
            anomaly, claims_by_id, hierarchy
        )
    clusters = hierarchy.get("clusters") or {}
    cluster_to_community = hierarchy.get("cluster_to_community") or {}
    anomaly_to_cluster = hierarchy.get("anomaly_to_cluster") or {}
    domains = hierarchy.get("domains") or {}
    communities = hierarchy.get("communities") or {}

    cluster_key = anomaly_to_cluster.get(anomaly.anomaly_id)
    cluster = clusters.get(cluster_key, {}) if cluster_key else {}

    community_id = cluster_to_community.get(cluster_key) if cluster_key else None
    community = communities.get(community_id, {}) if community_id else {}

    siblings: list[dict] = []
    if community_id:
        for ck, cdata in clusters.items():
            if ck == cluster_key:
                continue
            if cluster_to_community.get(ck) == community_id:
                siblings.append({
                    "key": ck,
                    "anomaly_count": len(cdata.get("anomaly_ids", [])),
                })
            if len(siblings) >= 5:
                break
    if not siblings:
        ranked = sorted(
            (
                (ck, len(cd.get("anomaly_ids", [])))
                for ck, cd in clusters.items()
                if ck != cluster_key
            ),
            key=lambda x: -x[1],
        )[:3]
        siblings = [{"key": ck, "anomaly_count": n} for ck, n in ranked]

    domain_name = "uncategorized"
    if cluster.get("claim_ids"):
        votes: Counter = Counter()
        for cid in cluster["claim_ids"]:
            cl = claims_by_id.get(cid)
            if cl is None:
                continue
            d = (cl.domain or "").strip().lower()
            if d:
                votes[d] += 1
        if votes:
            domain_name = votes.most_common(1)[0][0]
    domain = domains.get(domain_name, {})

    anchor_claims: list[dict] = []
    for cid in (cluster.get("sample_claim_ids") or [])[:3]:
        cl = claims_by_id.get(cid)
        if cl is None:
            continue
        anchor_claims.append({
            "claim_id": cl.claim_id,
            "paper_id": cl.paper_id,
            "method": cl.method or cl.subject_raw,
            "task": cl.task or cl.object_raw,
            "direction": cl.direction,
            "evidence_span": cl.evidence_span,
        })

    payload = {
        "anomaly": {
            "anomaly_id": anomaly.anomaly_id,
            "type": anomaly.type,
            "central_question": anomaly.central_question,
            "shared_entities": anomaly.shared_entities,
        },
        "domain": {
            "name": domain_name,
            "paper_count": domain.get("paper_count", 0),
            "top_methods": (domain.get("top_methods") or [])[:10],
            "top_tasks": (domain.get("top_tasks") or [])[:10],
            "anomaly_type_counts": domain.get("anomaly_type_counts", {}),
        },
        "community": {
            "id": community_id,
            "paper_count": community.get("paper_count", 0),
            "top_concepts": (community.get("top_concepts") or [])[:5],
        },
        "sibling_clusters": siblings,
        "anchor_claims": anchor_claims,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _prompt_payload_coarse_community_disconnect(
    anomaly: Anomaly,
    claims_by_id: dict[str, Claim],
    hierarchy: dict,
) -> str:
    """Coarse payload for ``community_disconnect`` anomalies.

    These anomalies don't have a single (method, task) cluster home — they
    describe two communities that share concepts but rarely cite each
    other. The payload surfaces the shared_concepts + community labels +
    sample claims + a framing_note that asks the LLM to propose a unifying
    mechanism. Because the orchestrator skips fine + synthesize for this
    anomaly type, this prompt's output IS the final hypothesis; the
    framing_note is what makes the cross-community ambition explicit.
    """
    shared = anomaly.shared_entities or {}
    raw_concepts = shared.get("shared_concepts", "")
    if isinstance(raw_concepts, str):
        shared_concepts = [c.strip() for c in raw_concepts.split(",") if c.strip()]
    elif isinstance(raw_concepts, (list, tuple, set)):
        shared_concepts = [str(c).strip() for c in raw_concepts if str(c).strip()]
    else:
        shared_concepts = []

    anchor_claims: list[dict] = []
    for cid in (anomaly.claim_ids or [])[:5]:
        cl = claims_by_id.get(cid)
        if cl is None:
            continue
        anchor_claims.append({
            "claim_id": cl.claim_id,
            "paper_id": cl.paper_id,
            "method": cl.method or cl.subject_raw,
            "task": cl.task or cl.object_raw,
            "direction": cl.direction,
            "evidence_span": cl.evidence_span,
        })

    domain_votes: Counter = Counter()
    for cid in anomaly.claim_ids or []:
        cl = claims_by_id.get(cid)
        if cl is None:
            continue
        d = (cl.domain or "").strip().lower()
        if d:
            domain_votes[d] += 1
    domains_top = [d for d, _ in domain_votes.most_common(3)]

    clusters = hierarchy.get("clusters") or {}
    top_clusters = sorted(
        clusters.items(),
        key=lambda kv: -len(kv[1].get("anomaly_ids", [])),
    )[:5]
    sibling_clusters = [
        {"key": k, "anomaly_count": len(v.get("anomaly_ids", []))}
        for k, v in top_clusters
    ]

    payload = {
        "anomaly": {
            "anomaly_id": anomaly.anomaly_id,
            "type": "community_disconnect",
            "central_question": anomaly.central_question,
            "community_from": shared.get("community_from", ""),
            "community_to": shared.get("community_to", ""),
        },
        "shared_concepts": shared_concepts,
        "anchor_claims": anchor_claims,
        "domains": domains_top,
        "sibling_clusters": sibling_clusters,
        "framing_note": (
            "This anomaly is a community_disconnect: two communities share "
            "concepts (see shared_concepts) but rarely cite each other. "
            "Propose a unifying mechanism, framework, or evaluation that "
            "would bridge them. Reference shared_concepts and anchor_claims "
            "as evidence in inspired_by."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _creator_user_prompt(
    anomaly: Anomaly,
    claims: list[Claim],
    open_questions: list[OpenQuestion],
) -> str:
    payload = {
        "anomaly": {
            "anomaly_id": anomaly.anomaly_id,
            "type": anomaly.type,
            "central_question": anomaly.central_question,
            "shared_entities": anomaly.shared_entities,
        },
        "claims": [
            {
                "claim_id": c.claim_id,
                "paper_id": c.paper_id,
                "method": c.method or c.subject_raw,
                "task": c.task or c.object_raw,
                "dataset": c.dataset,
                "metric": c.metric,
                "direction": c.direction,
                "magnitude_text": c.magnitude_text,
                "evidence_span": c.evidence_span,
            }
            for c in claims[:30]
        ],
        "open_questions": [
            {
                "open_question_id": oq.open_question_id,
                "paper_id": oq.paper_id,
                "kind": oq.kind,
                "text": oq.text,
                "related_method": oq.related_method,
                "related_task": oq.related_task,
                "related_dataset": oq.related_dataset,
                "evidence_span": oq.evidence_span,
            }
            for oq in open_questions[:25]
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
