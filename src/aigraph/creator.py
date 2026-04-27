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
