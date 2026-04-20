"""LLM-backed claim extractor. Plugs in behind :class:`ClaimExtractor`.

Defensive by design: the model must not invent missing details. Unknown fields
stay ``null``; ``evidence_span`` must be copied verbatim from the paper.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from pydantic import ValidationError

from .claim_schema import normalize_structured_claim_payload
from .extract import ClaimExtractor
from .llm_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    build_openai_client,
    call_llm_text,
    configured_api_key,
    configured_base_url,
    configured_model,
)
from .models import Claim, Paper, PaperReadCandidate, Setting


logger = logging.getLogger(__name__)

_ALLOWED_CLAIM_TYPES = {
    "performance_improvement",
    "limitation",
    "comparison",
    "setting_effect",
    "mechanism",
}
_ALLOWED_DIRECTIONS = {"positive", "negative", "mixed"}
_ALLOWED_TASK_TYPES = {
    "factual",
    "multi-hop",
    "long-context",
    "agentic",
    "reasoning",
    "evaluation",
}

# Fixed vocabularies used to cluster claims across papers.
CANONICAL_METHODS: tuple[str, ...] = (
    "RAG",
    "reranking",
    "query-rewriting",
    "long-context",
    "fine-tuning",
    "prompting",
    "agent",
    "distillation",
    "evaluation-method",
    "safety-alignment",
    "multimodal",
    "other",
)

CANONICAL_TASKS: tuple[str, ...] = (
    "factual-QA",
    "multi-hop-QA",
    "long-context-QA",
    "domain-QA",
    "code",
    "summarization",
    "reasoning",
    "hallucination-mitigation",
    "safety",
    "evaluation",
    "conversation",
    "other",
)

_CANONICAL_METHODS_LOWER = {m.lower(): m for m in CANONICAL_METHODS}
_CANONICAL_TASKS_LOWER = {t.lower(): t for t in CANONICAL_TASKS}


SYSTEM_PROMPT = (
    "You are a careful scientific claim extractor. Given an AI paper's title and "
    "abstract/text, extract 0 to 3 structured claims. Never invent details: if a "
    "field is not explicitly stated, set it to null. Copy evidence_span verbatim "
    "from the provided text body (no paraphrasing). Return a JSON object "
    "directly. Do not explain your reasoning. Do not include analysis. "
    'with one key "claims" whose value is a list. Each claim has these fields:\n'
    "- claim_text: string, a concise restatement of the claim\n"
    "- subject_raw: string or null\n"
    "- predicate: string or null\n"
    "- object_raw: string or null\n"
    "- dataset_raw: string or null\n"
    "- metric_raw: string or null\n"
    "- baseline_raw: string or null\n"
    "- magnitude_text: string or null (e.g. +8.2 EM, -3pp)\n"
    "- conditions: list of strings\n"
    "- scope: list of strings\n"
    "- claim_type: one of performance_improvement, limitation, comparison, "
    "setting_effect, mechanism\n"
    "- method: string or null (free-text, e.g., RAG, DPR, chain-of-thought)\n"
    "- canonical_method: MUST be one of "
    f"[{', '.join(CANONICAL_METHODS)}]. "
    "Pick the closest single category. Use 'other' only if truly none fits.\n"
    "- model: string or null (e.g., GPT-4, Llama-70B)\n"
    "- task: string or null (free-text, e.g., factual QA, multi-hop QA)\n"
    "- canonical_task: MUST be one of "
    f"[{', '.join(CANONICAL_TASKS)}]. "
    "Pick the closest single category. Use 'other' only if truly none fits. "
    "Domain-specific QA (medical, legal, finance, scientific) maps to 'domain-QA'.\n"
    "- dataset: string or null\n"
    "- metric: string or null\n"
    "- baseline: string or null\n"
    "- result: string or null (e.g., +8.2, -3.0, mixed)\n"
    "- direction: one of positive, negative, mixed\n"
    "- setting: object with fields retriever, top_k, context_length, task_type "
    "(each string or null). task_type must be one of factual, multi-hop, "
    "long-context, agentic, reasoning, evaluation, or null.\n"
    "- evidence_span: short quote copied verbatim from the abstract/text\n"
    "- evidence_source_field: text or abstract or null\n"
    "- evidence_sentence_index: integer or null\n"
    "- evidence_char_start: integer or null\n"
    "- evidence_char_end: integer or null\n"
    "- domain: string or null (e.g., finance, medicine, time series, robotics)\n"
    "- data_modality: string or null (e.g., text, time series, text + time series)\n"
    "- mechanism: string or null (explicit mechanism, e.g., event grounding, preference optimization)\n"
    "- failure_mode: string or null (explicit failure, e.g., temporal leakage, reward hacking)\n"
    "- evaluation_protocol: string or null (e.g., backtesting, rolling-window evaluation)\n"
    "- assumption: string or null (explicit assumption behind the claim)\n"
    "- risk_type: string or null (e.g., safety risk, financial risk, privacy risk)\n"
    "- temporal_property: string or null (e.g., non-stationarity, forecast horizon, regime shift)\n"
    'Output STRICT JSON only, no prose, no markdown fences. Example: {"claims": []} '
    "when no explicit claim is present."
)


class LLMClaimExtractor(ClaimExtractor):
    """Sends title+abstract to an LLM and parses strict JSON into :class:`Claim`."""

    def __init__(
        self,
        model: Optional[str] = None,
        client: Any | None = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = configured_model(model)
        self._client = client
        self._api_key = configured_api_key(api_key)
        self._base_url = configured_base_url(base_url)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = build_openai_client(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def extract(
        self,
        paper: Paper,
        start_index: int = 0,
        *,
        candidates: list[PaperReadCandidate] | None = None,
    ) -> list[Claim]:
        content = _paper_context(paper, candidates=candidates)
        if not content.strip():
            return []
        try:
            raw = self._call_llm(content)
        except Exception as e:  # pragma: no cover - network errors
            logger.warning("LLM call failed for %s: %s", paper.paper_id, e)
            return []
        return self._parse_response(raw, paper, start_index)

    def _call_llm(self, content: str) -> str:
        client = self._get_client()
        import os

        return call_llm_text(
            client,
            model=self.model,
            system=SYSTEM_PROMPT,
            user=content,
            temperature=0.0,
            max_tokens=int(os.environ.get("AIGRAPH_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        )

    def _parse_response(self, raw: str, paper: Paper, start_index: int) -> list[Claim]:
        payload = _load_json(raw)
        if payload is None:
            logger.warning("Could not parse LLM JSON for %s", paper.paper_id)
            return []
        items = payload.get("claims") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            logger.warning("Unexpected LLM payload shape for %s", paper.paper_id)
            return []

        claims: list[Claim] = []
        for i, item in enumerate(items[:3]):
            if not isinstance(item, dict):
                continue
            normalized = _normalize_claim_dict(item, paper)
            if normalized is None:
                continue
            claim_id = f"c{start_index + len(claims) + 1:03d}"
            normalized["claim_id"] = claim_id
            normalized["paper_id"] = paper.paper_id
            try:
                claims.append(Claim.model_validate(normalized))
            except ValidationError as e:
                logger.warning("Claim validation failed for %s: %s", paper.paper_id, e)
                continue
        return claims


def _paper_context(paper: Paper, *, candidates: list[PaperReadCandidate] | None = None) -> str:
    if candidates:
        abstract_prefix = (paper.abstract or "").strip()
        if len(abstract_prefix) > 600:
            abstract_prefix = abstract_prefix[:600].rsplit(" ", 1)[0] + "..."
        candidate_lines = []
        for idx, candidate in enumerate(candidates):
            candidate_lines.append(
                json.dumps(
                    {
                        "candidate_index": idx,
                        "sentence": candidate.sentence,
                        "evidence_span": candidate.evidence_span,
                        "subject_raw": candidate.subject_raw,
                        "predicate": candidate.predicate,
                        "object_raw": candidate.object_raw,
                        "dataset_raw": candidate.dataset_raw,
                        "metric_raw": candidate.metric_raw,
                        "baseline_raw": candidate.baseline_raw,
                        "direction": candidate.direction,
                        "magnitude_text": candidate.magnitude_text,
                        "conditions": candidate.conditions,
                        "scope": candidate.scope,
                        "candidate_score": candidate.candidate_score,
                        "selection_reason": candidate.selection_reason,
                    },
                    ensure_ascii=False,
                )
            )
        return (
            f"Title: {paper.title}\n\n"
            f"Abstract prefix:\n{abstract_prefix}\n\n"
            "Claim candidate pool (use only these grounded spans as evidence):\n"
            + "\n".join(candidate_lines)
        ).strip()
    body = paper.abstract or paper.text or ""
    return f"Title: {paper.title}\n\nAbstract/Text:\n{body}".strip()


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _load_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    text = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Look for the first balanced JSON object in the text.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _normalize_claim_dict(item: dict, paper: Paper) -> Optional[dict]:
    claim_type = item.get("claim_type") or "performance_improvement"
    if claim_type not in _ALLOWED_CLAIM_TYPES:
        claim_type = "performance_improvement"
    direction = item.get("direction") or "positive"
    if direction not in _ALLOWED_DIRECTIONS:
        direction = "positive"

    setting_raw = item.get("setting") or {}
    if not isinstance(setting_raw, dict):
        setting_raw = {}
    task_type = setting_raw.get("task_type")
    if task_type is not None and task_type not in _ALLOWED_TASK_TYPES:
        task_type = None

    normalized = normalize_structured_claim_payload(item, paper, trust_evidence_if_no_body=False)
    if normalized is None:
        return None

    return {
        **normalized,
        "claim_type": claim_type,
        "method": normalized.get("method") or _clean_str(item.get("method")),
        "canonical_method": _canonicalize(item.get("canonical_method"), _CANONICAL_METHODS_LOWER),
        "model": _clean_str(item.get("model")),
        "task": normalized.get("task") or _clean_str(item.get("task")),
        "canonical_task": _canonicalize(item.get("canonical_task"), _CANONICAL_TASKS_LOWER),
        "direction": direction,
        "setting": Setting(
            retriever=_clean_str(setting_raw.get("retriever")),
            top_k=_clean_str(setting_raw.get("top_k")),
            context_length=_clean_str(setting_raw.get("context_length")),
            task_type=task_type,
        ),
        "domain": _clean_str(item.get("domain")),
        "data_modality": _clean_str(item.get("data_modality")),
        "mechanism": _clean_str(item.get("mechanism")),
        "failure_mode": _clean_str(item.get("failure_mode")),
        "evaluation_protocol": _clean_str(item.get("evaluation_protocol")),
        "assumption": _clean_str(item.get("assumption")),
        "risk_type": _clean_str(item.get("risk_type")),
        "temporal_property": _clean_str(item.get("temporal_property")),
    }


def _canonicalize(value: Any, vocab_lower: dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    key = value.strip().lower()
    return vocab_lower.get(key)


def _clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value or value.lower() in ("null", "none", "n/a", "na", "unknown"):
        return None
    return value
