from __future__ import annotations

import re
from typing import Any

from .models import Paper


_WS_RE = re.compile(r"\s+")
_DELTA_RE = re.compile(r"([+\-−])\s*(\d+(?:\.\d+)?)\s*(pp|pts|points|%|percent|em|f1|accuracy|acc|win rate)?", re.IGNORECASE)
_SENTENCE_RE = re.compile(r".+?(?:[.!?](?:\s+|$)|$)", re.S)

_DATASET_ALIASES = {
    "nq": "naturalquestions",
    "natural questions": "naturalquestions",
    "naturalquestions": "naturalquestions",
    "hotpot qa": "hotpotqa",
    "hotpotqa": "hotpotqa",
    "truthful qa": "truthfulqa",
    "truthfulqa": "truthfulqa",
    "hh rlhf": "hh-rlhf",
    "hh-rlhf": "hh-rlhf",
}
_METRIC_ALIASES = {
    "exact match": "exact-match",
    "em": "exact-match",
    "f1": "f1",
    "accuracy": "accuracy",
    "acc": "accuracy",
    "win rate": "win-rate",
    "wins": "win-rate",
}


def claim_source_body(paper: Paper) -> tuple[str | None, str]:
    body = (paper.text or "").strip()
    if body:
        return "text", body
    abstract = (paper.abstract or "").strip()
    if abstract:
        return "abstract", abstract
    return None, ""


def sentence_spans(text: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    if not text:
        return spans
    for idx, match in enumerate(_SENTENCE_RE.finditer(text)):
        chunk = match.group(0)
        trimmed = chunk.strip()
        if not trimmed:
            continue
        leading_ws = len(chunk) - len(chunk.lstrip())
        trailing_ws = len(chunk) - len(chunk.rstrip())
        start = match.start() + leading_ws
        end = match.end() - trailing_ws
        spans.append({"sentence": trimmed, "index": idx, "start": start, "end": end})
    return spans


def normalize_structured_claim_payload(
    item: dict[str, Any],
    paper: Paper,
    *,
    trust_evidence_if_no_body: bool = False,
) -> dict[str, Any] | None:
    claim_text = _clean_str(item.get("claim_text"))
    subject_raw = _clean_str(item.get("subject_raw")) or _clean_str(item.get("method")) or _clean_str(item.get("model"))
    predicate = _clean_str(item.get("predicate")) or _infer_predicate(item)
    object_raw = _clean_str(item.get("object_raw")) or _clean_str(item.get("task"))
    dataset_raw = _clean_str(item.get("dataset_raw")) or _clean_str(item.get("dataset"))
    metric_raw = _clean_str(item.get("metric_raw")) or _clean_str(item.get("metric"))
    baseline_raw = _clean_str(item.get("baseline_raw")) or _clean_str(item.get("baseline"))
    result_text = _clean_str(item.get("result"))
    magnitude_text, magnitude_value, magnitude_unit = parse_magnitude(_clean_str(item.get("magnitude_text")) or result_text or claim_text or "")
    direction = _clean_str(item.get("direction")) or _infer_direction_from_magnitude(magnitude_value)

    if not claim_text:
        claim_text = compose_claim_text(subject_raw, predicate, object_raw, magnitude_text)
    if not claim_text:
        return None

    evidence_span = _clean_str(item.get("evidence_span")) or claim_text
    grounding = ground_evidence_span(
        paper,
        evidence_span,
        trust_if_no_body=trust_evidence_if_no_body,
    )

    return {
        "claim_text": claim_text,
        "method": _clean_str(item.get("method")) or subject_raw,
        "model": _clean_str(item.get("model")),
        "task": _clean_str(item.get("task")) or object_raw,
        "dataset": dataset_raw,
        "metric": metric_raw,
        "baseline": baseline_raw,
        "result": result_text or magnitude_text,
        "direction": direction,
        "subject_raw": subject_raw,
        "subject_canonical": canonicalize_entity(item.get("subject_canonical") or subject_raw),
        "predicate": predicate,
        "object_raw": object_raw,
        "object_canonical": canonicalize_entity(item.get("object_canonical") or object_raw),
        "dataset_raw": dataset_raw,
        "dataset_canonical": canonicalize_dataset(item.get("dataset_canonical") or dataset_raw),
        "metric_raw": metric_raw,
        "metric_canonical": canonicalize_metric(item.get("metric_canonical") or metric_raw),
        "baseline_raw": baseline_raw,
        "baseline_canonical": canonicalize_entity(item.get("baseline_canonical") or baseline_raw),
        "magnitude_text": magnitude_text,
        "magnitude_value": magnitude_value,
        "magnitude_unit": magnitude_unit,
        "conditions": normalize_string_list(item.get("conditions")),
        "scope": normalize_string_list(item.get("scope")),
        **grounding,
    }


def ground_evidence_span(
    paper: Paper,
    evidence_span: str | None,
    *,
    trust_if_no_body: bool = False,
) -> dict[str, Any]:
    span = (evidence_span or "").strip()
    source_field, body = claim_source_body(paper)
    if not span:
        return _empty_grounding()
    if not body:
        if trust_if_no_body:
            return {
                "evidence_span": span,
                "evidence_source_field": None,
                "evidence_sentence_index": None,
                "evidence_char_start": None,
                "evidence_char_end": None,
            }
        return _empty_grounding()

    exact = body.find(span)
    if exact != -1:
        sentence_index = None
        for sent in sentence_spans(body):
            if sent["start"] <= exact < sent["end"]:
                sentence_index = sent["index"]
                break
        return {
            "evidence_span": span,
            "evidence_source_field": source_field,
            "evidence_sentence_index": sentence_index,
            "evidence_char_start": exact,
            "evidence_char_end": exact + len(span),
        }

    normalized_body = normalize_free_text(body)
    normalized_span = normalize_free_text(span)
    if normalized_span and normalized_span in normalized_body:
        return {
            "evidence_span": span,
            "evidence_source_field": source_field,
            "evidence_sentence_index": None,
            "evidence_char_start": None,
            "evidence_char_end": None,
        }
    return _empty_grounding()


def parse_magnitude(text: str | None) -> tuple[str | None, float | None, str | None]:
    raw = (text or "").strip()
    if not raw:
        return None, None, None
    match = _DELTA_RE.search(raw)
    if not match:
        return None, None, None
    sign, value, unit = match.groups()
    scalar = float(value)
    if sign in {"-", "−"}:
        scalar = -scalar
    clean_unit = (unit or "").strip().lower() or None
    if clean_unit == "percent":
        clean_unit = "%"
    if clean_unit in {"pts", "points"}:
        clean_unit = "pp"
    return match.group(0).strip(), scalar, clean_unit


def canonicalize_dataset(value: Any) -> str | None:
    normalized = normalize_free_text(value)
    if not normalized:
        return None
    return _DATASET_ALIASES.get(normalized, normalized)


def canonicalize_metric(value: Any) -> str | None:
    normalized = normalize_free_text(value)
    if not normalized:
        return None
    return _METRIC_ALIASES.get(normalized, normalized)


def canonicalize_entity(value: Any) -> str | None:
    normalized = normalize_free_text(value)
    return normalized or None


def normalize_free_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return _WS_RE.sub(" ", text)


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[;\n,]+", value) if part.strip()]
        return parts
    if isinstance(value, (list, tuple, set)):
        cleaned: list[str] = []
        for item in value:
            text = _clean_str(item)
            if text:
                cleaned.append(text)
        return cleaned
    return []


def compose_claim_text(
    subject_raw: str | None,
    predicate: str | None,
    object_raw: str | None,
    magnitude_text: str | None,
) -> str:
    bits = [part for part in [subject_raw, predicate, object_raw] if part]
    text = " ".join(bits).strip()
    if magnitude_text:
        text = f"{text} ({magnitude_text})".strip()
    return text


def _empty_grounding() -> dict[str, Any]:
    return {
        "evidence_span": "",
        "evidence_source_field": None,
        "evidence_sentence_index": None,
        "evidence_char_start": None,
        "evidence_char_end": None,
    }


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "na", "unknown"}:
        return None
    return text


def _infer_predicate(item: dict[str, Any]) -> str | None:
    raw = _clean_str(item.get("predicate"))
    if raw:
        return raw
    direction = str(item.get("direction") or "").strip().lower()
    if direction == "positive":
        return "improves"
    if direction == "negative":
        return "degrades"
    if direction == "mixed":
        return "has mixed effect on"
    return None


def _infer_direction_from_magnitude(magnitude_value: float | None) -> str | None:
    if magnitude_value is None:
        return None
    if magnitude_value > 0:
        return "positive"
    if magnitude_value < 0:
        return "negative"
    return None
