from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .claim_schema import claim_source_body, normalize_string_list, parse_magnitude, sentence_spans
from .llm_client import DEFAULT_MAX_TOKENS, build_openai_client, call_llm_text, configured_api_key, configured_base_url
from .models import Direction, Paper, PaperReadCandidate


logger = logging.getLogger(__name__)

DEFAULT_READER_MODEL = "gpt-5.4-mini"
DEFAULT_READER_MODE = "mini"
DEFAULT_READER_MAX_CANDIDATES = 6
DEFAULT_READER_PREFILTER_SENTENCES = 16

_ALLOWED_DIRECTIONS = {"positive", "negative", "mixed"}
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

_DIRECTION_CUES = {
    "positive": ("improve", "improves", "boost", "boosts", "gains", "helps", "outperform"),
    "negative": ("hurts", "degrades", "reduces", "drops", "fails", "worse"),
    "mixed": ("mixed", "uneven", "noisy", "inconsistent", "introduces irrelevant"),
}
_DELTA_RE = re.compile(r"([+\-−])\s*(\d+(?:\.\d+)?)")
_CONDITION_RE = re.compile(r"\b(?:when|under|with|if|at|for)\b([^.;]+)", re.IGNORECASE)
_SCOPE_RE = re.compile(r"\b(?:only|especially|primarily|across|generalizes to|limited to)\b([^.;]+)", re.IGNORECASE)
_METRIC_CUE_RE = re.compile(r"\b(?:accuracy|exact match|em|f1|win rate|precision|recall|auc|bleu|rouge)\b", re.IGNORECASE)
_BASELINE_CUE_RE = re.compile(r"\b(?:over|than|against|compared with|compared to|baseline)\b", re.IGNORECASE)

_METHOD_KEYWORDS = [
    "RAG",
    "DPR",
    "BM25",
    "retrieval-augmented",
    "chain-of-thought",
    "reranking",
]
_MODEL_KEYWORDS = [
    "GPT-4",
    "GPT-3.5",
    "Llama",
    "Transformer",
    "Long-context",
]
_DATASET_KEYWORDS = [
    ("naturalquestions", "NaturalQuestions"),
    ("natural questions", "NaturalQuestions"),
    ("hotpotqa", "HotpotQA"),
    ("truthfulqa", "TruthfulQA"),
    ("hh-rlhf", "HH-RLHF"),
]
_METRIC_KEYWORDS = [
    ("exact match", "Exact Match"),
    ("em", "Exact Match"),
    ("accuracy", "Accuracy"),
    ("f1", "F1"),
    ("win rate", "Win Rate"),
]
_TASK_KEYWORDS = [
    ("multi-hop", "multi-hop QA"),
    ("hotpotqa", "multi-hop QA"),
    ("naturalquestions", "factual QA"),
    ("factual", "factual QA"),
    ("long-context", "long-context QA"),
    ("agentic", "agentic QA"),
]

MINI_READER_SYSTEM_PROMPT = (
    "You are a fast scientific paper reader. You will receive a paper title, a short abstract prefix, "
    "and a numbered pool of grounded candidate sentences. Select up to the requested number of sentences "
    "that most likely contain empirical result, limitation, comparison, or setting-effect claims. "
    "Return STRICT JSON only with a top-level key 'candidates'. Each item must have: "
    "candidate_index (integer, required), evidence_span (string copied verbatim from the candidate sentence "
    "or a verbatim substring), subject_raw, predicate, object_raw, dataset_raw, metric_raw, baseline_raw, "
    "direction (positive|negative|mixed|null), magnitude_text, conditions (list), scope (list), "
    "candidate_score (0..1), selection_reason. Do not invent fields that are not explicit."
)


@dataclass
class PaperReadResult:
    candidates: list[PaperReadCandidate]
    mode_used: str
    latency_sec: float
    prefilter_count: int = 0
    fallback_used: bool = False


class PaperReader(ABC):
    @abstractmethod
    def read(self, paper: Paper) -> PaperReadResult:
        """Return grounded candidate spans for one paper."""


class HeuristicPaperReader(PaperReader):
    def __init__(self, *, max_prefilter_sentences: int = DEFAULT_READER_PREFILTER_SENTENCES):
        self.max_prefilter_sentences = max(1, int(max_prefilter_sentences))

    def read(self, paper: Paper) -> PaperReadResult:
        start = time.perf_counter()
        source_field, body = claim_source_body(paper)
        if not body:
            return PaperReadResult(candidates=[], mode_used="heuristic", latency_sec=0.0, prefilter_count=0)

        ranked: list[PaperReadCandidate] = []
        for meta in sentence_spans(body):
            candidate = _candidate_from_sentence(
                meta["sentence"],
                source_field=source_field,
                sentence_index=meta["index"],
                char_start=meta["start"],
                char_end=meta["end"],
            )
            if candidate is None:
                continue
            ranked.append(candidate)

        ranked.sort(key=lambda item: (-float(item.candidate_score or 0.0), int(item.evidence_sentence_index or 0)))
        limited = ranked[: self.max_prefilter_sentences]
        return PaperReadResult(
            candidates=limited,
            mode_used="heuristic",
            latency_sec=round(time.perf_counter() - start, 4),
            prefilter_count=len(limited),
        )


class LLMPaperReaderMini(PaperReader):
    def __init__(
        self,
        *,
        model: str | None = None,
        max_candidates: int = DEFAULT_READER_MAX_CANDIDATES,
        candidates: list[PaperReadCandidate] | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = configured_reader_model(model)
        self.max_candidates = max(1, int(max_candidates))
        self.prefiltered_candidates = list(candidates or [])
        self._client = client
        self._api_key = configured_api_key(api_key)
        self._base_url = configured_base_url(base_url)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = build_openai_client(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def read(self, paper: Paper) -> PaperReadResult:
        start = time.perf_counter()
        if not self.prefiltered_candidates:
            return PaperReadResult(candidates=[], mode_used="mini", latency_sec=0.0, prefilter_count=0)
        try:
            raw = call_llm_text(
                self._get_client(),
                model=self.model,
                system=MINI_READER_SYSTEM_PROMPT,
                user=_mini_reader_prompt(paper, self.prefiltered_candidates, self.max_candidates),
                temperature=0.0,
                max_tokens=min(1600, int(os.environ.get("AIGRAPH_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))),
            )
        except Exception as exc:  # pragma: no cover - network failures
            logger.warning("Mini paper reader failed for %s: %s", paper.paper_id, exc)
            return PaperReadResult(
                candidates=[],
                mode_used="mini",
                latency_sec=round(time.perf_counter() - start, 4),
                prefilter_count=len(self.prefiltered_candidates),
                fallback_used=True,
            )

        parsed = _load_json(raw)
        if parsed is None:
            logger.warning("Mini paper reader returned malformed JSON for %s", paper.paper_id)
            return PaperReadResult(
                candidates=[],
                mode_used="mini",
                latency_sec=round(time.perf_counter() - start, 4),
                prefilter_count=len(self.prefiltered_candidates),
                fallback_used=True,
            )

        items = parsed.get("candidates") if isinstance(parsed, dict) else None
        if not isinstance(items, list):
            return PaperReadResult(
                candidates=[],
                mode_used="mini",
                latency_sec=round(time.perf_counter() - start, 4),
                prefilter_count=len(self.prefiltered_candidates),
                fallback_used=True,
            )

        selected: list[PaperReadCandidate] = []
        for item in items[: self.max_candidates]:
            normalized = _candidate_from_llm_item(item, self.prefiltered_candidates)
            if normalized is None:
                continue
            selected.append(normalized)

        return PaperReadResult(
            candidates=selected,
            mode_used="mini",
            latency_sec=round(time.perf_counter() - start, 4),
            prefilter_count=len(self.prefiltered_candidates),
            fallback_used=False,
        )


def configured_reader_mode(mode: str | None = None) -> str:
    value = str(mode or os.environ.get("AIGRAPH_READER_MODE") or DEFAULT_READER_MODE).strip().lower()
    return value if value in {"off", "heuristic", "mini"} else DEFAULT_READER_MODE


def configured_reader_model(model: str | None = None) -> str:
    return str(model or os.environ.get("AIGRAPH_READER_MODEL") or DEFAULT_READER_MODEL).strip()


def configured_reader_max_candidates(max_candidates: int | None = None) -> int:
    raw = max_candidates if max_candidates is not None else os.environ.get("AIGRAPH_READER_MAX_CANDIDATES")
    try:
        value = int(raw) if raw is not None else DEFAULT_READER_MAX_CANDIDATES
    except (TypeError, ValueError):
        value = DEFAULT_READER_MAX_CANDIDATES
    return max(1, min(12, value))


def configured_reader_prefilter_sentences(prefilter_sentences: int | None = None) -> int:
    raw = prefilter_sentences if prefilter_sentences is not None else os.environ.get("AIGRAPH_READER_PREFILTER_SENTENCES")
    try:
        value = int(raw) if raw is not None else DEFAULT_READER_PREFILTER_SENTENCES
    except (TypeError, ValueError):
        value = DEFAULT_READER_PREFILTER_SENTENCES
    return max(1, min(40, value))


def read_paper_candidates(
    paper: Paper,
    *,
    mode: str | None = None,
    model: str | None = None,
    max_candidates: int | None = None,
    prefilter_sentences: int | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> PaperReadResult:
    if paper.structured_hint:
        return PaperReadResult(candidates=[], mode_used="structured_hint", latency_sec=0.0, prefilter_count=0)

    resolved_mode = configured_reader_mode(mode)
    resolved_max_candidates = configured_reader_max_candidates(max_candidates)
    resolved_prefilter = configured_reader_prefilter_sentences(prefilter_sentences)

    if resolved_mode == "off":
        return PaperReadResult(candidates=[], mode_used="off", latency_sec=0.0, prefilter_count=0)

    heuristic = HeuristicPaperReader(max_prefilter_sentences=resolved_prefilter)
    heuristic_result = heuristic.read(paper)
    if resolved_mode == "heuristic":
        return PaperReadResult(
            candidates=heuristic_result.candidates[:resolved_max_candidates],
            mode_used="heuristic",
            latency_sec=heuristic_result.latency_sec,
            prefilter_count=heuristic_result.prefilter_count,
            fallback_used=False,
        )

    if not configured_api_key(api_key):
        return PaperReadResult(
            candidates=heuristic_result.candidates[:resolved_max_candidates],
            mode_used="heuristic",
            latency_sec=heuristic_result.latency_sec,
            prefilter_count=heuristic_result.prefilter_count,
            fallback_used=True,
        )

    mini = LLMPaperReaderMini(
        model=model,
        max_candidates=resolved_max_candidates,
        candidates=heuristic_result.candidates,
        client=client,
        api_key=api_key,
        base_url=base_url,
    )
    mini_result = mini.read(paper)
    if mini_result.candidates:
        return mini_result
    return PaperReadResult(
        candidates=heuristic_result.candidates[:resolved_max_candidates],
        mode_used="heuristic",
        latency_sec=round(heuristic_result.latency_sec + mini_result.latency_sec, 4),
        prefilter_count=heuristic_result.prefilter_count,
        fallback_used=True,
    )


def _candidate_from_sentence(
    sentence: str,
    *,
    source_field: str | None,
    sentence_index: int,
    char_start: int,
    char_end: int,
) -> PaperReadCandidate | None:
    direction = _guess_direction(sentence)
    magnitude_text, _, _ = parse_magnitude(sentence)
    conditions = _guess_conditions(sentence)
    scope = _guess_scope(sentence)
    metric = _guess_metric(sentence)
    baseline = _guess_baseline(sentence)
    subject = _guess_method(sentence) or _guess_model(sentence)
    task = _guess_task(sentence)
    score, reason = _score_candidate_sentence(
        sentence,
        direction=direction,
        magnitude_text=magnitude_text,
        metric=metric,
        baseline=baseline,
        conditions=conditions,
        scope=scope,
        subject=subject,
        task=task,
    )
    if score <= 0:
        return None
    return PaperReadCandidate(
        sentence=sentence.strip(),
        evidence_span=sentence.strip(),
        evidence_source_field=source_field,
        evidence_sentence_index=sentence_index,
        evidence_char_start=char_start,
        evidence_char_end=char_end,
        subject_raw=subject,
        predicate=_guess_predicate(sentence, direction),
        object_raw=task,
        dataset_raw=_guess_dataset(sentence),
        metric_raw=metric,
        baseline_raw=baseline,
        direction=direction,
        magnitude_text=magnitude_text,
        conditions=conditions,
        scope=scope,
        candidate_score=round(score, 4),
        selection_reason=reason,
    )


def _score_candidate_sentence(
    sentence: str,
    *,
    direction: str | None,
    magnitude_text: str | None,
    metric: str | None,
    baseline: str | None,
    conditions: list[str],
    scope: list[str],
    subject: str | None,
    task: str | None,
) -> tuple[float, str]:
    low = sentence.lower()
    score = 0.0
    reasons: list[str] = []
    if direction:
        score += 0.4
        reasons.append(f"{direction} cue")
    if magnitude_text:
        score += 0.9
        reasons.append("delta")
    if metric or _METRIC_CUE_RE.search(low):
        score += 0.8
        reasons.append("metric")
    if baseline or _BASELINE_CUE_RE.search(low):
        score += 0.6
        reasons.append("baseline/comparison")
    if conditions:
        score += 0.4
        reasons.append("conditions")
    if scope:
        score += 0.25
        reasons.append("scope")
    if subject:
        score += 0.35
        reasons.append("method/model")
    if task:
        score += 0.2
        reasons.append("task")
    if len(sentence.split()) > 45:
        score -= 0.15
    if not reasons:
        return 0.0, ""
    return score, ", ".join(reasons)


def _mini_reader_prompt(paper: Paper, candidates: list[PaperReadCandidate], max_candidates: int) -> str:
    abstract_prefix = (paper.abstract or "").strip()
    if len(abstract_prefix) > 600:
        abstract_prefix = abstract_prefix[:600].rsplit(" ", 1)[0] + "..."
    lines = []
    for idx, candidate in enumerate(candidates):
        lines.append(
            json.dumps(
                {
                    "candidate_index": idx,
                    "sentence": candidate.sentence,
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
                    "heuristic_score": candidate.candidate_score,
                    "heuristic_reason": candidate.selection_reason,
                },
                ensure_ascii=False,
            )
        )
    return (
        f"Title: {paper.title}\n\n"
        f"Abstract prefix:\n{abstract_prefix}\n\n"
        f"Select up to {max_candidates} candidate claim sentences from this pool.\n"
        "Candidate pool:\n"
        + "\n".join(lines)
    )


def _candidate_from_llm_item(item: Any, pool: list[PaperReadCandidate]) -> PaperReadCandidate | None:
    if not isinstance(item, dict):
        return None
    try:
        index = int(item.get("candidate_index"))
    except (TypeError, ValueError):
        return None
    if index < 0 or index >= len(pool):
        return None
    original = pool[index]
    evidence_span = str(item.get("evidence_span") or original.evidence_span).strip()
    if not evidence_span:
        return None
    local_start = original.sentence.find(evidence_span)
    if local_start == -1:
        local_start = 0
        evidence_span = original.evidence_span
    absolute_start = original.evidence_char_start
    absolute_end = original.evidence_char_end
    if absolute_start is not None:
        absolute_start += local_start
        absolute_end = absolute_start + len(evidence_span)
    direction = str(item.get("direction") or original.direction or "").strip().lower() or None
    if direction not in _ALLOWED_DIRECTIONS:
        direction = original.direction
    magnitude_text = str(item.get("magnitude_text") or original.magnitude_text or "").strip() or None
    score = item.get("candidate_score")
    try:
        numeric_score = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        numeric_score = float(original.candidate_score or 0.0)
    return PaperReadCandidate(
        sentence=original.sentence,
        evidence_span=evidence_span,
        evidence_source_field=original.evidence_source_field,
        evidence_sentence_index=original.evidence_sentence_index,
        evidence_char_start=absolute_start,
        evidence_char_end=absolute_end,
        subject_raw=_clean_str(item.get("subject_raw")) or original.subject_raw,
        predicate=_clean_str(item.get("predicate")) or original.predicate,
        object_raw=_clean_str(item.get("object_raw")) or original.object_raw,
        dataset_raw=_clean_str(item.get("dataset_raw")) or original.dataset_raw,
        metric_raw=_clean_str(item.get("metric_raw")) or original.metric_raw,
        baseline_raw=_clean_str(item.get("baseline_raw")) or original.baseline_raw,
        direction=direction,
        magnitude_text=magnitude_text,
        conditions=normalize_string_list(item.get("conditions")) or original.conditions,
        scope=normalize_string_list(item.get("scope")) or original.scope,
        candidate_score=round(numeric_score, 4),
        selection_reason=_clean_str(item.get("selection_reason")) or "mini reader selection",
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


def _guess_direction(sentence: str) -> Direction | None:
    low = sentence.lower()
    for direction, cues in _DIRECTION_CUES.items():
        if any(cue in low for cue in cues):
            return direction  # type: ignore[return-value]
    return None


def _guess_method(sentence: str) -> str | None:
    low = sentence.lower()
    for kw in _METHOD_KEYWORDS:
        if kw.lower() in low:
            return kw
    return None


def _guess_model(sentence: str) -> str | None:
    low = sentence.lower()
    for kw in _MODEL_KEYWORDS:
        if kw.lower() in low:
            return kw
    return None


def _guess_task(sentence: str) -> str | None:
    low = sentence.lower()
    for needle, canonical in _TASK_KEYWORDS:
        if needle in low:
            return canonical
    return None


def _guess_dataset(sentence: str) -> str | None:
    low = sentence.lower()
    for needle, canonical in _DATASET_KEYWORDS:
        if needle in low:
            return canonical
    return None


def _guess_metric(sentence: str) -> str | None:
    low = sentence.lower()
    for needle, canonical in _METRIC_KEYWORDS:
        if len(needle) <= 3:
            if re.search(rf"\b{re.escape(needle)}\b", low):
                return canonical
            continue
        if needle in low:
            return canonical
    if "%" in sentence:
        return "Percent"
    return None


def _guess_baseline(sentence: str) -> str | None:
    match = re.search(r"\b(?:over|than|against|compared with|compared to)\s+([^.,;]+)", sentence, re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _guess_predicate(sentence: str, direction: str | None) -> str | None:
    low = sentence.lower()
    cues = sorted(_DIRECTION_CUES.get(direction or "", ()), key=len, reverse=True)
    for cue in cues:
        if cue in low:
            return cue
    if direction == "positive":
        return "improves"
    if direction == "negative":
        return "degrades"
    if direction == "mixed":
        return "has mixed effect on"
    return None


def _guess_conditions(sentence: str) -> list[str]:
    return normalize_string_list([match.group(0).strip() for match in _CONDITION_RE.finditer(sentence)])


def _guess_scope(sentence: str) -> list[str]:
    return normalize_string_list([match.group(0).strip() for match in _SCOPE_RE.finditer(sentence)])


def _clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value or value.lower() in {"null", "none", "n/a", "na", "unknown"}:
        return None
    return value
