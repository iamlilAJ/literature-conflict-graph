"""LLM-backed possible-explanation generator for detected anomalies."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from pydantic import ValidationError

from .hypotheses import HypothesisGenerator, TemplateGenerator, _bridge
from .llm_client import (
    DEFAULT_MAX_TOKENS,
    build_openai_client,
    call_llm_text,
    configured_api_key,
    configured_base_url,
    configured_model,
)
from .llm_extract import _load_json
from .models import Anomaly, Claim, GraphBridge, Hypothesis


logger = logging.getLogger(__name__)


# Shared structural rules — these apply to every anomaly-type prompt and define
# the JSON contract + epistemic guardrails. Kept in one place so the schema,
# evidence-citation discipline, and "do not assert truth" rule live ONCE.
_SHARED_RULES = """SHARED RULES — apply to every response:
- Output STRICT JSON, schema { "hypotheses": [ ... ] }; no markdown, no fences, no prose
- Generate EXACTLY 3 hypotheses, each distinct from the others
- Each hypothesis must include all of: hypothesis, mechanism, explains_claims,
  predictions (exactly 2 short concrete strings), minimal_test, scope_conditions,
  evidence_gap, graph_bridge
- explains_claims must reference real claim_ids from the user payload (≥1 id)
- Do not assert any hypothesis as true — these are candidates for human review
- graph_bridge.from / graph_bridge.to should reference shared_entities or
  claim metadata from the user payload, not invented terms

JSON schema:
{
  "hypotheses": [
    {
      "hypothesis": "one-sentence statement",
      "mechanism": "causal mechanism or moderator",
      "explains_claims": ["c001", "c002"],
      "predictions": ["prediction 1", "prediction 2"],
      "minimal_test": "specific experiment or analysis",
      "scope_conditions": {"method": "...", "task": "..."},
      "evidence_gap": "what evidence is still missing",
      "graph_bridge": {"from": "source concept", "to": "target concept"}
    }
  ]
}"""


# Per-type framing strings. Each describes the cognitive task for THIS anomaly
# type. Concatenated with _SHARED_RULES at module-import time so the model
# anchors on the type-specific framing first, then the shared output rules act
# as the closing constraint block (recency-bias-friendly).

_FRAMING_BENCHMARK = """You generate competing explanations for a benchmark inconsistency.

Two or more papers report contradictory results for the same method on the
same task. Generate exactly 3 competing, testable explanations focused on
**moderator variables that could flip the sign** of the effect — dataset
composition, distribution shift, prompt format, retrieval recall, model
scale, decoding strategy, or evaluation protocol differences. Each
hypothesis is a back-explanation of why the contradiction exists; each
minimal_test is an experiment that holds the alleged moderator fixed and
checks whether the contradiction disappears."""

_FRAMING_REPLICATION = """You generate root-cause hypotheses for a replication failure.

Paper B explicitly attempts to reproduce Paper A's method on the same task
but reports the opposite direction. Generate exactly 3 candidate root
causes drawn from: implementation differences, hyperparameter choices,
data preprocessing or version drift, leakage or test-set contamination,
metric definition mismatch, seed variance, hardware/precision (fp16 vs
fp32 / bf16), or under-specified algorithm details. Each prediction must
be **distinguishable by re-running B's code on A's exact dataset, seed,
and hardware** — that is the empirical bar. minimal_test must describe
the controlled re-run that would isolate the responsible factor."""

_FRAMING_BRIDGE = """You generate forward-looking transfer hypotheses for a bridge opportunity.

Two clusters of work share underlying concepts but have no citation path
between them. Generate exactly 3 **forward-looking** hypotheses: what
would happen if methods/findings from one cluster were applied to the
other cluster's task? Each hypothesis is a research direction that has
NOT yet been tried, not a retrospective explanation of past results.
minimal_test must specify a concrete, runnable transfer experiment:
which method, which dataset, which baseline, which metric — phrased so a
researcher could start it tomorrow."""

_FRAMING_COMMUNITY = """You generate hypotheses about a community disconnect.

Two research communities address overlapping problems with shared
mechanisms or failure modes, yet do not cite each other. Generate
exactly 3 hypotheses split between (a) **why the disconnect persists**
(terminology drift, venue separation, language barrier, methodological
priors, generational lag) and (b) **what cross-pollination would look
like** (which technique from community A would be most useful in
community B, and vice versa). predictions should be testable via a
focused literature comparison or a small replication study that imports
one community's tooling into the other's benchmark."""

_FRAMING_EVIDENCE_GAP = """You generate hypotheses about an evidence gap.

All available claims point in one direction (positive or negative) but
the evidence base is thin or one-sided. Generate exactly 3 hypotheses
covering (a) **whether the effect is real and robust** vs an artifact of
selective reporting, file-drawer bias, or under-tested settings, and
(b) **what specific missing evidence** would confirm or refute it.
minimal_test must specify the exact data that would close the gap —
which experimental conditions are missing, which negative controls have
not been run, which adversarial settings have not been probed."""

_FRAMING_SETTING_MISMATCH = """You generate hypotheses for a setting mismatch.

The same method on the same task produces conflicting outcomes across
papers, and the runs differ along at least one Setting field
(retriever / top_k / context_length / task_type / etc). Generate
exactly 3 hypotheses about **which setting variable is responsible for
the divergence**. Each prediction must hold all other settings fixed and
vary only the candidate variable; minimal_test must be a parameter sweep
over that single setting on at least one of the original benchmarks."""

_FRAMING_METRIC_MISMATCH = """You generate hypotheses for a metric mismatch.

Same method, same task, different metrics, conflicting outcomes.
Generate exactly 3 hypotheses about **which metric is measuring what**
and how the metrics' definitional gap drives the divergence — e.g. one
metric rewards calibration where the other rewards rank, or one is
sensitive to surface form where the other is not. Each prediction must
involve **cross-scoring the same model outputs with both metrics** on a
shared eval set; minimal_test must specify how to obtain or generate
that paired prediction set."""


def _build_prompt(framing: str) -> str:
    """Concatenate per-type framing then shared rules. Format chosen so the
    output-shaping constraints are last (recency bias)."""
    return f"{framing.strip()}\n\n{_SHARED_RULES.strip()}\n"


# Map every AnomalyType value to its specialized prompt. The
# test_system_prompts_covers_every_anomaly_type test asserts this dict's keys
# match `typing.get_args(AnomalyType)` — adding a new anomaly type without a
# prompt will fail CI rather than silently fall through to the default.
SYSTEM_PROMPTS: dict[str, str] = {
    "benchmark_inconsistency": _build_prompt(_FRAMING_BENCHMARK),
    "impact_conflict": _build_prompt(_FRAMING_BENCHMARK),
    "replication_conflict": _build_prompt(_FRAMING_REPLICATION),
    "bridge_opportunity": _build_prompt(_FRAMING_BRIDGE),
    "community_disconnect": _build_prompt(_FRAMING_COMMUNITY),
    "evidence_gap": _build_prompt(_FRAMING_EVIDENCE_GAP),
    "setting_mismatch": _build_prompt(_FRAMING_SETTING_MISMATCH),
    "metric_mismatch": _build_prompt(_FRAMING_METRIC_MISMATCH),
}


# Fallback prompt — kept as the original generic text for safety. A new
# AnomalyType value not covered by SYSTEM_PROMPTS will route here and log
# once. See _select_prompt below.
_DEFAULT_PROMPT = """You generate possible explanations for conflicts in AI literature.

You will receive one detected anomaly and the evidence claims that define it.
Return STRICT JSON only, no markdown, no analysis prose.

Generate exactly 3 competing, testable explanations. Keep every string concise.
Each explanation must:
- explain the anomaly using the provided claims only;
- cite concrete claim_ids in explains_claims;
- be distinct from the others;
- include exactly two discriminative predictions;
- include a concrete minimal_test;
- state the remaining evidence_gap;
- avoid saying the claim is true; these are candidate explanations for human inspection.

JSON schema:
{
  "hypotheses": [
    {
      "hypothesis": "one-sentence explanation",
      "mechanism": "causal mechanism or moderator",
      "explains_claims": ["c001", "c002"],
      "predictions": ["prediction 1", "prediction 2"],
      "minimal_test": "specific experiment or analysis",
      "scope_conditions": {"key": "value"},
      "evidence_gap": "what evidence is still missing",
      "graph_bridge": {"from": "source concept", "to": "target concept"}
    }
  ]
}
"""


# Module-level dedup so a corpus with N anomalies of an unknown type produces
# one INFO line per type per process, not N. Module-level (not instance-level)
# so a pipeline that constructs many LLMHypothesisGenerator objects still
# benefits from the dedup.
_warned_unknown_types: set[str] = set()


def _select_prompt(anomaly_type: str) -> str:
    """Look up the prompt for an anomaly type; fall back to _DEFAULT_PROMPT
    on unknown types and log a single INFO line per unknown type per process."""
    prompt = SYSTEM_PROMPTS.get(anomaly_type)
    if prompt is not None:
        return prompt
    if anomaly_type not in _warned_unknown_types:
        _warned_unknown_types.add(anomaly_type)
        logger.info(
            "llm_hypotheses: no specialized prompt for anomaly type %r; "
            "falling back to _DEFAULT_PROMPT",
            anomaly_type,
        )
    return _DEFAULT_PROMPT


class LLMHypothesisGenerator(HypothesisGenerator):
    """Use an OpenAI-compatible chat model to generate anomaly explanations."""

    def __init__(
        self,
        model: Optional[str] = None,
        client: Any | None = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fallback: HypothesisGenerator | None = None,
    ):
        self.model = configured_model(model)
        self._client = client
        self._api_key = configured_api_key(api_key)
        self._base_url = configured_base_url(base_url)
        self.fallback = fallback if fallback is not None else TemplateGenerator()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = build_openai_client(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def generate(
        self,
        anomaly: Anomaly,
        claims_by_id: dict[str, Claim],
        start_index: int = 0,
    ) -> list[Hypothesis]:
        claims = [claims_by_id[cid] for cid in anomaly.claim_ids if cid in claims_by_id]
        if not claims:
            return []
        try:
            raw = self._call_llm(anomaly, claims)
            parsed = self._parse_response(raw, anomaly, claims_by_id, start_index)
        except Exception as e:  # pragma: no cover - defensive for network/model errors
            logger.warning("LLM hypothesis generation failed for %s: %s", anomaly.anomaly_id, e)
            parsed = []
        if parsed:
            return parsed
        return self.fallback.generate(anomaly, claims_by_id, start_index=start_index)

    def _call_llm(self, anomaly: Anomaly, claims: list[Claim]) -> str:
        client = self._get_client()
        return call_llm_text(
            client,
            model=self.model,
            system=_select_prompt(anomaly.type),
            user=_prompt_payload(anomaly, claims),
            temperature=float(os.environ.get("AIGRAPH_HYPOTHESIS_TEMPERATURE", "0.2")),
            max_tokens=int(os.environ.get("AIGRAPH_HYPOTHESIS_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        )

    def _parse_response(
        self,
        raw: str,
        anomaly: Anomaly,
        claims_by_id: dict[str, Claim],
        start_index: int,
    ) -> list[Hypothesis]:
        payload = _load_json(raw)
        items = payload.get("hypotheses") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            logger.warning("Unexpected LLM hypothesis payload for %s", anomaly.anomaly_id)
            return []

        allowed_claims = set(anomaly.claim_ids)
        out: list[Hypothesis] = []
        for item in items[:5]:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_hypothesis_dict(item, anomaly, allowed_claims)
            if normalized is None:
                continue
            normalized["hypothesis_id"] = f"h{start_index + len(out) + 1:03d}"
            normalized["anomaly_id"] = anomaly.anomaly_id
            try:
                out.append(Hypothesis.model_validate(normalized))
            except ValidationError as e:
                logger.warning("Hypothesis validation failed for %s: %s", anomaly.anomaly_id, e)
                continue
        return out


def _prompt_payload(anomaly: Anomaly, claims: list[Claim]) -> str:
    data = {
        "anomaly": {
            "anomaly_id": anomaly.anomaly_id,
            "type": anomaly.type,
            "central_question": anomaly.central_question,
            "positive_claims": anomaly.positive_claims,
            "negative_or_mixed_claims": anomaly.negative_claims,
            "shared_entities": anomaly.shared_entities,
            "varying_settings": anomaly.varying_settings,
        },
        "claims": [_claim_summary(c) for c in claims],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _claim_summary(c: Claim) -> dict[str, Any]:
    return {
        "claim_id": c.claim_id,
        "claim_text": c.claim_text,
        "direction": c.direction,
        "claim_type": c.claim_type,
        "method": c.method,
        "canonical_method": c.canonical_method,
        "model": c.model,
        "task": c.task,
        "canonical_task": c.canonical_task,
        "dataset": c.dataset,
        "metric": c.metric,
        "baseline": c.baseline,
        "result": c.result,
        "setting": c.setting.model_dump(),
        "evidence_span": c.evidence_span,
    }


def _normalize_hypothesis_dict(item: dict[str, Any], anomaly: Anomaly, allowed_claims: set[str]) -> dict[str, Any] | None:
    hypothesis = _clean_str(item.get("hypothesis"))
    if hypothesis is None:
        return None

    explains = item.get("explains_claims") or []
    if not isinstance(explains, list):
        explains = []
    explains = [str(cid) for cid in explains if str(cid) in allowed_claims]
    if not explains:
        explains = list(anomaly.claim_ids)

    predictions = item.get("predictions") or []
    if not isinstance(predictions, list):
        predictions = []
    predictions = [_clean_str(p) for p in predictions]
    predictions = [p for p in predictions if p is not None][:4]

    scope = item.get("scope_conditions") or {}
    if not isinstance(scope, dict):
        scope = {}
    scope = {str(k): str(v) for k, v in scope.items() if v is not None}

    bridge_raw = item.get("graph_bridge") or {}
    if not isinstance(bridge_raw, dict):
        bridge = _bridge(anomaly)
    else:
        bridge = GraphBridge(
            **{
                "from": _clean_str(bridge_raw.get("from")) or _bridge(anomaly).from_,
                "to": _clean_str(bridge_raw.get("to")) or _bridge(anomaly).to,
            }
        )

    return {
        "hypothesis": hypothesis,
        "mechanism": _clean_str(item.get("mechanism")) or "",
        "explains_claims": explains,
        "predictions": predictions,
        "minimal_test": _clean_str(item.get("minimal_test")) or "",
        "scope_conditions": scope,
        "evidence_gap": _clean_str(item.get("evidence_gap")) or "",
        "graph_bridge": bridge,
    }


def _clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value or value.lower() in ("null", "none", "n/a", "na", "unknown"):
        return None
    return value
