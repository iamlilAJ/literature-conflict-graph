"""LLM-backed possible-explanation generator for detected anomalies."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from pydantic import ValidationError

from .hypotheses import HypothesisGenerator, TemplateGenerator, _bridge
from .llm_extract import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TIMEOUT_SECONDS, _load_json
from .models import Anomaly, Claim, GraphBridge, Hypothesis


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You generate possible explanations for conflicts in AI literature.

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
        self.model = model or os.environ.get("AIGRAPH_MODEL") or DEFAULT_MODEL
        self._client = client
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("AIGRAPH_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.fallback = fallback if fallback is not None else TemplateGenerator()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - exercised in real runs
            raise RuntimeError(
                "openai package is required for LLMHypothesisGenerator. "
                "Install with `pip install -e '.[real]'`."
            ) from e
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        kwargs["timeout"] = float(os.environ.get("AIGRAPH_LLM_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
        self._client = OpenAI(**kwargs)
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
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _prompt_payload(anomaly, claims)},
            ],
            temperature=float(os.environ.get("AIGRAPH_HYPOTHESIS_TEMPERATURE", "0.2")),
            max_tokens=int(os.environ.get("AIGRAPH_HYPOTHESIS_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        )
        return resp.choices[0].message.content or ""

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
