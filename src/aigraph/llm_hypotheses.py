"""LLM-backed hypothesis generator for detected anomalies.

§7 THAW #4 (2026-06-22, see docs/v0.7-pipeline-freeze.md): the per-anomaly-type
RETROSPECTIVE back-explanation contract (8 type framings + a hard "EXACTLY 3"
rule) was replaced with a single FORWARD-DESIGN contract. For each anomaly the
generator now emits 2-4 deliberately shape-diverse FORWARD research hypotheses —
interior-optimum (a design knob with a sweet spot), mechanism (a specific causal
mechanism with differential predictions), and scaling/transfer — grounded in the
same claim cluster. At most one retrospective conflict-explanation may survive
per anomaly, so the delivered set cannot regress to the old monoculture.

Empirical justification (the §4 freeze bars quality edits WITHOUT measured
evidence): scripts/hyp_ab_generators.py ran a blind, calibration-passing A/B on
arxiv-reasoning-v0.7-100p (18 anomalies, all 6 live types, same model, same
anomaly seeds, judged by scripts/hyp_quality_oracle.py). Forward beat the frozen
generator on EVERY axis — forward_design +0.46, falsifiability +0.28 (→5.00),
mechanism_specificity +0.28, grounding +0.19, novelty +0.13 — and flipped the
delivered shape mix: conflict_attribution 37% → 4%. Owner-authorized thaw;
reproduce the pre-thaw control generator from the `v0.7-frozen` git tag.
"""

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


# --------------------------------------------------------------------------- #
# Forward-design generation contract (§7 Thaw #4).
#
# One type-agnostic system prompt: the anomaly's type + central_question are
# carried in the user payload (see _prompt_payload), so the model adapts per
# type without a separate prompt per type. The prompt BANS the old boilerplate
# explicitly and requires shape diversity.
# --------------------------------------------------------------------------- #
_SYSTEM = """You design FORWARD research hypotheses from a cluster of related ML/AI paper claims.

You are given one "anomaly" (a method/task cluster the literature has flagged)
plus the real claims that define it. Your job is NOT to explain why past papers
disagree. Your job is to propose concrete, testable research DIRECTIONS a team
could start next week, grounded in these specific claims.

BANNED (these are the failure mode you are replacing):
- "An unreported moderator variable / confound drives the conflicting results."
- "Replay all claims in a common harness with identical prompts and decoding."
- Any purely retrospective explanation of why existing numbers differ.
If your hypothesis would still be true after merely re-running old experiments
in a shared harness, it is BANNED — rewrite it as a forward design.

Produce 2-4 DISTINCT hypotheses that span DIFFERENT structural shapes. Cover at
least TWO of these shapes (never 3 of the same shape):

  [interior_optimum] A design knob with an internal sweet spot. State the knob
     (sample count, depth, #agents, temperature, retrieval-k, model scale...),
     the predicted optimum region, and the SPECIFIC mechanism that makes more
     of it eventually HURT. Form: "<knob> helps <task/metric> up to ~<N>,
     beyond which it degrades because <named mechanism>."

  [mechanism] A specific causal mechanism for why the method works, stated so it
     makes DIFFERENTIAL predictions: it should hold where task-property P is
     present and FAIL where P is absent. Name P concretely (e.g. "tasks with
     verifiable intermediate steps", "long-range coreference", "compositional
     depth > 3"). Not "some confound" — a named mechanism.

  [scaling_transfer] How the benefit scales with a measurable property, OR the
     result of porting the method to a NAMED new task/dataset. State the
     predicted direction and the property/target by name.

You MAY include at most ONE [mechanism]-style grounded explanation of a genuine
high-signal conflict if the claims strongly support it — but it must still make
a forward, falsifiable prediction, and it must not be the majority of the set.

GROUNDING RULES (hard):
- Every hypothesis cites >=1 real claim_id from the payload in explains_claims.
- Reference real methods / datasets / metrics from the payload, never invented.
- minimal_test names the dataset(s), the variable swept OR the mechanism-
  isolating manipulation, the metric, and what result would FALSIFY it.
- predictions: EXACTLY 2 short, discriminative strings, quantitative where the
  claims allow.
- Do not assert any hypothesis as true; these are candidates for human review.

The anomaly.signals object carries numeric context (evidence_impact,
recent_activity, impact_balance, citation_bridge_score, replication_score,
topology_score). Use it to choose which shapes to emphasise — e.g. a high
replication_score cluster is a good candidate for a mechanism hypothesis with a
controlled re-run as its minimal_test.

Output STRICT JSON ONLY, schema { "hypotheses": [ ... ] }, no markdown/prose:
{
  "hypotheses": [
    {
      "shape": "interior_optimum | mechanism | scaling_transfer",
      "hypothesis": "one-sentence forward design",
      "mechanism": "the specific causal mechanism",
      "explains_claims": ["c001"],
      "predictions": ["p1", "p2"],
      "minimal_test": "named dataset + swept variable + metric + falsifier",
      "scope_conditions": {"method": "...", "task": "..."},
      "evidence_gap": "what is still unmeasured",
      "graph_bridge": {"from": "source concept", "to": "target concept"}
    }
  ]
}
"""


def _temperature() -> float:
    """Generation temperature. Slightly warmer than the pre-thaw 0.2 — forward
    design benefits from spread while staying grounded. The legacy
    AIGRAPH_HYPOTHESIS_TEMPERATURE env var still applies (default now 0.4)."""
    return float(os.environ.get("AIGRAPH_HYPOTHESIS_TEMPERATURE", "0.4"))


def _max_tokens() -> int:
    """A 2-4 forward set is larger than the old 3-back-explanation set, and
    thinking models (Kimi) need headroom. Legacy AIGRAPH_HYPOTHESIS_MAX_TOKENS
    still applies."""
    return max(700, int(os.environ.get("AIGRAPH_HYPOTHESIS_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))))


def _max_per_anomaly() -> int:
    return max(1, int(os.environ.get("AIGRAPH_HYPOTHESIS_MAX_PER_ANOMALY", "4")))


_RETRO_MARKERS = (
    "unreported moderator",
    "a confound in",
    "common harness with identical prompts",
    "between-claim variance",
)


def _looks_retrospective(item: dict[str, Any]) -> bool:
    """Cheap lexical guard against the banned boilerplate sneaking back in."""
    blob = " ".join(
        str(item.get(k, "")) for k in ("hypothesis", "mechanism", "minimal_test")
    ).lower()
    return any(m in blob for m in _RETRO_MARKERS)


class LLMHypothesisGenerator(HypothesisGenerator):
    """Use an OpenAI-compatible chat model to generate forward-design
    hypotheses for an anomaly (§7 Thaw #4)."""

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
            system=_SYSTEM,
            user=_prompt_payload(anomaly, claims),
            temperature=_temperature(),
            max_tokens=_max_tokens(),
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
        cap = _max_per_anomaly()
        out: list[Hypothesis] = []
        conflict_kept = 0
        for item in items:
            if len(out) >= cap:
                break
            if not isinstance(item, dict):
                continue
            shape = str(item.get("shape", "")).strip().lower()
            # Guardrail: at most one retrospective conflict-attribution item may
            # survive per anomaly, so the generator cannot silently regress to
            # the pre-thaw dominant shape.
            if shape in {"conflict_attribution", "conflict", ""} and _looks_retrospective(item):
                if conflict_kept >= 1:
                    continue
                conflict_kept += 1
            normalized = _normalize_hypothesis_dict(item, anomaly, allowed_claims)
            if normalized is None:
                continue
            # Stamp the generator's own shape label so downstream/eval can read
            # it without re-judging. Lives in scope_conditions (already a free-
            # form str->str dict in the schema) to avoid any model change.
            if shape:
                normalized.setdefault("scope_conditions", {})
                normalized["scope_conditions"]["shape"] = shape
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
            "signals": _anomaly_signals(anomaly),
        },
        "claims": [_claim_summary(c) for c in claims],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def _anomaly_signals(anomaly: Anomaly) -> dict[str, float]:
    """Pack the numeric metadata fields _annotate_topology_scores writes onto
    each Anomaly. Uses getattr with 0.0 default so legacy fixtures or
    partially-constructed Anomalies (e.g. Anomaly.model_construct in tests)
    serialize cleanly. Rounds to 3 decimals — these are 0..~10 range floats
    and 3 decimals is enough resolution for the LLM to differentiate impact
    levels without burning tokens on noise digits."""
    return {
        "evidence_impact": round(float(getattr(anomaly, "evidence_impact", 0.0) or 0.0), 3),
        "recent_activity": round(float(getattr(anomaly, "recent_activity", 0.0) or 0.0), 3),
        "impact_balance": round(float(getattr(anomaly, "impact_balance", 0.0) or 0.0), 3),
        "citation_bridge_score": round(float(getattr(anomaly, "citation_bridge_score", 0.0) or 0.0), 3),
        "replication_score": round(float(getattr(anomaly, "replication_score", 0.0) or 0.0), 3),
        "topology_score": round(float(getattr(anomaly, "topology_score", 0.0) or 0.0), 3),
    }


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
