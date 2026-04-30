"""Template-driven hypothesis generation. Structured so an LLM generator can be dropped in."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from .models import Anomaly, Claim, GraphBridge, Hypothesis


class HypothesisGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        anomaly: Anomaly,
        claims_by_id: dict[str, Claim],
        start_index: int = 0,
    ) -> list[Hypothesis]:
        """Return candidate hypotheses for an anomaly."""


TemplateFn = Callable[[Anomaly, dict[str, Claim]], Optional[dict]]


def _method(a: Anomaly) -> str:
    return a.shared_entities.get("method") or a.shared_entities.get("method_from") or "the method"


def _task(a: Anomaly) -> str:
    return a.shared_entities.get("task") or a.shared_entities.get("task_from") or "the task"


def _is_metric_critique(a: Anomaly) -> bool:
    # Claims tagged with metric anomalies (e.g., EM/F1 vs evidence-chain).
    return "metric" in (_method(a) or "").lower()


def _claims_pos_neg(a: Anomaly, claims_by_id: dict[str, Claim]) -> tuple[list[Claim], list[Claim]]:
    pos = [claims_by_id[c] for c in a.positive_claims if c in claims_by_id]
    neg = [claims_by_id[c] for c in a.negative_claims if c in claims_by_id]
    return pos, neg


def _anomaly_claims(a: Anomaly, claims_by_id: dict[str, Claim]) -> list[Claim]:
    return [claims_by_id[cid] for cid in a.claim_ids if cid in claims_by_id]


def _unique(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        clean = value.strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _join(values: list[str], default: str, max_items: int = 2, overflow_label: str = "related settings") -> str:
    if not values:
        return default
    clipped = values[:max_items]
    if len(values) > max_items:
        if len(clipped) == 1:
            return f"{clipped[0]} and {overflow_label}"
        return ", ".join(clipped) + f", and {overflow_label}"
    return " and ".join(clipped)


def _claim_field(
    a: Anomaly,
    claims_by_id: dict[str, Claim],
    field: str,
    default: str,
    *,
    max_items: int = 2,
    overflow_label: str = "related settings",
) -> str:
    return _join(
        _unique([getattr(c, field) for c in _anomaly_claims(a, claims_by_id)]),
        default,
        max_items=max_items,
        overflow_label=overflow_label,
    )


def _metric(a: Anomaly, claims_by_id: dict[str, Claim]) -> str:
    return _claim_field(a, claims_by_id, "metric", "the reported evaluation metric", max_items=1, overflow_label="related metrics")


def _dataset(a: Anomaly, claims_by_id: dict[str, Claim]) -> str:
    return _claim_field(a, claims_by_id, "dataset", f"{_task(a)} benchmarks", max_items=1, overflow_label="related benchmarks")


def _bridge(a: Anomaly) -> GraphBridge:
    if a.type == "bridge_opportunity":
        return GraphBridge(
            **{
                "from": a.shared_entities.get("method_from", ""),
                "to": a.shared_entities.get("method_to", ""),
            }
        )
    return GraphBridge(**{"from": _method(a), "to": _task(a)})


def _metric_sensitivity(a: Anomaly, claims_by_id: dict[str, Claim]) -> Optional[dict]:
    claims = _anomaly_claims(a, claims_by_id)
    involves_em = any((c.metric or "").lower() in ("exact match", "em", "f1") for c in claims)
    if not (involves_em or _is_metric_critique(a)):
        return None
    metric = _metric(a, claims_by_id)
    dataset = _dataset(a, claims_by_id)
    return {
        "hypothesis": (
            f"The reported reversal is driven by {metric}: surface-oriented scores reward overlap, so "
            "evidence-enriched answers that paraphrase can score lower even when factually correct."
        ),
        "mechanism": (
            "Exact Match / F1 treats lexical overlap as ground truth and ignores whether the "
            "cited passages actually support the answer, rewarding short confident guesses."
        ),
        "predictions": [
            "Re-scoring with evidence-chain F1 or LLM-as-judge reduces the sign flip between claims.",
            f"Rankings of {_method(a)} variants on {dataset} shift when switching from {metric} to evidence-chain scoring.",
        ],
        "minimal_test": (
            f"Re-score the same {dataset} predictions with {metric} and evidence-chain F1; compute rank correlation."
        ),
        "scope_conditions": {"metric": metric},
        "evidence_gap": "Few benchmarks report both surface metrics and evidence-chain metrics on the same runs.",
    }


def _generic_moderator(a: Anomaly, claims_by_id: dict[str, Claim]) -> Optional[dict]:
    # Padding template so every anomaly has at least one hypothesis.
    dataset = _dataset(a, claims_by_id)
    metric = _metric(a, claims_by_id)
    return {
        "hypothesis": (
            f"An unreported moderator variable drives the conflicting results around {_method(a)} "
            f"on {_task(a)}."
        ),
        "mechanism": (
            "A confound in data preprocessing, prompt formatting, or decoding parameters correlates "
            "with outcome direction and is not held constant across the claims."
        ),
        "predictions": [
            "Holding prompt template and decoding fixed shrinks the between-claim variance by >50%.",
            "A covariate analysis reveals prompt/decoding parameters account for the sign flip.",
        ],
        "minimal_test": (
            f"Replay all claims on {dataset} in a common harness with identical prompts and decoding settings; "
            f"recompute {metric} deltas."
        ),
        "scope_conditions": {},
        "evidence_gap": "Prompt and decoding configurations are inconsistently reported across the claims.",
    }


TEMPLATES: list[TemplateFn] = [
    _metric_sensitivity,
    _generic_moderator,
]


class TemplateGenerator(HypothesisGenerator):
    def __init__(self, templates: list[TemplateFn] | None = None):
        self.templates = templates or TEMPLATES

    def generate(
        self,
        anomaly: Anomaly,
        claims_by_id: dict[str, Claim],
        start_index: int = 0,
    ) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        for fn in self.templates:
            data = fn(anomaly, claims_by_id)
            if data is None:
                continue
            hid = f"h{start_index + len(out) + 1:03d}"
            out.append(
                Hypothesis(
                    hypothesis_id=hid,
                    anomaly_id=anomaly.anomaly_id,
                    hypothesis=data["hypothesis"],
                    mechanism=data.get("mechanism", ""),
                    explains_claims=list(anomaly.claim_ids),
                    predictions=list(data.get("predictions", [])),
                    minimal_test=data.get("minimal_test", ""),
                    scope_conditions=dict(data.get("scope_conditions", {})),
                    evidence_gap=data.get("evidence_gap", ""),
                    graph_bridge=_bridge(anomaly),
                )
            )
        return out


def generate_hypotheses(
    anomalies: list[Anomaly],
    claims: list[Claim],
    generator: HypothesisGenerator | None = None,
) -> list[Hypothesis]:
    generator = generator or TemplateGenerator()
    claims_by_id = {c.claim_id: c for c in claims}
    out: list[Hypothesis] = []
    for anomaly in anomalies:
        out.extend(generator.generate(anomaly, claims_by_id, start_index=len(out)))
    return out
