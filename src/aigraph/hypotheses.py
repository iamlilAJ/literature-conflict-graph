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


def _is_retrieval(a: Anomaly) -> bool:
    m = (_method(a) or "").lower()
    return any(tok in m for tok in ("rag", "retriev", "rerank", "bm25", "dpr"))


def _is_multi_hop(a: Anomaly) -> bool:
    return "multi-hop" in (_task(a) or "").lower()


def _is_long_context(a: Anomaly) -> bool:
    return "long-context" in (_task(a) or "").lower() or "context_length" in a.varying_settings


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


def _setting_field(
    a: Anomaly,
    claims_by_id: dict[str, Claim],
    field: str,
    default: str,
    *,
    max_items: int = 2,
    overflow_label: str = "related settings",
) -> str:
    return _join(
        _unique([getattr(c.setting, field) for c in _anomaly_claims(a, claims_by_id)]),
        default,
        max_items=max_items,
        overflow_label=overflow_label,
    )


def _metric(a: Anomaly, claims_by_id: dict[str, Claim]) -> str:
    return _claim_field(a, claims_by_id, "metric", "the reported evaluation metric", max_items=1, overflow_label="related metrics")


def _dataset(a: Anomaly, claims_by_id: dict[str, Claim]) -> str:
    return _claim_field(a, claims_by_id, "dataset", f"{_task(a)} benchmarks", max_items=1, overflow_label="related benchmarks")


def _baseline(a: Anomaly, claims_by_id: dict[str, Claim]) -> str:
    return _claim_field(a, claims_by_id, "baseline", "the strongest available baseline", max_items=1, overflow_label="matched baselines")


def _result(a: Anomaly, claims_by_id: dict[str, Claim]) -> str:
    return _claim_field(a, claims_by_id, "result", "the reported gain", max_items=1, overflow_label="reported effects")


def _model(a: Anomaly, claims_by_id: dict[str, Claim]) -> str:
    return _claim_field(a, claims_by_id, "model", "the same generator family", max_items=1, overflow_label="related models")


def _retriever(a: Anomaly, claims_by_id: dict[str, Claim]) -> str:
    return _setting_field(a, claims_by_id, "retriever", "the retrieval pipeline", max_items=1, overflow_label="related retrieval settings")


def _bridge(a: Anomaly) -> GraphBridge:
    if a.type == "bridge_opportunity":
        return GraphBridge(
            **{
                "from": a.shared_entities.get("method_from", ""),
                "to": a.shared_entities.get("method_to", ""),
            }
        )
    return GraphBridge(**{"from": _method(a), "to": _task(a)})


def _retrieval_noise(a: Anomaly, claims_by_id: dict[str, Claim]) -> Optional[dict]:
    if not _is_retrieval(a):
        return None
    dataset = _dataset(a, claims_by_id)
    metric = _metric(a, claims_by_id)
    retriever = _retriever(a, claims_by_id)
    return {
        "hypothesis": (
            f"High top-k retrieval injects distractor passages that overwhelm the generator "
            f"on {_task(a)}, flipping the effect of {_method(a)} from positive to negative."
        ),
        "mechanism": (
            "Distractor evidence competes for the generator's attention; the signal-to-noise "
            "ratio of the retrieved set, not retrieval recall, governs downstream accuracy."
        ),
        "predictions": [
            f"Holding the retriever fixed, increasing top_k from 3 to 20 monotonically degrades {_task(a)} accuracy.",
            f"Entailment or reranker filtering improves {metric} without changing {retriever}.",
        ],
        "minimal_test": (
            f"Sweep top_k in {{1, 3, 5, 10, 20}} on {dataset} with a fixed generator and "
            f"{retriever}; measure {metric} plus evidence support."
        ),
        "scope_conditions": {"task_type": "multi-hop" if _is_multi_hop(a) else "factual", "retrieval_noise": "high"},
        "evidence_gap": "Most RAG papers hold top_k constant or sweep it only on factual QA, rarely on multi-hop.",
    }


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


def _model_scale(a: Anomaly, claims_by_id: dict[str, Claim]) -> Optional[dict]:
    if not _is_retrieval(a):
        return None
    dataset = _dataset(a, claims_by_id)
    metric = _metric(a, claims_by_id)
    model = _model(a, claims_by_id)
    return {
        "hypothesis": (
            f"The marginal benefit of {_method(a)} on {_task(a)} decreases with generator scale, "
            "because stronger base models internalize the facts that retrieval would otherwise supply."
        ),
        "mechanism": (
            "Pretraining corpora increasingly cover benchmark facts, so larger models rely less on "
            "external evidence for common-entity questions, shrinking the RAG lift."
        ),
        "predictions": [
            f"Plotting {_method(a)} gain vs. base-model size yields a decreasing curve on {_task(a)}.",
            f"The reported gap on {dataset} shrinks for stronger models than {model}.",
        ],
        "minimal_test": (
            f"Evaluate {_method(a)} and the same non-retrieval baseline across three model sizes on "
            f"{dataset}; report delta {metric} as a function of scale."
        ),
        "scope_conditions": {"base_model_size": "7B-70B", "task_type": "factual"},
        "evidence_gap": "Reported RAG numbers rarely hold base-model capacity constant across claims.",
    }


def _retriever_generator_misalignment(a: Anomaly, claims_by_id: dict[str, Claim]) -> Optional[dict]:
    if not (_is_retrieval(a) and _is_multi_hop(a)):
        return None
    dataset = _dataset(a, claims_by_id)
    metric = _metric(a, claims_by_id)
    return {
        "hypothesis": (
            "Lexical retrievers match surface tokens of the question but not the intermediate "
            "reasoning bridges required by multi-hop QA, so adding more passages cannot supply "
            "the missing chain."
        ),
        "mechanism": (
            "Multi-hop answers depend on a reasoning chain; retrievers trained on "
            "question-to-answer relevance don't surface the bridging entities."
        ),
        "predictions": [
            f"Generator-aware or entailment-filtered retrieval yields larger gains on {dataset} than raising top_k.",
            "Retrieval recall measured on bridging entities correlates with final EM more than overall recall does.",
        ],
        "minimal_test": (
            f"On {dataset}, compare a generator-aware reranker against a high-top_k retriever at equal "
            f"retrieval cost; report {metric} and bridge-entity recall."
        ),
        "scope_conditions": {"task_type": "multi-hop"},
        "evidence_gap": "Retrieval metrics are reported on question-answer relevance, not on bridging entities.",
    }


def _benchmark_contamination(a: Anomaly, claims_by_id: dict[str, Claim]) -> Optional[dict]:
    anomaly_text = " ".join(
        [
            a.central_question or "",
            _method(a),
            _task(a),
            _dataset(a, claims_by_id),
            _baseline(a, claims_by_id),
        ]
    ).lower()
    claim_text = " ".join((c.claim_text or "") for c in _anomaly_claims(a, claims_by_id)).lower()
    contamination_markers = (
        "pretrain",
        "contamin",
        "closed-book",
        "benchmark",
        "retrieval",
        "rag",
        "date-filter",
        "test-set",
    )
    if not any(marker in anomaly_text or marker in claim_text for marker in contamination_markers):
        return None
    dataset = _dataset(a, claims_by_id)
    metric = _metric(a, claims_by_id)
    result = _result(a, claims_by_id)
    baseline = _baseline(a, claims_by_id)
    return {
        "hypothesis": (
            "Partial test-set contamination in pretraining inflates closed-book baselines, which "
            "shrinks or reverses the apparent benefit of retrieval on standard benchmarks."
        ),
        "mechanism": (
            "Web-crawled pretraining data overlaps with benchmark test questions, letting larger "
            "models answer without retrieval and collapsing the measured RAG gain."
        ),
        "predictions": [
            f"On a held-out, date-filtered version of {dataset}, {_method(a)} gains move away from {result}.",
            f"{baseline} performance correlates with membership-inference scores on the evaluation set.",
        ],
        "minimal_test": (
            f"Build a post-cutoff evaluation set matching {dataset}; evaluate {_method(a)} and {baseline}; "
            f"compare delta {metric} to the original benchmark."
        ),
        "scope_conditions": {"task_type": "factual"},
        "evidence_gap": f"Few papers quantify pretraining overlap with {dataset}.",
    }


def _baseline_weakness(a: Anomaly, claims_by_id: dict[str, Claim]) -> Optional[dict]:
    baselines = {(c.baseline or "").lower() for c in _anomaly_claims(a, claims_by_id) if c.baseline}
    if not baselines:
        return None
    baseline = _baseline(a, claims_by_id)
    dataset = _dataset(a, claims_by_id)
    metric = _metric(a, claims_by_id)
    return {
        "hypothesis": (
            f"Gains attributed to {_method(a)} are inflated when compared against weak baselines; "
            "the sign of the effect depends primarily on baseline choice."
        ),
        "mechanism": (
            "Weak baselines (under-prompted closed-book LLMs, small models) leave more headroom, "
            "so any additional evidence appears to help even when retrieval is noisy."
        ),
        "predictions": [
            f"Replacing {baseline} with a stronger matched baseline reduces the reported {_method(a)} gain.",
            "The correlation between baseline strength and RAG gain is negative across papers in this anomaly.",
        ],
        "minimal_test": (
            f"Re-run the positive claims on {dataset} against {baseline} and a stronger matched baseline; "
            f"report the baseline-conditioned delta in {metric}."
        ),
        "scope_conditions": {"baseline_strength": "low_vs_high"},
        "evidence_gap": "Baseline strength is rarely controlled across RAG reports.",
    }


def _context_length_saturation(a: Anomaly, claims_by_id: dict[str, Claim]) -> Optional[dict]:
    claims = _anomaly_claims(a, claims_by_id)
    has_long_ctx = any((c.setting.context_length or "") in ("32k", "64k", "128k") for c in claims)
    if not (has_long_ctx or _is_long_context(a) or "context_length" in a.varying_settings):
        return None
    dataset = _dataset(a, claims_by_id)
    metric = _metric(a, claims_by_id)
    return {
        "hypothesis": (
            f"As generator context length grows, the marginal benefit of {_method(a)} on {_task(a)} "
            "saturates because in-context evidence can be supplied without retrieval."
        ),
        "mechanism": (
            "Long-context models can ingest the full corpus or a broad slice of it, so retrieval "
            "no longer uniquely provides the relevant passage."
        ),
        "predictions": [
            f"As context length increases, the delta between {_method(a)} and its baseline shrinks on {dataset}.",
            "The RAG gain as a function of context length forms a monotonically decreasing curve.",
        ],
        "minimal_test": (
            f"Fix the generator family and sweep context length in {{4k, 16k, 32k, 128k}} on {dataset}; "
            f"report {_method(a)} delta in {metric}."
        ),
        "scope_conditions": {"context_length": ">=32k"},
        "evidence_gap": "Most RAG papers benchmark on 4k-context generators only.",
    }


def _task_decomposition(a: Anomaly, claims_by_id: dict[str, Claim]) -> Optional[dict]:
    if not _is_multi_hop(a):
        return None
    dataset = _dataset(a, claims_by_id)
    metric = _metric(a, claims_by_id)
    return {
        "hypothesis": (
            "Retrieval-then-answer pipelines fail multi-hop QA because they retrieve once against the "
            "original question; an explicit decomposition step is required for the second hop."
        ),
        "mechanism": (
            "The second-hop evidence is keyed on a bridging entity that only appears after the first "
            "hop is resolved; single-shot retrieval cannot surface it."
        ),
        "predictions": [
            f"Iterative retrieval with sub-question decomposition improves {metric} on {dataset}.",
            f"Single-shot {_method(a)} performance on bridge-style questions is bounded by first-hop recall.",
        ],
        "minimal_test": (
            f"Compare single-shot {_method(a)} against a two-step decomposition pipeline on {dataset}; "
            f"report {metric} by question subtype."
        ),
        "scope_conditions": {"task_type": "multi-hop", "question_type": "bridge"},
        "evidence_gap": "Multi-hop RAG numbers rarely disaggregate bridge vs. comparison questions.",
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
    _retrieval_noise,
    _metric_sensitivity,
    _model_scale,
    _retriever_generator_misalignment,
    _benchmark_contamination,
    _baseline_weakness,
    _context_length_saturation,
    _task_decomposition,
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
