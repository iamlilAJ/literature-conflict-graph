from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


Direction = Literal["positive", "negative", "mixed"]
CorpusSyncStatus = Literal["queued", "complete", "failed"]
SectionCanonicalType = Literal[
    "abstract",
    "introduction",
    "method",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "other",
]
ClaimType = Literal[
    "performance_improvement",
    "limitation",
    "comparison",
    "setting_effect",
    "mechanism",
]
AnomalyType = Literal[
    "benchmark_inconsistency",
    "setting_mismatch",
    "bridge_opportunity",
    "metric_mismatch",
    "evidence_gap",
    "community_disconnect",
    "impact_conflict",
]
InsightType = Literal[
    "unifying_theory",
    "transfer_opportunity",
    "community_disconnect",
]


class LooseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Paper(LooseModel):
    paper_id: str
    title: str
    year: int
    venue: str
    url: Optional[str] = None
    doi: Optional[str] = None
    cited_by_count: int = 0
    referenced_works: list[str] = Field(default_factory=list)
    counts_by_year: list[dict] = Field(default_factory=list)
    selection_score: float = 0.0
    selection_reason: Optional[str] = None
    retrieval_channel: Optional[str] = None
    abstract: str = ""
    text: str = ""
    paper_role: Optional[str] = None
    paper_role_score: float = 0.0
    paper_role_signals: list[str] = Field(default_factory=list)
    openalex_id: Optional[str] = None
    arxiv_id_full: Optional[str] = None
    arxiv_id_base: Optional[str] = None
    corpus_tag: Optional[str] = None
    seed_reason: Optional[str] = None
    academic_impact: float = 0.0
    recency_score: float = 0.0
    reasoning_relevance: float = 0.0
    role_weight: float = 0.0
    priority_score: float = 0.0
    sync_status: Optional[CorpusSyncStatus] = None
    sync_attempt_count: int = 0
    first_seen_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    last_attempted_at: Optional[str] = None
    completed_at: Optional[str] = None
    # Optional list of claim dicts to make rule extraction deterministic.
    structured_hint: Optional[list[dict]] = None


class PaperReadCandidate(LooseModel):
    sentence: str
    evidence_span: str
    evidence_source_field: Optional[Literal["text", "abstract"]] = None
    evidence_sentence_index: Optional[int] = None
    evidence_char_start: Optional[int] = None
    evidence_char_end: Optional[int] = None
    subject_raw: Optional[str] = None
    predicate: Optional[str] = None
    object_raw: Optional[str] = None
    dataset_raw: Optional[str] = None
    metric_raw: Optional[str] = None
    baseline_raw: Optional[str] = None
    direction: Optional[Direction] = None
    magnitude_text: Optional[str] = None
    conditions: list[str] = Field(default_factory=list)
    scope: list[str] = Field(default_factory=list)
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    section_kind: Optional[str] = None
    candidate_score: float = 0.0
    selection_reason: Optional[str] = None


class PaperArtifactStatus(LooseModel):
    paper_id: str
    source_fetched: bool = False
    html_fetched: bool = False
    pdf_fetched: bool = False
    canonical_source: Optional[Literal["tex", "html", "pdf"]] = None
    parse_status: Literal["complete", "partial", "failed", "missing"] = "missing"
    parser_version: str = "corpus-v1"
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    text_length: int = 0
    section_count: int = 0
    sentence_count: int = 0
    source_url: Optional[str] = None
    html_url: Optional[str] = None
    pdf_url: Optional[str] = None


class PaperSection(LooseModel):
    section_id: str
    title: str
    kind: Literal["abstract", "section", "subsection", "appendix", "other"]
    level: int = 0
    parent_id: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    source: Literal["tex", "html", "pdf"] = "tex"
    canonical_type: SectionCanonicalType = "other"
    canonical_confidence: float = 0.0
    canonical_matched_by: str = "fallback_other"


class PaperSentence(LooseModel):
    sentence_id: str
    section_id: Optional[str] = None
    text: str
    char_start: int = 0
    char_end: int = 0
    sentence_index: int = 0
    section_sentence_index: int = 0


class Setting(LooseModel):
    retriever: Optional[str] = None
    top_k: Optional[str] = None
    context_length: Optional[str] = None
    task_type: Optional[str] = None


class Claim(LooseModel):
    claim_id: str
    paper_id: str
    claim_text: str
    claim_type: ClaimType = "performance_improvement"
    method: Optional[str] = None
    model: Optional[str] = None
    task: Optional[str] = None
    dataset: Optional[str] = None
    metric: Optional[str] = None
    baseline: Optional[str] = None
    result: Optional[str] = None
    direction: Direction = "positive"
    setting: Setting = Field(default_factory=Setting)
    evidence_span: str = ""
    subject_raw: Optional[str] = None
    subject_canonical: Optional[str] = None
    predicate: Optional[str] = None
    object_raw: Optional[str] = None
    object_canonical: Optional[str] = None
    dataset_raw: Optional[str] = None
    dataset_canonical: Optional[str] = None
    metric_raw: Optional[str] = None
    metric_canonical: Optional[str] = None
    baseline_raw: Optional[str] = None
    baseline_canonical: Optional[str] = None
    magnitude_text: Optional[str] = None
    magnitude_value: Optional[float] = None
    magnitude_unit: Optional[str] = None
    conditions: list[str] = Field(default_factory=list)
    scope: list[str] = Field(default_factory=list)
    evidence_source_field: Optional[Literal["text", "abstract"]] = None
    evidence_sentence_index: Optional[int] = None
    evidence_char_start: Optional[int] = None
    evidence_char_end: Optional[int] = None
    # Canonicalized cluster keys (filled by LLM extractor; optional for rule-based).
    canonical_method: Optional[str] = None
    canonical_task: Optional[str] = None
    # Lightweight semantic fields used for topology/insight detection.
    domain: Optional[str] = None
    data_modality: Optional[str] = None
    mechanism: Optional[str] = None
    failure_mode: Optional[str] = None
    evaluation_protocol: Optional[str] = None
    assumption: Optional[str] = None
    risk_type: Optional[str] = None
    temporal_property: Optional[str] = None


class Anomaly(LooseModel):
    anomaly_id: str
    type: AnomalyType
    central_question: str
    claim_ids: list[str] = Field(default_factory=list)
    positive_claims: list[str] = Field(default_factory=list)
    negative_claims: list[str] = Field(default_factory=list)
    shared_entities: dict[str, str] = Field(default_factory=dict)
    varying_settings: list[str] = Field(default_factory=list)
    local_graph_nodes: list[str] = Field(default_factory=list)
    local_graph_edges: list[dict] = Field(default_factory=list)
    evidence_impact: float = 0.0
    recent_activity: float = 0.0
    impact_balance: float = 0.0
    citation_bridge_score: float = 0.0
    topology_score: float = 0.0


class Insight(LooseModel):
    insight_id: str
    type: InsightType
    title: str
    insight: str
    communities: list[str] = Field(default_factory=list)
    shared_concepts: list[str] = Field(default_factory=list)
    evidence_claims: list[str] = Field(default_factory=list)
    evidence_papers: list[str] = Field(default_factory=list)
    citation_gap: str = ""
    unifying_frame: str = ""
    transfer_suggestions: list[str] = Field(default_factory=list)
    impact_score: float = 0.0
    topology_score: float = 0.0
    confidence_score: float = 0.0
    quality_score: float = 0.0


class GraphBridge(LooseModel):
    from_: str = Field(alias="from", default="")
    to: str = ""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class Hypothesis(LooseModel):
    hypothesis_id: str
    anomaly_id: str
    hypothesis: str
    mechanism: str = ""
    explains_claims: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    minimal_test: str = ""
    scope_conditions: dict[str, str] = Field(default_factory=dict)
    evidence_gap: str = ""
    graph_bridge: GraphBridge = Field(default_factory=GraphBridge)


class ScoreBreakdown(LooseModel):
    hypothesis_id: str
    explain: float = 0.0
    grounding: float = 0.0
    testability: float = 0.0
    novelty: float = 0.0
    cost: float = 0.0
    discriminability: float = 0.0
    impact: float = 0.0
    topology: float = 0.0
    utility: float = 0.0
