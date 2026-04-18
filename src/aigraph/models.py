from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


Direction = Literal["positive", "negative", "mixed"]
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
    # Optional list of claim dicts to make rule extraction deterministic.
    structured_hint: Optional[list[dict]] = None


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
