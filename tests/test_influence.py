"""Unit tests for src/aigraph/influence.py — Phase 1 (4-dim, non-LLM)."""

from __future__ import annotations

from aigraph.influence import (
    InfluenceScore,
    WEIGHTS_PHASE1,
    _build_claim_to_cluster_index,
    community_reach,
    compute_evidence_quality,
    grounding_depth,
    novelty_score,
    predict_influence_batch,
    predict_influence_phase1,
    scope_overreach,
)
from aigraph.models import Claim, Hypothesis


def _make_claim(
    c_id: str = "c1",
    paper_id: str = "p1",
    *,
    text: str = "RAG improves QA",
    canonical_method: str | None = "rag",
    canonical_task: str | None = "factual-qa",
    dataset_canonical: str | None = None,
    domain: str | None = None,
    evidence_span: str = "RAG outperforms baseline by 5 points on NaturalQuestions in our experiments",
    magnitude_value: float | None = None,
) -> Claim:
    return Claim(
        claim_id=c_id,
        paper_id=paper_id,
        claim_text=text,
        method="RAG",
        canonical_method=canonical_method,
        task="factual QA",
        canonical_task=canonical_task,
        dataset_canonical=dataset_canonical,
        domain=domain,
        direction="positive",
        evidence_span=evidence_span,
        magnitude_value=magnitude_value,
    )


def _make_hypothesis(
    h_id: str = "h001",
    *,
    explains_claims: list[str] | None = None,
    scope: dict[str, str] | None = None,
    novelty: dict | None = None,
) -> Hypothesis:
    h = Hypothesis(
        hypothesis_id=h_id,
        anomaly_id="a001",
        hypothesis="Test hypothesis",
        mechanism="Test mechanism",
        explains_claims=explains_claims or [],
        predictions=["Pred 1"],
        minimal_test="Run X on Y",
        scope_conditions=scope or {},
    )
    if novelty is not None:
        # Hypothesis is LooseModel extra="ignore" so this attaches in
        # memory but won't survive model_dump_json. novelty_score reads
        # it back via getattr — that's all we need for the test path.
        object.__setattr__(h, "novelty_check", novelty)
    return h


# -- community_reach ---------------------------------------------------------


def test_community_reach_partial_coverage():
    """Two claims spanning two of three communities -> reach = 2/3."""
    h = _make_hypothesis(explains_claims=["c1", "c2"])
    hierarchy = {
        "communities": {"c000": {}, "c001": {}, "c002": {}},
        "cluster_to_community": {"rag-qa": "c000", "cot-math": "c001"},
        "clusters": {
            "rag-qa": {"claim_ids": ["c1"]},
            "cot-math": {"claim_ids": ["c2"]},
            "extra-cluster": {"claim_ids": ["other"]},
        },
    }
    reach, n = community_reach(h, hierarchy)
    assert reach == 2 / 3
    assert n == 2


def test_community_reach_empty_hypothesis_returns_zero():
    """No explains_claims -> reach = 0, no division by zero."""
    h = _make_hypothesis(explains_claims=[])
    hierarchy = {"communities": {"c000": {}}, "cluster_to_community": {}, "clusters": {}}
    reach, n = community_reach(h, hierarchy)
    assert reach == 0.0
    assert n == 0


# -- novelty_score -----------------------------------------------------------


def test_novelty_score_uses_check_when_present():
    """novelty_check.is_novel=True with N similar papers -> 1/(1+N)."""
    h = _make_hypothesis(novelty={"is_novel": True, "similar_papers": [{"x": 1}]})
    score, is_n, n_sim = novelty_score(h)
    assert is_n is True
    assert n_sim == 1
    assert score == 0.5  # 1 / (1 + 1)


def test_novelty_score_falls_back_to_neutral_when_check_missing():
    """No novelty_check on the hypothesis -> neutral 0.5, is_novel=None."""
    h = _make_hypothesis()
    score, is_n, n_sim = novelty_score(h)
    assert is_n is None
    assert score == 0.5
    assert n_sim == 0


# -- grounding_depth + compute_evidence_quality ------------------------------


def test_grounding_depth_strong_evidence():
    """Substantive evidence_span + canonicals + magnitude -> well above
    the 0.5 baseline."""
    c1 = _make_claim(
        evidence_span="RAG demonstrates significant improvements on multilingual MedQA over a strong baseline",
        canonical_method="rag",
        canonical_task="medical-qa",
        dataset_canonical="medqa",
        magnitude_value=5.2,
    )
    h = _make_hypothesis(explains_claims=["c1"])
    score = grounding_depth(h, {"c1": c1})
    # 0.5 base + 0.20 (long span) + 0.10 (canonical) + 0.05 (dataset) + 0.10 (magnitude) = 0.95
    assert score == 0.95


def test_grounding_depth_weak_evidence_and_no_canonicals():
    """Short evidence, no canonicals, no dataset, no magnitude -> baseline only."""
    c1 = _make_claim(
        evidence_span="x",  # < 50 chars
        canonical_method=None,
        canonical_task=None,
        dataset_canonical=None,
        magnitude_value=None,
    )
    h = _make_hypothesis(explains_claims=["c1"])
    score = grounding_depth(h, {"c1": c1})
    assert score == 0.5


# -- scope_overreach ---------------------------------------------------------


def test_scope_overreach_match():
    """Every (key, value) in scope_conditions appears in observed scope ->
    overreach = 0."""
    c1 = _make_claim(canonical_method="rag", canonical_task="qa")
    h = _make_hypothesis(
        explains_claims=["c1"],
        scope={"method": "rag", "task": "qa"},
    )
    overreach = scope_overreach(h, {"c1": c1})
    assert overreach == 0.0


def test_scope_overreach_total_mismatch():
    """Scope claims a domain that isn't present anywhere in evidence ->
    overreach = 1.0."""
    c1 = _make_claim(canonical_method="rag", canonical_task="qa")
    h = _make_hypothesis(
        explains_claims=["c1"],
        scope={"domain": "neuroscience"},  # not in c1
    )
    overreach = scope_overreach(h, {"c1": c1})
    assert overreach == 1.0


# -- predict_influence_phase1 (full path) ------------------------------------


def test_predict_influence_phase1_combines_correctly():
    """Build a hypothesis where every dimension is high — verify total
    is in [0, 1] and individual fields populate as expected."""
    c1 = _make_claim(
        canonical_method="rag",
        canonical_task="qa",
        dataset_canonical="natural_questions",
        evidence_span="RAG demonstrates significant improvements on factual QA benchmarks over the closed-book baseline",
        magnitude_value=5.2,
    )
    h = _make_hypothesis(
        explains_claims=["c1"],
        scope={"method": "rag"},
        novelty={"is_novel": True, "similar_papers": []},
    )
    hierarchy = {
        "communities": {"c000": {}},
        "cluster_to_community": {"rag-qa": "c000"},
        "clusters": {"rag-qa": {"claim_ids": ["c1"]}},
    }

    score = predict_influence_phase1(h, hierarchy, {"c1": c1})

    assert isinstance(score, InfluenceScore)
    assert 0.0 <= score.total <= 1.0
    assert score.is_novel is True
    assert score.n_communities_touched == 1
    assert score.community_reach == 1.0
    assert score.novelty == 1.0  # is_novel=True with 0 similar
    assert score.grounding_depth == 0.95
    assert score.scope_overreach_risk == 0.0


def test_predict_influence_batch_uses_shared_index():
    """Batch path returns one score per input, in input order, using a
    single claim_to_cluster index. Smoke-check correctness; the
    optimization itself is internal."""
    c1 = _make_claim("c1", canonical_method="rag", canonical_task="qa")
    c2 = _make_claim("c2", canonical_method="cot", canonical_task="math")
    hierarchy = {
        "communities": {"c000": {}, "c001": {}},
        "cluster_to_community": {"rag-qa": "c000", "cot-math": "c001"},
        "clusters": {
            "rag-qa": {"claim_ids": ["c1"]},
            "cot-math": {"claim_ids": ["c2"]},
        },
    }
    h_a = _make_hypothesis("ha", explains_claims=["c1"])
    h_b = _make_hypothesis("hb", explains_claims=["c2"])

    scores = predict_influence_batch(
        [h_a, h_b], hierarchy, {"c1": c1, "c2": c2},
    )
    assert len(scores) == 2
    # Each hypothesis touches exactly 1 of 2 communities.
    assert scores[0].n_communities_touched == 1
    assert scores[1].n_communities_touched == 1
    assert scores[0].community_reach == 0.5
    assert scores[1].community_reach == 0.5


def test_claim_to_cluster_index_is_idempotent():
    """Building the index twice on the same hierarchy yields equal dicts."""
    hierarchy = {
        "clusters": {
            "rag-qa": {"claim_ids": ["c1", "c2"]},
            "cot-math": {"claim_ids": ["c3"]},
        },
    }
    a = _build_claim_to_cluster_index(hierarchy)
    b = _build_claim_to_cluster_index(hierarchy)
    assert a == b
    assert a == {"c1": "rag-qa", "c2": "rag-qa", "c3": "cot-math"}
