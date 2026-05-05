# Influence Prediction for Hypotheses — Design Doc

A concrete implementation design for predicting the **influence** of a
generated hypothesis, building on the theoretical framework in
[the theory paper](./theory-from-claims-to-selection.tex) (atomic claims
→ graph phenomena → graph-repair hypotheses → calibrated set selection).

> Status: design phase. Not yet implemented.
> Author: aigraph project. Last updated: v0.5.

---

## 1. Why this exists

The theoretical framework defines `Δ_G(h; G_z, z) = Φ(G_z^h) − Φ(G_z)`
— the predicted graph-level consequence of accepting hypothesis `h`.
This is what influence prediction *means* in the theory.

aigraph currently scores hypotheses on **static features** of the
hypothesis itself (`explains_claims` count, `predictions` count,
`grounding` quality, etc.), but does **not** compute the graph-level
consequence. Two specific gaps:

1. The hypothesis as graph repair `Δ_h` is implicit in
   `hypothesis.graph_bridge` and `hypothesis.predictions`, but never
   materialized as an actual `nx.MultiDiGraph` operation.
2. The selection objective uses MMR diversity on hypothesis text
   similarity, not on **predicted differential test outcomes**.

Filling these turns aigraph from "static hypothesis ranking" into
"forecasted research-direction influence ranking". This is a
qualitative change in the kind of decisions the system supports.

The user-facing benefit: instead of a black-box utility score, each
hypothesis gets a decomposed forecast:

```
Hypothesis a014: Multi-Tool Progressive Forking Search
  Predicted influence: 0.78  (top 12%)
  ├── Structural: closes 4/6 contradicts in cluster (a014, a017, a022, a035)
  ├── Reach: spans 3 communities (search-based reasoning, RL, planning)
  ├── Test efficiency: distinguishes from 7 other hypotheses
  ├── Novelty: ✅ no closely-similar arxiv paper
  └── Grounding: 0.92 (4 claims + 3 OQs)

  Risk:
  ├── Evidence weakness: 0.15 (low; all peer-reviewed)
  └── Scope overreach: 0.40 (claims 'general method' but math-only grounding)
```

This is the deliverable of this design.

---

## 2. The 5 influence dimensions

We decompose hypothesis influence into 5 independent dimensions plus a
risk modifier:

```
Influence(h) = Σ_k α_k · I_k(h) − β · Risk(h)

I_1 — Structural Impact      (graph-repair consequence)
I_2 — Community Reach         (cross-cluster span)
I_3 — Test Efficiency         (set-level discriminability)
I_4 — Novelty vs External     (arxiv-grounded novelty)
I_5 — Grounding Depth         (claim/OQ provenance quality)
```

Each dimension is independently computable from current corpus + graph
+ small new modules. None require labeled training data. None require
external services beyond what aigraph already uses (LLM, arxiv, S2).

---

### I₁ — Structural Impact

**Definition.** The local graph phenomenon strength reduction induced
by applying the hypothesis as a graph repair.

For an anomaly `z` of type Conflict / BridgeGap / ContextMismatch, we
build the augmented graph `G_z^h = G_z ⊕ Δ_h` and measure the
phenomenon-specific delta:

```
Conflict z:
   I₁_conflict(h) = (E_conf(G_z) − E_conf(G_z^h)) / E_conf(G_z)
   where E_conf is the conflict-edge weight sum

Bridge gap z:
   I₁_bridge(h) = (d(C_i, C_j; G_z) − d(C_i, C_j; G_z^h)) / d(C_i, C_j; G_z)
   where d is shortest-path between community centers

Context mismatch z:
   I₁_context(h) = Consistency(G_z | Ω_h) − Consistency(G_z)
   where Consistency measures direction agreement after scope filtering
```

**Implementation requires Gap 1: build G_z^h.** This is a new module
`graph_repair.py` that:

1. Parses `hypothesis.graph_bridge.from` / `.to` to identify the
   target nodes for the repair edge
2. Parses `hypothesis.predictions` for predicted method-task or
   method-metric relationships, adding tentative `supports` edges
3. Optionally adds a latent mediator node from
   `hypothesis.mechanism` (typed `Mechanism`, with `proposed_by_h`
   provenance)
4. Returns a deep-copy graph `G_z^h` with these additions

```python
# src/aigraph/graph_repair.py (new)
def apply_hypothesis_repair(g_z: nx.MultiDiGraph, h: Hypothesis) -> nx.MultiDiGraph:
    """Build G_z^h = G_z ⊕ Δ_h. Returns a new graph; original unchanged.

    Δ_h is inferred from h.graph_bridge, h.predictions, and h.mechanism:
      - graph_bridge.from -> graph_bridge.to: add 'explains' edge with
        polarity from h
      - For each prediction p in h.predictions, parse for (method, task,
        metric) tuples; if all three present in g_z, add a 'supports'
        edge from method->task with weight 0.5 (tentative)
      - h.mechanism: add a Mechanism node with id=f"Mechanism:{h.id}",
        connected to the explained_claims via 'mediates' edges
    """
```

The Δ_G calculation:

```python
def structural_impact(g_z, g_z_h, z) -> dict:
    """Returns the delta on phenomenon-specific metrics."""
    if z.type in {"benchmark_inconsistency", "impact_conflict",
                  "replication_conflict"}:
        e_before = sum(d.get("weight", 1) for u, v, d in g_z.edges(data=True)
                       if d.get("edge_type") == "contradicts")
        e_after = sum(d.get("weight", 1) for u, v, d in g_z_h.edges(data=True)
                      if d.get("edge_type") == "contradicts")
        return {"conflict_reduction": (e_before - e_after) / max(e_before, 1)}
    elif z.type == "bridge_opportunity":
        # cross-community distance on the citation subgraph
        ...
    elif z.type in {"setting_mismatch", "metric_mismatch"}:
        # scope-conditional consistency metric
        ...
```

**Computation cost.** O(|V| + |E|) for the repair, O(|E|) for delta
calculation. No LLM calls.

**Why it works.** A hypothesis that *claims* to resolve a conflict
must, when applied as a repair, actually decrease the local conflict
edge sum. If it doesn't, either the hypothesis isn't really
explanatory or the graph repair inference is incomplete. Either way,
it's diagnostic.

**Why it's hard to fake.** Generating a hypothesis with pretend graph
bridges that "look like" they resolve conflicts but actually don't
touch contradicts edges → I₁ = 0. The system measures what the LLM
*proposed* to bridge, not what the LLM *said* it would bridge.

---

### I₂ — Community Reach

**Definition.** The number of distinct corpus communities (or
clusters) that the hypothesis's grounding spans.

```
I₂(h) = |{community(c) : c ∈ inspired_by(h)}| / |communities|
```

A hypothesis that cites claims from a single cluster and 1 OQ is
locally important. One that cites claims from 4 different communities
is potentially a unifying mechanism — the kind of work that, if
verified, reshapes the field.

**Implementation.** Pure graph queries:

```python
def community_reach(h: Hypothesis, hierarchy: dict) -> float:
    """Compute fraction of communities touched by h.inspired_by."""
    inspired_ids = set(h.explains_claims) | set(getattr(h, "inspired_by", []))
    cluster_to_community = hierarchy.get("cluster_to_community", {})
    
    communities_touched = set()
    for c_id in inspired_ids:
        cluster_key = lookup_cluster_for_claim(c_id, hierarchy)
        community_id = cluster_to_community.get(cluster_key)
        if community_id:
            communities_touched.add(community_id)
    
    total_communities = len(hierarchy.get("communities", {}))
    return len(communities_touched) / max(total_communities, 1)
```

**No new infrastructure required.** `hierarchy.json` is already
generated (commit 7942a11). Just dict lookups.

**Why it's a meaningful signal.** Reach correlates with how broad the
research implications are. A hypothesis grounded only in a tight
cluster might be useful but localized. A multi-community hypothesis
either truly unifies or is overstating its scope (caught by Risk
component).

---

### I₃ — Test Efficiency

**Definition.** The number of *other* hypotheses the same minimal_test
discriminates from `h`.

```
I₃(h) = |{h_j ∈ H_z, j ≠ h : Diff(P_h(t_h), P_{h_j}(t_h)) > θ}| / |H_z|
```

If running the experiment described by `h.minimal_test` gives a result
that distinguishes `h` from 7 other competing hypotheses, that
experiment is much more valuable than one that distinguishes `h` from
1 other hypothesis.

This is the **information-theoretic** core of good experiment design:
a test that discriminates many hypotheses simultaneously is high-leverage.

**Implementation.** Pairwise LLM-judge calls:

```python
def test_discrimination(h_i: Hypothesis, h_j: Hypothesis,
                        client=None, model=None) -> bool:
    """Ask LLM whether running h_i.minimal_test would differentiate h_i
    and h_j based on their respective predictions."""
    prompt = f"""You judge whether an experiment differentiates two hypotheses.

Experiment: {h_i.minimal_test}

Hypothesis A: {h_i.hypothesis}
A's predictions: {h_i.predictions}

Hypothesis B: {h_j.hypothesis}
B's predictions: {h_j.predictions}

If we run the experiment, would A and B predict observably different outcomes?
Respond with strict JSON: {{"differentiates": true|false, "rationale": "..."}}
"""
    response = call_llm_text(client, model=model, system="...", user=prompt)
    return parse_json(response).get("differentiates", False)


def test_efficiency(h: Hypothesis, h_pool: list[Hypothesis], **kwargs) -> float:
    differentiated = sum(
        test_discrimination(h, h_j, **kwargs)
        for h_j in h_pool if h_j is not h
    )
    return differentiated / max(len(h_pool) - 1, 1)
```

**Cost.** O(N) LLM calls per hypothesis where N = pool size. For
|H_z| = 50, that's 50 calls per hypothesis × 50 hypotheses = 2500
calls. At minimal-effort pricing ~$0.005/call → ~$12.

**Caching.** Pair-symmetry (`Diff(h_i, h_j) = Diff(h_j, h_i)`) cuts
calls in half. Cache lookups by hash of `(h_i.id, h_j.id)`. After
first run for a hypothesis pool, subsequent calls free.

**Why it works.** This is the only dimension that really uses
hypothesis interaction. Without it, the system can't tell that 5
hypotheses all proposing the same minimal_test with the same
predictions are *not* 5 distinct candidates.

---

### I₄ — Novelty vs External

**Definition.** Boolean: is the hypothesis substantively new compared
to recent arxiv literature?

```
I₄(h) = 1[is_novel] · (1 / (1 + N_similar))
```

The novelty_check pipeline (commit 49d007a, just shipped) already does
this:

```python
def novelty_score(h: Hypothesis) -> float:
    nc = getattr(h, "novelty_check", None)
    if nc is None:
        return 0.5  # unknown
    if not nc.get("is_novel", False):
        return 0.0
    n_similar = len(nc.get("similar_papers", []))
    return 1.0 / (1 + n_similar)
```

**Implementation.** Just consume what `novelty_check` already attaches.

**Why it's signal.** A hypothesis that resolves a conflict beautifully
but already has 3 published implementations is much less useful as a
research direction than one that's structurally identical but
unattested.

**Limitation.** novelty_check uses arxiv API which has its own gaps.
A hypothesis published only in a conference proceedings or workshop
might evade detection. We accept this — better than no novelty signal.

---

### I₅ — Grounding Depth

**Definition.** Average evidence quality across explained claims.

```
I₅(h) = (1 / |C_h^+|) · Σ_{c ∈ C_h^+} Q(c)
```

where Q(c) = f(confidence, source_independence, evidence_type,
provenance_diversity) — exactly your theoretical Equation §5.4.

**Implementation.** Already partially in scoring.py (W_GROUNDING).
Reformulate as positive contribution rather than penalty:

```python
def grounding_depth(h: Hypothesis, claims_by_id: dict) -> float:
    explained = h.explains_claims or []
    if not explained:
        return 0.0
    qualities = []
    for c_id in explained:
        c = claims_by_id.get(c_id)
        if c is None:
            continue
        q = compute_evidence_quality(c)  # uses claim_schema utilities
        qualities.append(q)
    return sum(qualities) / max(len(qualities), 1)


def compute_evidence_quality(c: Claim) -> float:
    """Theoretical Q(c) per §5.4."""
    score = 0.5
    if c.evidence_span and len(c.evidence_span) > 50:
        score += 0.2  # has substantive evidence
    if c.confidence and c.confidence > 0.7:
        score += 0.15
    if c.canonical_method or c.canonical_task:
        score += 0.1  # has structured concepts
    if c.dataset_canonical:
        score += 0.05
    return min(1.0, score)
```

**Why split it from the existing W_GROUNDING.** The existing
W_GROUNDING uses claim count + anomaly overlap. This dimension
specifically measures **claim-level evidence quality** (how strong is
the evidence behind each cited claim), not just how many. Two
hypotheses with the same number of cited claims but different
evidence strength get different I₅.

---

### Risk components

These are subtractive — they reduce influence:

#### Risk.EvidenceWeakness

```
Risk_evidence(h) = 1 - I₅(h)
```

Just the inverse of grounding depth. A hypothesis grounded in weak
evidence has lower predicted impact even if all other dimensions
score high. We surface this explicitly so user can see the
"high-influence-but-low-evidence" outliers (which are exactly the
risky bets).

#### Risk.ScopeOverreach

```
Risk_scope(h) = 1 - intersection(h.scope_conditions, observed_scope_in_explains_claims) / total
```

If `h.scope_conditions` claims "applies to general LLM reasoning
tasks" but `h.explains_claims` only includes math reasoning papers,
that's overreach. Compute by:

1. Extract scope dimensions from each claim's `setting` (retriever,
   top_k, context_length, task_type) and `domain`/`task`/`dataset`
2. Compare with `h.scope_conditions`
3. Score as Jaccard distance from observed-to-claimed scope

```python
def scope_overreach(h: Hypothesis, claims_by_id: dict) -> float:
    """Fraction of claimed scope dimensions NOT supported by explains_claims."""
    claimed_scope = set((h.scope_conditions or {}).items())
    if not claimed_scope:
        return 0.0  # no overreach if no explicit scope claim
    
    observed_scope = set()
    for c_id in (h.explains_claims or []):
        c = claims_by_id.get(c_id)
        if c is None:
            continue
        if c.canonical_task:
            observed_scope.add(("task", c.canonical_task))
        if c.canonical_method:
            observed_scope.add(("method", c.canonical_method))
        if c.dataset_canonical:
            observed_scope.add(("dataset", c.dataset_canonical))
        if c.domain:
            observed_scope.add(("domain", c.domain))
    
    if not observed_scope:
        return 1.0  # no observed scope = total overreach
    
    intersection = claimed_scope & observed_scope
    return 1 - len(intersection) / len(claimed_scope)
```

This is a **soft** measure — Jaccard match on canonical strings.
Not perfect but identifies obvious overreach (claim about "general
reasoning" with grounding only in arithmetic).

---

## 3. Combined formula

The final influence formula:

```
Influence(h) = α₁ · I₁(h)              # Structural Impact
              + α₂ · I₂(h)              # Community Reach
              + α₃ · I₃(h)              # Test Efficiency
              + α₄ · I₄(h)              # Novelty vs External
              + α₅ · I₅(h)              # Grounding Depth
              − β₁ · Risk_evidence(h)
              − β₂ · Risk_scope(h)
```

Initial weights (from rough proportionality argument, to be tuned):

```
α₁ = 0.30 (structural impact is the most "earned" signal)
α₂ = 0.15
α₃ = 0.20 (high — measures the actual experimental value)
α₄ = 0.20
α₅ = 0.15
β₁ = 0.10
β₂ = 0.10
```

Sum of α's = 1.0; β's are pure penalty.

Output range: roughly [-0.2, 1.0] with most hypotheses in [0.1, 0.7].

---

## 4. Module API

New module `src/aigraph/influence.py`:

```python
"""Predict the multi-dimensional influence of a generated hypothesis.

Decomposes influence into 5 dimensions + 2 risk components, each
independently computable from the local graph + corpus context.
"""

from typing import NamedTuple, Optional
import networkx as nx

from .models import Hypothesis, Anomaly, Claim


class InfluenceScore(NamedTuple):
    structural_impact: float        # I₁: graph-repair consequence
    community_reach: float          # I₂: cross-cluster span
    test_efficiency: float          # I₃: pairwise discrimination
    novelty: float                  # I₄: arxiv-external novelty
    grounding_depth: float          # I₅: evidence quality average
    risk_evidence_weakness: float   # subtractive
    risk_scope_overreach: float     # subtractive
    total: float                    # weighted sum


WEIGHTS = {
    "structural": 0.30,
    "reach": 0.15,
    "efficiency": 0.20,
    "novelty": 0.20,
    "grounding": 0.15,
    "evidence_weakness": 0.10,
    "scope_overreach": 0.10,
}


def predict_influence(
    h: Hypothesis,
    g_z: nx.MultiDiGraph,
    z: Anomaly,
    h_pool: list[Hypothesis],
    hierarchy: dict,
    claims_by_id: dict[str, Claim],
    *,
    client=None,
    model: Optional[str] = None,
    weights: dict = WEIGHTS,
) -> InfluenceScore:
    """Compute the 5-dimensional + 2-risk influence score for h.
    
    Args:
        h: target hypothesis
        g_z: local graph centered on z
        z: the anomaly h addresses
        h_pool: peer hypotheses (for test efficiency)
        hierarchy: precomputed hierarchy.json
        claims_by_id: claim lookup
        client/model: LLM for test_efficiency
        weights: combination weights (defaults to WEIGHTS)
    """
    g_z_h = apply_hypothesis_repair(g_z, h)
    
    structural = _structural_impact(g_z, g_z_h, z)
    reach = _community_reach(h, hierarchy)
    efficiency = _test_efficiency(h, h_pool, client=client, model=model)
    novelty = _novelty_score(h)
    grounding = _grounding_depth(h, claims_by_id)
    
    risk_e = 1 - grounding
    risk_s = _scope_overreach(h, claims_by_id)
    
    total = (
        weights["structural"] * structural +
        weights["reach"] * reach +
        weights["efficiency"] * efficiency +
        weights["novelty"] * novelty +
        weights["grounding"] * grounding -
        weights["evidence_weakness"] * risk_e -
        weights["scope_overreach"] * risk_s
    )
    
    return InfluenceScore(
        structural_impact=structural,
        community_reach=reach,
        test_efficiency=efficiency,
        novelty=novelty,
        grounding_depth=grounding,
        risk_evidence_weakness=risk_e,
        risk_scope_overreach=risk_s,
        total=total,
    )


def predict_influence_batch(
    hypotheses: list[Hypothesis],
    g_z: nx.MultiDiGraph,
    z: Anomaly,
    hierarchy: dict,
    claims_by_id: dict[str, Claim],
    **kwargs,
) -> list[InfluenceScore]:
    """Cache-efficient batch prediction. Test efficiency benefits
    from pair-caching across the batch."""
    ...
```

CLI:

```python
@app.command("predict-influence")
def predict_influence_cmd(
    hypotheses: Path,
    graph: Path,
    anomalies: Path,
    claims: Path,
    hierarchy: Path,
    output: Path,
    model: Optional[str] = None,
):
    """For each hypothesis, attach an influence_score field."""
```

---

## 5. Validation methodology

How do we know the influence prediction is meaningful?

### Internal consistency checks

These don't validate the model is "right" but ensure it's well-formed:

1. **Monotonicity**: If `h₁` strictly dominates `h₂` on all 5
   dimensions and is no worse on risk, then `Influence(h₁) >
   Influence(h₂)`. Verify with sanity tests.

2. **Bounds**: Each `I_k ∈ [0, 1]`, each Risk ∈ [0, 1], total ∈
   [-0.2, 1.0]. Test fixture verifies.

3. **Determinism on repeated runs**: All non-LLM components must
   produce identical scores given identical inputs. Test with seed
   control.

### Backtest validation

The real test: does predicted influence correlate with actual
research impact?

**Procedure**:

1. **Build training corpus**: papers from 2020-2022. Generate
   hypotheses on this corpus.
2. **Build evaluation corpus**: papers from 2023-2025 (subsequent
   3 years).
3. **Predicted influence**: Apply `predict_influence` to all
   2020-2022 hypotheses.
4. **Actual influence proxy**: For each predicted hypothesis,
   search the 2023-2025 corpus for papers that:
   - Cite the original `inspired_by` papers
   - Use the proposed method (extract method names from
     `h.proposed_method`)
   - Reference the resolved anomaly type
5. **Score actual influence**: Count of post-2022 papers matching
   the hypothesis × their average citation count.

```python
def actual_influence(h: Hypothesis, future_corpus: list[Paper]) -> float:
    related_papers = find_papers_related_to(h, future_corpus)
    if not related_papers:
        return 0.0
    return sum(p.cited_by_count for p in related_papers) / len(related_papers)
```

6. **Compare**: Pearson / Spearman correlation between
   `predicted_influence` and `actual_influence` across all
   hypotheses.

**Acceptance criterion**: ρ > 0.4 (moderate positive). Below 0.2
suggests the model is not capturing real influence; tune α weights.

### Ablation study

Test which dimensions actually contribute:

1. Measure baseline: full Influence formula
2. Drop each dimension `I_k`: re-measure correlation
3. Drop pattern reveals which dimensions are most predictive

If, e.g., dropping `I₁` (Structural) and `I₃` (Test Efficiency)
preserves >90% of the correlation, those two dimensions are the
"real" predictive signal. The others might still be useful for
interpretability but aren't load-bearing.

---

## 6. Rollout plan

### Phase 1 — Cheap dimensions (1 week)

Implement only:
- `I₂` Community Reach
- `I₄` Novelty (already shipped)
- `I₅` Grounding Depth
- `Risk_scope` Scope Overreach

These need **zero LLM calls**, no Gap 1 implementation. ~200 lines
of code total. 8-10 tests.

Output: A 4-dimensional simplified influence score, ranking
hypotheses purely by graph + claim quality features. Already
useful — surfaces hypotheses with broad community reach AND deep
grounding AND novelty AND no overreach.

### Phase 2 — Add Structural Impact (1-2 weeks)

Implement:
- `graph_repair.py:apply_hypothesis_repair`
- `I₁` Structural Impact

This is the centerpiece. ~400 lines + tests.

Output: Full structural-aware influence. Hypotheses that *actually*
close graph contradictions get higher scores than hypotheses that
just *describe* themselves as conflict-resolving.

### Phase 3 — Add Test Efficiency (1 week)

Implement:
- `I₃` Test Efficiency with LLM-judge pairwise + caching

~150 lines + tests + cost estimation.

Output: Full 5-dimensional influence. Set selection becomes
information-theoretic.

### Phase 4 — Backtest validation (1-2 weeks)

Run validation methodology end-to-end on the existing 4895-paper
corpus. Tune weights based on results.

Output: Empirical evidence that influence prediction works. Or
honest documentation of what it doesn't capture.

### Phase 5 — Integrate into selection (1 week)

Replace or supplement `scoring.py`'s utility with influence-based
ranking. Update `report.py` to display the 5-dimensional
breakdown per hypothesis.

Output: User-visible influence scores with breakdown.

---

## 7. Open questions and risks

### Will the structural repair inference be too brittle?

The `apply_hypothesis_repair` function infers `Δ_h` from
`hypothesis.graph_bridge` and `hypothesis.predictions` — both of
which are *natural language* fields produced by an LLM. Parsing them
into precise graph operations is heuristic.

**Mitigation**: Be conservative. If parse fails, treat as no repair
(`I₁ = 0`). Don't fabricate edges from ambiguous text. Clear
documentation: this is a *minimum* lower bound, not a complete
characterization.

### Are α weights stable across corpora?

The weights are tuned on a single corpus. Different research domains
(reasoning vs medical vs finance) might want different weights.

**Mitigation**: Make weights configurable (CLI flag). Initial
implementation uses corpus-specific defaults. Validation phase
includes per-corpus tuning.

### Backtest may suffer from selection bias

If 2020-2022 hypotheses we generate happen to cluster around a
specific topic, the 2023-2025 follow-on papers we find might be
biased by what was popular regardless of our predictions.

**Mitigation**: Stratified sampling across anomaly types and
clusters when generating the test set. Report per-anomaly-type
correlations as well as overall.

### Test efficiency cost scaling

For a hypothesis pool of size N, computing `I₃` for all hypotheses
requires N(N-1)/2 LLM calls. At N=480 (full multi-grain run), that's
~115k calls.

**Mitigation**:
- Cache aggressively (first run = full cost; subsequent re-runs of
  the same pool = free)
- Allow pool subsampling: compare each hypothesis only to top-K
  most-similar (by embedding) peers, not all N
- Allow disabled mode: `predict_influence(... compute_test_efficiency=False)`
  for cost-sensitive deployments

### What about EvidenceWeakness as a phenomenon type?

Theory section §5.4 lists EvidenceWeakness as the 4th phenomenon
type, but aigraph's anomaly detector doesn't have an
`evidence_weakness` type. This design includes it as a Risk
component, not as a phenomenon trigger.

**Open question**: Should aigraph add `evidence_weakness` as an 8th
anomaly type that triggers evidence-artifact hypotheses? Or is
folding it into the Risk modifier sufficient?

The current design says: Risk-level modifier is sufficient.
Phenomenon detection is for triggering hypothesis generation, but
"evidence weakness" doesn't propose new explanations — it modifies
how seriously to take existing ones.

### What about Δ_G measuring "future testability"?

Theory talks about "discriminability" partly capturing what tests
will distinguish hypotheses. But we don't measure how the entire
field's *future tests* (not just tests we can run today) would
distinguish hypotheses.

**Mitigation**: Out of scope. This is a fundamentally limited
information problem. We measure what we can — same-test
discrimination among existing hypotheses.

---

## 8. Connection to theory

Each dimension explicitly maps to a section of the theory paper:

| Dimension | Theory section | Theoretical formula |
|---|---|---|
| I₁ Structural | §6.1, §7 (Δ_G) | `Δ_G(h) = Φ(G_z^h) − Φ(G_z)` |
| I₂ Reach | §5 (graph topology) | `\|{community(c) : c ∈ inspired_by(h)}\|` |
| I₃ Efficiency | §7.2 (Discriminability) | `D(h_i, h_j) = max_t Diff(P_{h_i}(t), P_{h_j}(t)) / Cost(t)` |
| I₄ Novelty | §6 (testable, novel) | (extension; not in theory paper) |
| I₅ Grounding | §6 (Grounding) | `(1 / |C_h^+|) Σ Q(c)` |
| Risk_evidence | §5.4 (EvidenceWeakness) | `1 - Q(H)` |
| Risk_scope | §6 (Ω_h) | (Jaccard distance, derived) |

**What this design adds beyond the theory:**

- **Novelty vs External (I₄)**: external arxiv search not modeled
  in theory. Justified because the theory's "claims as
  evidence-assessable" framework can be extended to "external
  literature as a proxy for evidence" — papers we don't have are
  evidence we don't have.

- **Risk_scope (Risk overreach)**: explicit operationalization of
  scope grounding as a Jaccard mismatch.

**What this design omits from theory:**

- **Calibrated probability layer p̂_φ ∝ exp(Û/τ)**: future work,
  requires labeled training data we don't yet have.

- **Set-level Coverage objective F̂(S)**: this design predicts
  per-hypothesis influence; set selection still uses MMR. A future
  extension could replace MMR with influence-aware set selection
  using the union-of-coverage formula from theory §7.1.

---

## 9. What this delivers

After Phase 1-3 implementation, the system can answer questions
like:

> "Which 5 hypotheses, if pursued, would maximally restructure the
> field's understanding of multi-step reasoning?"

Currently aigraph answers a related but weaker question:

> "Which 5 hypotheses scored highest by an 8-dim utility?"

The difference is whether the system has any model of *what would
happen* if the hypothesis were true. The influence framework gives
that.

For grant writing, paper roadmaps, or strategic research direction,
this is the difference between "interesting ideas, ranked" and
"ideas with predicted impact, ranked".

---

## 10. Implementation ordering for v0.6

Suggest splitting into 3 PRs:

```
v0.6.0 — feat/influence-prediction-phase1
  + src/aigraph/influence.py (Phase 1: I₂, I₄, I₅, Risk_scope)
  + cli.py: predict-influence subcommand
  + tests/test_influence.py (10 tests)
  Net: +400 lines, 0 new dependencies

v0.6.1 — feat/influence-prediction-structural
  + src/aigraph/graph_repair.py (Gap 1 implementation)
  + influence.py: I₁ Structural added
  + tests: structural impact validation
  Net: +500 lines

v0.6.2 — feat/influence-prediction-efficiency
  + influence.py: I₃ Test Efficiency
  + caching layer for pairwise LLM judgments
  + cost cap mechanism
  Net: +250 lines + LLM cost
```

After all three: backtest validation + report integration.

Total scope: ~1200 lines of new code, ~25 tests, ~1 month engineering
calendar time, ~$50-100 in LLM costs for backtest validation.

---

## Appendix: Comparison with existing scoring

aigraph's current `scoring.py` has 8 dimensions with hand-set
weights. The proposed influence prediction has 5 dimensions + 2
risk modifiers. Some overlap, some new:

| Current scoring | Proposed influence | Difference |
|---|---|---|
| W_EXPLAIN | (not in influence) | Theory says coverage is set-level; here per-element |
| W_GROUNDING | I₅ Grounding Depth | Same |
| W_TESTABILITY | (subsumed by I₃ Test Efficiency) | Influence measures pairwise |
| W_NOVELTY | I₄ Novelty (external) | Internal vs external distinction |
| W_DISCRIMINABILITY | I₃ Test Efficiency | Same (discriminability is the right name) |
| W_IMPACT | I₁ Structural Impact | Renamed; theory-grounded |
| W_TOPOLOGY | (subsumed by I₂ Community Reach) | Reach is more interpretable |
| W_COST | Risk_scope Overreach | Different concept; cost was lexical |

Recommendation: when influence prediction is mature, replace
scoring.py's utility with influence-based selection. Until then,
keep both: scoring for production selection, influence for
analytical decomposition shown in reports.

---

*Last updated: v0.5. Schema-versioned aspects (anomaly types in
`models.AnomalyType`, edge types in `graph.py`) are canonical if
this document drifts.*
