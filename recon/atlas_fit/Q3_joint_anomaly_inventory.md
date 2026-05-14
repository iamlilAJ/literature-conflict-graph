# Q3 — Joint-graph anomaly inventory

Design-only deliverable. Inventory, not implementation. Each cell is
"the cheapest condition that would let this detector fire", not a
production spec.

## Variants of existing 8 detectors

| # | Existing detector | Joint-graph variant | Trigger condition | Why stronger |
|---|---|---|---|---|
| 1 | `benchmark_inconsistency` | `cross-paper-benchmark` | Same `(canonical_method, canonical_task)` cluster shows mixed-direction claims AND Atlas edge `improves(B,A)` OR `compares(B,A)` exists. | Atlas confirms the two papers are aware of each other; rules out "two papers in different sub-communities that happen to claim opposite things". The conflict is *acknowledged*, not artifactual. |
| 2 | `impact_conflict` | `cited-impact-conflict` | Same as (1) but require `paper_a.citation_count ≥ p75` AND `paper_b.citation_count ≥ p75` (both high-impact, using Atlas's `citation_count`). | Replaces aigraph's noisy `paper_impact_sum` heuristic with Atlas's clean Semantic-Scholar cite count — same cohort, same number, lower noise. |
| 3 | `setting_mismatch` | `versioned-setting-mismatch` | Detect aigraph's setting_mismatch on a `(method, task)` cluster, AND Atlas has an `extends(B,A)` edge where `paper_b.bottleneck_json` mentions the differing setting (e.g. `top_k`, `context_length`). | Atlas's `bottleneck_json` directly tells you which setting B identified as A's weakness. Removes the "we guessed the setting matters" leap. |
| 4 | `metric_mismatch` | `evaluator-disagree-on-metric` | aigraph's metric_mismatch on cluster, AND Atlas `compares(B,A)` edge exists where `paper_b.validation` field mentions the metric. | Atlas knows which paper validated against which metric. Aligns with aigraph's metric-direction flip. |
| 5 | `evidence_gap` | `bottleneck-without-resolution` | aigraph's evidence_gap (cluster with few claims around high-active entities), AND Atlas has ≥3 inbound edges to one of the cluster papers ALL with non-empty `bottleneck_json`. | Atlas tells you the cluster has been *observed by others* to have weaknesses, even if its own claims don't reflect them. Stronger evidence that the gap is real, not just under-extracted. |
| 6 | `community_disconnect` | `cross-method-community-disconnect` | aigraph's existing detector, BUT define communities using Atlas `method_id` co-occurrence in `paper_methods` rather than Louvain on citation graph. | Atlas's method-level community structure is denser and pre-computed; aigraph's Louvain on citation graph is sparse on small corpora (we routinely see 0 communities on < 1500-paper cohorts). |
| 7 | `bridge_opportunity` | `method-bridge-via-atlas-co-use` | aigraph's bridge_opportunity, AND ≥1 paper in BOTH clusters appears in Atlas with `paper_methods` linking to a SHARED method_id that's never been compared. | Atlas's `paper_methods` table is the *direct* signal for "two methods that have not been benchmarked together". aigraph's bridge_opportunity infers this from token-overlap; Atlas confirms it. |
| 8 | `replication_conflict` | `replication-vs-replaces` | aigraph's replication_conflict, AND Atlas has NO `replaces(B,A)` edge despite the conflict. | If aigraph sees a replication conflict (B claims A's result fails to reproduce) AND Atlas has no `replaces` edge from B to A, the field hasn't yet acknowledged the failure as a methodology supersession. Combines acknowledged-by-text (aigraph) and unacknowledged-by-structure (Atlas) signals. |

## Three new anomaly types that ONLY exist on the joint graph

### `unresolved_replacement`

**Trigger:** Atlas asserts `replaces(B,A)` (B's method replaces A's), but
aigraph's claim graph shows:
- A's `(method, task)` cluster still has more positive-stance claims than
  negative within window {year(B), year(B)+2}, OR
- B's `(method, task)` cluster has unresolved negative claims from any
  paper.

**Meaning:** The field structurally moved on (Atlas says so), but the
field's *textual* belief about A vs B is still in flux. Either A is
being kept around despite being replaced (vestigial usage), or B's
replacement is contested.

**Why joint:** Pure aigraph cannot anchor "the field structurally moved
on" (no `replaces` edges); pure Atlas cannot anchor "claim-level belief
about A vs B" (no stance signal at claim granularity).

**Expected scale:** rare. On the 540p run we have 4 replaces edges in
the intersection (see Q4 finding) → optimistically ≤ 5-10 candidates at
540p; would grow at 5000-paper scale.

### `bottleneck_open_q_alignment`

**Trigger:** For paper A, Atlas has an inbound `improves(B,A)` edge with
`bottleneck_json` describing A's weakness W_atlas. aigraph also extracts
an `open_question` from A's text whose semantic content overlaps with
W_atlas (Jaccard over content-tokens ≥ 0.4 or LLM similarity score
≥ 0.7).

**Meaning:** Consensus weakness — both A's own authors (limitation
section) and B's authors (external observation) flag the SAME weakness.
This is the gold-standard signal that the weakness is real.

**Why joint:** Independent agreement across first-party (aigraph) and
third-party (Atlas) views. Either alone is suggestive; both together is
strong evidence.

**Expected scale:** depends on Atlas-bottleneck quality (paper says
99.8% non-empty) and aigraph-open_question depth. Rough estimate at
540p scale: 30-80 candidates given 369 intersected edges × 8% match
rate.

### `silent_replacement`

**Trigger:** aigraph detects that B's claims `contradict` A's claims on
the same `(method, task)` cluster (i.e. the existing
`benchmark_inconsistency`/`impact_conflict` types), AND Atlas has NO
edge from B to A (no `replaces`, no `improves`, no `compares`).

**Meaning:** B has empirically shown A doesn't work, but the field
hasn't structurally acknowledged it. Could indicate:
- A's authors haven't seen B (unlikely if year(B) > year(A) + 1)
- The community considers B's evidence non-canonical
- B is a too-recent or too-niche paper to have caught on

This is the *weakness-detection* type — surfaces unaddressed conflicts.

**Why joint:** Requires aigraph's stance graph AND Atlas's "no edge"
signal. Atlas alone would not flag this — they only assert positive
edges. aigraph alone has no notion of structural acknowledgment.

**Expected scale:** depends on Atlas recall on `compares` edges. With
2.4M `compares` edges, Atlas's coverage is dense. So this anomaly type
fires when aigraph DOES detect a conflict and Atlas DOESN'T have the
edge — which on a well-covered cohort should be a small set
(e.g. 5-15 candidates on 540p).

## Implementation cost estimate (if J2 is adopted)

| Anomaly type | Add to `anomalies.py`? | New code | Atlas data needed |
|---|---|---|---|
| All 8 variants of existing | No — sit in a new `joint_anomalies.py` module; existing 8 stay frozen | ~250 LOC | `paper_evolution_edges`, `paper_methods` |
| `unresolved_replacement` | New module | ~80 LOC | `paper_evolution_edges.replaces` filter |
| `bottleneck_open_q_alignment` | New module | ~120 LOC, +open_question extractor (exists) | `paper_evolution_edges.bottleneck_json` |
| `silent_replacement` | New module | ~80 LOC | `paper_evolution_edges` (any-edge index) |

Total new code: ~530 LOC in a single new `joint_anomalies.py`. v0.7-frozen
modules untouched.

## Cross-cut: what aigraph's data needs from Atlas's parquet

For ALL the above to work, aigraph needs a stable mapping from its
`Paper.paper_id` (currently `arxiv:{base_id}v{N}`) to Atlas's
`paper_id` (currently `s2_{...}` or `jsonl_{slug}`-style identifiers
per the local mirror).

Cohort overlap on 540p × Atlas-papers = 445/474 papers via `arxiv_id`
(see Q-corpus-intersection finding). So the join key is `arxiv_id_base`,
not paper_id. A loose normalization is required (strip version,
lowercase).
