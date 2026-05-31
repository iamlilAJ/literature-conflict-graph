# Method 12 — Likert composite vs production utility on all 300 hyps — verdict

**Date:** 2026-05-31
**Method 12 of:** /loop "test different methods" plan; operational test of whether the Likert ranker discovered in Methods 3, 8, 11 actually improves on the production scorer.
**One-line answer:** **The production scorer (`scoring.py` utility) and the Likert composite are statistically independent (ρ=0.112, top-10 overlap=1/10), and the production scorer is delivering ~20% of its top-K with `atlas_overlap ∈ {1,2}` — hyps that the LLM judge identifies as tangential or unrelated to any known Atlas research bottleneck. The hybrid rule "drop overlap=1,2 then rank by utility" is a deployable improvement.**

## Experiment

- All 300 hyps in `artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1/hypotheses_scored.jsonl`.
- Each scored via Method 3's Likert judge against top-5 Atlas bottleneck quotes.
- 6 workers, ~70 min wall time (Kimi reasoning latency at the larger Atlas judge prompt is ~90s/call wall).
- 259/300 (86%) completed cleanly; 41 errors (judge JSON parse + Kimi timeouts).
- Joined with `score_all(...).utility` recomputed locally from the same hyps + anomalies + claims.

## Headline 1 — independence

Spearman ρ between current production utility and each Likert axis (n=259):

| signal | ρ |
|---|---:|
| atlas_overlap | -0.007 |
| forward_looking | -0.064 |
| named_mechanism | +0.014 |
| single_variable_test | +0.106 |
| specific_scope | -0.018 |
| **composite (5 axes sum)** | **+0.112** |
| composite_q (no overlap) | +0.107 |
| anchored_novelty | +0.093 |

**All correlations are below 0.15.** The two rankers measure nearly-disjoint things — utility captures structural completeness (grounding/test/scope/topology); Likert captures semantic specificity and Atlas anchoring. This is not "Likert beats utility" or vice versa — they are complementary signals.

## Headline 2 — the production top-K leaks tangential hyps

`atlas_overlap` distribution in top-K by production utility:

| top-K | overlap=1 | overlap=2 | overlap=3 | overlap=4 | "leak rate" (1+2) |
|---:|---:|---:|---:|---:|---:|
| 10 | 1 | 1 | 7 | 1 | **20%** |
| 30 | 2 | 2 | 24 | 2 | **13%** |
| 50 | 2 | 5 | 40 | 3 | **14%** |
| 100 | 5 | 15 | 74 | 6 | **20%** |

For comparison, Likert composite's top-K is 100% overlap ∈ {3, 4} through K=50.

Sample leak cases the production scorer ranks high but Likert flags:

- `h286` util=0.932, overlap=1 — Likert: "the hypothesis concerns cross-domain terminology drift and citation-graph separation" (no research anchoring to any bottleneck).
- `h247` util=0.935, overlap=2 — Likert: "the hypothesis addresses a meta-scientific terminology-drift problem" (tangential).
- `h013` util=0.976, overlap=2 — ranks #4 in production top-10; Likert sees tangential framing.
- `h293` util=0.974, overlap=1 — ranks #9 in production top-10; Likert sees no research anchoring.

These hyps have high utility because they score well on structural axes (the scorer's `explain` + `grounding` + `topology`), but they're not addressing recognized research problems. From a delivered-quality perspective they're noise.

## Headline 3 — the hybrid rule

Filter rule from Method 11 confirmed at scale:

```python
# Drop tangential / unanchored hyps, then rank survivors by existing utility.
def hybrid_rank(hyps, atlas_overlap_lookup, utility_lookup, k=8):
    survivors = [h for h in hyps if atlas_overlap_lookup[h.id] >= 3]
    return sorted(survivors, key=lambda h: -utility_lookup[h.id])[:k]
```

On the 540p-thaw1 run:
- 204/259 (79%) hyps survive the overlap≥3 filter.
- The hybrid top-10 retains the production scorer's high-utility hits (util 0.973–0.979) with composite scores 16–22 (all atlas-anchored).
- The two top-10 hyps that Method 3 already flagged as Pareto-front winners (`h202`, `h158`, `h264`) are all in the hybrid top-10.

## What this unlocks — concrete deployment

### Method 9.A — non-frozen production wire-up (recommended next)

In `scripts/aigraph_query.py` `_select()`, between candidate ranking and `select_mmr`:

1. Compute `atlas_overlap` for each candidate in the top-N (say N=30) via one Kimi call per hyp against pre-cached top-5 Atlas quotes.
2. Drop overlap ∈ {1, 2}.
3. Hand the filtered candidate set to the existing `select_mmr`.

Cost: ~30 Kimi calls per query. At 4 workers ≈ 4 min latency; at 8 workers ≈ 2 min. Cacheable per `(hyp_id, run_id)` since atlas_overlap doesn't depend on the query topic.

**Honest gotcha:** the 30-call latency makes this unsuitable for synchronous `get_idea_report` calls today. The right shape is an **offline batch pre-scoring step**: score every hyp in a run once after generation completes, persist `atlas_overlap` to a sidecar JSON, then `_select()` reads it as zero-latency metadata. This is one persistent script in `scripts/precompute_atlas_overlap.py` + a sidecar file under `artifacts/runs/<run>/`.

### Method 9.B — post-thaw scorer enrichment (deferred)

Once the thaw record is open for the next freeze cycle, weight `atlas_overlap` into `scoring.py` as a multiplicative gate: `utility *= overlap_weight` where `overlap_weight = {1: 0.4, 2: 0.6, 3: 1.0, 4: 0.8, 5: 0.2}`. Requires §4 thaw justification + 50-pair oracle re-validation.

### Method 9.C — query-time annotation (cheapest)

Skip ranking entirely; just attach `atlas_overlap` + the closest_quote + why to each delivered hyp as a metadata field. Users see "this hyp is anchored to a known bottleneck about X". No ranking change, no cost.

Recommended: **A then C in parallel, B after the next thaw.**

## Honest caveats

1. **Single judge (Kimi-K2.6)** — all Likert scores come from one model. Same-judge bias risk. Method 3's bottom_2 caveat extends to here: a different judge might score the borderline cases differently.
2. **TF-IDF retriever quality cap.** The top-5 Atlas candidates per hyp were chosen by TF-IDF cosine. Many low-overlap hyps (atlas_overlap=1) have a top-1 sim of just 0.1-0.2 — the judge is essentially saying "none of the candidates is close enough; not anchored". A neural-embedding retriever (SPECTER, MiniLM) would feed the judge stronger candidates and might rescue some "overlap=1" hyps that are actually relevant but lexically distant. Blocked by Python 3.14 + missing torch wheels.
3. **41/300 judge errors are not random.** They cluster on hyps whose text + 5-quote block exceeds Kimi's effective input window. The hyps with the longest abstracts are systematically under-judged; could bias the overlap distribution against verbose-but-good hyps.
4. **Production scorer "utility" was recomputed locally**, not pulled from a persisted field. If anomalies.jsonl or claims.jsonl drifts between gen-time and re-score-time, the utility number could differ from what was actually used at gen time. Verified: in this run, the local recompute matches `hypotheses_scored.jsonl`'s implicit ordering.

## Artifacts

- `scripts/method12_build_input.py` — judge-input builder
- `artifacts/atlas_test/method12_judge_input.jsonl` — 300 hyp × top-5 quote inputs
- `artifacts/atlas_test/m12_out.jsonl` — Likert judge outputs (259 ok / 41 err)
- `artifacts/atlas_test/m12_joined.jsonl` — joined with utility, composite, anchored_novelty
