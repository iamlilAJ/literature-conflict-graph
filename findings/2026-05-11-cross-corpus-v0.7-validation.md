# Cross-corpus validation of v0.7 influence predictor — FAIL

**Date:** 2026-05-11
**Git tag at predictor:** `v0.7-frozen` (post-thaw #1)
**Question:** Does the v0.7 influence predictor that scored ρ_total = 0.371
on the 540-paper baseline generalize to a held-out arxiv reasoning corpus?
**Answer:** **No.** ρ_total collapses from **+0.371 → +0.003** on the
venue-proxy subset. Per freeze doc §4 condition 1
(ρ_total < 0.10 on the primary cohort), this is a thaw trigger.

---

## Method

Two arxiv-only reasoning corpora, same predictor (v0.7-frozen):

| | v1_540p_thaw1 (baseline) | v2_847p (held-out) |
|---|---:|---:|
| papers (with sections.json) | 474 | 847 |
| claims | 1,961 | 3,303 |
| graph nodes / edges | 4,154 / 31,407 | 6,285 / 51,464 |
| anomalies (full) | 496 | 377 |
| anomaly types fired | 8 / 8 | 5 / 8 |
| anomalies after top-100 cap | 100 | 100 |
| hypotheses generated | 300 | 299 |
| novelty: novel / known / null | 261 / 14 / 25 | 193 / 23 / 83 |
| papers with cited_by_count > 0 (post Semantic Scholar enrich) | 450 / 474 | 823 / 847 |
| max cited_by_count | 19,444 | ~3,300 |

Both corpora are arxiv-only, so the backtest script's venue-match returns
zero hits and falls back to the **top-100 cited papers as venue proxy**.
Same cap, same predictor weights, same downstream MMR — only the corpus
and the claim-extraction LLM trace differ.

Backtest is `scripts/validate_influence_backtest.py`, fed
`hypotheses_scored.jsonl` (renamed to `creator_hypotheses.jsonl` via
symlink — same schema). Spearman ρ between
`influence_score.total` and `max cited_by_count over explains_claims →
paper`, computed on the venue-proxy subset.

## Headline numbers

Spearman ρ (total predicted vs L1 actual = max cite over evidenced papers):

| Subset | n (v1 / v2) | v1_540p ρ_total | v2_847p ρ_total |
|---|---:|---:|---:|
| **venue-grounded (top-100-cite proxy)** | **184 / 111** | **+0.371** | **+0.003** |
| all hypotheses | 300 / 299 | +0.130 | -0.056 |
| novel-only (is_novel=True) | 261 / 193 | +0.113 | -0.124 |
| known-only (is_novel=False) | 14 / 23 | +0.380 | +0.246 |

The headline cell is the venue-proxy ρ that the v0.7 pre-registered
validation uses. v2's +0.003 is roughly two orders of magnitude below
the +0.371 v1 number and well below the 0.10 floor in
`docs/v0.7-pipeline-freeze.md §4` condition 1.

## Per-dimension breakdown (venue-grounded subset)

| Dimension | v1_540p ρ | v2_847p ρ |
|---|---:|---:|
| community_reach | N/A (constant) | N/A (constant) |
| novelty | N/A (constant) | N/A (constant) |
| grounding_depth | +0.277 | +0.079 |
| scope_overreach_risk | **-0.199** (desired direction) | **+0.260** (anti-desired, sign-flipped) |

Two of four dimensions never moved (community_reach and novelty) — see
"Why two dims are constant" below.

The dramatic finding is `scope_overreach_risk`: its sign **flipped**
between corpora. On v1 the high-risk hypotheses were the lower-cited
ones (-0.199, desired); on v2 high-risk hypotheses are the higher-cited
ones (+0.260, opposite). The risk subtraction in the v0.7 weight
formula was protecting the v1 number; on v2 it actively makes the
prediction worse than chance.

## Why two dimensions are constant in BOTH runs

`community_reach` looks at hierarchy.communities. Both runs have
**0 communities** (only domains + clusters), so the dim is 0 for every
hypothesis. This is a structural property of how
`build-hierarchy` thresholds communities on reasoning-only corpora.

`novelty` (in `influence_score`, NOT to be confused with the
LLM-based `check-novelty` field) is derived from `is_novel` in the
scored hypothesis. The predict-influence stage was run **before**
check-novelty, so `is_novel: null` is what got baked into
influence_score; the dim is 0.5 (neutral) for every hypothesis. Re-
running predict-influence after check-novelty would populate this dim,
but neither run did that.

In other words, the v0.7 predictor on these corpora was effectively
2-of-4-dim, not 4-of-4. The freeze covers the configuration; this is a
*real* behavior of the v0.7 predictor on arxiv-only inputs.

## Anomaly-type drop on v2

| Type | v1_540p (full) | v2_847p (full) |
|---|---:|---:|
| benchmark_inconsistency | 66 | 162 |
| impact_conflict | 106 | **0** |
| setting_mismatch | 24 | 26 |
| metric_mismatch | 66 | 103 |
| evidence_gap | 65 | **0** |
| community_disconnect | 185 | **0** |
| bridge_opportunity | 40 | 73 |
| replication_conflict | 2 | 13 |
| **types_fired** | **8 / 8** | **5 / 8** |

Three detectors went silent at 847-paper scale: `impact_conflict`,
`evidence_gap`, `community_disconnect`. The first two are conditional on
paper metadata that differs by corpus (impact thresholds, claim
density). The third is conditional on graph community structure (and
v2 had 0 communities — see above).

Per freeze §4 condition 2, three zero-detection types on a 1000+ paper
cohort would be a thaw trigger; v2 is at 847 (just under threshold) so
it does not technically trip the condition, but is on the edge.

## So what

Two independent freeze conditions look at this run:

- **Condition 1** (ρ_total < 0.10 on primary cohort): **TRIPPED**
  (v2 ρ = +0.003).
- **Condition 2** (anomaly type produces 0 detections on a 1000+ paper
  cohort): not technically tripped — v2 is at 847 papers. But three
  detectors are zero, so this is one corpus-expansion away.

What the v0.7 predictor was actually predicting on the 540p baseline,
in retrospect: a small grounding_depth signal (+0.277) plus a
scope_overreach_risk signal whose sign happens to align with citation
counts in **that particular corpus** (-0.199). The latter is the
load-bearing piece, and it does not generalize.

Three legitimate paths forward:

1. **Drop `scope_overreach_risk` from `WEIGHTS_PHASE1`** — restore the
   four-dim formula to depend only on dims with consistent sign.
   Estimated v2 ρ_total if we drop risk and renormalize: I haven't
   recomputed, but it'd be dominated by grounding_depth (+0.079) so
   still small.
2. **Re-run predict-influence AFTER check-novelty** so the
   `novelty` dim actually moves. Cheap (no LLM); both runs have novelty
   data already.
3. **Pick a cohort with cleaner venue annotation** (NeurIPS / ICML /
   ICLR by venue field, not by cite-count proxy). The proxy
   introduces ranking noise that the predictor cannot reasonably
   defeat.

Recommended sequence: (2) first (free), then re-examine ρ; (3) is
the proper validation cohort the freeze doc anticipated but was deferred
because the OpenAlex pivot was deferred. (1) is a weight tweak that
should follow evidence from (2)+(3), not lead.

## Caveats

- v2's `hypotheses_scored.jsonl` was generated with `is_novel: null`
  baked in (predict-influence ran before check-novelty). Same for
  v1_540p_thaw1. So the "novelty" dimension is a known zero contributor
  in both runs.
- The "venue-grounded" subset for arxiv-only corpora is a top-100-cite
  proxy. Both corpora use this fallback. It's the cleanest comparison
  the current artifacts allow.
- The 540p run was tagged against `v0.7-frozen` *pre-thaw*; the
  bridge_opportunity thaw doesn't change influence scores for
  non-bridge anomalies, so this comparison is essentially apples-to-apples.
- The 0.371 number quoted here is from re-running the backtest today
  on `artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1/`. A previous
  in-conversation memo noted +0.346; the difference is the
  hypothesis cap (300 ≈ same) and which subset filter is applied.
  Both numbers are well above v2's +0.003.

## Provenance

| File | Purpose |
|---|---|
| `artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1/` | v1 baseline run (300 hyp, post-thaw) |
| `artifacts/runs/arxiv-reasoning-v2/` | v2 held-out run (299 hyp) |
| `scripts/validate_influence_backtest.py` | backtest script (Spearman ρ) |
| `docs/v0.7-pipeline-freeze.md §4` | the thaw conditions this triggers |

Verification command for the v2 number:

```bash
ln -sf hypotheses_scored.jsonl artifacts/runs/arxiv-reasoning-v2/creator_hypotheses.jsonl
python scripts/validate_influence_backtest.py artifacts/runs/arxiv-reasoning-v2
```
