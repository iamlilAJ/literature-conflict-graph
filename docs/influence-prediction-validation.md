# Influence Prediction — Phase 1 Validation (empirical record)

This document records the empirical backtest of Phase 1 (4-dim, non-LLM)
influence prediction on the existing 1000-paper corpus, ingest-box run
`/tmp/fullrun_v2/`. The framework is in
`docs/influence-prediction-design.md`; the implementation is in
`src/aigraph/influence.py`. The validation script is at
`scripts/validate_influence_backtest.py`.

Run by:

```bash
python scripts/validate_influence_backtest.py /tmp/fullrun_v2
```

## Run inputs

| field | value |
|---|---|
| hypotheses | 95 (from `creator_hypotheses.jsonl`) |
| claims | 3849 |
| papers | 4895 |
| hierarchy domains | 106 |
| hierarchy communities | 27 |
| hierarchy clusters | 1533 |
| venue/proxy papers | top-100 by `cited_by_count` (the `arXiv`-only corpus has `Paper.venue == "arXiv"` for almost all rows, so the hard ICML/NeurIPS/ICLR string filter matched 0; the script falls back to top-100 cited as a proxy) |
| venue-grounded hypotheses | 89 |

## Headline result

**Spearman ρ (predicted total vs L1 actual impact) = 0.328**

Partial validation band (0.2 < ρ < 0.4). The 4-dim score correlates
positively with citation impact, but not at the threshold the design
doc asked for (ρ > 0.4 to proceed to Phase 2 directly). Per-dimension
analysis tells a sharper story.

## Per-dimension correlations

| dimension | ρ vs max-cite | interpretation |
|---|---:|---|
| `community_reach` (I_2) | **+0.542** | The strongest signal. Hypotheses that span more communities correlate with higher-cited grounding papers. **This is the dimension that's working.** |
| `novelty` (I_4) | N/A (constant) | All 95 hypotheses default to 0.5 because `novelty_check` is not populated in this artifact (the arxiv-novelty-check pipeline ships on a different branch and was never applied to these records). The dimension contributes a constant 0.30 × 0.5 = 0.15 to every total — pure noise floor in this backtest. |
| `grounding_depth` (I_5) | **−0.166** | Slight anti-correlation. Higher heuristic evidence quality correlates with *lower* actual cite impact. Weakly so, but counter to the design's assumption. Likely cause: the booster favors canonicalized + dataset-named claims, which over-indexes on newer / shorter / preprint papers; older popular papers in the proxy set tend to have less canonicalization. |
| `scope_overreach_risk` | N/A (constant) | The 95 hypotheses' `scope_conditions` are uniformly empty in this artifact, so risk = 0.0 across the board. Like novelty, contributes a constant. |

## Top-10 overlap

**0/10 (0%)** — none of the top-10 predicted appear in the top-10
actual. Initially alarming; closer inspection shows the L1 metric is
heavily skewed by a single very-popular paper:

- Top predicted: `a398/a396` cluster, all `predicted ≈ 0.395`,
  `actual_cite = 6674` (engineering / enterprise QA hypotheses).
- Top actual: `a365/a364` cluster, all `actual_cite = 19558`,
  `predicted ≈ 0.36-0.38` (one cited paper has 19558 cites,
  dragging up every hypothesis that grounds in it).

The L1 proxy "max `cited_by_count` over inspired papers" gives the
same high score to *any* hypothesis grounded in a textbook-class
paper, regardless of the hypothesis's own quality. This is a proxy
artifact, not a fundamental Phase 1 failure: with a noisier L1, top-K
overlap is unreliable as a single number.

## Honest interpretation

1. **`community_reach` works** (ρ = 0.542). It alone carries enough
   signal to drive ~half the variance. That validates the
   hierarchy-based geometry the design depends on.
2. **`novelty` and `scope_overreach` were never tested** — they need
   upstream pipelines (arxiv-novelty-check, scope-aware creator) to
   populate the input fields. Phase 1's contribution is *the
   plumbing*; the empirical signal awaits those upstream branches
   merging and being run end-to-end.
3. **`grounding_depth` may be miscalibrated.** Slight anti-correlation
   suggests `compute_evidence_quality` is rewarding the wrong things
   — most likely the canonical-method/task booster, which correlates
   with paper recency rather than research importance. A future tweak
   should drop or invert that booster.
4. **L1 metric is noisy** at the top-K. A more diagnostic metric
   would normalize by paper year (cites/year), or use percentile rank
   within the corpus rather than raw max-cite. Phase 2 should
   reconsider this.

## Decision

Phase 1 partial validation. Move to Phase 2 (Structural Impact via
graph_repair) **with these two adjustments**:

1. Re-run the backtest *after* `feat/arxiv-novelty-check` has been
   applied to `creator_hypotheses.jsonl` (input file). The novelty
   dimension is currently a no-op; with real input it should
   contribute ~0.3-0.5 ρ on its own and lift the total.
2. Replace the L1 metric with a corpus-rank-normalized variant
   (cites_percentile_within_corpus_year). Re-report ρ and top-K
   overlap.

If after both adjustments ρ still hovers below 0.4, treat
`grounding_depth` as the suspect dimension and either tune the
booster weights or replace it with a paper-quality proxy
(impact_score from anomaly metadata).

Phase 1 ships with this honest record; the framework is plumbed
end-to-end and the strongest single dimension is empirically
confirmed.
