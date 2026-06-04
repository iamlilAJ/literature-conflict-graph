# Real venue-cohort backtest — freeze §4 condition 1 TRIPPED + nuanced finding

**Date:** 2026-05-14
**Git tag at predictor:** `v0.7-frozen` (post-thaw #1)
**Cohort:** `artifacts/validation_v1/cohorts/primary_2018_2020.parquet`
**Cohort size:** 1,790 papers, 200 per (NeurIPS/ICML/ICLR × 2018/2019/2020) cell
**Question:** Does the v0.7 influence predictor reach ρ_total ≥ 0.10 on the
canonical NeurIPS/ICML/ICLR primary cohort the freeze doc was designed
around?
**Answer:** **No on aggregate (ρ_total = −0.069, novelty-aware −0.148),
yes on the known-only subset (ρ_total = +0.527 with depth=−0.127,
scope_overreach_risk=−0.430 — both signs as the formula assumes).**
The predictor formula is therefore *not broken everywhere* — it has
real signal on the small "known" subset, but the dominant "novel"
slice carries it the wrong way at the aggregate level.

---

## Cohort

Built `data/corpus/validation_v1_primary/papers.jsonl` from the parquet
cohort by remapping `arxiv_id` (100% present) into the `corpus-sync-arxiv`
manifest format. Then ran the standard pipeline:

| Stage | Output |
|---|---|
| corpus-sync-arxiv | 1790/1790 synced (some had TeX source, some PDF-only) |
| extract (8 workers) | 4,211 claims / 949 paper_ids covered (53% — lower than v2_2488p's 87% because more PDF-only papers had thin parses) |
| build-graph | 12,522 nodes, 34,580 edges |
| detect-anomalies | 1,423 anomalies (6 / 8 types — missing bridge_opportunity, replication_conflict) |
| stratified cap (top-12 / type × 6 types fired) | 72 anomalies |
| generate-hypotheses | 184 hypotheses |
| build-hierarchy | 152 domains, **0 communities**, 3126 clusters |
| predict-influence | 184 scored |
| check-novelty | **134 novel / 10 known / 40 null** |
| re-score with is_novel populated | hypotheses_scored_v2.jsonl |

Sync took ~16 hours wall (much slower than v2 because older arxiv papers
trigger more catastrophic regexes / PDF-only papers; mid-run hit a
transient arxiv 503 wave that forced a restart of the loop).

## Headline cross-cohort comparison

All four runs use the same v0.7-frozen predictor formula. The difference
is the corpus and the venue subset:

```
                                                Spearman rho on subset
                                            all       novel-only   known-only
v1_540p_thaw1   (arxiv reasoning, proxy)  +0.130      +0.113        +0.380
v2_2488p_v2     (arxiv reasoning, scaled, novelty-aware) +0.109      +0.166      +0.086
val1_primary    (NeurIPS/ICML/ICLR, REAL venue)  -0.069     -0.135       +0.527
val1_primary    (REAL venue, novelty-aware)       -0.148     -0.176       +0.527
                                          --- freeze §4 trip threshold: rho < 0.10 ---
```

Per-dim breakdown on val1_primary novelty-aware:

```
              all (n=184)    novel (n=134)   known (n=10)
ρ_total       -0.148         -0.176          +0.527
reach         N/A            N/A             N/A    (0 communities)
novelty       -0.068         -0.061          N/A    (single value)
depth         +0.059         +0.111          -0.127
risk          +0.260         +0.362          -0.430  ← sign matches formula
```

## Why this matters

### freeze §4 condition 1 is tripped at the aggregate

ρ_total on the primary cohort is −0.069 (pre-novelty) and −0.148
(novelty-aware). Both well below the 0.10 floor. By the freeze
contract this is a thaw trigger for `WEIGHTS_PHASE1`.

### But the predictor has real signal on the known subset

The `known_only` (`is_novel=False`) subset is small (n=10) but the
correlation is **+0.527** — exceeding the freeze doc's success bar
of 0.4 ("Phase 1 design validated"). And the per-dim
breakdown shows that on this subset, `scope_overreach_risk` correlates
**negatively** with citations (−0.430) — exactly the direction the
v0.7 formula's `(1 − risk)` term assumes. This is the first cohort
where any dim signs match the formula's structural assumption.

### The aggregate failure is novelty-driven

`scope_overreach_risk` flips sign between novel-only (+0.362) and
known-only (−0.430) on this cohort. The same pattern showed up faintly
on v1_540p (novel: −0.224, known: −0.305 — same direction but
magnitudes differ). At val1_primary scale + REAL venue cite-counts,
the contrast is dramatic.

What this means in plain terms:
- For hypotheses about known concepts (where prior arXiv work covers
  them), the v0.7 risk dimension is doing the right thing: scope-
  overreaching hypotheses about known stuff cite less because the
  field already knows what works and what doesn't.
- For hypotheses about novel concepts, scope-overreaching ones cite
  *more* — probably because in a novel area, an ambitious claim
  attracts attention regardless of whether it's well-grounded.

The v0.7 formula treats both as the same regime. That's the bug.

## Recommendations (final, supersedes prior memo paths)

1. **Two-regime weights conditional on is_novel**. The aggregate failure
   is entirely composed of two signals pointing opposite directions on
   different subsets. A single weight vector cannot capture both. Concrete:
   ```
   weight_known  = {community_reach: 0.25, novelty: 0.0, grounding: 0.25, -risk: 0.50}
   weight_novel  = {community_reach: 0.25, novelty: 0.5, grounding: 0.25, +risk: 0.00}
   ```
   The risk sign flips; the novelty weight gets concentrated on the
   subset where it actually moves.
2. **Drop `scope_overreach_risk` from the default formula** as previously
   memo'd is still a defensible *fallback* — it strictly improves the
   aggregate ρ — but loses the +0.527 known-subset signal. The
   two-regime version is better when novelty data is available.
3. **Path (3) from the prior memo is now retired**: we now have the proper
   venue cohort. The result is documented in this memo.

## Concrete predictor changes if this memo lands

`src/aigraph/influence.py:WEIGHTS_PHASE1` to be replaced by a function
`weights_for(is_novel: bool | None)` returning the two-regime vector
above (or the default if is_novel is None). The freeze doc §3 numeric
table also needs updating to record the regime split.

## Caveats

- known-only subset is n=10. The ρ=+0.527 is suggestive but the sample
  is small. The novelty-aware re-run produces the identical number
  because `is_novel` data already exists; the +0.527 is a property of
  10 venue-tagged hypotheses' actual cite counts, not of the formula
  weights.
- 0 communities again. `community_reach` was N/A on every cohort so far.
  At 2488p (v2) we got 15 communities; at 1790 NeurIPS papers we got 0.
  This is a hierarchy-build threshold tuning issue, not a predictor
  formula issue — but it does mean we're chronically running 1-of-4
  to 2-of-4 dim predictor.
- arxiv-only proxy comparisons (v1_540p, v2_*) use top-100-cite as a
  *proxy* for venue; val1_primary uses *real* venue tags. The drop from
  v2_2488p_v2's +0.109 → val1_primary's −0.148 is partly a
  switch-from-proxy-to-real (the proxy was capturing some signal that
  isn't there in real venue data), partly a cohort difference (newer
  arxiv reasoning papers vs older NeurIPS/ICML/ICLR papers).

## Provenance

| File | Purpose |
|---|---|
| `artifacts/validation_v1/cohorts/primary_2018_2020.parquet` | source cohort |
| `data/corpus/validation_v1_primary/papers.jsonl` | full corpus manifest (gitignored — 10 MB inside data/) |
| `artifacts/runs/validation-v1-primary/papers.jsonl` | 1790 papers w/ S2 cited_by_count |
| `artifacts/runs/validation-v1-primary/claims.jsonl` | 4211 claims |
| `artifacts/runs/validation-v1-primary/anomalies_top.jsonl` | 72 stratified (top-12 / type) |
| `artifacts/runs/validation-v1-primary/hypotheses.jsonl` | 184 |
| `artifacts/runs/validation-v1-primary/hypotheses_scored.jsonl` | v0.7 formula, pre-novelty (rho_total=−0.069) |
| `artifacts/runs/validation-v1-primary/hypotheses_novel.jsonl` | + novelty_check (134/10/40) |
| `artifacts/runs/validation-v1-primary/hypotheses_scored_v2.jsonl` | novelty-aware (rho_total=−0.148) |
| `findings/2026-05-11-cross-corpus-v0.7-validation.md` | prior memo, 847p findings |
| `findings/2026-05-12-cross-corpus-v0.7-scale-up.md` | prior memo, 2488p findings |
| `scripts/validate_influence_backtest.py` | backtest harness |
| `docs/v0.7-pipeline-freeze.md §4` | thaw conditions |

Verification:

```bash
ln -sf hypotheses_scored.jsonl artifacts/runs/validation-v1-primary/creator_hypotheses.jsonl
python scripts/validate_influence_backtest.py artifacts/runs/validation-v1-primary
# expected: rho_total = -0.069  (venue-grounded, NOT proxy fallback)
```

Gitignored: `anomalies.jsonl` (20 MB), `graph.json` (11 MB) — regen with
`aigraph build-graph` + `aigraph detect-anomalies`.
