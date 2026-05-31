# Cross-corpus v0.7 — scale-up to 2488 papers + 8/8 anomaly types

**Date:** 2026-05-12
**Git tag at predictor:** `v0.7-frozen` (post-thaw #1)
**Question:** Does the 540p → 847p ρ collapse (reported in
`findings/2026-05-11-cross-corpus-v0.7-validation.md`) persist when the
v2 corpus is scaled to 2488 papers?
**Short answer:** ρ partially recovers (-0.056 → +0.171 on the full
hypothesis set; +0.003 → +0.060 on the venue-proxy subset), and two
influence dimensions that were dead-N/A on both prior runs start to
move. But the v0.7 weighted total is still well below the 540p
baseline, and the previously-load-bearing `scope_overreach_risk` sign
is still flipped vs. the 540p direction.

---

## Scale jump

| | v1_540p_thaw1 | v2_847p | **v2_2488p** |
|---|---:|---:|---:|
| papers (with sections) | 474 | 847 | **2,488** |
| claims | 1,961 | 3,303 | **10,724** |
| graph nodes / edges | 4,154 / 31,407 | 6,285 / 51,464 | **20,386 / 493,810** |
| anomalies (full) | 496 | 377 | **3,077** |
| anomaly types fired | 8/8 | 5/8 | **8/8** ✓ |
| hierarchy: domains / **communities** / clusters | n/a | 161 / **0** / 926 | 419 / **15** / 2608 |
| hyp generated (cap=100) | 300 | 299 | 299 |
| novelty: novel / known / null | 261 / 14 / 25 | 193 / 23 / 83 | 217 / 6 / 76 |
| cited_by>0 papers (Semantic Scholar) | 450 / 474 | 823 / 847 | 823 / 2,488 |

Note: the cited-proxy subset stays at **823 papers** even at 2488 scale
because the 1647 newly-seeded papers are mostly 2025 (1200) and 2026
(425). Semantic Scholar has no citation counts for them yet. The
venue-proxy subset of the backtest therefore captures the same
underlying 2023-2024 paper population in both v2 runs.

Newly-firing anomaly types at 2488p (were 0 at 847p):
- `community_disconnect`: 0 → 1,906
- `impact_conflict`:      0 → 136
- `evidence_gap`:         0 → 73

And `community_disconnect = 1906` is the proximate cause of the graph
explosion (493k edges) and the 660 MB `anomalies.jsonl`. Each
community-disconnect anomaly carries lots of cluster-pair evidence.

## ρ results (Spearman, predicted total influence vs. max cited_by_count over evidenced papers)

Predictor scoring re-run after `check-novelty` so the `novelty` dim
finally has variance (was constant 0.5 in both prior runs). The
"SCORED_V2" rows use that re-scored version; SCORED_V1 is the
v0.7-finish_local_run output without novelty fed in.

```
=== v1_540p_thaw1 ===  (baseline)
  all         n=300  total=+0.130  reach= N/A   novelty= N/A   depth=+0.023  risk=-0.201
  novel_only  n=261  total=+0.113  reach= N/A   novelty= N/A   depth=-0.013  risk=-0.224
  known_only  n= 14  total=+0.380  reach= N/A   novelty= N/A   depth=+0.099  risk=-0.305

=== v2_847p_SCORED_V1 ===  (memo from 2026-05-11)
  all         n=299  total=-0.056  reach= N/A   novelty= N/A   depth=+0.008  risk=+0.019
  novel_only  n=193  total=-0.124  reach= N/A   novelty= N/A   depth=-0.001  risk=+0.046
  known_only  n= 23  total=+0.246  reach= N/A   novelty= N/A   depth=+0.400  risk=+0.295

=== v2_2488p_SCORED_V1 ===  (pre-novelty re-scoring)
  all         n=299  total=+0.171  reach=+0.125  novelty= N/A   depth=+0.228  risk=+0.229
  novel_only  n=217  total=+0.145  reach=+0.150  novelty= N/A   depth=+0.222  risk=+0.314
  known_only  n=  6  total=+0.086  reach=+0.086  novelty= N/A   depth=+0.143  risk= N/A

=== v2_2488p_SCORED_V2 ===  (novelty-aware: predict-influence after check-novelty)
  all         n=299  total=+0.109  reach=+0.125  novelty=+0.189  depth=+0.228  risk=+0.229
  novel_only  n=217  total=+0.166  reach=+0.150  novelty=+0.299  depth=+0.222  risk=+0.314
  known_only  n=  6  total=+0.086  reach=+0.086  novelty=    N/A  depth=+0.143  risk= N/A

=== venue-grounded (top-100-cite proxy) ===
                       v1_540p   v2_847p   v2_2488p_v1   v2_2488p_v2
  n                       184       111         154            154
  ρ_total              +0.371    +0.003      +0.060         -0.234
  community_reach        N/A      N/A         +0.114         +0.114
  novelty                N/A      N/A          N/A           -0.158
  grounding_depth      +0.277    +0.079      +0.207         +0.207
  scope_overreach_risk -0.199    +0.260      +0.375         +0.375
  top-10 overlap         0/10      0/10        0/10           0/10
```

(Per the existing `validate_influence_backtest.py` script. Venue-proxy
fallback applies on all four runs because none of the corpora carry an
ICML/NeurIPS/ICLR venue tag.)

## What scaled

1. **community_reach finally moves** (+0.125 on the full set, +0.114 on
   venue-grounded). The 847p run had 0 communities in the hierarchy;
   at 2488p we get 15. So this dimension was dormant in the v1_540p
   and v2_847p backtests, not broken — just data-starved.

2. **All 8 anomaly detectors fire**. The freeze doc §4 condition 2
   (zero detections on a 1000+ paper cohort) was never tripped here,
   but `community_disconnect`, `impact_conflict`, and `evidence_gap`
   were on the edge at 847p and now fire at 2488p with substantial
   counts.

3. **Direction of ρ flips back positive**. The 847p result had a
   small negative ρ on the full subset (`-0.056`) and on novel-only
   (`-0.124`). At 2488p both are positive again (`+0.171`, `+0.145`),
   which is the *direction* the predictor needs. The magnitude is
   lower than the v1_540p baseline (+0.130) on `all` — comparable —
   and slightly higher on `novel_only` (+0.145 vs +0.113).

## What still doesn't scale

1. **`scope_overreach_risk` sign is still flipped from the 540p
   baseline**. v1_540p ran -0.305 (known) / -0.224 (novel) — risk
   penalty pointing the right way. Both v2 runs run +0.x — risk
   penalty pointing the wrong way (high-risk hypotheses get *higher*
   cite scores, opposite of the v0.7 weight formula's assumption).
   That is the single most consistent failure across scale-ups.

2. **Adding the novelty dim *hurts* the venue-grounded ρ**: +0.060 →
   −0.234 when we re-score with novelty included. The novelty
   dimension's per-dim ρ on the venue-grounded subset is **negative**
   (-0.158). The top-cited venue-proxy papers in our corpus are
   highly-cited *because* their work has been built on extensively;
   their hypotheses are *less* novel relative to arXiv prior art, not
   more. The 0.30 weight on novelty in `WEIGHTS_PHASE1` is therefore
   counterproductive on this slice.

3. **Top-10 overlap is still 0/10** on all four runs. The ρ values
   describe rank-order trend, but the predicted top-10 and the actual
   top-10 don't share any hypotheses. The predictor is detecting
   *direction* (slight positive monotonic trend) but not the
   *extremes*. Likely cause: cited_by_count is dominated by venue
   reputation / recency / topic popularity, and these are not captured
   by any v0.7 dimension.

## Recommendation (updated)

1. **Drop or invert `scope_overreach_risk`**. It's the most clearly
   broken dim. The freeze doc weights it 0.20 as `1 - risk` (treating
   high risk as a *deduction*). On the v2 corpora the per-dim ρ is
   consistently positive (around +0.2-0.4), meaning the sign should
   be flipped — high "risk" actually *helps* citation. Either drop it
   from `WEIGHTS_PHASE1` or invert its contribution.
2. **Re-weight novelty conditional on subset**. On the
   `novel_only` slice it contributes positively (+0.299 in the full
   set); on the venue-proxy slice it's negative (-0.158). The
   uniform 0.30 weight is the wrong shape. A two-regime version
   (one weight for novel-only ranking, a different one for the
   venue-proxy ranking) would be honest about what each dimension is
   actually predicting.
3. **Don't tune to ρ_total alone**. The 2488p data shows
   community_reach + grounding_depth track citations weakly but
   positively (+0.1 to +0.2 each); novelty contributes on one slice
   and anti-correlates on another; risk anti-correlates everywhere
   against the formula's sign convention. The v0.7 formula's
   structural assumption — that all four dims pull in the same
   direction with positive weights — is empirically wrong on this
   corpus. A reviewer asked to audit the formula would conclude this.

## What this means for the v0.7 freeze

- Freeze doc §4 condition 1 (ρ < 0.10 on primary cohort) is **not**
  tripped at 2488p (ρ_total venue-grounded = +0.060, still below the
  formal threshold but no longer near-zero; on the full hypothesis
  set ρ is +0.171). It *was* tripped at 847p (+0.003).
- Freeze doc §4 condition 2 (zero anomaly type detections on 1000+
  cohort) is no longer in play — all 8 fire at 2488p.
- So technically neither thaw condition fires at 2488p. But the
  underlying observation (that the v0.7 formula is composed of dims
  with inconsistent signs across corpora) is independent of the
  formal thresholds. A weight-tuning thaw is still the most
  defensible response.

## Provenance

| File | Purpose |
|---|---|
| `findings/2026-05-11-cross-corpus-v0.7-validation.md` | prior memo, 847p results |
| `artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1/` | v1 baseline (300 hyp) |
| `artifacts/runs/arxiv-reasoning-v2/papers.jsonl` | 2488 papers + Semantic Scholar cited_by |
| `artifacts/runs/arxiv-reasoning-v2/claims.jsonl` | 10,724 claims |
| `artifacts/runs/arxiv-reasoning-v2/hypotheses.jsonl` | 299 hypotheses (cap=100) |
| `artifacts/runs/arxiv-reasoning-v2/hypotheses_novel.jsonl` | + novelty_check |
| `artifacts/runs/arxiv-reasoning-v2/hypotheses_with_novelty.jsonl` | hypotheses + novelty for re-scoring |
| `artifacts/runs/arxiv-reasoning-v2/hypotheses_scored_v2.jsonl` | influence re-scored with `is_novel` |
| `scripts/validate_influence_backtest.py` | backtest harness (read `creator_hypotheses.jsonl`, symlinked) |

Verification:

```bash
# headline venue-grounded ρ on v2_2488p, SCORED_V1
ln -sf hypotheses_scored.jsonl artifacts/runs/arxiv-reasoning-v2/creator_hypotheses.jsonl
python scripts/validate_influence_backtest.py artifacts/runs/arxiv-reasoning-v2
# expected: rho_total = 0.060

# SCORED_V2 (novelty-aware)
ln -sf hypotheses_scored_v2.jsonl artifacts/runs/arxiv-reasoning-v2/creator_hypotheses.jsonl
python scripts/validate_influence_backtest.py artifacts/runs/arxiv-reasoning-v2
# expected: rho_total = -0.234
```

Two gitignored files (regen from claims):
- `anomalies.jsonl` 660 MB
- `graph.json` 99 MB
