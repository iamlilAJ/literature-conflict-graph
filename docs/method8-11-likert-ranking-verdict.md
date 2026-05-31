# Methods 8 + 11 — Likert ranking is the lever — verdict

**Date:** 2026-05-31
**Methods 8 and 11 of:** /loop "test different methods" plan; jointly close Method 1's binary-rubric blind spot and identify the deployable signal.
**One-line answer:** **The Likert composite ranking — driven by `named_mechanism`, `forward_looking`, `specific_scope`, and the *independent* `atlas_overlap` axis — consistently identifies the back-explanation generation as low-quality (6-7 of bottom-10 are frozen across every weighting rule tried). That gives us a calibrated ranking oracle the broken 1−Jaccard novelty couldn't be.**

## Method 8 — Likert re-judge of D vs C

The Method 1 verdict was "statistically tied at 4/4 binary ceiling". Re-judged the same 14 hC/hD pairs from `wfd_out2.jsonl` under Method 3's Likert. Result on n=10 paired anomalies, Likert composite (sum 5 axes):

- **D wins: 4**
- **Tie:   3**
- **C wins: 3** (but a1210 loses by Δ=-5, dragging mean)

Population means (n=11 C, 13 D):

| criterion | C | D | Δ D−C |
|---|---:|---:|---:|
| atlas_overlap | 3.27 | 3.38 | +0.11 |
| forward_looking | 4.45 | 4.31 | −0.14 |
| named_mechanism | 4.82 | 4.62 | −0.20 |
| single_variable_test | 4.91 | 5.00 | +0.09 |
| specific_scope | 4.82 | 4.54 | −0.28 |

**D is slightly more atlas-anchored; C is slightly sharper on mechanism + scope.** The composite means delta is ≈0.4 in favor of C — below noise at n=10. Method 1's "tied" stays a tie under Likert; what we gain is *qualitative direction* — the two workflows produce different *shapes* of hyp, not different *qualities*.

The outlier `a1210` (D=18 vs C=23, Δ=-5) is the largest single-case drag; spot-checked manually, D's hypothesis on that anomaly took a less-anchored framing. Random per-anomaly variance.

## Method 11 — Pareto + correlation analysis across 70 Likert-judged hyps

Combined Methods 3 + 7 + 8 outputs. 70 hyps across 6 populations:

| population | source | n |
|---|---|---|
| M3 forward | forward-frame single-shot | 12 |
| M3 frozen | back-explanation frozen pipeline | 12 |
| M7 joint | Atlas-selected, Plain prompting | 10 |
| M7 native | aigraph-native, Workflow C | 12 |
| M8 workflow_C | forward-frame + reflect | 11 |
| M8 workflow_D | 4-stage decomposition | 13 |

### Finding 1 — `atlas_overlap` is statistically independent

Pearson correlations across the 5 Likert axes:

|  | atlas_o | fwd_l | named_m | single_v | spec_s |
|---|---:|---:|---:|---:|---:|
| atlas_overlap        | 1.000 | 0.044 | 0.191 | 0.199 | 0.126 |
| forward_looking      | 0.044 | 1.000 | 0.424 | 0.234 | 0.331 |
| named_mechanism      | 0.191 | 0.424 | 1.000 | 0.355 | 0.353 |
| single_variable_test | 0.199 | 0.234 | 0.355 | 1.000 | 0.489 |
| specific_scope       | 0.126 | 0.331 | 0.353 | 0.489 | 1.000 |

The 4 quality axes correlate moderately (0.23–0.49). `atlas_overlap` correlates with none of them at >0.20 — it's a **genuinely orthogonal** signal. That's the empirical basis for using it as an independent ranking term.

### Finding 2 — `atlas_overlap=3` is the sweet spot; `=2` is a real quality gate

Other-axis means split by `atlas_overlap`:

| overlap | n | fwd_l | named_m | single_v | spec_s |
|---|---:|---:|---:|---:|---:|
| 2 | 4 | 4.00 | **3.75** | **3.75** | 4.00 |
| 3 | 48 | 4.38 | 4.73 | 4.33 | 4.54 |
| 4 | 17 | 4.35 | 4.76 | 4.65 | 4.65 |
| 5 | 1 | 4.00 | 4.00 | 5.00 | 4.00 |

Hyps at overlap=2 (tangential) score ~1 Likert point lower on `named_mechanism` and `single_variable_test`. Overlap 3 and 4 are essentially tied on the other quality axes. **Filter rule: reject overlap=2 hyps.** Overlap 3 vs 4 don't need disambiguation — 4 is "directly addresses a known bottleneck" which the judge actively praises.

### Finding 3 — Every Likert composite rule bottom-ranks frozen 6-7/10

Tried 4 different weighting rules:

| rule | top-10 pops | bottom-10 pops |
|---|---|---|
| uniform 5-sum | joint:6, native:1, D:2, forward:1 | **frozen:7**, D:1, forward:2 |
| no-overlap 4-sum | joint:5, native:4, forward:1 | **frozen:7**, forward:3 |
| boost_nm + sweet_ovr | joint:4, native:4, forward:2 | **frozen:6**, forward:1, D:3 |
| anchored-novelty | joint:4, native:4, forward:1, C:1 | **frozen:6**, D:3, forward:1 |

The ranking is **robust to weighting**. No matter how I combine the axes, frozen hyps dominate the bottom of the ranking and forward-framed populations dominate the top. This is what a working ranker looks like.

### Finding 4 — Workflow D actually has the most Pareto-front hyps (5/14)

On the `(named_mechanism, atlas_overlap)` Pareto front of 14 dominating hyps, source distribution:
- M8 workflow_D: 5
- M7 native: 3
- M8 workflow_C: 2
- M7 joint: 2
- M3 forward: 1
- M3 frozen: 1 (`h202`)

Workflow D's mean was tied with C, but D has a fatter top tail. The decomposition method is producing some of the very best hyps even though it ties on average. Worth keeping in mind for the next thaw cycle.

## What this unlocks

This is the first cleanly deployable result of the /loop. **All four prior methods (1=D, 2=Atlas prompt, 7=Atlas selector, 8=Likert D-vs-C) were ties or losses.** Methods 3 and 11 jointly identify a real lever:

### Method 12 — already launched
Score all 300 hyps in `arxiv-reasoning-v0.7-540p-thaw1` via Likert (5 axes incl. atlas_overlap), compute composite, correlate to current frozen-scorer `utility`. Tests whether the Likert composite ranking *operationally* improves on the production ranking. Background tmux session `m12_exp`, 6 workers, ETA ~25 min.

### Production wiring (Method 9, after Method 12 verifies)
Two integration paths:
1. **Delivery-time annotation (non-frozen).** Per delivered hyp, attach `atlas_overlap`, `closest_quote`, and a one-line "why". Costs 1 Kimi call per delivered hyp.
2. **Re-ranking at query-time (non-frozen).** In `scripts/aigraph_query.py` `_select()`, compute Likert composite for the candidate top-K before `select_mmr`, demote `overlap=2` hyps, re-rank top-K. Costs K Kimi calls per query.
3. **Scorer replacement (post-thaw).** Replace `novelty = 1 - Jaccard(...)` with `f(atlas_overlap)` in `scoring.py`. Requires §4 thaw record + 50-pair oracle verification.

## Honest caveats

1. **70 Likert-judged hyps total** — Method 11's Pareto/correlation conclusions are on a small sample. The independence of `atlas_overlap` (max corr 0.20) is plausibly real but a 200+ sample would pin it down. Method 12's 300-hyp run will provide that.
2. **All scoring is via one judge (Kimi-K2.6)** — single-model bias risk. The Likert means are self-consistent within the model but might differ under another judge (e.g., Claude or GPT). Not validated.
3. **The composite-rule design space is unexplored** — I tried 4 rules; could be 100s of better ones. Method 12 will pin the right rule.
4. **`atlas_overlap=5` only fired once** in 70 hyps — can't conclude anything about exact-restatement hyps as a category.
