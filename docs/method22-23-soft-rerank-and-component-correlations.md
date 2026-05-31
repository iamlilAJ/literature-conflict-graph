# Methods 22 + 23 — Soft re-rank ablation and scorer-component decomposition

**Date:** 2026-05-31
**Methods 22, 23 of:** /loop "test different methods"; close the question of whether the binary filter is the *best* form of the Atlas signal, and decompose *what* atlas_overlap captures vs the frozen scorer.

## Method 22 — Soft re-rank doesn't beat hard filter

Three arms on the same 12 topics × top-8 = 96 slots: OFF (no filter), HARD (Method 13 binary drop overlap<3), SOFT (utility × overlap_weight where weight curve is `{1:0.4, 2:0.6, 3:1.0, 4:0.9, 5:0.5}` from Method 11's anchored_novelty rule).

| arm | leak rate (overlap=1 or 2 in top-8) |
|---|---:|
| OFF | 22/96 (23%) |
| HARD | **0/96 (0%)** |
| SOFT | 1/96 (1%) |

The 1/96 SOFT leak is one overlap=2 hyp with high enough utility to clear `utility × 0.6` and still make top-8. HARD's deterministic threshold is more reliable.

Slot churn: OFF↔HARD 63/192 (33%); OFF↔SOFT 58/192 (30%); HARD↔SOFT 57/192 (30%). HARD and SOFT pick fairly different top-8 sets — both are valid, but HARD's guarantee dominates.

**Verdict:** Keep the hard filter from Method 13. SOFT could be useful as a *tie-breaker among overlap=3 hyps* in a future refinement, but isn't worth replacing HARD.

## Method 23 — atlas_overlap correlates with quality components, NOT with utility

Spearman ρ on the 259 production hyps where both `atlas_overlap` (Likert) and `ScoreBreakdown` components are known.

| frozen-scorer component | ρ (atlas_overlap, x) | reading |
|---|---:|---|
| **grounding** | **+0.629** | strong positive |
| **testability** | **+0.629** | strong positive |
| **topology** | **−0.627** | strong NEGATIVE |
| **impact** | **+0.576** | strong positive |
| discriminability | +0.361 | moderate positive |
| explain | +0.343 | moderate positive |
| cost | +0.220 | weak positive |
| novelty (1−Jaccard) | −0.226 | weak negative |
| **utility (composite)** | **−0.007** | **near zero** |

**This explains Method 12's central puzzle.** The frozen utility composite is uncorrelated with atlas_overlap not because Atlas captures something orthogonal — it captures something the components DO measure individually, but the weighted sum **cancels the signal**. +0.6 from grounding plus −0.6 from topology nets out to zero in the weighted composite.

### Per-bucket means

| overlap | n | explain | grounding | testability | novelty | discrim. | impact | topology | cost | utility |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11 | 0.761 | 1.00 | 1.00 | 0.890 | 0.951 | 0.999 | 0.845 | 0.205 | 0.874 |
| 2 | 44 | 0.861 | 1.00 | 1.00 | 0.898 | 0.940 | 0.997 | 0.856 | 0.227 | 0.899 |
| 3 | 187 | 0.869 | 1.00 | 1.00 | 0.889 | 0.948 | 0.999 | 0.841 | 0.201 | 0.904 |
| 4 | 17 | **0.941** | 1.00 | 1.00 | 0.881 | 0.951 | 1.00 | 0.842 | 0.250 | **0.916** |

`grounding` and `testability` are at ceiling for every bucket — the rank correlations are picking up sub-ceiling variance the means hide. `explain` shows a clean monotone climb 0.76 → 0.94 across overlap=1→4. This is the most actionable per-component finding: **`atlas_overlap` is approximately a smoothed, semantically-grounded `explain` signal** — but with stronger discrimination at the low end (overlap=1's `explain` 0.761 is below the noise floor of the 1−Jaccard novelty).

### Why atlas_overlap is not "just better novelty"

Naïvely you might assume Atlas captures novelty since it's measuring against a corpus of known open problems. The data refutes this: ρ(overlap, novelty) is **−0.23 (weakly negative)**. Atlas-anchored hyps are *slightly less* Jaccard-novel — because they share more vocabulary with the bottleneck corpus, just like Method 11 already showed. Atlas adds **anchoring**, not novelty. Real novelty is still measured against the *cited corpus*, not the *known-bottleneck corpus*.

## Combined implication

The hard filter from Method 13 is the right deployment shape because:

1. The signal it gates (`atlas_overlap ≥ 3`) is independent of the composite utility, so the filter doesn't double-penalize anything the scorer is already weighing.
2. The signal correlates strongly with multiple quality components (explain, grounding, testability, impact) — these are *real* quality dimensions the production scorer is *also* tracking but masking via the topology and novelty counter-weights.
3. Adding soft weighting (Method 22) doesn't measurably improve over hard filtering.

The Atlas filter functions as **a guardrail against the frozen scorer's known false-positive mode**: high-utility-low-anchoring hyps that the cost/topology/novelty cancellation lets through. Methods 12, 14, 15, 22, 23 all converge on this story.

## What's still untested

- **Method 10** — Atlas conflict-graph (9.6M edges parquet) as `graph_bridge` validator. Distinct Atlas surface from the 1,607 bottleneck quotes. Requires pyarrow.
- **Method 24** — per-anomaly-type breakdown of atlas_overlap (does community_disconnect produce more overlap=1 hyps than evidence_gap?).
- **Cross-judge ensembling** — same prompt to a non-Kimi LLM (no other working endpoint today).
- **Filter applied to a different run** — generalization check on val1-primary (needs new sidecar computation, ~40 min Kimi time).

## Artifacts

- `scripts/method22_soft_rerank.py` — soft vs hard ablation
- `scripts/method23_atlas_vs_scorer_components.py` — Spearman + bucket means
- `artifacts/atlas_test/method22_results.json` — per-topic OFF/HARD/SOFT delivered top-8
