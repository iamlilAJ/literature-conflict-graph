# Method 2 — Atlas-grounded prompting vs Plain — verdict

**Date:** 2026-05-31
**Method 2 of:** [quality-next-levers-research-brief.md](quality-next-levers-research-brief.md), part of the /loop "test different methods" iteration.
**One-line answer:** **plain prompting wins 7:4 in pairwise** on 11 verified A/B cases (both arms 4/4 rubric ceiling). Atlas bottleneck context, in its current form, makes hypotheses *broader and more architectural* — the judge prefers the tighter single-variable mechanism style that plain prompting yields.

## Experiment

- 16 joint anomalies (`artifacts/atlas_test/method2_input.json`), stratified 2-per-dimension across the 8 most-frequent bottleneck dimensions.
- Same forward-frame + reflect workflow (production winner = "C") in both arms; only the *context* differs.
- Arm P (plain): only the original-paper limitation quote shown to the model — Atlas bottleneck_signals are *stripped* from the central_question.
- Arm A (atlas): full Atlas-grounded context shown — central_question + later-paper bottleneck quotes + dimension + severity + explicit instruction to "address BOTH the original limitation AND the later paper's bottleneck".
- Same binary 4-axis rubric + blind pairwise P-vs-A.
- Kimi-K2.6, 4 workers, ~15 min wall.

## Numbers (n=11, 5 errors)

Errors: 4 "A draft empty" + 1 "P draft empty". The Atlas-grounded prompt's larger input chews more reasoning budget — that's a real ops cost.

| component | P (plain) | A (atlas) | Δ |
|---|---:|---:|---:|
| forward_looking | 1.00 | 1.00 | 0 |
| named_mechanism | 1.00 | 1.00 | 0 |
| single_variable_test | 1.00 | 1.00 | 0 |
| specific_scope | 1.00 | 1.00 | 0 |
| **composite (0-4)** | **4.00** | **4.00** | **0** |

Binary rubric ceiling — same problem documented in [`workflow-d-vs-c-verdict.md`](workflow-d-vs-c-verdict.md) and the [[binary-rubric-ceilings]] memo.

**Blind pairwise P vs A: P 7, A 4, tie 0**

## What pair_why actually says

Reading the 11 justifications splits cleanly along a **shape axis**, not a quality axis.

### P wins (7/11) — the judge consistently praises:
- "tightly controlled, single-variable diagnostic experiment that directly isolates the hypothesized antagonistic-gradient mechanism"
- "sharply focused, mechanistic explanation … proposes an isolable single-variable test"
- "precise, counter-intuitive dynamical mechanism (overshoot-induced rotation driving a spectral sign flip)"
- "exact gradient-scaling formula controlled by a single coefficient λ"
- "precise mechanistic failure mode (cross-task gradient interference) and a clean single-variable ablation"
- "precise, falsifiable causal mechanism (gradient-update cancellation) … rigorous single-variable experimental control"

### A wins (4/11) — the judge consistently praises:
- "concrete architectural innovation—an amortized low-rank hypernetwork with explicit complexity bounds and a systematic rank-ablation plan"
- "causally isolable, theoretically motivated architectural extension to address a core representational limitation"
- "novel, exact graph-theoretic constraint and a clean rewiring experiment that isolates residual-path continuity as a causal driver"
- "new, generalizable model structure that couples utilities via auxiliary attributes to resolve **TWO DISTINCT BOTTLENECKS**"

Plain prompting steers the model toward **single-variable mechanism + diagnostic experiment**. Atlas grounding steers it toward **architectural innovation that addresses multiple bottlenecks**. Both are valid scientific hypotheses; the pair judge prefers the former 7:4. Whether end-users prefer the same balance is open.

### Dimension matters

Per-dimension breakdown (n=11 split across 8 dimensions, so low n each):

| dimension | n | P wins | A wins |
|---|---:|---:|---:|
| **expressiveness** | 2 | 0 | **2** |
| computational_complexity | 2 | 1 | 1 |
| data_efficiency | 1 | 0 | 1 |
| training_stability | 2 | **2** | 0 |
| accuracy | 1 | 1 | 0 |
| generalization | 1 | 1 | 0 |
| scalability | 1 | 1 | 0 |
| memory_efficiency | 1 | 1 | 0 |

The only dimension where A consistently wins is **expressiveness** — and reading those cases, "extend the model's representational power" is naturally an architectural answer, so the Atlas push toward architectural framings aligns with the dimension's nature. For **training_stability**, **accuracy**, **generalization**, **scalability**, **memory_efficiency**, the judge wants the diagnostic-experiment shape and plain wins.

## Implication for the /loop

This confirms the standing memo [[atlas-bottleneck-align-tested]]: **Atlas's value is anomaly-SELECTION, not prompt-INJECTION**. Two methods follow:
- **Method 6 (refined direction)**: use Atlas severity / dimension to *rank which aigraph anomalies* feed the generator, not to enrich the per-anomaly context. Tests whether Atlas-selected anomalies produce inherently better hypotheses regardless of grounding.
- **Method 3**: use Atlas open-questions as an *anti-novelty* corpus — if a generated hypothesis closely matches an already-stated Atlas open-question, mark it NOT novel. Atlas as evaluator, not generator-context.

Method 3 sketches next iteration. Method 6 is the structural follow-up.

## Honest negatives

1. **Binary rubric still saturates** — both arms 4/4 perfect. Method 2's verdict is entirely from pairwise reasoning, which is the only working signal in this setting. Likert upgrade (task #31) is now blocking any future generation A/B.
2. **A's draft-empty rate (4/16) is structural** — larger-context prompts eat Kimi's reasoning budget faster. Any Atlas-grounded production use needs higher per-call `max_tokens` than plain prompts.
3. **n=11 per pairwise winner split is tiny** — the 7:4 split is suggestive, not significant. A 50-pair version would tighten the case but costs ~1 hr of Kimi time.
