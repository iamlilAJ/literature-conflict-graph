# Workflow D (decomposition) vs Workflow C (reflect) — verdict

**Date:** 2026-05-31
**Method 1 of:** [quality-next-levers-research-brief.md](quality-next-levers-research-brief.md) (the /loop "test different methods" plan).
**One-line answer:** **statistically tied** — both at 4/4 binary rubric ceiling on 14 verified anomalies; pair judge picks C 8:6 (not significant). Decomposition does not measurably improve on the production forward-frame + reflect workflow at this rubric resolution. The framing rewrite is the real lever, not the workflow shape.

## Experiment

- 16 same anomalies as the prior A/B/C experiment (`artifacts/atlas_test/genexp16.json`)
- Workflow D: 4-stage decomposition (brainstorm → sharpen → test → draft), inspired by [arxiv 2601.09714](https://arxiv.org/html/2601.09714) which reported decomposition ≈ 2× novelty over reflection at 30-proposal scale.
- Workflow C: forward-framed single-shot + reflect (current production winner).
- All 4 LLM calls per D run + 2 judge calls + Workflow C result reused from prior experiment.
- Kimi-K2.6, 4 workers, 16 anomalies. Run 2 results (Run 1 had 9/16 token-budget stage errors, fixed by bumping per-stage `max_tokens` to 5K–6K).

## Numbers (n=14, 2 stage errors)

| component | C (this subset) | D | Δ D-C |
|---|---:|---:|---:|
| forward_looking | 0.93 | 1.00 | +0.07 |
| named_mechanism | 0.93 | 1.00 | +0.07 |
| single_variable_test | 0.93 | 1.00 | +0.07 |
| specific_scope | 0.93 | 1.00 | +0.07 |
| **composite (0-4)** | **3.71** | **4.00** | **+0.29** |

Blind pairwise C vs D: **C 8, D 6, tie 0**.

## What's actually going on

The +0.29 composite delta comes from D never dropping a binary criterion;
C drops one criterion in 2/14 cases. That's at the binary rubric's noise
floor — both arms saturate. The pair judge's 8:6 split is more
informative since it's forced to pick. Sample `pair_why` justifications:

**C wins (reasoned reasons):**
- "X proposes a precise causal bottleneck with exact, falsifiable scaling
  predictions and a clean experimental manipulation"
- "X identifies a precise mechanistic bottleneck and proposes a concrete
  architectural replacement with a quantified, falsifiable prediction"
- "X specifies a controlled ablation and a sharp crossover boundary to
  isolate the causal failure mode"

**D wins (reasoned reasons):**
- "Y offers a more forward-looking, integrative prediction by linking a
  single dynamical control parameter (step size) to a sharp
  geometric-dynamic phase transition"
- "Y advances a specific, falsifiable experiment that isolates a causal
  mechanism—closed-loop intermediate supervisory feedback"
- "Y posits a novel, counter-intuitive mechanism—domain-confounder
  amplification—and derives a precise crossover prediction"

The judge's language is symmetric: both arms produce concrete, falsifiable,
mechanism-named hypotheses. C's reflect step nudges generic drafts toward
specificity; D's brainstorm+sharpen step front-loads divergence then commits
to one. Net quality is comparable.

## Why this differs from the arxiv 2601.09714 result

That paper reports decomposition ≈ 4.17/5 novelty vs reflection ≈ 2.17/5.
Three reasons the gap shrinks here:
1. **Our C is already forward-framed** — the 2024 reflection baseline in
   that paper probably used a generic critique step, not a forward-framing
   prompt. Our C inherits all the gain from the prior framing rewrite.
2. **Our rubric is binary 0/1; theirs is 5-point Likert.** Saturated 4/4
   scores can't show the within-ceiling differences a Likert would.
3. **Their "decomposition" includes retrieval-augmented sub-question
   exploration** (literature search per sub-question). Our D is
   intra-LLM decomposition only — no extra evidence is brought in.

The honest reading: at our scale and rubric, **the framing dominates the
workflow shape**. Extra LLM calls don't recover further headroom.

## Implications

1. **Workflow D does not justify the extra LLM cost.** 4 calls per anomaly
   for D vs 2 for C, no measurable quality lift. Production stays on C.
2. **Binary rubric ceiling is the real obstacle.** Future generation A/Bs
   need Likert 1-5 per criterion, or the comparison is meaningless. Tracked
   as task #31.
3. **The next levers must change WHAT goes in, not HOW it gets processed.**
   That points squarely at the Atlas data tracks (Methods 2-5 in the
   research brief) — Atlas-seeded prompting, Atlas as novelty oracle,
   Atlas-augmented grounding, joint Atlas anomaly mining.

## Artifacts

- [`scripts/run_workflow_d_experiment.py`](../scripts/run_workflow_d_experiment.py)
- [`scripts/analyze_workflow_d.py`](../scripts/analyze_workflow_d.py)
- [`artifacts/atlas_test/wfd_out2.jsonl`](../artifacts/atlas_test/wfd_out2.jsonl) — per-anomaly hC, hD, brainstorm, rubric, pair_why
