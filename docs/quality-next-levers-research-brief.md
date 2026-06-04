# Next levers for hypothesis quality — arxiv-grounded research brief

**Date:** 2026-05-31
**Status:** ground truth + literature scan complete; Workflow D scaffolded but not yet run.
**Predecessors:**
[hypothesis-quality-diagnosis.md](hypothesis-quality-diagnosis.md),
[hypothesis-generation-workflow-experiment.md](hypothesis-generation-workflow-experiment.md),
[forward-framing-end-to-end-status.md](forward-framing-end-to-end-status.md).

We shipped forward-framing (Step 1+2+3 prior). End-to-end carry-through is
confirmed (+0.053 utility, MMR 4.4× over-selection). The thaw branch
`v0.7-thaw/forward-framing` sits gated on a 50-pair human oracle. **Open
question this brief addresses:** what's the next lever, before we re-merge?

## 1. Arxiv scan — what 2025–26 shows works

Three findings shape the rest of this brief:

1. **Decomposition workflows roughly double novelty vs reflection.**
   ["Evaluating Novelty in AI-Generated Research Plans Using Multi-Workflow
   LLM Pipelines"](https://arxiv.org/html/2601.09714) (2026) ran exactly the
   workflow-comparison we did, at larger scale, on 30 proposals × 5
   workflows. Their numbers (5-point Likert, novelty axis):

   | Workflow | Novelty | Feasibility | Impact |
   |---|---|---|---|
   | Reflection | 2.17 | 2.50 | 2.33 |
   | Sakana AI v2 | 3.50 | 2.67 | 3.83 |
   | GPT Deep Research (decomposition) | 3.83 | 3.00 | 4.00 |
   | Google Co-Scientist (decomposition + multi-agent) | 4.17 | 3.00 | 3.83 |
   | Gemini 3 Pro (long-context) | 4.17 | 2.83 | 3.33 |

   Reflection is the *worst* among real workflows. Our Workflow C is
   reflection-based. **Our next lever is decomposition.**

2. **Si et al. 2024 (Stanford 100-researcher blind study)** —
   [arxiv 2409.04109](https://arxiv.org/pdf/2409.04109) — established the
   bar: LLM ideas already beat humans on novelty (p < 0.05) but lose on
   feasibility. The frontier today is making LLM ideas *executable*, not
   more novel.

3. **HindSight protocol** —
   [arxiv 2603.15164](https://arxiv.org/pdf/2603.15164) — pre-cut corpus at
   a date, generate from "before" only, match the generated ideas against
   "after" papers via SPECTER embedding cosine. Our corpus supports this
   (133 future papers in `arxiv-reasoning-v0.7-540p-thaw1` after a 2025-01
   cutoff).

   Other useful pointers:
   [The Ideation–Execution Gap](https://arxiv.org/html/2506.20803v1) —
   ideation scores collapse once ideas are actually executed; novelty alone
   misleads.
   [ScholarEval](https://arxiv.org/pdf/2510.16234) — literature-grounded
   evaluation. We already do this (claims-grounded).
   [Chain of Ideas](https://arxiv.org/pdf/2410.13185) — agent-based ideation
   pattern that maps cleanly onto an aigraph anomaly→hypothesis stage.

## 2. TF-IDF HindSight is the wrong metric (verified empirically)

I built a stdlib TF-IDF + cosine implementation of HindSight first
([`scripts/score_hindsight.py`](../scripts/score_hindsight.py)) because the
aigraph venv has no torch/sentence-transformers on Python 3.14. Result:

| population | n | mean best-sim | median | frac > 0.35 |
|---|---:|---:|---:|---:|
| frozen (back-explanation) | 300 | 0.160 ± 0.062 | 0.145 | 0.7% |
| forward (Workflow C) | 24 | 0.124 ± 0.060 | 0.101 | 0.0% |

Δmean = **-0.036, permutation p = 0.005**. Forward hyps score *lower* on
TF-IDF HindSight — the opposite of the Kimi rubric finding and the opposite
of frozen-scorer utility.

**Diagnosis: TF-IDF rewards jargon density, not idea match.** Inspecting the
top-5 matches per population shows neither population's "matches" share
actual mechanism content with the so-called matching future paper — frozen
just shares more stock ML vocabulary ("training data composition",
"evaluation metric", "moderator") which is broadly distributed across the
future corpus too. Forward hyps strip that boilerplate by design (the whole
point), so their TF-IDF cosine drops. The signal is noise on both sides;
frozen just has marginally more noise.

**Implication:** for a useful label-free quality oracle we need either
neural embeddings (SPECTER or sentence-transformers MiniLM — blocked by
Python 3.14 wheels) or LLM-judge matching ("does this hyp test the same
idea as one of these 5 candidate future papers?"). Recorded as a permanent
artefact: [`artifacts/atlas_test/hindsight_scores.jsonl`](../artifacts/atlas_test/hindsight_scores.jsonl)
(324 hyps × {best_sim, best_paper_id}), useful for ablation when the
metric is fixed.

## 3. Recommended next experiment — Workflow D (Decomposition)

Workflow C (forward-framed reflect) is the current production winner from
the prior A/B (rubric_mean 3.75/5, vs A=1.06, B=4.00). The literature
(§1.1) puts decomposition at ~+1.8 on the same 5-point axis. Concrete spec:

### Pipeline

For each anomaly:
1. **Mechanism brainstorm** — single Kimi call, no JSON schema, prompt
   the model to list 4–5 candidate causal mechanisms in 1 line each. No
   commitment to format, no schema. Cheap and high-divergence.
2. **Mechanism pick + sharpen** — second Kimi call: present the 4–5
   candidates, ask the model to pick the strongest single-variable causal
   story and name the mechanism precisely (one sentence, no boilerplate).
3. **Test design** — third Kimi call: given the sharpened mechanism,
   design the experiment that varies *one* thing and would discriminate
   this mechanism from the alternatives in §1.
4. **Assemble** — fourth Kimi call: emit structured Hypothesis JSON
   (existing schema) using §2's mechanism and §3's test. This step is the
   only one that must produce schema-clean output, so it can be temperature
   ≈ 0.1 while §1 is at 0.6–0.8.

### Why this should win
- §1 forces *divergence first* (4–5 mechanisms before commitment), which is
  what the reflection workflow's single-shot generation can't do.
- §2 forces *single-mechanism commitment*, which addresses the "EXACTLY 3
  near-dups" failure mode at the structural level (we already loosened the
  prompt rule on the thaw branch; this is the structural fix).
- §3 separates "what's the idea" from "how do I write a JSON schema",
  removing schema-pressure from the creative step.

### Evaluation
- Re-run the 16-anomaly subset with Workflows {A=baseline, C=reflect,
  D=decomposition}; Kimi-judge rubric (forward_looking, named_mechanism,
  single_variable_test, specific_scope, overall) — same protocol as the
  prior A/B/C experiment.
- Also: pass D outputs through `score_forward_vs_frozen.py` to measure
  carry-through (utility / MMR) on the merged pool.
- LLM-judge HindSight as a stretch goal once the rubric A/B confirms D wins.

### Cost
- ~4 LLM calls per anomaly vs C's 2 (gen + reflect) → 2× the LLM cost.
- At 16 anomalies × 4 calls × ~3K tokens ≈ 200K tokens — small.

## 4. Scaffolded but not yet run

[`scripts/gen_workflow_d.py`](../scripts/gen_workflow_d.py) — local-side
generator skeleton; mirrors the §3 stages. Ready to scp+run on
`8.208.118.99` against Kimi-K2.6.

Decision pending: run Workflow D on the 16-anomaly subset (cheap, ~10 min),
or first invest in neural-embedding HindSight (better oracle, ~30 min plus
torch install risk on 3.14).

## 5. What's explicitly NOT recommended this round
- **Memento.** Per the diagnosis, doesn't address generation-time root
  causes; risks Goodharting on the current scorer.
- **EXACTLY-N retuning past N=1–3.** Already done on the thaw branch.
- **Scorer reweighting in `scoring.py`.** Frozen; needs thaw + oracle.
- **scope_overreach fix.** Verified dead-end; only feeds explainability.

## Sources

- [arxiv:2601.09714 — Evaluating Novelty in AI-Generated Research Plans](https://arxiv.org/html/2601.09714) — the 5-workflow comparison
- [arxiv:2409.04109 — Can LLMs Generate Novel Research Ideas?](https://arxiv.org/pdf/2409.04109) — Si et al. 100-researcher blind study
- [arxiv:2603.15164 — HindSight](https://arxiv.org/pdf/2603.15164) — temporal-holdout idea evaluation
- [arxiv:2506.20803 — Ideation–Execution Gap](https://arxiv.org/html/2506.20803v1) — novelty alone misleads
- [arxiv:2510.16234 — ScholarEval](https://arxiv.org/pdf/2510.16234) — literature-grounded evaluation
- [arxiv:2410.13185 — Chain of Ideas](https://arxiv.org/pdf/2410.13185) — agent-based ideation
- [arxiv:2504.05496 — Survey on Hypothesis Generation for Scientific Discovery](https://arxiv.org/html/2504.05496v1) — general taxonomy
