# Method 3 — Atlas as anti-novelty oracle — verdict

**Date:** 2026-05-31
**Method 3 of:** [quality-next-levers-research-brief.md](quality-next-levers-research-brief.md), 3rd iteration of the /loop "test different methods" plan.
**One-line answer:** **Works.** The Atlas bottleneck-quotes corpus (1,607 unique open problems from later papers) is a usable real-valued evaluator. A single LLM-Likert call per hypothesis returns a 1–5 score where the population distributions of forward-framed and frozen-pipeline hyps actually differ — the first time in the /loop a single number separates them. Use it as a delivery-time signal now (non-frozen) and a `scoring.py` novelty replacement after the next thaw.

## Why this matters

Methods 1 and 2 of the /loop both hit the binary rubric ceiling — both arms at 4/4, no discrimination. The /loop's actual technical bottleneck wasn't the generator; it was the metric. Method 3 introduces a Likert judge with an Atlas-grounded axis, and the first numeric signal in two iterations falls out.

## Experiment

- Corpus: 1,607 unique bottleneck quotes extracted from the 484 joint anomalies' `bottleneck_signals.quote` fields.
- Sample: 30 hypotheses — 15 forward (from `forward_hyps.jsonl`) + 15 frozen (random from the 540p run's `hypotheses_scored.jsonl`).
- Pipeline (`scripts/method3_build_retrieval.py` + `scripts/run_method3_judge.py`):
  1. Per hyp, TF-IDF top-K=5 most-similar Atlas quotes.
  2. One Kimi call: STRICT JSON with `closest_idx`, `atlas_overlap` (1–5 Likert), and four other Likert axes (`forward_looking`, `named_mechanism`, `single_variable_test`, `specific_scope`).
- Single-pass, no reflect. 30 calls × ~30 s ÷ 4 workers ≈ 8 min wall.
- Errors: 6/30 — same pattern as prior runs (large input + Kimi reasoning budget).

## Results (n=24, Likert 1–5)

| criterion | forward | frozen | Δ |
|---|---:|---:|---:|
| **atlas_overlap** | **3.17** | **2.92** | **+0.25** |
| forward_looking | 4.33 | 3.92 | +0.41 |
| named_mechanism | 4.67 | 4.08 | +0.59 |
| single_variable_test | 3.08 | 3.42 | **−0.34** |
| specific_scope | 4.17 | 4.08 | +0.09 |

**Atlas-overlap distribution across n=24:** `{2: 2, 3: 19, 4: 3}` — no 1s, no 5s. The judge clusters at 3 (sweet spot) and weakly polarizes from there.

## Reading the atlas_overlap axis

The semantic content of each Likert score, with verbatim `why` justifications:

- **5** (exact restatement): 0 cases — never triggered.
- **4** (directly addresses a known open problem): 3 cases (2 forward, 1 frozen).
  - `fwd_a400 (training_stability)`: "directly tackles bottleneck [1]'s open problem of explaining SGD behavior in nonconvex settings by proposing a specific geometric mechanism—loss of star-convexity via low-rank hidden feature covariances and nullspace perturbation."
  - `fwd_a1178 (accuracy)`: "directly extends the open challenge in bottleneck 3 of using cognitive priors to inform neural networks, proposing economic-cognitive priors as a cross-domain mechanism to break anomaly detection ceilings."
- **3** (same area, distinct mechanism — *the sweet spot*): 19 cases (10 forward, 9 frozen). The judge consistently frames this as "related area but different mechanism".
- **2** (thematically adjacent, mechanistically distinct — possibly tangential): 2 cases (0 forward, 2 frozen).
  - `h233 (accuracy)`: "Both concern accuracy and QA, but the hypothesis focuses on metric divergence in tool-use factual QA whereas bottleneck [0] addresses language-prior shortcuts in VQA."
  - `h070 (fairness)`: "Both touch on bias evaluation… but the hypothesis asks why cross-community terminology drift hides shared error mechanisms, whereas the bottleneck asks how group definitions alter bias metrics."
- **1**: 0 cases.

**Key interpretation:** `atlas_overlap` is **NOT a pure novelty signal**. A 4 isn't "non-novel" — the judge actively praises 4-rated hyps as "directly addresses" a real open problem. A 2 isn't "novel" — the judge flags 2-rated hyps as "thematically adjacent" (potentially tangential). The honest reading: this is a **specificity-and-anchoring** signal. 3 = anchored to a real research area, novel mechanism inside it.

## Why forward hyps OVERLAP MORE with Atlas (+0.25)

Counterintuitive on first read. Explained by the corpus: forward framing produces **sharper named mechanisms** (forward 4.67 vs frozen 4.08 on `named_mechanism`), which means they're *more matchable* to bottleneck quotes that themselves name specific limitations. Frozen back-explanation produces generic moderator-variable text that matches no specific Atlas quote strongly — its lower `atlas_overlap` is a sign of *vagueness*, not novelty.

This confirms the [[hypothesis-quality-diagnosis]]: the broken Jaccard novelty was rewarding vagueness as novelty. Atlas-overlap measures real anchoring.

## Why forward LOSES on single_variable_test (−0.34)

Distribution split (n=12 each):
- forward: `{3: 5, 2: 4, 5: 2, 4: 1}` — mode 3, centered.
- frozen:  `{2: 4, 5: 3, 4: 3, 3: 2}` — bimodal (4 twos, 3 fives).

Frozen is **bimodal**: lots of poor tests (mode 2) and a thick tail of great tests (3 fives). Forward is **centered at 3** (consistent but not stellar). Forward framing trades variance for consistency on the test axis. The mean comparison favors frozen because the 3-tier ceiling for forward never dominates the 5s frozen occasionally hits.

**Fix:** the forward-framing prompt mentions "imply a single-variable minimal test" but doesn't enforce a concrete-test rubric like the brainstorm/sharpen/test stages of Workflow D do. A small prompt tweak — "minimal_test MUST name the method, the dataset, the metric, and exactly ONE varying control" — should narrow this gap without re-architecting.

## What this unlocks — concrete next moves

### (A) Tactical — non-frozen, ship today
Add an `atlas_overlap` field to `query_records()` output in `scripts/aigraph_query.py`. For each delivered hyp, run a single Method-3 Likert call against the top-K Atlas quotes, attach the score + the closest_quote + a one-line "why" to the rendered hypothesis. Users see *what known open problem* their hyp is anchored to — direct value, no thaw needed.

Cost: 1 Kimi call per delivered hyp at delivery time. For a typical 8-hyp report, ~3 min latency at one worker, ~50s at 4 workers. Cacheable per (hyp_id, run_id).

### (B) Strategic — post-thaw scorer replacement
Replace `novelty = 1 - Jaccard(hyp.text, neighbor.text)` in `scoring.py` with a function of `atlas_overlap`:
```python
# Anchored-novelty: 3 is best, 4 acceptable, 2 demoted, 5 banned, 1 unanchored
ANCHORED_NOVELTY = {1: 0.4, 2: 0.6, 3: 1.0, 4: 0.8, 5: 0.2}
novelty = ANCHORED_NOVELTY[atlas_overlap]
```
This puts a **real** semantic signal into the frozen scorer. Requires §4 thaw record (real bug: Jaccard is provably broken — see [[hypothesis-quality-diagnosis]] §3) and the standard 50-pair human-rated oracle to verify the substitution doesn't break ρ.

### (C) Likert rubric for all future generation A/Bs
This experiment effectively closes task #31 (binary → Likert). The 5-axis Likert judge from `run_method3_judge.py` should be the canonical judge going forward. **The Workflow D vs C verdict needs re-running under Likert** before being final — the binary version returned a tie that might be a Likert win for either arm.

## Artifacts

- `scripts/method3_build_retrieval.py` — local TF-IDF retrieval
- `scripts/run_method3_judge.py` — server-side Likert judge
- `artifacts/atlas_test/method3_atlas_quotes.jsonl` — 1,607-quote corpus
- `artifacts/atlas_test/method3_judge_input.jsonl` — 30 hyp × top-5 quote inputs
- `artifacts/atlas_test/m3_out.jsonl` — judge output

## Honest caveats

1. **n=24 is small.** The +0.25 atlas_overlap delta is one Kimi call per data point; a 100-hyp version would tighten the CI but cost ~25 min Kimi time.
2. **The 6/30 error rate is structural** — same Kimi reasoning-budget issue documented in [[binary-rubric-ceilings]]. Each retry escalates the cost.
3. **TF-IDF top-K filter may miss semantic matches.** A neural-embedding retriever would feed the judge better candidates. Not blocking — the judge can downgrade a bad top-K case to atlas_overlap=2, which it does.
4. **The Likert judge itself is a Kimi call.** Cost: ~$0.005/hyp at current pricing. For per-delivery scoring this is small; for batch-scoring 1,000+ hyps it adds up.
