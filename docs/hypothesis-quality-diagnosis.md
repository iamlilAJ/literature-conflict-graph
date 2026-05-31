# Why aigraph's hypotheses aren't good enough — and whether the Memento workflow fixes it

**Date:** 2026-05-29
**Method:** 6-agent diagnosis workflow (4 parallel readers → synthesis → adversarial critic) + independent ground-truth verification of every disputed numeric claim. Cohort: `validation-v1-primary` (184 hyps = 61 anomalies × 3).
**One-line answer:** The hypotheses are weak because the generator is structurally **caged into back-explaining pre-detected conflicts** with a hard-coded 3-variant moderator template and a fit-not-quality scorer — all in *frozen* code. The Memento workflow does **not** address this (it's an unwired user-feedback *recall* layer, unproven for quality) and would **backfire** if bolted onto the current ranking. Fix the generator + scorer first; stage Memento later, narrowly, gated.

> **Provenance note:** the diagnosis workflow's synthesis contained several **fabricated/misattributed numbers** (caught by the adversarial critic, then verified by me). Corrections are folded in below. Trust this doc's numbers, not the raw synthesis.

## Root causes (verified, strongest first)

1. **Back-explanation task contract (master cause).** The generator's job is "explain why these 2–3 papers conflict." 184/184 hyps are bound to `explains_claims`; output is reactive conflict-rationalization, not forward research directions. Confirmed in `llm_hypotheses.py` (the dominant framings literally instruct "back-explanation of why the contradiction exists"). Tell: even the BRIDGE/COMMUNITY framings that *do* ask for forward-looking ideas still produce almost none → the cage is the upstream anomaly structure + dominant reactive framing, not wording. **Memento: does NOT fix** (frozen prompt contract, not memory).

2. **Hard-coded "EXACTLY 3 per anomaly" off a fixed moderator menu.** `_SHARED_RULES` (llm_hypotheses.py:33) forces exactly 3; the 3 are near-duplicate reasoning frames differing only by which moderator they pick (firsthand: h001/h002/h003 on anomaly a145 = geometry / protocol / history-window). Lexically distinct (Jaccard ~0.06) but structurally identical. **Memento: does NOT fix** (prompt constant).

3. **Scorer/ranking measures fit + token-overlap, not scientific quality — and `scope_overreach` is an inverted incentive.** `scoring.py` weights fit-aligned terms ~0.60 vs novelty ~0.12; "novelty" = 1−Jaccard (fresh *words*, not ideas). `influence.py` `scope_overreach` returns **0.0 risk for empty scope** (best) and penalizes specific scopes that don't canonical-string-match — a real deterministic bug. *Verified impact:* only 1/184 hyps has empty scope (so no mass collapse), but it gets an above-average boost (ranks #4, 0.578 vs 0.352 mean), and a degenerate self-conflict (`planning/planning`) ranks #2. **Memento: does NOT fix**, and **actively backfires** if its retrieval key is this score (Goodhart).

4. **Fully stateless generation** — no feedback/critique/refine loop, no exemplars; temp 0.2 + rigid 8-field schema compress variance. **Memento: this is the one gap CBR is designed for — but unproven for quality (see below).**

### Corrected claims (synthesis was wrong; verified by me)
- ❌ "vacuous h175 ranks #1" → **#1 is h121** (real scope); h175 ranks **#4**.
- ❌ "novelty inert, is_novel=None for all 184" → **is_novel = 134 True / 40 None / 10 False** (live).
- ❌ "ranking collapses to short scope strings" → no length gradient; only 1 empty-scope hyp exists.
- ⚠️ Atlas A/B "proven causal / joint 22-22 / lose 9-5" → the raw A/B was **54% errors**, low-n; downgrade to "noisy suggestion that CQ-framing matters, context-padding doesn't." (Run conflation: 540p=300 hyps vs val1=184.)

## What the Memento work actually is

`aigraph_mem/src/aigraph/memory_feedback.py` (404 LOC) = a **user-feedback recall layer**: stores `FeedbackRecord(hypothesis_id, verdict[good/bad/important], reason)` into `memento_v4` causal memory (domain-partitioned sessions, BM25 + exact-id rerank). **Proven:** recall is reliable under noise (4/4 real smoke, Kimi-K2.6, Gate 0–3, non-regressive). **Explicitly NOT proven (its own report §6):** that feedback memory improves idea quality. **Not wired into any `.py`** in the live pipeline. Its seed data even encodes the disease: `h001=good (concrete top_k sweep)` vs `h005=bad (generic moderator, no mechanism)`.

## Verdict on "does the Memento workflow make hypotheses better?"

**No, not as a generation-time memory/context layer** — the root causes live in frozen prompts, weights, and a metric, none of which a store/recall layer touches; and the closest tested analog (Atlas context-padding) gave zero per-hyp lift. Memento is genuinely useful for **one** narrow, evidence-aligned thing: case-based **central-question re-templating** + **feedback-validated anomaly/candidate selection** — but that's unproven (no closed loop built) and dangerous if keyed on the current (broken) score.

## Recommendation (staged, gated)

0. **Fix `scope_overreach` first** (influence.py — outside the frozen v0.7 anomaly/generation contract, deterministic, no labels needed). It's an inverted incentive and the precondition for trusting any ranking or quality oracle.
1. **Fix the anomaly / central-question generator** (highest ROI, no Memento): kill degenerate X-on-X self-conflicts (`planning/planning`), re-template the 6 rigid CQ shapes. The only lever the Atlas A/B even hinted was causal.
2. **Loosen generation** (needs a v0.7 freeze thaw): drop "EXACTLY 3", add a forward-looking slot for non-bridge types, replace Jaccard-novelty with embedding/LLM-judge similarity.
3. **Build a quality oracle** (50–100 human-rated hyps + blind rubric: named mechanism? single-variable test? specific scope?) — hard precondition for any memory loop.
4. **Only then stage Memento**, narrowly: CQ-retemplating retriever + selection bias keyed on *validated* quality (never the current score), behind `AIGRAPH_ENABLE_MEMORY_TOOLS`, merged only on a blind no-memory-vs-memory A/B with a stated min valid-n; abort on no lift (exactly as the Atlas context test would have).
