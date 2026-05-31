# Generation-workflow experiment — does changing HOW we generate fix the quality?

**Date:** 2026-05-30
**Cohort:** N=16 anomalies sampled from `validation-v1-primary` (community_disconnect-heavy, matching the run distribution).
**Model (gen + judge):** Kimi-K2.6 via the OMC LiteLLM endpoint, on the remote box (this Mac can't reach an LLM endpoint; gpt-5.4 is dead).
**Artifacts:** `artifacts/atlas_test/genexp16_out.jsonl` (raw), `genexp_input.json`, remote script `/tmp/genexp_remote.py`.

## Design — model-controlled, blind rubric A/B/C

For each of 16 anomalies (same `central_question` + claim set), three model-controlled generations and a blind rubric:

- **Arm A** — frozen back-explanation framing ("output ONE hypothesis that explains WHY these papers conflict — the moderator or condition that reconciles the contradiction").
- **Arm B** — forward-looking single-shot ("propose ONE specific, FORWARD-LOOKING, testable research hypothesis — a new mechanism or direction to investigate, NOT merely an explanation").
- **Arm C** — Arm B + one reflect/revise pass (Memento workflow lesson: critique on 4 axes, then output an improved version).

**Blind rubric (4 binary criteria, scored on each arm's final hypothesis):**
forward_looking, named_mechanism, single_variable_test, specific_scope → composite 0–4.
**Plus headline blind A-vs-C pairwise** (randomized X/Y order).

This isolates two factors: (a) the **framing** change A→B (the diagnosed master cause #1: back-explanation contract) and (b) the **reflection loop** B→C (the Memento workflow lesson).

## Result — decisive: framing is the win; reflection is a slight regression

16/16 completed, 0 errors.

| Arm | composite mean (out of 4) | reading |
|---|---:|---|
| **A** frozen back-explanation | **1.06** | fails ~3 of 4 criteria on average |
| **B** forward-looking single-shot | **4.00** | perfect on every criterion, every anomaly |
| **C** forward + reflect (Memento) | **3.75** | slight regression — reflect over-edits |

**Blind pairwise A vs C: C wins 16/16 (100%).** Even with the slight reflect regression, the full improved workflow clobbers the frozen baseline.

### Per-criterion breakdown — where exactly the frozen generator fails

| criterion | A | B | Δ |
|---|---:|---:|---:|
| forward_looking | **0.06** | 1.00 | +0.94 |
| single_variable_test | **0.06** | 1.00 | +0.94 |
| specific_scope | 0.44 | 1.00 | +0.56 |
| named_mechanism | 0.50 | 1.00 | +0.50 |

The frozen generator is **structurally caged** into reactive explanation: it almost never (1/16) proposes a forward-looking direction or a single-variable testable experiment. Mechanism and scope are roughly 50/50.

### Concrete example (anomaly a900, community_disconnect)

**A (frozen, 0/4):** "The conflict reflects a structural divide: private top-k selection requires explicit, finite candidate rankings… while private parameter estimation relies on implicit asymptotic bounds. A shared mechanism is therefore only v[ague]…" — explanatory, abstract, no test.

**B (forward, 4/4):** "The data-dependent candidate-restriction rule from private top-k selection — limiting outputs to domain elements strictly above the (k+1)-th value — causally compresses effective sensitivity by discarding low-signal indices, and transplanting this rule as a pre-estimation filter on parameter coordinates can unify reliability…" — concrete mechanism, transferable, testable.

**C (forward + reflect, 4/4):** "Applying a data-dependent hard-threshold pre-filter — analogous to the top-k rule that discards entries below the (k+1)-th order statistic — to private sparse Ising model estimation reduces effective ℓ₀-sensitivity by bounding the support set prior to noise injection." — sharper still, narrower scope.

## Interpretation

1. **The diagnosis's master cause (#1 back-explanation contract) is confirmed empirically and is the only thing that matters.** The framing change A→B is +2.94 composite points and 100% pairwise win — every quality gain comes from dropping the "explain why they conflict" framing in favour of "propose a forward-looking testable direction."
2. **EXACTLY-3 was not the bottleneck** — single-best forward-looking already wins decisively without dropping the count separately. (Dropping it is still desirable to avoid the 3 near-dup frames; the query-layer cap from commit 68f3aa5 already handles the delivered-set side.)
3. **Reflection is a slight regression here (4.00 → 3.75).** Kimi's revise occasionally over-edits and loses a criterion (e.g. specificity for breadth). The Memento *workflow lesson* survives in spirit (the gain came from the *framing*, the most upstream change), but the reflect step itself doesn't help on top of a strong single-shot. Don't ship it.
4. **The frozen generator's forward_looking = 0.06/1 and single_variable_test = 0.06/1** is the cleanest possible signal that the frozen contract is structurally producing reactive, untestable output at scale — *exactly* the kind of evidence the freeze §4 anticipates ("predictor fundamentally broken").

## Recommendation

1. **Open a `v0.7-thaw/forward-framing` branch off the `v0.7-frozen` tag.** Rewrite `llm_hypotheses.py`'s framing rules to the forward-looking prompt (drop "explain why they conflict"; require a named causal mechanism + single-variable minimal test + specific scope). Keep schema; drop EXACTLY-3 → 1–3.
2. **Build the human-rated quality oracle** (50 hypothesis pairs blind-ranked by 2 reviewers using the same 4-criterion rubric) to confirm the Kimi-judge result transfers to human judgment, *before* merging.
3. **Skip the reflect step** in this round — empirically doesn't help; postpone Memento-style iteration to a separate experiment with a stronger judge model.
4. **Merge via a §7 thaw record**: cite this experiment's A=1.06 vs B=4.00 + the oracle's confirmation as the §4 condition-1 evidence ("predictor fundamentally broken on forward_looking + single_variable_test criteria at N=16 with composite mean 1.06/4").

## Honest limitations

- Single judge (Kimi K2.6, the same model that generated). Human-oracle confirmation is the precondition for merging.
- N=16, community_disconnect-heavy. Useful as a first decisive signal but a wider type-stratified set (impact_conflict, metric_mismatch, evidence_gap) would tighten the case before the merge.
- Quality criteria are binary; gradations would surface subtler differences.
- The reflect arm used Kimi for both gen and revise; a stronger reviser (Claude/GPT-class) might add value. Don't conclude "reflection is bad," only "this specific Kimi reflect step doesn't help here."
