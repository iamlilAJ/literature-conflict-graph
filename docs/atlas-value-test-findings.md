# Atlas → aigraph: empirical value test (bottleneck_open_q_alignment)

**Date:** 2026-05-24
**Cohort:** `validation-v1-primary` (1790 NeurIPS/ICML/ICLR 2018-2020 papers, 4211 claims)
**Judge model:** Kimi-K2.6 (OMC LiteLLM endpoint; this Mac can't reach the
LLM, so candidates were built locally and judged on the remote box).
**Artifacts:** `artifacts/atlas_test/{val1_candidates,val1_judged,val1_phase2}.jsonl`,
script `scripts/atlas_bottleneck_align_test.py` (+ remote judges in `/tmp`).
**Builds on:** the J0→J2-prototype recon (`recon/atlas-aigraph-fit` branch,
`docs/atlas-aigraph-fit-recon.md`), whose 30-sample Q2 found 53% complementary.

## What was tested

The recon's single most-supported joint anomaly, `bottleneck_open_q_alignment`:
align Atlas's **third-party** bottleneck (paper B asserts paper P's weakness,
from `bottleneck_json` on evolution edges) with aigraph's **first-party**
weakness (P's own negative-direction + `limitation` claims). Two phases:

- **Phase 1 — does the signal exist & is it additive?** Count candidates,
  then have the LLM 4-class each (same_signal / complementary / unrelated /
  contradictory).
- **Phase 2 — does it improve the deliverable?** Blind A/B: generate a
  hypothesis from aigraph-weakness-only (baseline) vs aigraph+Atlas-bottleneck
  (joint); judge which is more specific / testable / grounded.

## Phase 1 result — STRONG GO

- **70,997** Atlas inbound bottleneck edges land on the cohort.
- **484 candidate papers** have BOTH an Atlas third-party bottleneck AND an
  aigraph first-party weakness (27% of the 1790 papers). This is the
  `bottleneck_open_q_alignment` firing count — far above the recon's "≥5
  candidates" bar and it kills the sparsity worry from the 540p run.
- Of **353 valid** LLM judgments (129 ERROR = endpoint timeouts on the slow
  reasoning calls, 2 parse errors — infra artifacts, not signal):

  | label | n | % of valid |
  |---|---:|---:|
  | **complementary** | 306 | **86.7%** |
  | unrelated | 26 | 7.4% |
  | same_signal | 20 | 5.7% |
  | contradictory | 1 | 0.3% |

  Atlas's third-party bottleneck is overwhelmingly **additive** to aigraph's
  first-party weakness — it surfaces *new, compatible* weakness dimensions
  aigraph misses, almost never redundant (5.7%) or conflicting (0.3%). The
  recon's 30-sample 53% was a floor; at scale it's 87%.

## Phase 2 result — naive context-injection gives NO lift

Blind A/B on 24 complementary papers (23 valid):

| winner | n | % |
|---|---:|---:|
| baseline (aigraph-only) | 11 | 47.8% |
| joint (aigraph + Atlas bottleneck) | 9 | 39.1% |
| tie | 3 | 13.0% |

Joint did **not** beat baseline (slight baseline edge, within noise at n=23),
and ≥1 "joint win" was degenerate (empty baseline from a generation timeout),
so the real joint edge is even weaker. Reading the pairs: appending the
third-party bottleneck text pushed the generator toward **broader, more
ambitious** hypotheses (e.g. "adversarial domain-confusion latent space")
that the judge found **less tightly grounded** than the focused baseline,
which stuck to the paper's own specific violated assumption.

## Conclusion — where Atlas's value actually is

**Atlas's bottleneck data is a candidate-SELECTION signal, not a
prompt-padding signal.**

- ✅ **Use it to decide WHAT to hypothesize about.** `bottleneck_open_q_alignment`
  as a new anomaly type is well-supported: 484 high-quality candidates, 87%
  complementary. It expands aigraph's weakness-grounded anomaly coverage with
  a third-party signal aigraph structurally lacks (no `contradicts`/bottleneck
  edges of its own). This is the real "效果" lift — better anomalies in, before
  any generation.
- ❌ **Don't naively append bottleneck text to the generation prompt.** No
  measurable hypothesis-quality gain; if anything it dilutes specificity.

## Recommended next steps

1. **Implement `bottleneck_open_q_alignment` as a real detector** in a new
   `joint_anomalies.py` (no frozen-module changes), emitting the 484-style
   candidates as anomalies with the structured `{dimension, severity, quote}`
   from `bottleneck_json` attached. This is the J2 prototype the recon
   recommended, now empirically justified.
2. **Smarter Phase-2 integration before concluding J3 dead:** the test used
   raw text-append. A targeted framing ("propose a hypothesis that *resolves*
   the third-party-identified bottleneck on dimension D") or using the
   structured severity/dimension fields might beat baseline where free-text
   append did not. Re-run the A/B with that framing before deciding whether
   Atlas context helps generation.
3. Re-judge the 129 Phase-1 ERROR rows (endpoint timeouts) to tighten the
   distribution — though 353 valid is already decisive.

## Post-implementation test (2026-05-27): does integration make output BETTER?

After implementing the detector (`joint_anomalies.py`), a head-to-head on
val1-primary. Artifacts: `artifacts/atlas_test/testB_*.{json,jsonl}`,
script `scripts/atlas_bottleneck_align_test.py` + remote judges.

**Test A — coverage (0 LLM, decisive):**
- baseline (8 frozen types): 1423 anomalies covering **638 papers (36%)**.
- joint: 484 anomalies → **154 net-new papers** (zero baseline anomaly) +
  330 papers augmented. Run coverage **36% → 44%**.
- ✅ Integration clearly adds *more* weakness-grounded candidates.

**Test B — quality A/B (Kimi-K2.6, model-controlled, blind):**
- **B1 (net-new papers): joint-grounded hyp vs generic ungrounded hyp.**
  Valid n=14 (26/40 lost to LLM timeouts): **generic 9, joint 5** — the
  Atlas-grounded hypothesis LOST to a generic "advance this paper" prompt.
- **B2 (overlap papers): joint vs baseline anomaly.** Valid n=23 (17 errors):
  **baseline 13, joint 10** — no lift (consistent with the earlier phase-2).

**Root cause of the quality miss (definitive): broken central_question.**
**40/40 B1 and 35/40 B2 joint CQs read "Paper P studies _other_ on …"** —
the CQ is templated from the weakness claim's `method` field, which is the
degenerate `"other"`/empty value (the same extractor-hygiene problem Thaw #2
only partly fixed). So the joint hypotheses were handicapped by clunky
framing, not by the Atlas signal itself — the clean `bottleneck_json`
(dimension/severity/quote) was in the prompt but the CQ spine was contaminated.

### CQ fix + clean re-test (2026-05-28)

The `central_question` was rebuilt around the clean bottleneck
(`{dimension, quote}`) as the spine, dropping the degenerate `method` slot
(commit pending). Re-ran B1/B2 with a 420s timeout (Kimi's reasoning was
timing out 70% of calls at 150s). Errors dropped to ~2%, giving clean
samples:

| arm | before fix | **after fix (clean, larger n)** |
|---|---|---|
| **B1** net-new: joint vs generic | generic 9 / joint 5 (n=14) | **joint 22 / generic 22 / tie 5 (n=49)** — dead even |
| **B2** overlap: joint vs baseline | baseline 13 / joint 10 (n=23) | **joint 9 / baseline 9 / tie 1 (n=19)** — dead even |

The fix erased the quality penalty: the broken "studies _other_ on …" CQ was
the entire reason joint lost the first time. With clean CQs, joint hypotheses
are **statistically tied** with both a generic prompt (net-new) and the
baseline anomaly (overlap).

### FINAL VERDICT: integration is a net positive, but modest
- **Coverage — better (clear):** +154 net-new papers (24% more covered),
  36% → 44% run coverage. Real, bankable.
- **Per-hypothesis quality — no change:** joint hypotheses tie generic
  (net-new, n=49) and tie baseline (overlap, n=19). Atlas grounding neither
  raises nor lowers per-idea quality once the CQ is clean.
- **Net effect:** integration gives you MORE weakness-grounded ideas — on
  papers the frozen pipeline produces nothing for — at the SAME quality bar.
  It does NOT produce higher-quality ideas per se. "More, equally good," not
  "better."
- Phase-1's 87%-complementary signal is real (the third-party bottleneck IS a
  distinct dimension) but does not, by itself, translate into measurably
  better generated hypotheses.

Caveat: single judge (Kimi K2.6, also the generator); the production model
(gpt-5.4) endpoint is dead, so this couldn't use it. Direction is consistent
across 4 runs though.

## Caveats

- Judge = Kimi-K2.6 (reasoning model; needed `max_tokens≥3000` or reasoning
  ate the budget → the first run's 99% parse-errors). Single-judge, no human
  validation of the 4-class labels.
- Phase 2 n=23 is small; baseline-vs-joint is within noise — the honest claim
  is "no lift," not "Atlas hurts."
- "First-party weakness" = negative-direction + `limitation` claims as a proxy
  for open-questions (not extracted on this run), same proxy the recon used.
