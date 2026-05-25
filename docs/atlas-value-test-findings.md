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

## Caveats

- Judge = Kimi-K2.6 (reasoning model; needed `max_tokens≥3000` or reasoning
  ate the budget → the first run's 99% parse-errors). Single-judge, no human
  validation of the 4-class labels.
- Phase 2 n=23 is small; baseline-vs-joint is within noise — the honest claim
  is "no lift," not "Atlas hurts."
- "First-party weakness" = negative-direction + `limitation` claims as a proxy
  for open-questions (not extracted on this run), same proxy the recon used.
