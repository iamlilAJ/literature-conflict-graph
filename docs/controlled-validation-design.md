# Controlled Validation of Influence Prediction — Design Doc

A controlled experimental design for empirically validating that aigraph's
5-dimensional influence prediction (`predict_influence_phase1`) tracks
real scientific impact, using a multi-year cohort of papers from the top
AI venues (NeurIPS / ICML / ICLR).

> Status: design phase, post-pivot. Companion to
> [docs/influence-prediction-design.md](./influence-prediction-design.md)
> and [docs/intern-atlas-pivot.md](./intern-atlas-pivot.md).
> Last updated: v0.2 (May 8, 2026 — Intern-Atlas pivot amendments).
>
> **Reading order note:** §1–§14 are the original design. §3, §4, §5, §6,
> and §9 carry inline amendments showing what the Intern-Atlas pivot
> changed (May 1, 2026 release of `OpenRaiser/Intern-Atlas`). §16 at the
> bottom is the consolidated pivot record. If you only have time to read
> one section, read §16.

---

## 1. Why this exists

We have a 5-dim score (`community_reach`, `novelty`, `grounding_depth`,
`scope_overreach`, future `structural_impact`) and a smoke-grade
backtest on a hand-curated fixture. That's enough for sanity, not enough
to claim the score predicts influence in the real world.

A real validation needs three things the fixture does not provide:

1. **Sample size.** A fixture of ~30 hypotheses can't separate signal
   from noise at the per-dimension level. We saw this in PR #14: the
   `ρ_grounding_depth = -0.166` anomaly went un-explained because the
   denominator was too small to know if it was real or a single
   outlier.
2. **Independent ground truth.** The fixture's "ground truth" was
   constructed by the same humans who chose the cohort. Real validation
   needs an outcome variable whose value was determined *after* the
   prediction window closes — i.e., citations, follow-on work — that
   nobody on the project influenced.
3. **Controlled comparison across cohorts.** A single cohort cannot
   distinguish "the algorithm works" from "the algorithm overfits to
   one slice of literature". We need stratified samples across venues
   and years so we can decompose `ρ_total` into venue and year
   components.

Filling these turns the influence dimension from "we wrote the math
down and it correlates on 30 hand-picked examples" into "we predicted
1500 papers, locked the predictions, and citation-based outcomes
3+ years later confirm a stable ranking correlation". The latter is
the kind of evidence that should appear in a paper-grade validation
section; the former is unit-test-grade.

---

## 2. The core methodological challenge — 2023 LLM discontinuity

Any validation that uses **post-2022 citation outcomes** as ground
truth has to grapple with the fact that 2023 was a regime shift. The
ChatGPT (Nov 2022), GPT-4 (Mar 2023), and the LLM-instruction-tuning
boom inflated downstream citations of any paper that happened to be
LLM-adjacent — prompting, in-context learning, transformer training
tricks, retrieval-augmented methods, instruction tuning — by amounts
that have *nothing to do with the underlying scientific quality of the
hypothesis aigraph would have generated about that paper*.

Concretely: a 2021 paper introducing some prompt-engineering trick
might collect 2000 citations in 2023-2025 not because the trick was
substantively important, but because the LLM ecosystem that consumed
that trick exploded. A contemporaneous 2021 paper introducing an
equally important graph-neural-net method gets 200 citations in the
same window. Their difference is topic-shift dividend, not influence
prediction signal.

If we naïvely use `ρ(predicted_influence, cited_by_count_2023_2025)`
as our headline metric, we will measure two things superimposed —
"did our score predict scientific impact" and "did our score correlate
with whether the topic happened to become the hottest in the field".
The second confounds the first, and the second is also exactly the
thing aigraph does **not** try to predict (we score graph-internal
properties at submission time, not external topic-shift dynamics).

This doc is structured around mitigating that confound, not pretending
it isn't there.

---

## 3. Cohort design

### 3.1 Primary cohort — pre-LLM-boom (2018-2020)

```
Years      : 2018, 2019, 2020
Venues     : NeurIPS, ICML, ICLR
Sample size: 200 papers per (venue, year) cell
Total      : 9 cells × 200 = 1800 papers
Outcome window: 2020-2023 (3 years per paper)
```

Why this cohort: every paper's 3-year citation window closes **before
or at the start of** the 2023 LLM regime. Long-term citation patterns
have stabilized — these papers have been in the field for 5-7 years —
and the dominant signal in their citation count is "did the work
matter to the contemporaneous community", not "did the topic become
the next hype cycle".

This is the cohort the headline `ρ` should be computed on.

> **Pivot amendment (v0.2).** The actual cohort is built by
> `scripts/cohort_from_intern_atlas.py` from the Intern-Atlas
> `papers` parquet, not by OpenAlex fetch. After applying the quality
> filter (abstract ≥ 80 chars, ≥ 1 ID, non-null citation_count,
> venue_canonical ∈ {NeurIPS, ICML, ICLR}), seed=17 sampling at 200/cell
> yielded **1790 papers, not 1800** — ICLR 2018 only had 190 papers
> passing the filter (vs 531 raw). All other cells reached the 200
> target. This is well above the n=82 power threshold (§6.2), but
> per-cell bootstrap CIs for ICLR 2018 will be slightly wider — to be
> respected when reporting per-cell results, not pooled across.

### 3.2 Robustness cohort — peri- and post-boom (2020-2023)

```
Cohort A (peri-boom): 2020-2022 papers, 2022-2024 outcome
Cohort B (post-boom): 2022-2023 papers, 2024-2025 outcome
Same venue / sample size structure as primary.
```

Purpose: report `ρ_pre`, `ρ_peri`, `ρ_post` side by side. Three
possible patterns and what each means:

| Pattern | Interpretation |
|---|---|
| All three ρ stable and similar | Algorithm robust to topic shift — strongest claim |
| ρ_pre > ρ_peri > ρ_post (monotone decline) | Algorithm captures scientific quality, regime shift dominates outcome metric — confound is in ground truth, not predictor |
| ρ_pre and ρ_post stable, ρ_peri spikes/drops | Year-cohort effect; investigate the specific year |
| ρ_pre weak | Predictor doesn't work on its cleanest test — bigger problem than the LLM confound |

The multi-cohort report is itself a publishable finding regardless of
which pattern wins, because each pattern characterizes a different
true property of the predictor.

### 3.3 Stratified sampling within (venue, year) cells

For each cell, draw a random sample of 200 papers subject to filters:

- has English abstract ≥ 80 chars (originally specified as 80 words; the
  pivot tightened this to chars to match the Intern-Atlas parquet field)
- has ≥ 1 ID out of {arxiv_id, doi, openalex_id, s2_id} (originally ≥ 2;
  relaxed to 1 because Intern-Atlas paper rows typically carry only one
  ID — see comment in `cohort_from_intern_atlas.py`)
- non-null `citation_count` (will be the L1 outcome)
- `venue_canonical` matches the target set exactly

200 per cell sized so that, after filter losses, each cell has ≥ 150
usable papers — enough to compute per-cell `ρ` with a credible
confidence interval (see §6.2 for power analysis). The aigraph
claim-extractor and main-track filters from the original design are
applied **after** sampling, in §4.1 step 2, rather than gating the
cohort itself; this lets us report drop ratios per cohort honestly.

---

## 4. Pipeline

### 4.1 Per-paper prediction pipeline (training/prediction time)

For each paper in cohort:

1. **Load** abstract + metadata + citation_count from the Intern-Atlas
   parquet via `corpus.intern_atlas_loader` (planned, see
   intern-atlas-pivot.md §5.2). The original design fetched via
   OpenAlex + S2; the pivot replaces this with a parquet read that's
   strictly faster, free, and produces the same Paper model. OpenAlex
   fetch remains as a fallback path.
2. **Extract claims** via the existing claim-extraction LLM call.
3. **Build local graph** — papers + claims for the (venue, year)
   cell, plus all OpenAlex citations / co-citations between them.
4. **Detect anomalies** on the cell-level graph.
5. **Generate hypotheses** — both anomaly-driven (`llm_hypotheses`)
   and creator-mode (multi-grain `creator`).
6. **Score every hypothesis** with `predict_influence_phase1` →
   record per-paper a list of `InfluenceScore` objects, plus the
   max and mean rolled up to paper level.
7. **Lock predictions** — write `predictions_v1.jsonl` with hash of
   inputs and timestamp. Once written this file is read-only.

The "lock" step is not a code-style nicety. The whole point of the
backtest is that nobody should be able to look at the outcome data
before fixing the predictor. Treating `predictions_v1.jsonl` as
immutable is what gives the experiment its statistical credibility.

### 4.2 Per-paper ground-truth pipeline (evaluation time, run later)

For each paper:

1. **Pull engagement data**:
   - L1: directly from `papers.cited_by_count` and
     `papers.influential_citation_count` in the Intern-Atlas parquet.
     No external API call needed. The original OpenAlex citation pull
     is dropped.
   - L2: from `paper_evolution_edges` (Intern-Atlas), filtering rows
     where `paper_b_id` is the source paper and `evolution_relation` is
     in the engaged set (§5.2).
2. Compute three ground-truth tiers (§5).
3. Aggregate to (paper, hypothesis) and (paper, dim) levels.

These two pipelines are **separate scripts** with separate input
locks. The prediction pipeline does not see citation outcomes; the
ground-truth pipeline does not see predictions until both have
written their respective output files.

---

## 5. Ground truth — three tiers

Citation count is a noisy proxy for scientific impact. Rather than
defend a single metric we're going to report three, of increasing
cleanliness and decreasing scale.

### 5.1 L1 — citation-count proxy

Per-paper outcome: `cited_by_count` from the Intern-Atlas `papers`
parquet. We additionally report `influential_citation_count` (the
Semantic-Scholar-derived sub-count of citations that S2's classifier
flagged as influential — strictly better signal than raw count for
small extra cost).

- Cheapest. Available for all 1790 primary-cohort papers, free, no
  API call.
- Most confounded — counts hype-driven citations equally with
  substantive ones, especially for peri-/post-boom cohorts.
- Use for: stratified analyses, large-N robustness checks.

Both `cited_by_count` and `influential_citation_count` are reported as
separate ρ values in the headline table; the latter is the recommended
primary metric, the former for transparency. Original design assumed
fetching cite counts from OpenAlex per-paper post hoc; pivot removes
that step entirely.

### 5.2 L2 — engaged citation count (typed-edge engagement)

Per-paper outcome: count of incoming `paper_evolution_edges` whose
`evolution_relation` is in the **engaged set**:

```
engaged = {extends, improves, replaces, adapts}
```

These four are Intern-Atlas's `strong_causal` subset — they represent
genuine methodological lineage, not background citations. Background-
cite types (`compares`, `uses_component`, `background`) are excluded —
recon §Q5 confirmed `compares` alone is 58% of all edges, so without
this filter L2 collapses into noise.

- Computed for all 1790 primary-cohort papers from a single parquet
  scan. Cost: $0 (vs original design's ~$50/cohort for citing-paper
  claim extraction).
- Cleaner than L1: the 4-type filter is Intern-Atlas's own
  validated taxonomy (paper §3.2), not our heuristic, so this is
  somebody else's signal we are consuming, not our own circular logic.
- Use for: per-dimension `ρ_grounding_depth`, `ρ_structural_impact`
  validation — these dimensions specifically should track engaged
  citation, not raw cite count.

Original design specified "shared method or dataset entity" matching
between citing and cited papers, requiring per-citing-paper claim
extraction. Pivot uses Intern-Atlas's pre-computed typed edges
instead, which is both cheaper and (because their extraction was at
4M-edge scale with 99.8% completeness) higher-quality than the
ad-hoc heuristic the original design proposed.

### 5.3 L3 — manual annotation (top-50 + bottom-50)

Sample: take the top 50 and bottom 50 papers by predicted influence
score in the primary cohort. (Optionally extend to the middle 50.)

For each paper + its top-1 aigraph hypothesis:
- Read the aigraph hypothesis.
- Read the abstracts of the 5 highest-predicted-influence outcome
  papers in the citation graph.
- Annotate 0/1/2:
  - **0** — the hypothesis does not anticipate any actual follow-on
    direction.
  - **1** — the hypothesis is mentioned/relevant in some follow-on
    direction but not central.
  - **2** — the hypothesis genuinely anticipates a major follow-on
    direction (specific method, dataset, or claim).

Two annotators per paper, disagreement adjudicated by a third.

- Smallest sample (100-150 papers).
- Highest signal-to-noise.
- Use for: top-K precision claims ("our top-10 contains how many
  L3=2 papers?"), and as a sanity gate that the L1 and L2 metrics
  are not measuring something orthogonal to actual influence.

### 5.4 Outcome metric matrix

```
                       L1 (cite count)   L2 (engaged)   L3 (annotated)
Primary cohort         all 1800          all 1800       top+bottom 50
Peri-boom cohort       all 1800          all 1800       top 50 only
Post-boom cohort       all 1800          all 1800       —
```

---

## 6. Metrics

### 6.1 Headline metrics

For the primary cohort, on the L1 outcome:

```
ρ_total          : Spearman between predicted_influence and cited_by_count
                   (and separately, influential_citation_count)
ρ_dim_k for each dim k : Spearman between dim_k score and cited_by_count
top-K precision  : at K = 10, 50, 100  (fraction of top-K predictions
                   that fall in top-K of outcome)
```

For the primary cohort, on the L2 outcome: same four metrics.

For L3: top-K precision at K=10 (using L3-annotated tier).

**Baselines reported alongside aigraph's headline numbers** (added by
pivot — the comparators define what a "good" ρ means):

| Baseline | What it predicts | Why we report it |
|---|---|---|
| Random shuffle | random | Floor — anything beating this is non-zero |
| `cited_by_count` at submission time (Y0) | cite carry-over | Hard baseline; the predictor must beat "papers that were already cited stay cited" |
| OpenAlex paper-level PageRank | structural centrality | Tests whether our influence score is just centrality dressed up |
| **Intern-Atlas `/api/eval`** | their 5-dim score on the same hypotheses | Direct competitor; if their API returns ρ ≈ ours, our story leans on §4.1+§4.2 of intern-atlas-pivot.md |

The Intern-Atlas baseline is not optional — it is the most informative
single comparator now that they exist. If their reported ρ=0.81 with
expert review holds on our cohort, the influence-prediction story
needs to either (a) match it or (b) explicitly contribute orthogonal
information (different from theirs, useful in addition).

### 6.2 Confidence intervals

Bootstrap by paper (not hypothesis) — resample with replacement at the
paper level to preserve the within-paper hypothesis correlation
structure. Report 95% CI on every ρ.

Power calculation (one-tailed Spearman, α=0.05, β=0.2, expected
ρ=0.30): n=82 papers needed. Our cell size is 200, so each (venue,
year) cell on its own should be well-powered to detect a moderate
effect. The total cohort of 1800 papers gives us ~22× overhead for
weaker effects (down to ρ ≈ 0.07).

### 6.3 Stratified breakdowns

Run the headline metrics on every subgroup:
- Per (venue, year) cell — 9 cells × per-dim breakdown.
- Per topic stratum (LLM-relevant vs non-LLM, see §7).
- Per paper-role (survey, empirical, theoretical — using the existing
  `paper_role` field).

### 6.4 Per-dimension validity check

The 5 dimensions should ideally be **complementary** rather than
redundant. Two diagnostics:

1. **Marginal contribution.** Drop each dimension in turn and
   recompute `ρ_total`. A dimension that adds no marginal value is a
   candidate for removal.
2. **Inter-dim correlation.** Spearman between every pair of dim
   scores. High correlation (|ρ| > 0.7) means we're double-counting.

Both reported in the validation paper as a per-dimension robustness
section.

---

## 7. Topic stratification (LLM vs non-LLM split)

For the peri-boom and post-boom cohorts where the 2023 confound
applies, split each cohort into two strata:

- **LLM-relevant**: paper title or abstract contains any of:
  `language model`, `LLM`, `large language model`, `transformer`,
  `in-context`, `prompt`, `instruction tuning`, `RLHF`, `chain of
  thought`, `retrieval-augmented`, `RAG`. Also include any paper that
  declares one of {GPT-2, GPT-3, BERT, T5, LLaMA, etc.} as a method.
- **Non-LLM**: everything else.

Report `ρ_LLM` and `ρ_non-LLM` separately for the same cohort. Two
patterns to interpret:

- `ρ_LLM` ≈ `ρ_non-LLM` → the predictor is robust across topics; the
  LLM-confound concern was overstated.
- `ρ_LLM` < `ρ_non-LLM` → the predictor works on stable topics but
  fails when topic-shift dominates the outcome — important
  limitation, surfaceable in the discussion section.
- `ρ_LLM` > `ρ_non-LLM` → unexpected; investigate whether the
  predictor is accidentally tracking topic-shift indicators.

The keyword filter is intentionally simple; we'd rather have a noisy
but transparent topic split than an LLM-judged one whose decisions are
not auditable.

---

## 8. Anti-confound design choices

### 8.1 Self-citation filter

Drop citations where the citing paper shares ≥ 50% of its authors
with the cited paper. Self-citation count tracks publication volume,
not influence.

### 8.2 Outcome window normalization

Outcome metrics (L1, L2) are normalized by the median outcome of
papers in the same (venue, year, topic-stratum) cell. This converts
absolute counts into relative rankings within the topic environment
the paper was published in — a 2020 NeurIPS paper with 80 cites is
above-average for non-LLM 2020 NeurIPS work, even though 80 cites is
unimpressive for an LLM-adjacent 2022 paper.

Spearman ρ on the normalized outcome is the headline; raw is also
reported for transparency.

### 8.3 Outcome-window leakage

The graph aigraph builds for hypothesis generation must use **only**
citations whose source year ≤ the prediction year. We have to
explicitly filter the OpenAlex citation list. Without this, the
predictor would be looking at outcome data that hasn't happened yet
from the prediction window's perspective — classic look-ahead bias.

This is implemented as a `prediction_year_cutoff` parameter on the
graph builder, set per-cohort. Add a unit test that asserts no edge
in the cohort graph has a source paper published after the cutoff.

### 8.4 Annotator blinding (L3 only)

L3 annotators see only the abstract and the aigraph hypothesis. They
do **not** see:
- the predicted influence score,
- the actual citation count,
- whether the paper was in the top-50 or bottom-50 sample.

The randomization key linking paper → top/bottom assignment is held
in a separate file and revealed only after annotation is complete.

---

## 9. Cost and timeline

### 9.1 Cost (LLM + API budget)

Original design (kept for reference):

```
Claims extraction         : 1800 papers × $0.04         = $72
Hypothesis generation     : 1800 × ~3 hyps × $0.02      = $108
Novelty check (arxiv+LLM) : 1800 × $0.03                = $54
Citation enrichment       : free (OpenAlex + S2)
Citing-paper claims (L2)  : ~30k citing papers × keyword filter to ~5k × $0.01 = $50
L3 annotation labor       : 100 papers × 2 annotators × ~15min = manual cost
Subtotal LLM + API        : ~$285  → with buffer ≈ $370
Three cohorts             : ≈ $1100
```

Pivot-revised:

```
Cohort fetch              : $0      (was $0; now strictly $0, no API)
Citation enrichment       : $0      (Intern-Atlas papers carries cite count)
Citing-paper claims (L2)  : $0      (Intern-Atlas typed edges replace this)
Claims extraction         : $72     (unchanged)
Hypothesis generation     : $108    (unchanged)
Novelty check             : $54     (unchanged)
Intern-Atlas /api/eval    : tbd     (waitlist; assume free for academic use)
L3 annotation labor       : same    (manual)
Subtotal LLM + API        : ~$235   (~$370 with buffer)
Three cohorts             : ≈ $700  (~$400 saved vs original)
```

The $400 saved comes from dropping (a) per-paper OpenAlex/S2 fetch
work, (b) per-citing-paper claim extraction for L2. Both were already
$0 LLM cost in the original design's "Citation enrichment" line, but
the wall-clock time saved is the bigger win — no rate-limit pacing.

Use the saved budget on **L3 annotation pool expansion** from 100 →
150-200 papers, which is the most-defensible part of the validation
story now that data costs are zero.

### 9.2 Compute timeline

```
Week 1: Cohort fetch, claims extraction, graph build
Week 2: Hypothesis generation, influence scoring, predictions locked
Week 3: Outcome pipeline — citation enrichment, L2 method matching
Week 4: Metrics computation, plots, stratified analysis
Week 5: L3 annotation (parallel with Week 4 if annotators available)
Week 6: Writeup, robustness checks, discussion of LLM confound
```

Six weeks per cohort, with the prediction stage front-loaded so
predictions are locked before any outcome work starts. Cohorts can
overlap — Cohort A in weeks 1-6, Cohort B in weeks 3-8.

### 9.3 Rate limits

- OpenAlex: 100k requests/day no-auth, plenty.
- Semantic Scholar: 1 req/sec without API key, get an API key for
  10 req/sec. Apply at start of week 1.
- LLM provider: existing aigraph batch infrastructure handles this.

---

## 10. Implementation plan

### 10.1 New scripts to write

```
scripts/
├── cohort_fetch.py                # Pull (venue, year, sample) from OpenAlex
├── cohort_filter.py               # Apply abstract/claim/track filters
├── run_full_validation.py         # End-to-end prediction pipeline
├── lock_predictions.py            # Hash + freeze predictions_v1.jsonl
├── outcome_pipeline.py            # Pull citations, compute L1/L2
├── manual_annotation_setup.py     # Generate L3 annotation tasks (anonymized)
├── compute_metrics.py             # ρ, top-K, CI, all stratified breakdowns
└── plot_validation.py             # Per-dim ρ, per-cohort ρ, scatter, bootstrap CI
```

### 10.2 New config / artifacts

```
artifacts/validation_v1/
├── cohorts/
│   ├── primary_2018_2020.parquet         # paper_id, venue, year, abstract, ...
│   ├── peri_boom_2020_2022.parquet
│   └── post_boom_2022_2023.parquet
├── predictions/
│   ├── primary_v1.jsonl                  # locked
│   ├── peri_boom_v1.jsonl
│   └── post_boom_v1.jsonl
├── outcomes/
│   ├── primary_l1.parquet
│   ├── primary_l2.parquet
│   └── primary_l3.csv
└── reports/
    ├── headline_table.csv
    ├── stratified.csv
    ├── per_dim_breakdown.csv
    └── plots/
```

### 10.3 Code changes in src/aigraph

Three small changes to existing modules to support the experiment:

1. `graph.build_graph` gains a `prediction_year_cutoff: int | None`
   parameter that filters out edges whose source paper post-dates the
   cutoff. Default `None` preserves current behavior.
2. `influence.predict_influence_phase1` adds an optional `cohort_id`
   tag onto `InfluenceScore` so downstream aggregation can group.
3. `models.InfluenceScore` gets a `prediction_locked_at` timestamp.

### 10.4 Test plan

- Unit test: prediction_year_cutoff drops the right edges.
- Unit test: locked-predictions hash is stable across runs given
  identical inputs.
- Integration test: run full pipeline on a 5-paper synthetic cohort,
  assert outputs land in expected paths and CI bootstrap returns
  sensible CIs.
- Determinism test: run twice; predictions byte-identical when the
  same seed is set.

---

## 11. Pre-registered hypotheses

To prevent the experiment from drifting into post-hoc rationalization
of whatever ρ comes out, write down predictions before the outcome
data arrives:

1. **H1**: `ρ_total` on the primary cohort with L1 outcome will be
   ≥ 0.25.
2. **H2**: `ρ_total` will be **higher** on L2 than on L1 (engaged
   citations are a cleaner signal than raw count).
3. **H3**: `ρ_pre > ρ_peri > ρ_post` — the LLM-confound prediction.
4. **H4**: `ρ_grounding_depth > 0` on the primary cohort — closing
   out the negative-`ρ` anomaly from PR #14 specifically.
5. **H5**: `ρ_non-LLM > ρ_LLM` in the peri-boom cohort.

Pre-register these in a `docs/validation_preregistration_v1.md` and
git-commit before locking predictions. The header of the eventual
results section should list which were confirmed.

---

## 12. Limitations to acknowledge in the writeup

1. **Citation count ≠ quality.** Even with self-cite filtering and
   topic normalization, citation count tracks community attention,
   not quality. Lipton's "troubling trends" critique applies. L3
   annotation partially addresses this; we should not pretend the
   problem is solved.
2. **Top venues already pre-filter for impact.** Our cohort is
   already a curated set of papers that passed peer review at top
   venues, so the variance in outcomes is compressed. ρ measured here
   may be a lower bound for the variance one would see on a less
   curated cohort.
3. **Look-ahead in hypothesis generation.** aigraph in 2026 reads
   2018 papers using a 2026-era LLM. The hypothesis quality might
   benefit from knowing what the field looked like in 2024 even if
   the citation graph is properly cut at 2020. Mitigation: in the
   discussion, run a small ablation with a smaller / earlier-trained
   LLM and compare hypothesis content overlap.
4. **No adversarial baseline.** A baseline that just predicts
   "papers in popular topics get cited more" would beat random; a
   baseline that uses graph centrality alone would beat that. We
   should report `ρ_baseline` for at least:
   - venue+year random baseline
   - PageRank on the local graph
   - count of `cited_by_count` at submission time
   so the reader can locate our predictor's improvement on a
   baseline-aware scale.
5. **Sample is AI papers only.** Generalization to other fields is
   speculative; the design is constructed for AI literature where
   our claim extractor is calibrated.

---

## 13. What success looks like

A version of this validation that we'd be willing to put in a paper:

- Three cohorts × three outcome tiers × five dimensions all reported.
- Headline `ρ_total` ≥ 0.30 on the primary cohort, L2 outcome.
- All 5 pre-registered H1-H5 either confirmed or explicitly
  un-confirmed in the writeup with a paragraph of analysis.
- Per-dimension table showing each dim contributes positive marginal ρ.
- LLM-confound discussion section with quantitative LLM-vs-non-LLM ρ
  comparison.
- L3 manual annotation top-10 precision ≥ 50%.

A failure mode that would still be publishable:

- ρ_total ≥ 0.20 on primary cohort.
- LLM confound is real and large (ρ_LLM weak in peri-boom) — we
  document this carefully and frame the predictor as "scientific
  quality predictor robust outside topic-shift regimes". Honest
  characterization of where the algorithm works and where it does
  not is itself a contribution.

A failure mode that would force a redesign:

- ρ_total < 0.10 on the primary cohort.
- This means the score does not track real impact even on the
  cleanest cohort and the dimensions need to be rethought before a
  second validation attempt.

---

## 14. Open questions for the design

These are not blocking but should be settled before locking the
prediction pipeline.

1. **Hypothesis-to-paper aggregation.** A paper produces multiple
   hypotheses. Do we use `max(influence)` or `mean(influence)` as
   the paper-level prediction? Argument for max: only the *best*
   hypothesis in a paper matters. Argument for mean: a paper that
   produces uniformly mediocre hypotheses is less impactful than one
   with one stellar and several mediocre. Default proposal: report
   both, lead with `max`.
2. **Citation graph snapshot date.** Do we use OpenAlex's view of
   the citation graph as of mid-2026 (most complete) or as of the
   end of the outcome window? The former gives more data, the latter
   is more faithful to the controlled-experiment ideal. Default
   proposal: pull both, headline metric uses the end-of-window
   snapshot, robustness uses the latest snapshot.
3. **L3 annotator pool.** Internal team only, or external? External
   is more independent but slower and more expensive. Default
   proposal: internal team for the first pass, external audit on a
   20-paper subsample for cross-validation.

---

## 15. Decision request — closed

**Original asks:** approval of §3 cohort structure, §5 three-tier
ground truth, §11 pre-registration commitment, and prioritization
vs Phase 2 (`structural_impact`).

**Resolution (May 8, 2026):** All four points superseded by the
Intern-Atlas pivot.

- §3 cohort: built and verified — 1790 / 1800 / 1200 papers across
  primary / peri-boom / post-boom. Single ICLR 2018 cell undershoot at
  190; everything else at 200/cell target.
- §5 three-tier: L1 and L2 now both consume Intern-Atlas data
  directly; L3 manual annotation expanded to 150 papers using saved
  budget.
- §11 pre-registration: still binding. To be written as a separate doc
  (`docs/validation_preregistration_v1.md`) before any prediction
  artifact is locked.
- Phase 2 prioritization: deferred. Phase 2 (`structural_impact`)
  starts after the Phase 1 4-dim validation runs through the new
  pipeline at least once on the primary cohort.

See §16 for the consolidated pivot record.

---

## 16. Pivot in light of Intern-Atlas (May 2026)

**Trigger:** May 1, 2026 release of `OpenRaiser/Intern-Atlas` — a
4.2M-paper / 4.14M-edge typed methodological evolution graph from
Shanghai AI Lab, MIT-licensed, with reported Spearman ρ=0.81 between
their 5-dim idea evaluator and human expert review.

**Decision:** hybrid integration. Consume their data as cohort +
outcome source; keep aigraph's claim-level conflict pipeline as
differentiated value. Full strategic rationale in
[docs/intern-atlas-pivot.md](./intern-atlas-pivot.md).

**What changed in this design doc:**

| Section | Original | After pivot |
|---|---|---|
| §3.1 cohort fetch | OpenAlex + S2 enrichment | `cohort_from_intern_atlas.py` parquet read |
| §3.3 filters | English abstract ≥ 80 words; ≥ 2 IDs | abstract ≥ 80 chars; ≥ 1 ID (Intern-Atlas reality) |
| §4.1 step 1 | OpenAlex fetch per paper | Parquet load via `corpus.intern_atlas_loader` |
| §4.2 step 1 | Per-paper OpenAlex citation pull | `papers.cited_by_count` direct + `paper_evolution_edges` for L2 |
| §5.1 L1 | `cited_by_count_in_window` | `cited_by_count` + `influential_citation_count` |
| §5.2 L2 | "shared method or dataset" via per-citing-paper claim extraction | Engaged-edge filter on Intern-Atlas's pre-computed typed edges |
| §6.1 baselines | random + PageRank + cite carry-over | + Intern-Atlas `/api/eval` baseline (now mandatory) |
| §9.1 cost | $1100 / 3 cohorts | $700 / 3 cohorts |
| §9.2 timeline | 6 weeks per cohort | 3-4 weeks per cohort (cohort fetch + outcome pull collapse to ~hours) |

**What did NOT change:**

- §2 (LLM-discontinuity confound argument) — still applies, this is
  why the 2018-2020 primary cohort matters
- §7 (LLM vs non-LLM topic stratification) — still applies
- §8 (anti-confound choices) — all still apply, including the
  prediction_year_cutoff guard against look-ahead leakage
- §11 pre-registered hypotheses H1–H5 — unchanged, will be committed
  as `validation_preregistration_v1.md` before predictions lock
- §13 success criteria — unchanged

**New risks introduced by the pivot** (also documented in
intern-atlas-pivot.md §8):

1. Their `/api/eval` ρ on our cohort might be very high. If it
   matches or beats our predictor's ρ, the influence-prediction story
   collapses to "we're a free open re-implementation" — not a paper
   contribution. Mitigation: lead the writeup with conflict detection
   (§4.1 of pivot doc), not influence prediction.
2. Schema change on their parquet breaks our loader. Mitigation: pin
   to a specific HF revision in `corpus.intern_atlas_loader`.
3. Their v2 might add a `contradicts` edge type. Mitigation: ship our
   conflict layer in enough depth that even if they add a shallow
   flag, our claim-level reasoning is meaningfully deeper.

**Cohort actuals** (recorded for reproducibility):

```
seed=17, run on May 8 2026
papers parquet revision: <pin in _summary.json>
filter drop ratio: 99.35% (4.2M → ~27k pass venue + quality filter)

primary_2018_2020:    1790 papers   (8 cells × 200 + ICLR 2018 × 190)
peri_boom_2020_2022:  1800 papers   (9 cells × 200)
post_boom_2022_2023:  1200 papers   (6 cells × 200)
```

Local at `artifacts/validation_v1/cohorts/`, gitignored.
`_summary.json` records the seed, filter loss ratio, and source
parquet mtime/size for tamper detection.
