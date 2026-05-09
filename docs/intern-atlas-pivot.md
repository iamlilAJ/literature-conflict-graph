# aigraph × Intern-Atlas — Pivot Decision Doc

A decision record for repositioning aigraph after the public release of
[Intern-Atlas](https://huggingface.co/datasets/OpenRaiser/Intern-Atlas)
on May 1, 2026 (paper [arxiv:2604.28158](https://arxiv.org/abs/2604.28158)).

> Status: pivot adopted. Last updated: v0.1.
> Companion documents:
> [influence-prediction-design.md](./influence-prediction-design.md),
> [controlled-validation-design.md](./controlled-validation-design.md).

---

## 1. The trigger

On May 1, 2026 the OpenRaiser team (Shanghai AI Lab + collaborators)
published a 4.2M-paper / 4.14M-edge typed methodological evolution graph
under MIT license, with the accompanying paper claiming Spearman
ρ = 0.81 between their 5-dim idea evaluator and human expert review.

Before that release, the strategic story for aigraph was: "we extract
typed conflict graphs from AI literature and predict hypothesis
influence". Three of the four words in that sentence — typed, graphs,
literature, influence — now have a public, scaled, validated implementation
that did not exist a week ago. **aigraph cannot continue as if Intern-Atlas
does not exist.** This doc records what that means.

---

## 2. What Intern-Atlas actually shipped (recon results)

Empirical inspection of the HF parquet (May 8, 2026) confirmed:

| Aspect | Status |
|---|---|
| License | MIT — commercial use OK, no copyleft |
| Year coverage | Modern, 2018-2025 each year ≥ 195k papers |
| Venue coverage (NeurIPS/ICML/ICLR 2018-2023) | 500-3500 per (venue, year) cell, well above any cohort we'd want |
| Typed extraction completeness | **99.8% non-empty** for impact_json / bottleneck_json / mechanism_json |
| Edge typology | 7 causal types: extends / improves / replaces / adapts / uses_component / compares / background — **no contradicts** |
| Sample edge quality | High (specific bottlenecks, numeric impact deltas), with ~10-20% paper_b artifacts (URLs not papers) |
| Pipeline source code | **Not released** — extraction prompts, SGT-MCTS, evaluator are API-only |
| Method registry (V_M, 8155 methods) | **Not released** — only 60-row method_relations seed published |

Two facts are decisive:

1. **The paper-level data layer is fully open and production-grade.** It
   covers our cohort needs and is strictly better than what we could
   build ourselves at our scale.
2. **The pipeline + method-level graph + evaluator are NOT open.** They
   exist as services behind an API in beta. Any architecture that depends
   on those is depending on a black-box external service.

---

## 3. Decision: Hybrid — consume their data, keep our differentiated logic

We adopt **hybrid integration** rather than full pivot or full
differentiate:

- **Consume** their `papers` config as cohort + corpus metadata source
- **Consume** their `paper_evolution_edges` config as a typed-citation
  baseline we compare against (not as our primary stance source — see §4)
- **Reference** their `/api/eval` as a baseline competitor in influence
  prediction validation (not as a hard dependency)
- **Keep** aigraph's own claim extraction, citation_stance, anomalies,
  influence prediction, and creator pipeline — all four are either
  qualitatively different (claim-level vs method-level) or operate on
  axes Intern-Atlas does not cover (conflict, open-questions)

This preserves aigraph's differentiation surface while eliminating the
weeks of corpus-building and validation-fetching work that are now
redundant.

---

## 4. Differentiation surface — what aigraph still uniquely provides

After honest scrutiny, three differentiators hold (one of them at
half-strength):

### 4.1 Conflict-aware layer (full strength)

Intern-Atlas's seven edge types are all causal. There is no
representation of:
- **contradicts**: paper A and paper B disagree on the same claim
- **benchmark_inconsistency**: two papers report incompatible results on
  the same task/dataset
- **replication_conflict**: a paper's claimed result does not reproduce
  in a follow-on paper
- **metric_mismatch / setting_mismatch**: papers compare under
  incompatible measurement protocols

aigraph's anomaly pipeline produces all five. This is not "aigraph also
does typed edges with finer granularity"; it is a structurally distinct
layer that Intern-Atlas's design does not expose. Conflict structure is
the strongest single defensible angle.

### 4.2 Author-acknowledged gaps via `open_questions` extraction (full strength)

aigraph extracts open questions and limitations directly from paper
text (typically the limitations / future work sections). Intern-Atlas
extracts bottlenecks **between** papers (paper B claims paper A's
bottleneck). These are complementary signals:

- Intern-Atlas: external observation of A's weakness, written by B
- aigraph: self-reported weakness, written by A's authors

A creator-mode hypothesis pipeline that conditions on **both** is
structurally richer than either alone.

### 4.3 Open + locally-runnable influence/idea evaluation (half strength)

Intern-Atlas's 5-dim evaluator is API-only and a black box. aigraph's
`influence.py` Phase 1 4-dim is open, deterministic, and reproducible
without external service calls. The differentiation here is not
"aigraph predicts better"; it is:

- aigraph evaluates **without** an external API (important for OMC
  talents that may run in air-gapped contexts)
- aigraph's scoring weights are inspectable and tunable per use case
- aigraph predicts **future impact** of a hypothesis given a corpus
  state, vs Intern-Atlas evaluates a finished idea text against a
  static graph — different questions, both interesting

This plank cannot be the lead story; it supports the other two.

### 4.4 What is NOT a differentiator (be honest)

- **"We extract typed edges with bottleneck/mechanism/impact"** — they
  extract this at 99.8% completeness on 4M edges. We don't compete here.
- **"We have multi-grain creator pipeline"** — they have a 4-strategy
  idea generator. The structural-vs-strategic distinction is too
  internal to be a public-facing claim.
- **"We use richer claim granularity"** — claim-level vs method-level is
  real, but until validated as predictive of better hypotheses, it is
  internal scaffolding rather than a marketed feature.

---

## 5. Concrete repository changes

### 5.1 New files

```
scripts/cohort_from_intern_atlas.py       — replaces planned cohort fetcher
docs/intern-atlas-pivot.md                — this file
```

### 5.2 Modified files

```
docs/controlled-validation-design.md
  §3.1  cohort fetch              → point to scripts/cohort_from_intern_atlas.py
  §5.1  L1 outcome (cite count)   → use Intern-Atlas papers.cited_by_count and
                                     papers.influential_citation_count directly
  §5.2  L2 engaged citation       → use paper_evolution_edges with
                                     evolution_relation IN (extends, improves,
                                     replaces, adapts) as engagement signal
  §6.1  baselines                 → add Intern-Atlas /api/eval as comparison
                                     baseline in influence ρ table
  §15   decision request          → mark closed; new §16 below

  §16 (new) Pivot in light of Intern-Atlas
      Brief explanation of why §3 / §5 changed, pointing to this doc.

src/aigraph/corpus.py
  Add intern_atlas_loader path. When AIGRAPH_USE_INTERN_ATLAS=1 set,
  replace S2/OpenAlex fetch with parquet load. Keep existing path as
  fallback.

scripts/validate_influence_backtest.py
  Add --baseline-eval intern-atlas flag that hits /api/eval for the same
  hypotheses and reports ρ comparison.
```

### 5.3 Files to deprecate (mark, not delete yet)

- The "1800 paper × 3 cohort × OpenAlex fetch" plan in
  controlled-validation-design.md §3 is replaced by the cohort script
  above. The original §3 numbers are kept intact; §16 explains the
  change of source.

---

## 6. What the validation table now looks like

Old plan: aigraph predicts → OpenAlex citation outcomes → ρ

New plan: aigraph predicts → Intern-Atlas-derived outcomes (3 tiers)
+ baseline comparison:

| Predictor | Outcome | ρ (target) |
|---|---|---|
| aigraph influence_phase1 | Intern-Atlas L1 (cited_by_count) | ≥ 0.30 |
| aigraph influence_phase1 | Intern-Atlas L2 (engaged citations: extends/improves/replaces/adapts edges) | ≥ 0.40 |
| aigraph influence_phase1 | Intern-Atlas L3 (manual annotation, 100 papers) | top-10 precision ≥ 50% |
| Intern-Atlas /api/eval (baseline) | Intern-Atlas L1 | reported as comparison |
| OpenAlex PageRank (baseline) | Intern-Atlas L1 | reported as comparison |

The Intern-Atlas /api/eval baseline is what aigraph has to beat or
meaningfully complement. If aigraph ρ matches their ρ on the same
cohort, the differentiation argument has to lean fully on §4.1 and §4.2
(conflict + open_questions), not on raw scoring quality.

---

## 7. OMC talent (literature-researcher-talent) implications

The OMC talent's value proposition shifts:

Before:
- "aigraph extracts typed citation graph + generates hypotheses
  conditioned on conflicts and open questions"

After:
- "aigraph adds a **conflict-aware claim layer + author-acknowledged
  gap layer** on top of Intern-Atlas's typed methodological evolution
  graph. When Intern-Atlas API is unavailable or rate-limited, the
  talent remains fully functional from the local Intern-Atlas parquet
  cache."

The talent should declare Intern-Atlas as a soft dependency (preferred
data source, with a fallback to OpenAlex/S2), not a hard one. This
matters for Talent Market reliability.

---

## 8. Risks and limitations of this pivot

### 8.1 Their /api/eval might score too well to differentiate against

Their reported ρ = 0.81 is from a 100-paper expert sample. On the
larger 1800-paper cohort, ρ may drop. Or it may stay high. We do not
know until POC #13 runs. If ρ ≈ 0.81 on our cohort too, our predictor
needs to either match it or contribute orthogonal information; if our
ρ < 0.5 on the same cohort, the influence prediction story needs
either a redesign or an explicit "we are not in this competition,
our story is conflict detection" framing.

### 8.2 Intern-Atlas might add `contradicts` edges in v2

Their pipeline could plausibly extend to conflict edges in a future
release. Our differentiation story has a half-life. Two responses:
- Ship a paper before that happens (publication is a defensive move)
- Build the conflict layer at sufficient depth that even if they add a
  shallow `contradicts` flag, our reasoning stack is meaningfully deeper

### 8.3 We are now downstream of an external dataset's update cadence

If OpenRaiser stops updating Intern-Atlas, our cohort source goes
stale. Mitigation: pin to a specific HF revision in the loader, and
keep the OpenAlex/S2 fetch path as a fallback that we can revive if
needed.

### 8.4 The OMC talent is now coupled to their parquet schema

A schema change on their side breaks our talent. Mitigation: schema
adapter layer in `corpus.py` that translates Intern-Atlas v1 to
aigraph's internal Paper / Claim models. When their v2 lands, adapter
absorbs the change rather than every downstream module needing edits.

---

## 9. What does NOT change

- aigraph's core graph model (`models.py`, `graph.py`)
- The anomaly pipeline (`anomalies.py`)
- Hypothesis generation prompts and creator pipeline
- The Phase 1 influence prediction code
- The OMC talent's overall structure and entrypoint

The pivot is data-source + validation + positioning. It is not a
rewrite.

---

## 10. Decision request — closed

§15 of `controlled-validation-design.md` requested approval for a
1800-paper × 3 cohort plan with $1100 budget over 6 weeks. That is now
**superseded** by:

- Cohort source: free, 5 min via cohort script
- LLM cost: same per-paper (claims + hypothesis generation), but cohort
  prep and outcome fetching cost goes to ~$0
- Timeline: 6 weeks → 3 weeks for the same deliverable

This is the second-order benefit of the pivot. Use the saved budget /
time for the manual L3 annotation tier (50-100 papers × 2 annotators)
which becomes the most-defensible part of the validation story.
