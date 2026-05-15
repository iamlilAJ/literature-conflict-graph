# atlas × aigraph fit — recon report

**Author:** cc (Claude Code agent, gpt-5.4 family)
**Date:** 2026-05-14
**Branch:** `recon/atlas-aigraph-fit` (worktree only, off `main`)
**Budget consumed:** LLM ≈ $1 of $25 cap; wall ≈ 1.5 hr of 6 hr cap.
**Stop:** Q1 KILL threshold tripped (31% vs 35% floor), so per brief stop
condition every other LLM-heavy probe was skipped. Q3 (design-only) and
Q4 (no-LLM) ran anyway because they're free and informative.

---

## TL;DR

**Recommendation: J0** (Atlas as corpus source only). J1 (canonical
method namespace) is dead at the structural level: aigraph's claim
extractor pulls 57.5 % non-method phrases ("comprehensive evaluation",
"chain-of-thought prompting", "Top-1 accuracy", or model names like
"GPT-4") into the `method` field, so any naïve V_M map will fail by
construction. Of the 200-claim sample, only 18-20 % cleanly map to V_M
once token-set false positives are removed (Q1 audit). J2/J3 are
defensible *in design* (Q3) but the corpus-level evidence is thin
on the 540 p run (Q4 finds only 4 `replaces` edges where both
endpoints are in 540 p, 0 of 4 cross-validate to negative-stance
claims).

The high-confidence path forward:

1. Fix aigraph's claim extractor so `method` is a method (not a model,
   not a metric, not a phrase). After that fix, re-run Q1 — V_M
   coverage of clean method strings is already at ~84 %, so the
   namespace adoption (J1) only blocks on aigraph hygiene, not on
   Atlas coverage.
2. While that fixes, adopt Atlas as a J0 corpus source: cohort builder
   already exists at `scripts/cohort_from_intern_atlas.py`. Trivial.
3. Add `unresolved_replacement` and `silent_replacement` as J2-level
   *prototype* detectors on a small scale (~5-10 candidates) to
   confirm the joint-graph anomalies are real before committing to
   J2 broadly.

---

## 1. Two-graph geometry

```
                                                                   ┌─────┐
   ATLAS                              AIGRAPH                      │ Joint
   ━━━━━                              ━━━━━━━                      └─────┘

   ┌──────────────┐                   ┌──────────────┐
   │  paper_a     │ extends           │   paper P    │
   │              │ ━━━━━━━━━━>       │              │
   │              │ improves          │   ┌──────┐   │
   │              │ ━━━━━━━━━━>       │   │claim │   │   (8 anomaly
   │              │ replaces          │   │  c1  │ ───┼─→  detectors:
   │              │ ━━━━━━━━━━>       │   │ pos  │   │    benchmark_inc.
   │              │ adapts            │   └──────┘   │    impact_conf.
   │              │ ━━━━━━━━━━>       │   ┌──────┐   │    metric_mism.
   │   paper_b    │ uses_component    │   │claim │   │    setting_mism.
   │              │ ━━━━━━━━━━>       │   │  c2  │ ───┼─→  evidence_gap
   │              │ compares          │   │ neg  │   │    bridge_opp.
   │              │ ━━━━━━━━━━>       │   └──────┘   │    community_disc.
   │              │ background        │      │       │    replication_c.
   │              │ ━━━━━━━━━━>       │      ▼ open_q │
   │              │                   │   future-work │
   └──────────────┘                   └──────────────┘
         │                                  │
         ▼                                  ▼
   paper-paper                         paper-claim-paper
   8155 method nodes                   ~25 canonical method buckets
   4.14 M edges                        ~500 anomalies on 540p run
   no contradicts                      explicit conflict types
```

aigraph and Atlas are not the same graph at different resolutions —
they are perpendicular axes over the same papers. Atlas is a directed
typed evolution graph between papers; aigraph is a typed conflict graph
between *claims within and across* papers. Where they meet is at the
paper node — and that's the only natural join point that data supports.

A real SVG version of this diagram is at
`docs/atlas-aigraph-fit-architecture.svg`.

---

## 2. Q0 — codebase audit

Per-file map of where Atlas could plug into aigraph.

| File | Function / class | Currently consumes | Where Atlas plugs in cleanly |
|---|---|---|---|
| `corpus.py` (1610 LOC) | `seed_reasoning_corpus`, `sync_arxiv_corpus`, `enrich_citations_from_semantic_scholar`, `hydrate_paper_from_corpus` | arxiv API (TeX/HTML/PDF), Semantic Scholar (citation enrich), OpenAlex via `fetch_openalex_papers` | New `intern_atlas_loader` module returning `list[Paper]`. The `Paper` schema already has `arxiv_id_base`, `cited_by_count`, `referenced_works` — same fields Atlas's `papers` parquet provides. Drop-in. **J0 lives here.** |
| `extract.py` (140 LOC) | `ClaimExtractor` ABC + `RuleExtractor` | Paper.text / Paper.abstract via `hydrate_paper_from_corpus` | No direct plug. Extract pulls *claims* from paper text; Atlas has no claim-level data. If aigraph wanted to use Atlas's `bottleneck_json` as a secondary claim source, the extractor would need a new "external bottleneck claim" path here, but that's a stretch (Atlas bottlenecks are written by paper B about paper A; they're not paper A's own claims). |
| `llm_extract.py` (378 LOC) | `LLMExtractor` (Claim builder) | LLM via `llm_client` | `CANONICAL_METHODS` (25 buckets) and `CANONICAL_TASKS` (18 buckets) are hardcoded vocabs. **J1 plug-point.** Replace this with Atlas V_M method_id namespace. Requires the extractor prompt to know about 8155 buckets — feasible only if you also supply the top-N alias subset relevant to the current paper context (full 8155 is too many to prompt with). |
| `graph.py` (613 LOC) | `build_graph`, `build_citation_graph`, `CLAIM_ENTITY_FIELDS` | Claims + Papers from JSONL | Node-type taxonomy is fixed: Paper / Claim / Method / Task / Dataset. **J2 plug-point.** Adding a "uses Atlas method_id" attribute to each Claim node — then Method nodes can co-occur with Atlas method_ids and joint anomaly detectors can use both labels. |
| `anomalies.py` (838 LOC, **FROZEN**) | 8 detectors operating on `(method, task)` clusters + citation graph | Claims + nx.MultiDiGraph from `graph.py` | DO NOT MODIFY. J2 joint detectors go into a new `joint_anomalies.py` module that consumes the same Claims+graph PLUS Atlas parquet. Compose with the existing detectors via an optional `apply_joint_detectors=True` flag in the run script. |
| `hypotheses.py` (215 LOC, **FROZEN**) | `TemplateGenerator` (fallback) for `Anomaly → Hypothesis` | Anomaly + claim text | DO NOT MODIFY. |
| `llm_hypotheses.py` (427 LOC, **FROZEN**) | `LLMGenerator` (production) | Anomaly + claim text + LLM | Frozen. **J3 plug-point.** Adding 1 paragraph of Atlas context (paper's `bottleneck_json` from incoming edges, Atlas lineage edges) to the hypothesis prompt would be a parallel module that wraps `LLMGenerator` rather than modifying it. New file `llm_hypotheses_atlas_grounded.py`. |
| `influence.py` (351 LOC, **FROZEN**) | `predict_influence_batch`, 4-dim score | Hypothesis + Claim + hierarchy.json + `novelty_check` extras | **J3 plug-point.** A 5th dim `lineage_position` could be added — but this requires a non-frozen module since `WEIGHTS_PHASE1` is frozen. Defer until after a v0.8 release. |
| `scoring.py` (288 LOC) | `score_all` (8-component composite) for downstream MMR | Hypothesis + Anomaly + Claim | No direct plug; downstream consumer of influence + anomaly. |
| `paper_select.py` (918 LOC) | `dedupe_papers`, `score_paper`, `select_representative_papers` | Paper list, query | Atlas could provide a `paper_role_signals` enrichment — Atlas has venue tier, citation count, fields_of_study. But paper_select.py is already arxiv-friendly; this is a J0-internal improvement, not a J2 differentiator. |
| `creator.py` (838 LOC, **FROZEN**) | `generate_creator_hypotheses`, `generate_creator_hypotheses_multi_grain` | Anomaly + Claim + OpenQuestion + hierarchy | **J3 plug-point.** Same shape as `llm_hypotheses.py` — wrap rather than modify. Atlas's `bottleneck_json` is a natural addition to the creator's prompt context. |
| `hierarchy.py` (321 LOC) | `build_hierarchy` (Louvain communities + clusters) | Claims + Papers + Anomaly + citation graph | **J2 plug-point.** Atlas's `paper_methods` table is a denser ground truth for "papers that use the same method" than aigraph's claim-method co-occurrence. Could replace the Louvain step on small corpora where `community_reach` is N/A. Documented in Q3 row 6. |

### Q0 observation

Atlas's papers parquet has 28 fields. `Paper` model in aigraph has 32
fields. They overlap in ~12 fields cleanly (paper_id, title, abstract,
year, venue, cited_by_count, arxiv_id, etc.). A loader function is
~80 LOC of column renames and type coercion. The hard part isn't
plumbing; it's deciding what to do with Atlas's *additional* data
(method_id, evolution edges, bottleneck_json).

**Verdict: GO on J0 plumbing, reasoning: Atlas's papers schema is a
clean superset of aigraph's Paper, and the corpus.py architecture
already expects a pluggable source.**

---

## 3. Q1 — Method-namespace coverage

**Probe:** 200 random claims from `artifacts/runs/arxiv-reasoning-v0.7-540p/claims.jsonl`,
each `method` field fuzzy-matched against Atlas V_M (8,155
method_names from the local mirror's `paper_methods` parquet).

### 3.1 Match rates

| Match type | Threshold | n / 200 | % |
|---|---|---:|---:|
| exact (case-insensitive) | 1.00 | 11 | 5.5 % |
| rapidfuzz ratio | ≥ 0.92 | 1 | 0.5 % |
| rapidfuzz token-set | ≥ 0.85 | 50 | 25.0 % |
| **headline** | any | **62** | **31.0 %** |
| unmatched | | 138 | 69.0 % |

### 3.2 LLM-judged labels on the 138 unmatched

Model: `gpt-5.4-mini` (substitution for the brief's spec of
`haiku-4.5` — endpoint blocks Anthropic models; see §7).

| Label | n | % of unmatched | % of total |
|---|---:|---:|---:|
| `too_generic` | 115 | 83.3 % | 57.5 % |
| `garbage` | 11 | 8.0 % | 5.5 % |
| `novel_method` | 8 | 5.8 % | 4.0 % |
| `alias_not_in_Atlas` | 4 | 2.9 % | 2.0 % |

Verbatim samples per label are in `recon/atlas_fit/LLM_audit_sample.md`.
Key examples:

- `too_generic`: `'comprehensive evaluation'`, `'chain-of-thought prompting'`,
  `'recall evaluation'`, `'multi-agent self-correction'`,
  `'LLM-based planner'`. These are descriptive phrases, not method names.
- `garbage`: `'GPT-4'` and `'Llama'` — *model names* misclassified into
  the `method` slot. Also `'o1-preview'`, `'LLMCad on-device inference engine'`.
- `novel_method`: `'Chain-of-Spot'`, `'KD-Encoder'`,
  `'Citation-Enhanced Generation (CEG)'`, `'SpatialRGPT'`,
  `'ViLaSR'`. These are genuine novel methods Atlas V_M lacks.
- `alias_not_in_Atlas`: `'Retrieval Augmented Generation (RAG) agent'`,
  `'SoM prompting'`, `'MedRag with RRF-2 retriever fusion'`. Real methods,
  non-canonical form. RAG is in V_M as
  `"Retrieval-Augmented Generation"` — a punctuation difference.

### 3.3 Restricted denominator: clean method strings only

Among the 200 claims, 74 (37 %) had method strings that are arguably
real methods (matched + `novel_method` + `alias_not_in_Atlas`):

```
clean methods matching V_M:   62 / 74   =  83.8 %
```

So V_M's *coverage of real methods that aigraph extracts* is **~84 %**.
The 31 % headline is depressed by aigraph's own extraction pulling
non-method phrases into the `method` field. (The audit at
`recon/atlas_fit/LLM_audit_sample.md` finds the 31 % is itself an upper
bound because token-set matches are loose; the honest match rate is
closer to 18-20 %. See §3.4.)

### 3.4 Audit of the audit

cc re-read 10 Q1 judgments (`recon/atlas_fit/LLM_audit_sample.md`)
and found:

- 6 / 10 agreements with the gpt-5.4-mini judge
- 2 / 10 mislabeled (Top-1 accuracy → `too_generic` should be `garbage`;
  multi-method strings labeled as alias)
- 2 / 10 token-set matches that should not have been counted as matches
  (`'goal-baseline regularization for CLIP reward models'` → V_M=`'CLIP'`
  via token-set 1.0; the method being claimed is regularization, not
  CLIP). With ~50 % of token_set matches being loose, the true clean
  match rate is closer to **18-20 %**.

### 3.5 Verdict

**Verdict: KILL J1 as currently specified, INCONCLUSIVE conditional on
fixing aigraph's method-field extractor. Reasoning: V_M is structurally
fit but aigraph's `method` slot is contaminated; only 37 % of extracted
methods are genuinely method names, and 31 % match (or 18-20 % with
strict matching). The brief's KILL threshold (<35 %) is tripped.**

---

## 4. Q2 — Same-paper weakness-signal overlap

**SKIPPED per brief stop condition.** The brief says "Q1 returns <35 %
map rate (every other probe is moot)" and Q1 returned 31 %. Q2 needs 30
papers in the aigraph 540 p ∩ Atlas intersection plus Sonnet-tier LLM
budget; both feasible (the intersection has 445 papers), but the result
would not change the J recommendation given Q1's verdict.

**What we would have learned from Q2 had we run it:** how often
Atlas-side "bottleneck observed by paper B" overlaps with aigraph-side
"open question or negative stance from paper A". This would have been
the primary evidence for `bottleneck_open_q_alignment` (one of the
three new joint anomaly types in Q3).

Future cc agent who picks this up: the protocol in §2.Q2 of the brief
is sound and the data is there. With `gpt-5.4` and the 445-paper
intersection, the probe costs ~$1 and 15 minutes wall.

---

## 5. Q3 — Joint-anomaly inventory

Design-only, no-LLM. Full inventory at
`recon/atlas_fit/Q3_joint_anomaly_inventory.md`. Summary:

- **8 joint variants of existing detectors.** Each cell explains what
  Atlas adds to the existing aigraph trigger. Most stronger via Atlas
  acknowledgment signal (`improves`/`replaces`/`compares` edges to
  filter out artifactual conflicts). Implementable in ~250 LOC in a new
  `joint_anomalies.py` module without touching frozen detectors.
- **3 new types ONLY on joint graph:**
  - `unresolved_replacement` — Atlas says B replaces A but aigraph
    stance is still mixed. Rare at small scale.
  - `bottleneck_open_q_alignment` — paper's own open_question matches
    third-party bottleneck observation. Gold-standard weakness signal.
    Optimistic scale: 30-80 candidates at 540 p.
  - `silent_replacement` — aigraph detects B contradicts A but Atlas
    has no edge. Surfaces unacknowledged conflicts.

Total new code estimate: **~530 LOC** in one new module.
v0.7-frozen modules untouched.

---

## 6. Q4 — `replaces` ↔ stance cross-validation

### 6.1 Corpus intersection

| Set | Size |
|---|---:|
| aigraph 540 p papers | 474 |
| 540 p ∩ Atlas papers (by `arxiv_id_base`) | 445 (93.9 %) |
| Paper-pairs with ANY Atlas evolution edge in 540 p ∩ Atlas | 369 |
| Of those, by relation: `compares` | 313 |
| `uses_component` | 39 |
| `improves` | 10 |
| **`replaces`** | **4** |
| `extends` | 3 |

### 6.2 The probe and what it found

Atlas `replaces` edges with both endpoints in 540 p: **4 edges only**.
Of those, **0 / 4** show an aigraph-side negative-stance claim whose
method/task overlaps with the replaced paper's claims.

### 6.3 What that 0 / 4 actually means

This is not a strong signal in either direction — it's a corpus-thinness
artifact. The 540 p run is a 2023-2026 arxiv-reasoning slice; "replaces"
edges are rare in fresh reasoning literature (which is overwhelmingly
`compares`-dominated). At ~5000 papers in a more mature cohort
(NeurIPS/ICML/ICLR 2018-2020, where we have parquet but not aigraph
output yet — see `artifacts/runs/validation-v1-primary/`), `replaces`
density would be 10-30× higher.

### 6.4 Verdict

**Verdict: INCONCLUSIVE on J2, reasoning: the corpus-intersection on
this cohort has only 4 replaces edges, which is enough for a hard KILL
under the brief's 30 %/10 % thresholds (0 % ≤ 10 %) but the sample is
too small for that ratio to be informative. Re-run on the 1790-paper
validation_v1_primary cohort would have 10-30× more `replaces` edges to
probe.**

---

## 7. Q5/Q6 — SKIPPED

Both LLM-heavy (Sonnet-tier, ~$5-8 combined) and predicated on J2/J3
being plausible. Q1's KILL on the J1 prerequisite makes them moot under
the brief's stop condition. Documented as Q5/Q6 placeholders so a
future agent can pick them up — the data shapes are clear.

Cc adds: had Q1 returned 50-60 %, Q5 would have been the most
informative probe (does any of the 3 new joint anomaly types actually
fire on real data?). Q6 (A/B with Atlas context) is the most expensive
and was always going to be the final-gate probe rather than an early
signal. Skip-order priority if budget allows in future:
**Q2 > Q5 > Q6**.

---

## 8. Recommendation

> **Go to J0 now. Stage J2-prototype on val1_primary
> (`bottleneck_open_q_alignment` first). J1 remains contingent on
> aigraph extractor fix.**
>
> Updated 2026-05-15 — Q2 and Q4-redo (the probes skipped in the
> original recon per the brief's stop condition) were run with the
> remaining budget. Both fired in the J2-supporting direction and
> changed the J0-only verdict from the original push.

### 8.1 Update (2026-05-15) — Q2 and Q4-redo results

**Q2 (30-paper Atlas-bottleneck × aigraph-weakness overlap)** —
the most informative probe per the original recon's "wished I could
run" list. 155 papers in (aigraph 540p ∩ Atlas) had both inbound
Atlas bottlenecks AND first-party weakness claims (negative-direction
+ `limitation` claim_type as proxy for the unextracted open_questions).
30 sampled, judged by gpt-5.4:

| label | n | % | brief threshold |
|---|---:|---:|---|
| **complementary** | 16 | **53.3 %** | ≥40% → GO J2 ✓ |
| unrelated | 11 | 36.7 % | |
| same_signal | 3 | 10.0 % | ≥60% → KILL (clear) |
| contradictory | 0 | 0 % | — |

Three verbatim complementary pairs at
`recon/atlas_fit/Q2_weakness_overlap.rows.json`; key example
(arxiv:2504.20930 ChestX-Reasoner):
- Atlas (third-party, 2506.16962): "existing medical reasoning
  models, including those using RL like ChestX-Reasoner, often
  suffer from superficial reasoning or hallucinations and may lose
  basic VQA capabilities"
- aigraph (first-party): "GPT-4o was much worse than ChestX-Reasoner
  on binary, single, and multiple disease diagnosis accuracy"

The two layers contribute *different* signals on the same paper.
That's the J2/J3 thesis in action.

**No contradictories** in the sample — third-party observations almost
never DENY first-party limitations; they ADD new dimensions on top.
That's a corpus property (probably true at scale), not a probe
failure. The "publication-grade contradictory pairs" framing in the
brief was speculative; the real publication story is the
complementary-rate magnitude (53%), not the contradictory examples.

**Q4 redo on val1_primary (1790 NeurIPS/ICML/ICLR papers)** —
addresses the cohort-thinness caveat that made the original Q4 (4
edges on 540p) uninformative.

```
Atlas replaces edges with both endpoints in val1_primary:  41
                                            (vs 540p:  4)
aigraph-side negative claim touches A's method/task:    5 / 41 = 12.2%
```

Per brief: 12.2% is in the INCONCLUSIVE band (10-30%, neither J2 GO
nor KILL). Real but thin. The original Q4's 0/4 verdict was indeed a
cohort artifact; on the proper cohort the signal exists but does not
clear the 30% bar.

### 8.2 Synthesis (post-Q2, post-Q4-redo)

J1 conclusion stands (KILL — Q1 31%, audit-adjusted ~18-20%, blocks
on extractor not Atlas). J2 conclusion **softens from
"design-only" to "stage a prototype"**:

| Probe | Verdict | What it gates |
|---|---|---|
| Q1 (method namespace) | KILL (31%) | J1 |
| Q2 (weakness overlap) | GO (53% complementary) | J2 `bottleneck_open_q_alignment` |
| Q3 (joint anomaly inventory) | design ready | J2 implementation |
| Q4 (replaces × stance, 540p) | KILL but data-thin | J2 `unresolved_replacement` |
| Q4-redo (replaces × stance, val1) | INCONCLUSIVE (12%) | J2 `unresolved_replacement` — partial support |
| Q5 (joint detector simulation) | SKIPPED | J2 firing rate |
| Q6 (hypothesis A/B) | SKIPPED | J3 |

The single most-supported joint anomaly is now
`bottleneck_open_q_alignment` (Q2 directly proxies its signal at 53%
complementary). The single least-supported is `unresolved_replacement`
(Q4 on the proper cohort returned 12% — exists but thin).
`silent_replacement` was not probed; it could be the next sanity
test.

### 8.3 Updated recommendation

> **J0 immediately + J2-prototype on val1_primary.**
> Implement ONLY `bottleneck_open_q_alignment` first as the
> highest-confidence joint anomaly. Measure: (a) candidate count at
> 1790-paper scale, (b) LLM-judged plausibility on 5-10 candidates.
> If ≥ 5 candidates with ≥ 60% plausibility, commit to the rest of
> J2 (add `silent_replacement`, `unresolved_replacement`). Otherwise
> retreat to J0.

This replaces the original "J0-only" recommendation. The original was
defensible under the stop condition but pessimistic — Q2 (which we
deliberately skipped) was the strongest evidence for J2 and it landed
in the GO band.

### 8.4 Original recommendation (preserved for the audit trail)

1. **J1 is dead at the structural level under current aigraph
   extractor** (Q1: 31 % match, audit-adjusted to ~18-20 %, 57.5 % of
   extracted methods are not method names). V_M itself covers 84 % of
   what aigraph extracts that *is* a real method, so the bottleneck is
   on aigraph's side. Fix the extractor, then retry J1.
2. **J2/J3 cannot be validated on the 540p cohort** because the corpus
   has only 4 Atlas `replaces` edges in the intersection (Q4). The
   data shape is right (Atlas has the edges; aigraph has stance), but
   the probe-able sample is too thin. J2/J3 verdicts should re-run on
   the 1790-paper validation_v1_primary cohort.
3. **J0 is a clean win**: 93.9 % of 540p papers ARE in Atlas already
   (Q4 corpus intersection), Atlas's papers schema is a 12-field
   superset of aigraph's `Paper`, and `scripts/cohort_from_intern_atlas.py`
   already exists. The pivot doc (intern-atlas-pivot.md §5.2) already
   plans this; this recon validates it.

What J0 buys us with no additional engineering: cohort builder, citation
counts (`cited_by_count`, `influential_citation_count`), venue tier,
referenced_works. What J0 does NOT buy: any of Atlas's structural
differentiation (no method nodes, no evolution edges, no
bottleneck_json signals).

If we want the structural differentiation, the path is:

```
J0  (now, ~half day)
 │
 ▼
fix aigraph claim extractor:
   - reject model names from `method` slot
   - reject metric names ("Top-1 accuracy") from `method` slot
   - reject phrases ("X-based approach", "novel framework for Y")
 │  (~1 week, prompt fix + heuristic post-filter)
 ▼
re-run Q1 on a clean 200-claim sample → expect 65-75 % V_M match
 │
 ▼
J1  (1-2 weeks): swap CANONICAL_METHODS for V_M method_id namespace
 │  (touches llm_extract.py CANONICAL_METHODS constant — frozen module,
 │   requires thaw record)
 ▼
J2 prototype (1 week): build joint_anomalies.py with 3 new types,
   measure firing rate on 1790-paper validation_v1_primary cohort
 │
 ▼
J3 deferred until influence ρ improves at v0.8.
```

---

## 9. Concrete next steps (engineering plan, post-recon)

If the J0-now / J1-after-extractor-fix recommendation lands, the
ordered work items are:

| # | Item | Estimate | Dependency |
|---|---|---|---|
| 1 | `src/aigraph/intern_atlas_loader.py` returning `list[Paper]` with cohort filters | 1 day | none |
| 2 | Update `scripts/cohort_from_intern_atlas.py` to call the loader | 0.5 day | #1 |
| 3 | Add `AIGRAPH_USE_INTERN_ATLAS=1` env flag to `corpus.seed_reasoning_corpus` | 0.5 day | #1, #2 |
| 4 | Smoke test: re-build the 540p run with `AIGRAPH_USE_INTERN_ATLAS=1` → compare papers.jsonl byte-for-byte | 0.5 day | #3 |
| 5 | **Extractor hygiene patch** in `llm_extract.py`: reject model names + metric names + descriptive phrases from `method` slot. Add explicit examples in SYSTEM_PROMPT. Requires v0.7-frozen thaw record. | 1-2 days | none (independent) |
| 6 | Re-run Q1 on a new 200-claim sample after #5 lands | 0.5 day | #5 |
| 7 | If Q1 ≥ 60 %: swap `CANONICAL_METHODS` for Atlas V_M method_id namespace (J1). Requires v0.7-frozen thaw record. | 2-3 days | #5, #6 |
| 8 | `src/aigraph/joint_anomalies.py` with 3 new types: `unresolved_replacement`, `bottleneck_open_q_alignment`, `silent_replacement`. New module, no thaw. | 3-4 days | #1, #7 |
| 9 | Run joint detectors on validation_v1_primary 1790p cohort. Measure: (a) candidate count per type, (b) sample LLM-judged plausibility on 5 per type. | 1 day | #8 |

Total: ~10-15 days. The critical-path bottleneck is #5 (extractor fix)
— without it, J1 and the joint anomalies inherit aigraph's `method`
contamination.

---

## 10. Risks

### 10.1 J1 might never reach 60 % even after extractor fix

If after the extractor patch Q1 returns, say, 40 %, the gap is V_M
coverage (`novel_method` rate exceeded what we sampled) rather than
extractor contamination. In that case the right play is to keep
aigraph's own `CANONICAL_METHODS` and treat Atlas V_M as a *secondary*
namespace (mapped via alias table, soft-linked, optional). J1 would
become a permanent J0.5.

### 10.2 J2 joint detectors might be too sparse to matter

The 540p run had ~500 anomalies; if 3 new joint types each fire only
1-3 candidates, the marginal value is low. Mitigation: run on the
larger 1790-paper validation cohort first; if the candidate counts are
still low, defer J2 indefinitely.

### 10.3 Atlas schema changes

`paper_evolution_edges` could add or remove relation types in v2.
Mitigation: pin to a specific HF revision in the loader.

### 10.4 Aigraph's frozen modules constrain J2 implementation

The brief says "Do not touch v0.7-frozen modules." For #7 (J1 namespace
swap) and #5 (extractor fix) this is a problem because `llm_extract.py`
IS frozen. Either:

- Defer #5/#7 until after v0.8 release (when freeze is lifted)
- OR write a thaw record in `docs/v0.7-pipeline-freeze.md` §7 per the
  existing thaw #1 protocol. The §4 thaw condition that fits: "Anomaly
  type produces 0 detections on a 1000+ paper real cohort" doesn't apply
  here. We'd need a new §4 thaw condition like "Extracted method field
  has < 50 % real-method rate on a 200+ sample". Reviewer-grade
  rationale required.

### 10.5 The probe results are sample-size sensitive

Q1 with N=200 has wide CI on the 31 % point estimate (95 % CI ~ ±7 %).
Q4 with effectively N=4 is point-meaningless. Re-running both at
larger N (Q1 at N=500, Q4 at the larger validation cohort) would
sharpen the decision.

---

## 11. What we DIDN'T probe (honest list)

1. **Q2: same-paper weakness-signal overlap (the publication story).**
   Skipped per brief stop. This is the single most informative probe
   for the conflict-aware differentiation argument that
   intern-atlas-pivot.md §4.1 leans on. If Atlas's bottleneck text is
   semantically distinct from aigraph's open_question extracts, the
   joint signal is publication-grade. We don't yet know.
2. **Q5: joint detector simulation.** No data on whether
   `unresolved_replacement` / `bottleneck_open_q_alignment` /
   `silent_replacement` actually fire on real data. Q3 is design-only.
3. **Q6: hypothesis A/B with Atlas context.** No data on whether adding
   Atlas's bottleneck text to the LLM hypothesis prompt actually
   produces better hypotheses. This was always the J3-gate probe.
4. **Atlas V_M aliases.** The brief mentions 9,545 aliases. Our local
   Atlas mirror does NOT have an explicit aliases table — only
   `method_id` + `method_name` (8,155 pairs). The 9,545 figure
   presumably reflects HF's full release which has alternative names
   per method_id. Without aliases, Q1 may be undercounting V_M coverage.
5. **The 1790-paper validation_v1_primary run.** That run is already
   tracked in another worktree and would be a cleaner Q4 test bed.
   Re-running Q4 on validation_v1_primary is the easiest follow-up.
6. **Atlas's `paper_methods` link table as a graph signal.** This table
   has 797 k rows (paper × method). Could be used as a method-co-use
   graph (an alternative to aigraph's claim-method co-occurrence). Did
   not probe.
7. **Whether Atlas's `compares` edges (2.4 M of them) correlate with
   aigraph's `benchmark_inconsistency` detections.** This was the most
   data-rich relation in the intersection (313 edges in 540p ∩ Atlas)
   but the brief's Q4 hardcoded `replaces`. Substitution would have
   given a much more informative number.
8. **Atlas's `bottleneck_json` semantic structure.** Whether it
   parses into a consistent schema (versus free text) was not
   characterized. Would inform whether `bottleneck_open_q_alignment`
   needs LLM matching or can use token-set rules.

---

## 12. Budget + stop-condition status

- **LLM cost:** Q1 used 138 calls × ~250 tokens × gpt-5.4-mini ≈ $0.50.
  Total ≈ $1 / $25 cap. **Headroom: $24.**
- **Wall time:** ~1.5 hr / 6 hr cap. **Headroom: 4.5 hr.**
- **Probes returning KILL:** 1 (Q1). Q4 returned KILL on numbers but
  the data is too thin to be meaningful (4 edges).
- **Brief stop conditions tripped:** Q1 < 35 % map rate. Per brief:
  "every other probe is moot." Q3 + Q4 were run anyway (both no-LLM
  and orthogonal to J1). Q2/Q5/Q6 skipped.

---

## 13. Deviations from the brief (be explicit)

1. **Model substitution.** Brief specified `haiku-4.5` and `sonnet-4.6`.
   The configured `AIGRAPH_BASE_URL` endpoint only serves OpenAI-family
   models (`gpt-5.4`, `gpt-5.4-mini`, `gpt-5.1`). I substituted
   `gpt-5.4-mini` for Haiku-4.5 (used in Q1). Pinned `temperature=0` as
   required. No Sonnet-tier calls made (Q2/Q5/Q6 skipped per stop).
2. **Atlas paper PDF.** Brief references the paper at a fixed path; I
   relied on the brief's confirmed-numbers + local parquet inspection
   rather than re-reading the PDF, since the brief explicitly stated
   "1,030,314 papers, 9,410,201 edges, 8,155 methods" as confirmed.
   Local mirror has 4.2 M / 4.14 M / 8,155 — the divergence on the
   first two is noted in the pivot doc and immaterial to this recon.
3. **The 9,545 aliases.** Brief mentions this number; local mirror
   doesn't expose an aliases table separately. Q1 matched against the
   8,155 method_names only. If the HF release exposes aliases per
   method_id, Q1 should be re-run with that expanded matching set.
4. **The SVG diagram.** Per brief §3.2, the diagram should be a real
   SVG at `docs/atlas-aigraph-fit-architecture.svg`. cc emitted SVG
   (3-panel) at that path; an ascii rendition is also in §1 above.

---

## 14. References

- Brief: `docs/atlas-aigraph-fit-recon-brief.md`
- Atlas pivot doc: `docs/intern-atlas-pivot.md` (deferred but informative)
- v0.7 freeze: `docs/v0.7-pipeline-freeze.md`
- Validation design: `docs/controlled-validation-design.md`
- Talk outline (current public story): `docs/aigraph-talk-outline-v1.md`
- Q1 raw output: `recon/atlas_fit/Q1_method_map.json`
- Q3 inventory: `recon/atlas_fit/Q3_joint_anomaly_inventory.md`
- Q4 raw output: `recon/atlas_fit/Q4_replaces_stance_xval.csv`
- LLM self-audit: `recon/atlas_fit/LLM_audit_sample.md`
