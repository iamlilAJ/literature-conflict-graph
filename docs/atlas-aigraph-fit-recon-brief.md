# cc Recon Brief — atlas × aigraph structural fit

> **Owner**: cc (Claude Code agent)
> **Goal**: Decide HOW deeply to integrate Intern-Atlas into aigraph,
> not WHETHER. Produce a self-contained recon report that lets the
> human pick a join level (J0 / J1 / J2 / J3) with eyes open.
> **Deliverable**: `docs/atlas-aigraph-fit-recon.md` (8–12 pages),
> + 1 architecture diagram, + a `recon/atlas_fit/` artifact folder with
> all probe outputs.
> **Worktree**: create new worktree on branch `recon/atlas-aigraph-fit`.
> Do NOT touch v0.7-frozen modules or `main`. Worktree only.
> **Budget**: LLM ≤ $25, wall ≤ 6 hr. Stop and report if you blow either.
> **Style**: probes first, conclusions last. If a probe contradicts the
> brief, write down the contradiction — do not flatten findings.

---

## 0. Context you need before starting

Read in this order, do not skip:

1. `docs/intern-atlas-pivot.md` — what we already know about Atlas,
   especially §4 (differentiation) and §8 (risks).
2. `docs/aigraph-talk-outline-v1.md` — current public story; the recon
   should be consistent with what's been claimed publicly or explicitly
   flag where it changes the story.
3. `docs/work-status.md` — current stream state, what's parked vs active.
4. `docs/controlled-validation-design.md` §16 — pivot record.
5. `src/aigraph/models.py` — aigraph's Paper / Claim / Hypothesis schema.
6. `src/aigraph/anomalies.py` — the 8 detectors and the (claim, paper)
   surfaces they touch.
7. `src/aigraph/corpus.py` — the only place that currently knows about
   external data sources.
8. The Intern-Atlas paper at `/Users/liuanjie/Library/Application Support/Claude/local-agent-mode-sessions/.../uploads/2604.28158v2 (1).pdf`
   — §3.1, §3.2, §F (audit accuracy). Confirmed actual numbers:
   1,030,314 papers, 9,410,201 edges, 8,155 methods, 9,545 aliases.
   7 edge types: extends, improves, replaces, adapts, uses_component,
   compares, background. No `contradicts`.

Atlas parquet is at `data/intern_atlas/` (HF mirror). Use polars,
not pandas.

---

## 1. The question you are answering

aigraph and Atlas are two graphs over the same underlying literature
but at different granularity:

- Atlas: forward causal lineage at **method** level (8,155 nodes)
- aigraph: stance + opens at **claim** level (atomic statements)

These are geometrically orthogonal: Atlas describes *how methods
evolved*; aigraph describes *who agrees with whom about a claim*. The
question is whether and how to **join** them.

Four candidate join levels:

| Level | Description |
|---|---|
| **J0** | Atlas as corpus source only. No structural integration. |
| **J1** | Adopt Atlas V_M as canonical method namespace in aigraph claims. Same nodes, no new edges. |
| **J2** | Joint typed multigraph. Atlas 7 edge types + aigraph 5 stance types share node namespace. New anomaly types enabled. |
| **J3** | Hypotheses grounded on 4 axes: explains_claims, touches_methods, addresses_bottleneck, fills_open_question. |

J0 ⊂ J1 ⊂ J2 ⊂ J3. Your job is to find the highest J that the data
actually supports, and to flag where each level breaks.

---

## 2. Probes — run in order, stop on hard fail

Each probe must end with a single sentence in its own paragraph:
**`Verdict: <GO | KILL | INCONCLUSIVE>, reasoning: ...`**.

### Q0. Codebase integration-surface audit  (no LLM)

Before any data probe, audit aigraph's codebase and produce a table:

| File | Function / class | Currently consumes | Place where Atlas plugs in cleanly |
|---|---|---|---|

Cover at minimum: corpus.py, extract.py, llm_extract.py, graph.py,
anomalies.py, hypotheses.py, influence.py, scoring.py, paper_select.py.

Output: section §2 of the report. ~30 min.

### Q1. Method-namespace coverage  (LLM = Haiku, N=200)

For 200 random claims from `artifacts/runs/arxiv-reasoning-v0.7-540p/`
selected outputs, take each claim's `method` field. For each:

1. Fuzzy match → Atlas V_M (canonical + 9545 aliases). Report
   match rate at thresholds {exact, normalized ratio ≥ 0.92,
   token-set ≥ 0.85}.
2. For the unmatched ones, LLM judge: `{novel_method,
   alias_not_in_Atlas, too_generic_to_be_a_method, garbage}`.

Output:
- match-rate histogram
- top-20 unmatched methods with judge labels
- aggregate: "X% of aigraph-extracted methods cleanly map to V_M"

**GO threshold for J1**: ≥ 60% clean map. KILL if < 35%.

### Q2. Same-paper weakness-signal overlap  (LLM = Sonnet, N=30)

Find 30 papers in the intersection of (aigraph 540p corpus) ∩ (Atlas
papers). For each paper P:

- Atlas side: pull all edges `(paper_b → P)` and extract the `bottleneck`
  field from each evidence record. These are *third-party-observed*
  weaknesses of P.
- aigraph side: pull P's `open_questions` and any negative-stance claims
  attributed to P. These are *first-party-acknowledged* or *third-party-
  asserted* weaknesses.

LLM judge each pair on the relation:
`{same_signal, complementary, unrelated, contradictory}`,
plus a 1-sentence explanation.

Output:
- 30-row table
- aggregate distribution
- 3 verbatim "complementary" examples (paste into report)
- 3 verbatim "contradictory" examples (these are gold — they're the
  publication story)

**GO threshold for J2/J3**: ≥ 40% complementary OR contradictory.
KILL if ≥ 60% same_signal (then we're redundant).

### Q3. Joint-anomaly inventory  (no LLM)

For each of the 8 existing anomaly detectors, propose a "joint graph"
variant. For each, specify:

| Existing detector | Joint-graph variant | Trigger condition | Why it's stronger |
|---|---|---|---|

Plus 3 new anomaly types that ONLY exist on the joint graph:

- `unresolved_replacement`: Atlas says B replaces A, but aigraph claims
  about A remain unresolved or B's claims contested
- `bottleneck_open_q_alignment`: Atlas's bottleneck on A ≈ aigraph's
  open_question from A  →  consensus weakness
- `silent_replacement`: aigraph stance shows B contradicts A's claims,
  but Atlas has no `replaces` edge B→A  →  field hasn't acknowledged

Output: §3 of report. Inventory, not implementation. ~45 min.

### Q4. `replaces` ↔ stance cross-validation  (no LLM)

Sample 50 Atlas `replaces` edges where both endpoints are in our 540p.
For each: does aigraph independently extract any negative-stance claim
B→A? Report cross-validation rate.

This is the cheapest probe and most directly answers "do these two
graphs see the same world".

**Soft signal**: > 30% cross-val supports J2; < 10% suggests aigraph
and Atlas are reading the literature very differently.

### Q5. Joint detector simulation  (LLM = Sonnet, N=5)

For the 3 new joint-anomaly types in Q3, simulate the detector on the
540p ∩ Atlas joint sample. For each: how many candidates fire? Output
5 real candidates with full provenance.

**GO threshold for J2**: each new type fires ≥ 5 candidates with
plausible LLM-judged plausibility. KILL if < 2.

### Q6. Hypothesis grounding A/B test  (LLM = Sonnet, N=10)

Take 10 anomalies from the 540p run that have hypotheses generated.
For each, re-generate the hypothesis under two conditions:

- **A**: aigraph-only prompt (current)
- **B**: same prompt + 1 paragraph of "Atlas context": the relevant
  Atlas bottleneck text, lineage edges, and method-level position

LLM judge (Sonnet, different prompt from generator) reads both,
blinded to which is which, picks `{A_better, B_better, tie}` with
a 50-word justification.

**GO threshold for J3**: B wins ≥ 60% with median justification length
≥ 30 words. INCONCLUSIVE if 40–60. KILL if B wins < 40%.

---

## 3. Required outputs

### 3.1 The report — `docs/atlas-aigraph-fit-recon.md`

Sections in this order:

1. **Two-graph geometry** — diagram (markdown ascii or `.svg` link)
2. **Q0 codebase audit** — the integration-surface table
3. **Q1–Q6 findings** — each ≤ 1 page, numeric, with verbatim examples
4. **Recommendation** — single line: "Go to J{n}, here's why" — must
   cite specific Q-findings, not handwave
5. **Concrete next steps** — for the recommended J level, list the
   exact files/PRs needed, with effort estimate per item
6. **Risks** — failure modes that would invalidate the recommendation
7. **What we DIDN'T probe** — list of questions you wanted to answer
   but couldn't within budget. Honesty here matters.

### 3.2 The diagram — `docs/atlas-aigraph-fit-architecture.svg`

A real SVG (not ASCII). 3 panels:
- Panel A: Atlas alone (method nodes + 7 typed edges)
- Panel B: aigraph alone (paper + claim nodes + stance edges)
- Panel C: joint graph at recommended J level, edges color-coded by
  source-graph

This will go into the next deck version. Keep it clean, legible at
1080p, sans-serif.

### 3.3 Artifacts — `artifacts/recon/atlas_fit/`

- `Q1_method_map.json` — all 200 fuzzy match + judge labels
- `Q2_weakness_overlap.csv` — 30-row table
- `Q3_joint_anomaly_inventory.md` — designed types
- `Q4_replaces_stance_xval.csv` — 50-row table
- `Q5_joint_detector_candidates.json` — 5 candidates per new type
- `Q6_ab_test.csv` — 10 paired hypotheses with judge verdicts
- `LLM_audit_sample.md` — your own audit: pick 10 LLM judgments
  from across Q1/Q2/Q5/Q6, re-read with fresh prompt, flag any you
  disagree with. This is the spot-check on your own judge.

---

## 4. Rules of the road

- **Worktree only.** `git worktree add ../aigraph-recon-fit recon/atlas-aigraph-fit`.
  Branch off `main`. Do not modify `v0.7-frozen` modules.
- **Determinism.** All LLM calls temp=0. Pin model versions
  (sonnet-4.6, haiku-4.5). Record model + prompt + version in every
  artifact JSON.
- **No re-running of the conflict_finder.** That path is dead.
- **Stop conditions**: any of —
  - budget hit ($25 LLM, 6 hr wall)
  - 2 consecutive probes return KILL
  - Q1 returns < 35% map rate (every other probe is moot)
- **Honest writing.** If a probe makes you LESS confident in our
  current story, write that down. The deck just shipped saying
  things; the recon is the chance to correct them before they
  become public.
- **No code in the report.** The report is a *decision document*, not
  an engineering plan. The engineering plan is §5 ("concrete next
  steps") and is one-line per item.
- **You may add probes.** If during Q0 you find a structural surface
  that needs its own probe (e.g., the influence formula's
  community_reach dim might natively want Atlas method-community
  data), add Q7 / Q8 with the same Verdict-paragraph discipline.
  Just stay under budget.

---

## 5. When you finish

Push the worktree branch, do NOT open PR. Report back with:

1. Recommended J level + one-sentence reasoning
2. Top 3 surprises (things you expected to find but didn't, or vice
   versa)
3. The 3 best example pairs from Q2 (publication-grade material)
4. Stop-condition status (budget left, time left)
5. The 2 probes you wished you could run but didn't

Then we discuss before any v0.8-aimed implementation work begins.
