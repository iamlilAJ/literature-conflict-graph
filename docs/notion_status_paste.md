# aigraph — Project Status

> **粘贴这份到 Notion**:打开 Notion → 新建 page → 命名 `aigraph` 或 `Project Status` → 把下面所有内容(从下一行到文末)复制粘贴进去。Notion 会自动渲染 markdown 表格 + headers + 引用块。
>
> 如果想以后让 cc 直接更新 Notion,创建后 page 右上 `⋯` → `Connect to` → 选 `Claude` MCP,共享后 cc 拿到 page URL 就能写。

---

> Mirror of `docs/work-status.md` from the repo. Refresh at session boundaries.
> Source of truth lives in git; this page is a derived dashboard.
>
> **Last refresh:** 2026-05-08
> **Repo:** https://github.com/iamlilAJ/literature-conflict-graph
> **GitHub Project:** https://github.com/users/iamlilAJ/projects/2

## Quick view

| Stream | Active | Blocked | Done this week | Parked |
| --- | --- | --- | --- | --- |
| S1. Intern-Atlas integration | 1 | 1 | 5 | 0 |
| S2. Controlled-validation execution | 1 | 0 | 1 | 0 |
| S3. OMC talent maintenance | 0 | 0 | 0 | 2 |
| S4. aigraph core | 0 | 0 | 6 PRs | 3 |
| S5. Infrastructure | 0 | 0 | 3 | 0 |

Legend: ▶ active · ⊗ blocked · ✓ done · ⏸ parked · ◯ next-up

---

## TL;DR — what shipped today (2026-05-08)

Full strategic pivot to consume **Intern-Atlas** (Shanghai AI Lab, MIT-licensed, 4.2M papers + 4.14M typed edges, released May 1).

- ✓ Recon: 99.8% typed-extraction completeness, 7-class edges, ICLR/ICML/NeurIPS coverage sufficient
- ✓ `docs/intern-atlas-pivot.md` — strategic positioning, 3 differentiator planks (collapsed to 2.5)
- ✓ `docs/controlled-validation-design.md` v0.2 — 5 sections amended + new §16 pivot record
- ✓ `scripts/cohort_from_intern_atlas.py` — 4790-paper cohort sampled
- ✓ `scripts/atlas_conflict_finder.py` — pure-polars conflict miner from Atlas's `impact_json`, no LLM, no API key
- ✓ Branch hygiene: 13 stale branches deleted (local + remote)
- ✓ GitHub Project board created: https://github.com/users/iamlilAJ/projects/2
- ✓ `docs/work-status.md` — single source of truth tracker

**Net commits:** `dd972d8` + `0f5c42a`

---

## S1. Intern-Atlas integration

Dominant active stream this week. Reposition aigraph as a specialist on top of Intern-Atlas's open paper graph.

| Item | Status | Notes |
| --- | --- | --- |
| HF dataset recon | ✓ | 99.8% typed extraction, 7-class edges |
| `intern-atlas-pivot.md` | ✓ | dd972d8 |
| `cohort_from_intern_atlas.py` script + run | ✓ | 4790 papers across 3 cohorts |
| `controlled-validation-design.md` amendments | ✓ | 0f5c42a |
| `atlas_conflict_finder.py` — pure-polars conflict miner | ✓ | Ready to run |
| `corpus.intern_atlas_loader` (#14) | ◯ | Half-day refactor; replaces S2/OpenAlex per-paper fetch |
| POC `/api/eval` baseline | ⊗ | **Blocked on Intern-Atlas API key** (waitlist) |
| `validate_influence_backtest --baseline-eval intern-atlas` | ⏸ | Depends on `/api/eval` unblocking |
| OMC talent README sync to IA-soft-dep | ⏸ | Other repo, lower priority |

**Unblock paths:**
- For `/api/eval`: apply at https://intern-atlas.opendatalab.org.cn/api waitlist (1-3 days)
- If denied: degrade to consuming `papers.cited_by_count` correlation only

---

## S2. Controlled-validation execution

Doc design locked (§16). Now needs prediction + outcome runs.

| Item | Status | Notes |
| --- | --- | --- |
| Design doc amendments | ✓ | 0f5c42a |
| `validation_preregistration_v1.md` | ◯ | 30 min draft; H1-H5 already in §11 |
| Prediction pipeline run on primary cohort | ⏸ | Blocked on #14; ~4-6 hr wall + ~$70 LLM cost |
| Outcome compute (L1 + L2) | ⏸ | Cheap once pipeline runs; pure parquet read |
| Metrics + bootstrap CI + plots | ⏸ | scripts/compute_metrics.py + plot_validation.py — neither written |
| L3 manual annotation (150 papers × 2 annotators) | ⏸ | Needs annotator pool |
| `prediction_year_cutoff` graph builder guard | ⏸ | Hard gate against look-ahead leakage |

**Critical-path:** #14 corpus loader → cutoff guard → prediction run → outcome compute → metrics + plots → preregistration commit (concurrent with prediction lock) → L3 annotation (concurrent)

---

## S3. OMC talent (literature-researcher-talent)

Cross-repo. After IA pivot the talent should declare IA as soft-dependency.

| Item | Status | Notes |
| --- | --- | --- |
| Audit current talent.yaml + entrypoint | ⏸ | Needs cc with HF/GitHub access to clone |
| Update talent README for IA-soft-dependency | ⏸ | After audit |
| Schema adapter (IA Paper → aigraph Paper) | ⏸ | Plumbing for above |

**Why parked:** S1+S2 are the critical path.

---

## S4. aigraph core (maintenance + deferred)

| Item | Status | Notes |
| --- | --- | --- |
| Hypothesis extras allow (PR #15) | ✓ | bbb4850 |
| arxiv https + redirect (PR #16) | ✓ | f564545 |
| arxiv retry/backoff (PR #17) | ✓ | 56af695 |
| Hierarchy + multi-grain creator (PR #11) | ✓ | ba3f18c |
| Influence Phase 1 (PR #14) | ✓ | 6754599 |
| Citation stance artifact-read (PR #9) | ✓ | shipped earlier |
| Phase 2 `structural_impact` | ⏸ | Wait for Phase 1 cohort validation |
| `ρ_grounding_depth = -0.166` anomaly | ⏸ | May resolve naturally on new cohort |
| Top-10 precision improvement (currently 20%) | ⏸ | Wait for new ρ |

---

## S5. Infrastructure

| Item | Status | Notes |
| --- | --- | --- |
| Local branch cleanup | ✓ | 13 deleted, stash dropped |
| Remote branch deletion | ✓ | Final remote: main + gh-pages only |
| GitHub Project board setup | ✓ | https://github.com/users/iamlilAJ/projects/2 — 10 issues seeded |

---

## Differentiation story (from `intern-atlas-pivot.md` §4)

Atlas covers 80% of what aigraph initially aimed at. The remaining 20% — defensible plank — is **2.5 dimensions**:

1. **Conflict-aware layer (full strength)** — Atlas's 7 edge types are all causal (extends/improves/replaces/adapts/uses_component/compares/background). No `contradicts`. aigraph's `anomalies.py` produces 5 conflict types (impact_conflict / replication_conflict / benchmark_inconsistency / metric_mismatch / setting_mismatch) Atlas doesn't expose.

2. **Author-acknowledged gaps via `open_questions` extraction (full strength)** — Atlas extracts bottlenecks **between** papers (B describing A's weakness). aigraph extracts limitations **within** a paper (author's own future-work / limitations). Complementary signal.

3. **Open + locally-runnable influence/idea evaluation (half strength)** — Atlas's 5-dim evaluator (ρ=0.81 with experts) is API-only blackbox. aigraph's Phase 1 4-dim is open, deterministic, locally-runnable. Important for OMC talents in air-gapped contexts; not the lead story.

---

## Next session priorities (ranked)

1. **Run `atlas_conflict_finder.py`** on the 4M-edge data — first empirical test of differentiator plank #1, no LLM, free.
2. **Implement `corpus.intern_atlas_loader` (#14)** — half day, unblocks S2 critical path.
3. **Apply for Intern-Atlas API key** — 5 min, async; unblocks /api/eval baseline (1-3 days).
4. **Draft `validation_preregistration_v1.md`** — 30 min; H1-H5 already in design doc §11.

---

## Maintenance rules

1. **One row per stream per status** — don't proliferate.
2. **⏸ Parked over delete** — keep memory of decisions.
3. **Each Active item must have one Next-action sentence** — pick up cold.
4. **Each Blocked item must have an Unblock path** — what + who.
5. **Refresh at session boundaries** — top: skim Next priorities; bottom: update statuses.

---

## Data inventory (gitignored, local)

| Path | Size | Content |
| --- | --- | --- |
| `data/intern_atlas/` | 2.9 GB | 4 configs: 4.2M papers + 4.1M edges + 797k methods + 60 method-relations |
| `data/corpus/arxiv_reasoning/` | 3.0 GB | Legacy corpus, 2746 papers + full text |
| `artifacts/validation_v1/cohorts/` | 2 MB | 4790 papers across 3 stratified cohorts |

---

## Doc inventory (committed)

- [`intern-atlas-pivot.md`](https://github.com/iamlilAJ/literature-conflict-graph/blob/main/docs/intern-atlas-pivot.md) — 291 lines, strategic positioning
- [`controlled-validation-design.md`](https://github.com/iamlilAJ/literature-conflict-graph/blob/main/docs/controlled-validation-design.md) — 780 lines, v0.2 with pivot amendments
- [`influence-prediction-design.md`](https://github.com/iamlilAJ/literature-conflict-graph/blob/main/docs/influence-prediction-design.md) — 904 lines, 5-dim scoring design
- [`influence-prediction-validation.md`](https://github.com/iamlilAJ/literature-conflict-graph/blob/main/docs/influence-prediction-validation.md) — 105 lines, Phase 1 backtest record
- [`work-status.md`](https://github.com/iamlilAJ/literature-conflict-graph/blob/main/docs/work-status.md) — this dashboard's source
