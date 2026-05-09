# aigraph Work Status

Single source of truth for what's in flight, what's blocked, what's
parked. Refresh this doc at the start and end of every session.

> Last updated: **2026-05-08**.
> Owner of refresh: whoever closes the session.

---

## Quick view

| Stream | Active | Blocked | Done this week | Parked |
|---|---:|---:|---:|---:|
| S1. Intern-Atlas integration | 1 | 1 | 4 | 0 |
| S2. Controlled-validation execution | 1 | 0 | 1 | 0 |
| S3. OMC talent maintenance | 0 | 0 | 0 | 2 |
| S4. aigraph core | 0 | 0 | 6 PRs | 3 |
| S5. Infrastructure | 0 | 1 | 1 | 0 |

Legend: ▶ active, ⊗ blocked, ✓ done, ⏸ parked, ◯ next-up

---

## S1. Intern-Atlas integration

This is the dominant active stream this week. Reposition aigraph as a
specialist on top of Intern-Atlas's open paper graph.

| Item | Status | Notes |
|---|---|---|
| HF dataset recon (#11) | ✓ | 99.8% typed extraction, 7-class edges, ICLR/ICML/NeurIPS coverage confirmed |
| `intern-atlas-pivot.md` written + merged (#12) | ✓ | dd972d8 |
| `cohort_from_intern_atlas.py` script + run (#16) | ✓ | dd972d8; cohorts at `artifacts/validation_v1/cohorts/`, gitignored |
| `controlled-validation-design.md` amendments (#15) | ✓ | uncommitted, to commit as next push |
| `corpus.intern_atlas_loader` (#14) | ◯ | Half-day refactor. Adds env switch `AIGRAPH_USE_INTERN_ATLAS=1`. Replaces S2/OpenAlex per-paper fetch with parquet load. Needs new module + unit tests + Paper schema adapter |
| POC `/api/eval` baseline (#13) | ⊗ | **Blocked on Intern-Atlas API key** (waitlist application). Without API access we can't measure ρ_aigraph vs ρ_intern_atlas head-to-head |
| `validate_influence_backtest --baseline-eval intern-atlas` flag | ⏸ | Depends on #13 unblocking |
| OMC talent README sync to IA dep | ⏸ | In other repo, lower priority |

**Unblock paths:**
- For #13: apply for API key at https://intern-atlas.opendatalab.org.cn
  /api waitlist. Manual step, takes 1-3 days typically.
- If API access denied: fall back to consuming their static evaluation
  scores from `papers.cited_by_count` correlation only — degraded but
  workable.

---

## S2. Controlled-validation execution

The doc design is locked (§16 of controlled-validation-design.md).
Now needs the actual prediction + outcome runs.

| Item | Status | Notes |
|---|---|---|
| Design doc amendments | ✓ | Uncommitted, this session |
| `validation_preregistration_v1.md` | ◯ | Must be committed BEFORE predictions are locked. ~30 min to draft (5 hypotheses H1-H5 already written in §11) |
| Prediction pipeline run on primary cohort | ⏸ | Needs #14 done first; ~4-6 hour wall time + ~$70 LLM cost |
| Outcome compute (L1 + L2) | ⏸ | Cheap once pipeline runs; pure parquet read |
| Metrics + bootstrap CI + plots | ⏸ | scripts/compute_metrics.py + scripts/plot_validation.py — neither written |
| L3 manual annotation (150 papers × 2 annotators) | ⏸ | Needs annotator pool. Was 100 papers in original budget; pivot upgraded to 150 with saved $400 |
| `prediction_year_cutoff` graph builder guard | ⏸ | Required hard gate against look-ahead leakage (§8.3) — small change in `graph.build_graph` |

**Critical-path order:**
```
#14 corpus loader  →  prediction_year_cutoff guard
                  →  prediction run (lock predictions)
                  →  outcome compute
                  →  metrics + plots
                  →  preregistration commit (concurrent with prediction lock)
                  →  L3 annotation pool kickoff (concurrent)
```

---

## S3. OMC talent (literature-researcher-talent)

Cross-repo work. The talent currently bundles aigraph and assumes its
own corpus. After IA pivot it should declare IA as soft-dependency.

| Item | Status | Notes |
|---|---|---|
| Audit current talent.yaml + entrypoint vs aigraph v0.7 | ⏸ | Not started; needs cc with HF/GitHub access to clone the talent repo |
| Update talent README for IA-soft-dependency | ⏸ | After audit |
| Schema adapter (IA Paper → aigraph Paper) | ⏸ | Plumbing for above |

**Why parked:** S1+S2 are the critical path. Talent maintenance can
wait until aigraph v0.7 is shippable (post-#14, post-prediction-run).

---

## S4. aigraph core (maintenance + deferred features)

| Item | Status | Notes |
|---|---|---|
| Hypothesis extras allow (PR #15) | ✓ | bbb4850 |
| arxiv https + redirect (PR #16) | ✓ | f564545 |
| arxiv retry/backoff (PR #17) | ✓ | 56af695 |
| Hierarchy + multi-grain creator (PR #11) | ✓ | ba3f18c |
| Influence Phase 1 (PR #14) | ✓ | 6754599 |
| Citation stance artifact-read (PR #9) | ✓ | shipped earlier |
| Phase 2 `structural_impact` | ⏸ | Parked until Phase 1 validates on real cohort. Phase 2 design choices need validation findings as input |
| `ρ_grounding_depth = -0.166` anomaly investigation | ⏸ | From PR #14 backtest. May resolve naturally on the new cohort — wait for new data before spending cycles |
| Top-10 precision improvement (currently 20%) | ⏸ | Same — wait for new ρ on Intern-Atlas cohort before tuning |

---

## S5. Infrastructure

| Item | Status | Notes |
|---|---|---|
| Local branch cleanup (#10) | ✓ | 13 branches deleted, stash dropped |
| Remote branch deletion via `cleanup_branches.sh` | ✓ | Executed 2026-05-08 in user-shell session: 7 branches were already gone (squash-merge auto-cleaned), 6 deleted explicitly. Script removed after success. Final remote: `main` + `gh-pages` only |
| Sandbox network egress | ⊗ | "All domains" toggle on but proxy still 403s. May need session restart or admin policy override |

---

## This session's outputs (2026-05-08)

**Committed:**
- `dd972d8` Intern-Atlas pivot: cohort builder + position doc + gitignore

**Uncommitted (to push next):**
- `docs/controlled-validation-design.md` — §3/§4/§5/§6/§9 amendments + new §16 pivot record

**Files written / modified:**
- `scripts/cohort_from_intern_atlas.py` (cohort builder, ran successfully)
- `docs/intern-atlas-pivot.md` (291 lines, strategic positioning)
- `docs/controlled-validation-design.md` (amendments)
- ~~`cleanup_branches.sh`~~ (executed and removed)
- `docs/work-status.md` (this file)

---

## Next session priorities (ranked)

1. **Commit + push the design doc amendments** (5 min, closes #15
   loop). One-liner: `git commit -am "docs: amend
   controlled-validation-design for Intern-Atlas pivot (closes #15)"`
2. **Apply for Intern-Atlas API key** (manual, blocking #13). Fire
   and forget; no agent work involved.
3. **Implement #14 `corpus.intern_atlas_loader`** (half day). Concrete,
   self-contained, unblocks S2 critical path.
4. **Draft `validation_preregistration_v1.md`** (30 min). Must commit
   before predictions lock; H1-H5 already in §11 of design doc.

---

## Maintenance rules (so this doc stays useful)

1. **One row per stream per status** — don't proliferate categories.
2. **Move items to ⏸ Parked** rather than deleting; we want a memory
   of "we considered this, deferred it, why".
3. **Each Active item must have one Next-action sentence** that the
   next session can pick up cold without re-reading anything.
4. **Each Blocked item must have an Unblock path** explaining what
   external thing has to happen and who does it.
5. **Refresh at session boundaries** — top of session: skim "Next
   priorities"; bottom of session: update statuses.
