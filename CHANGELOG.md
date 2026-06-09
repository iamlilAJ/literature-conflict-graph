# Changelog

All notable changes to **aigraph** (`literature-conflict-graph`) are recorded here.
Versions follow the `v0.7` line that the codebase, docs, and branches already use
as the frozen-pipeline codename — see the note on `v0.7-frozen` below.

## [0.7.0] — 2026-06-09

First tagged release on the **0.7 line**. The package version is realigned from
the stale `0.1.0` / `0.5.0.dev0` / `v0.4.0` sources onto the single `0.7` codename
used throughout the repo (`docs/v0.7-pipeline-freeze.md`, `stable/v0.7-runner-local`,
the frozen-module contract). Collects 60+ commits of MCP / serving work since `v0.4.0`:

- **Semantic relevance gate** (#58) — LLM 0–3 topic-relevance gate on corpus
  retrieval; fail-open, recall-floor-preserving.
- **Quality pass v1** (#59) — extraction robustness + retry/coverage report (#49),
  run-local taxonomy backfill (#50), open-questions / creator hypotheses (#51),
  multi-layer novelty audit with explicit `unknown` state (#52).
- **Runs dashboard** (#60–#64) — which requests ran and how they flowed through the
  11-stage pipeline; live auto-refresh, idea → source-paper drill-down, the
  star / conflict graph (星球图), and the final generated ideas surfaced.
- **Query log** (#66) — read-only grounding calls (`query_hypotheses`,
  `get_idea_report`) are now visible in the dashboard.
- **`research_e2e`** (#67) — one-shot `topic → {idea report, ideas, star-graph + frontend HTML}`.
- **Docs** (#65, #68) — MCP client-guide refresh (dashboard, env vars, git-only
  deploy, `research_e2e` for niche Stage-3 topics).

> **Note — the `0.5` / `0.6` gap.** `0.5.0.dev0` was a dev marker that was never
> released and `0.6` never existed; the line moves straight to `0.7.0` to match the
> long-standing `v0.7` codename. Nothing is missing.

## `v0.7-frozen` — 2026-05-09 (freeze marker, **not** a release)

A tag marking the **frozen-module baseline** for controlled validation, *not* a
version on the release line. At freeze time the package was ~`0.4`; the "0.7" here
is the pipeline codename. The frozen modules (`extract`, `llm_extract`, `graph`,
`anomalies`, `hypotheses`, `llm_hypotheses`, `creator`, `influence`, `scoring`,
`paper_select`) may not change without a recorded thaw. See
`docs/v0.7-pipeline-freeze.md`.

## [0.4.0] — 2026-05-31

Atlas overlap filter, TF-IDF semantic opt-in, calibration (#37).

## [0.3.0] — 2026-05-08

Controlled-validation design docs; corpus / extraction refinements.
(This is the version the `literature-researcher-talent` wrapper historically pinned.)

## [0.2.0] — 2026-04-27

Creator-mode hypothesis pipeline + 24/7 arXiv corpus builder. First public GitHub Release.

## [0.1.0]

Initial graph-based literature conflict explorer.

[0.7.0]: https://github.com/iamlilAJ/literature-conflict-graph/releases/tag/v0.7.0
[0.4.0]: https://github.com/iamlilAJ/literature-conflict-graph/releases/tag/v0.4.0
[0.3.0]: https://github.com/iamlilAJ/literature-conflict-graph/releases/tag/v0.3.0
[0.2.0]: https://github.com/iamlilAJ/literature-conflict-graph/releases/tag/v0.2.0
