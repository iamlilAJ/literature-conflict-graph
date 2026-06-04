# aigraph idea-generation pipeline — architecture

How a topic becomes research ideas, end to end. The system has two halves:
**① build a corpus** (write; heavy LLM; minutes) and **② serve ideas** (read;
mostly 0 LLM; sub-second). A corpus is built once and queried many times.

```
topic
  │
  ▼
┌──────────── ① BUILD CORPUS (run_pipeline, src/aigraph/server.py) ────────────┐
│  fetch papers → extract claims → build graph → detect anomalies              │
│              → generate critic hypotheses + insights → score/select          │
│              → overview + report + graph                                     │
│  everything persisted under artifacts/runs/<run_id>/                         │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────── ② SERVE IDEAS (read the cached files) ───────────────────────────┐
│  query_hypotheses · get_idea_report · get_conflict_graph                      │
│  generate_ideas (6-tier cascade) · research_ideas (resolve-or-build + cascade)│
└───────────────────────────────────────────────────────────────────────────────┘
```

## Phase ① — build the corpus (`run_pipeline`)

`start_run(topic)` enqueues a request; a single background worker
(`SearchService._worker_loop`) runs `run_pipeline` (server.py:309) through these
ordered stages. Output files land in `artifacts/runs/<run_id>/`.

| # | stage | function | LLM? | writes |
|---|---|---|---|---|
| 1 | fetch papers | `fetch_arxiv/openalex_papers` (+ topic decompose in `submit`) | topic decompose: 1 call | `papers.jsonl` |
| 2 | **extract claims** | `extract_claims_with_status` | **1 call per paper** (thread-pooled) | `claims.jsonl` |
| 3 | build graph | `build_graph` / `save_graph` | no | `graph.json` |
| 4 | **detect anomalies** | `detect_anomalies` | no (rules) | `anomalies.jsonl` |
| 5 | critic hypotheses | `generate_hypotheses(generator=TemplateGenerator())` | no (template) | `hypotheses.jsonl` |
| 6 | insights | `generate_insights` + `prune_insights` (LLM rewrites prose) | a few calls | `raw_insights.jsonl`, `insights.jsonl` |
| 7 | score + select | `score_all` + `select_mmr` | no | — |
| 8 | overview | `build_search_overview` | no | `overview.json` |
| 9 | report + viz | `render_report`, `render_visualization` | no | `selected_hypotheses.md`, `index.html` |
| 10 | community ingest | `ingest_run` | no | `_community/` |

The LLM cost concentrates in stages 1, 2, 6. **Stage 2 is the bottleneck for
reliability** — it makes one LLM call per paper, so a rate-limited / slow
endpoint fails the whole run (`status:"error"`, message
`"the provider is rate-limited right now"` or `"read operation timed out"`).

### A Claim carries far more than a sentence

`extract_claims_with_status` produces a `Claim` per paper with ~40 fields
(models.py:188): not just `claim_text`, but `method`, `task`, `dataset`,
`metric`, `result`, `direction`, plus extracted `claim_type` (one of
`performance_improvement|limitation|comparison|setting_effect|mechanism`),
`mechanism`, `failure_mode`, `assumption`, `evaluation_protocol`,
`canonical_method`, `canonical_task`, `domain`, `risk_type`. These extra fields
are what the cascade's fallback tiers mine when no anomalies form.

## The anomaly bottleneck (why phase ① can yield 0 hypotheses)

Stages 4→5 are the crux: **hypotheses depend entirely on anomalies.**

```
detect_anomalies → generate_hypotheses(anomalies, …)
```

`detect_anomalies` is rule-based: it recognizes 7 typed cross-paper conflict
patterns, the canonical one being ≥2 claims that share the same `method`+`task`
but disagree (`benchmark_inconsistency` / `impact_conflict`); the others are
`setting_mismatch`, `metric_mismatch`, `evidence_gap`, `community_disconnect`,
`bridge_opportunity`, `replication_conflict`. A **too-new / too-homogeneous corpus** — where every paper invents
its own method name (FORGE, SAGER, MobEvolve, …) so no two claims share
`method`+`task` — forms **0 anomalies → 0 hypotheses → empty
`selected_hypotheses.md`**.

Crucially, **both** critic and creator generation are anomaly-gated:
`generate_creator_hypotheses` (creator.py:255) *iterates over anomalies* too,
so it also returns nothing with 0 anomalies — regardless of how many open
questions exist. Only **insights** (community-based, anomaly-independent;
insights.py never reads its `anomalies` arg) survive a 0-anomaly corpus.

That is the exact failure a user hits: the run "succeeded" (papers, claims,
1 insight) but `selected_hypotheses.md` is empty, so it *looks* like nothing
was produced.

## Phase ② — serve ideas with a non-empty guarantee (`generate_ideas`)

`scripts/idea_cascade.py` `generate_ideas(run_dir, topic, min_ideas)` cascades
six tiers, highest-signal first, stopping once it has `min_ideas`:

| tier | label | source | LLM? | needs anomalies? |
|---|---|---|---|---|
| A | critic-conflict | `hypotheses.jsonl` via `query_records(hyp_kind="critic")` | no (read) | yes → 0 on barren corpus |
| B | creator-newmethod | `creator_hypotheses.jsonl` | no (read) | yes → 0 on barren corpus |
| C | community-bridge | `insights.jsonl` (each `transfer_suggestion` is a direction) | no (read) | **no** |
| D | method-extension | per-paper LLM "extend this method" | **yes** | no |
| E | limitation-forward | LLM over `limitation`/`failure_mode` claims | **yes** | no |
| F | paper-seeded | deterministic from abstracts/limitations | **no** | no |

- **A/B/C are 0-LLM reads** of artifacts that phase ① already computed.
- **D/E call the LLM** at query time, then **cache** results to
  `<run>/forward_ideas.jsonl`; a repeat call for the same run reads the cache
  and is 0-LLM, sub-second.
- **F is the deterministic backstop.** It turns each top topic-ranked paper's
  abstract (or self-reported limitation, or title) into a replicate-and-ablate
  direction with **no LLM round-trip**. This makes the non-empty guarantee
  *structural*: it holds even if the LLM endpoint is unreachable or returns
  unparseable/truncated JSON.

So "0 LLM" never means the system avoids LLMs — it means *this particular
`generate_ideas` call* didn't make one, because the expensive work was already
done at build time (A/B/C) or on a prior call (D/E cache). The LLM spend is
front-loaded into building the corpus and the first idea generation.

### Worked example (the friend's run)

```
"agentic memory for self-evolving agent"  (20 papers, 53 claims, 39 distinct methods)
  phase ① → 0 anomalies → 0 hypotheses, but 1 community insight
  phase ② generate_ideas:
     Tier A/B = 0   (no anomalies)
     Tier C   = 1   (SAGER↔FORGE cross-community bridge from insights.jsonl)
     Tier D   = 4   (per-paper LLM extensions: extends OEP, SEARL, MobEvolve, …)
     → 5 ideas, guaranteed_nonempty = true
```

## `research_ideas` — the one-shot wrapper

`research_ideas(topic, …)` glues phases ① and ② together:

1. **reuse** (default): `resolve_best_run` finds the best existing corpus whose
   topic matches (query-token coverage ≥ 0.34, requires non-empty
   `claims.jsonl`, ranks by completeness so a finished run beats a failed
   papers-only run) → runs `generate_ideas` immediately, 0 build cost.
2. **build**: otherwise submit a fresh corpus build, poll up to `wait_seconds`,
   then cascade. `wait_seconds=0` returns `{status:"building", run_id}` so the
   caller polls `get_run_status` then calls `generate_ideas`.

Reuse threshold caveat: a short topic sharing one content word with an existing
run (e.g. "bayesian **memory**" vs "agentic **memory**…") clears 0.34 and
reuses the wrong corpus — pass `reuse:false` for a genuinely new topic.

## Run directory contents (the "database")

```
artifacts/runs/<run_id>/
  query.txt                 the topic
  status.json               build progress / done / error
  papers.jsonl              fetched papers (title, abstract, selection_score…)
  claims.jsonl              ~40-field structured claims, 1+ per paper
  graph.json                method/task/dataset claim graph
  anomalies.jsonl           7-type cross-paper conflicts (may be empty)
  hypotheses.jsonl          critic hypotheses (empty if no anomalies)
  creator_hypotheses.jsonl  creator hypotheses (only if backfilled)
  insights.jsonl            cross-community bridges (anomaly-independent)
  overview.json             headline, top_papers, reading_path, hidden_bridges
  selected_hypotheses.md    rendered Stage-3 report
  index.html                interactive D3 graph
  forward_ideas.jsonl       cached Tier D/E ideas (written by generate_ideas)
  atlas_overlap.jsonl       per-hyp Likert anchoring sidecar (if precomputed)
```

## Where each MCP tool sits

| tool | phase | LLM at call time |
|---|---|---|
| `start_run` | ① trigger build | heavy (background worker) |
| `get_run_status` | ① poll | no |
| `list_runs`, `get_run_summary` | ② list | no |
| `query_hypotheses`, `get_idea_report`, `get_conflict_graph` | ② read cached hyps/graph | no |
| `generate_ideas` | ② cascade on a run | mostly 0; D/E once then cached |
| `research_ideas` | ①+② resolve-or-build → cascade | reuse: 0; new build: heavy |

## Related

- `docs/aigraph-mcp-client-guide.md` — connection patterns, tool reference, ops.
- `docs/method13-atlas-overlap-filter-deployed.md` — the Atlas overlap filter on
  the read path (`get_idea_report` / `query` drop tangential hypotheses).
- `docs/method12-likert-vs-utility-verdict.md` — why the Atlas filter exists.
