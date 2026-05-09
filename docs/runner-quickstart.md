# aigraph runner — quickstart

A user with their own LLM access + a paper corpus can run the full
v0.7-frozen idea-generation pipeline end-to-end in a single command.

> Tagged against [`v0.7-frozen`](./v0.7-pipeline-freeze.md). Pipeline
> internals are sealed; what changes between runs is data (`--corpus`)
> and orchestration parameters (`--keywords`, `--max-papers`,
> `--max-anomalies`).

## TL;DR

```bash
# 1. configure your LLM in .env (one-time)
cat > .env <<'EOF'
OPENAI_API_KEY=sk-...
AIGRAPH_BASE_URL=https://your-llm-endpoint/v1
AIGRAPH_MODEL=your-model-id
EOF

# 2. these two env vars stop reasoning models from timing out on
#    long claim-extraction calls. Strongly recommended.
export AIGRAPH_LLM_TIMEOUT=180
export AIGRAPH_REASONING_EFFORT=minimal

# 3. run end-to-end on a local corpus
python3 scripts/run_local_corpus.py \
    --corpus data/corpus/arxiv_reasoning \
    --max-papers 100 \
    --year-min 2023 \
    --keywords "reasoning,chain-of-thought,planning,tool use,agent" \
    --out artifacts/runs/my-run \
    --generator llm
```

Output lands in `artifacts/runs/my-run/`. Top-10 ideas with evidence
+ minimal-test annotations are in `selected_hypotheses.md`.

## Corpus contract

Whatever you pass to `--corpus` must be a directory with:

```
<corpus>/
├── papers.jsonl                    # one Paper record per line
└── artifacts/
    └── <safe_paper_id>/
        ├── sections.json           # parsed paper sections (required)
        ├── sentences.json          # per-sentence index (optional)
        └── text.json               # raw text (optional)
```

`<safe_paper_id>` is `paper_id` with `:` replaced by `__` and `/`
replaced by `_`. The bundled `data/corpus/arxiv_reasoning/` follows
this layout for 540 papers.

The Paper record is the `aigraph.models.Paper` schema. Minimum
required fields: `paper_id`, `title`, `year`, `venue`, `abstract`.

## What the runner does

```
filter (year + keywords + has-sections + sort + take top N)
    │
    ▼
[1] extract       claims via LLM, reading from sections.json
[2] build-graph   typed graph of paper / claim / method / task / dataset
[3] detect-anomalies   8-type anomaly detection
[4] generate-hypotheses   per-anomaly LLM hypothesis (3 / anomaly avg)
[5] build-hierarchy   community / cluster aggregation
[6] predict-influence   4-dim Phase 1 score per hypothesis
[7] select   MMR top-K with diversity
    │
    ▼
artifacts/runs/<id>/selected_hypotheses.md
```

## Two-script split

Use **`run_local_corpus.py`** for fresh end-to-end. It runs all 7
stages.

Use **`finish_local_run.py`** when you have a half-built run (papers,
claims, graph, anomalies already emitted) and want to skip ahead. Also
useful when anomaly count blows up — it caps to top-N by
`topology_score` before generate-hypotheses.

```bash
# fresh end-to-end
python3 scripts/run_local_corpus.py --corpus ... --out artifacts/runs/X

# resume from existing partial run (anomalies > spec, cap at top-50)
python3 scripts/finish_local_run.py --out artifacts/runs/X --max-anomalies 50
```

`run_local_corpus.py` does NOT cap anomalies by default — if your
corpus produces > ~100 anomalies, the serial generate-hypotheses stage
will be slow. Either rerun `finish_local_run.py` with a cap, or
modify the script for your scale.

## Output layout

```
artifacts/runs/<id>/
├── papers.jsonl                # selected subset
├── claims.jsonl                # extracted (LLM)
├── graph.json                  # typed graph
├── anomalies.jsonl             # full set
├── anomalies_top.jsonl         # capped subset (only if finish_local_run cap'd)
├── hypotheses.jsonl            # generated
├── hypotheses_scored.jsonl     # + influence_score
├── hierarchy.json              # cluster aggregation
├── selected_hypotheses.md      # top-K rendered report (the deliverable)
└── run_metadata.json           # provenance: git_sha + git_tag + counts + wall
```

## Replacing data or algorithm

**To change data**: point `--corpus` at a different directory. Same
script, same scripts, same v0.7-frozen modules.

**To change algorithm**: branch off `v0.7-frozen`, modify any of the
10 listed-as-frozen modules in `docs/v0.7-pipeline-freeze.md` §1, run
the same scripts, tag a new version. The runner scripts themselves
do not need changes.

## Cost / wall time reference

Empirical numbers from the bundled `arxiv-reasoning-v0.7-100p` run:

```
papers:        100   (filtered from 2746 with full-text)
claims:        407   (4.07 / paper, full-text extraction)
anomalies:     91    (6 of 8 types fired)
top anomalies: 50    (cap by topology_score)
hypotheses:    ~150  (3 per anomaly avg)
selected:      10

extract:       21 min  (4 workers)
hyp-gen:       25 min  (serial)
other stages:  ~2 min
total wall:    ~50 min

LLM cost:      ~$5-7 total
```

These numbers depend heavily on (model, paper length, anomaly count).
Treat as orders of magnitude.

## Troubleshooting

- **All extract calls time out**: bump `AIGRAPH_LLM_TIMEOUT` (default
  45s) and set `AIGRAPH_REASONING_EFFORT=minimal`. Reasoning models
  on long papers exceed 45s easily.
- **Anomaly count > 100**: use `finish_local_run.py --max-anomalies
  50` for resume. Generate-hypotheses is serial; > 100 makes it
  hours.
- **0 anomalies of a specific type**: the freeze doc §4 condition 2
  flags this on a 1000+ paper cohort. On smaller cohorts (100
  papers), 2-3 zero types is normal.
- **No artifacts found** for a paper: check `<corpus>/artifacts/<safe_id>/`
  exists with `sections.json`. The filter in `run_local_corpus.py`
  drops these.
