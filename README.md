# aigraph

A graph-based literature conflict explorer for AI papers.

`aigraph` turns a small paper collection into typed claims, builds a heterogeneous
claim graph, detects local conflict/gap regions, and renders an interactive graph
explorer. The goal is to help people see where a research community disagrees,
which evidence supports each side, and what checks might be worth running next.

This is an alpha research prototype. It is not a claim that the system can
automatically determine scientific truth.

## What It Does

```text
papers -> claims -> claim graph -> conflicts/gaps -> possible explanations -> visualization
```

- Fetches abstract-level paper metadata from OpenAlex.
- Extracts structured claims with either deterministic rules or an LLM.
- Builds a typed graph over papers, claims, methods, tasks, datasets, metrics,
  baselines, and settings.
- Detects benchmark inconsistencies, setting mismatches, and bridge opportunities.
- Generates possible explanations and minimal follow-up checks.
- Renders a static HTML graph explorer with paper links and evidence claims.

## Install

```bash
python -m pip install -e .
python -m pip install -e '.[real]'  # OpenAlex + OpenAI-compatible LLM extraction
python -m pip install -e '.[dev]'   # tests
```

Requires Python 3.10+.

## Quick Demo, No API Key

```bash
aigraph run-demo
aigraph visualize --input-dir outputs --output outputs/index.html
```

Open `outputs/index.html` in a browser.

The synthetic demo is deterministic and runs without network access or API keys.

## Included Real-Paper Demo

The repository includes a small sanitized RAG demo in `examples/rag_demo/`.
It was produced from OpenAlex metadata plus LLM claim extraction over a handful
of retrieval-augmented generation papers.

```bash
open examples/rag_demo/index.html
```

The demo includes:

- 5 real paper records with OpenAlex links.
- 9 extracted claims.
- 39 graph nodes and 50 graph edges.
- 1 detected benchmark inconsistency around RAG on domain QA.
- 5 possible explanations / follow-up checks.

To keep the repository lightweight and redistribution-friendly, this example
keeps paper titles, years, venues, and links, but omits full abstracts and paper
text. The extracted claims still include short evidence spans for inspection.

## Real-Paper Demo

Create a local `.env` file or export environment variables:

```bash
OPENAI_API_KEY=...
AIGRAPH_MODEL=gpt-5.4-mini
AIGRAPH_BASE_URL=
AIGRAPH_MAILTO=you@example.com
```

Then run:

```bash
aigraph run-real-demo \
  --query "retrieval augmented generation large language models" \
  --from-year 2020 \
  --to-year 2026 \
  --limit 5 \
  --output-dir outputs/openalex_rag

aigraph visualize \
  --input-dir outputs/openalex_rag \
  --output outputs/openalex_rag/index.html
```

LLM extraction calls the model once per paper. Start with `--limit 5` or
`--limit 10` before scaling up.

## CLI

```bash
aigraph fetch-openalex --query "retrieval augmented generation" --limit 20 --output data/papers.jsonl
aigraph extract --input data/papers.jsonl --output outputs/claims.jsonl --extractor llm
aigraph build-graph --claims outputs/claims.jsonl --output outputs/graph.json
aigraph detect-anomalies --graph outputs/graph.json --claims outputs/claims.jsonl --output outputs/anomalies.jsonl
aigraph generate-hypotheses --anomalies outputs/anomalies.jsonl --claims outputs/claims.jsonl --output outputs/hypotheses.jsonl
aigraph select --hypotheses outputs/hypotheses.jsonl --claims outputs/claims.jsonl --anomalies outputs/anomalies.jsonl --output outputs/report.md
aigraph visualize --input-dir outputs --output outputs/index.html
```

## Output Files

- `papers.jsonl`: input paper metadata.
- `claims.jsonl`: extracted typed claims with evidence spans.
- `graph.json`: NetworkX node-link graph.
- `anomalies.jsonl`: detected conflict/gap regions.
- `hypotheses.jsonl`: possible explanations and follow-up checks.
- `selected_hypotheses.md`: scored Markdown report.
- `index.html`: static D3 graph explorer.

## Current Limitations

- Abstract-level only by default: no PDF parsing, tables, figures, or section-level grounding.
- LLM-extracted claims can be noisy and should be verified by humans.
- Claim links currently resolve to source papers, not exact PDF paragraphs.
- Canonicalization of methods/tasks is heuristic.
- Hypothesis/explanation generation is template-based in the open-source MVP.
- Scoring weights are hand-set, not learned from human feedback.

## Development

```bash
pytest -q
```

Tests use fake clients for OpenAlex and LLM calls; they do not make network or API requests.

## License

MIT
