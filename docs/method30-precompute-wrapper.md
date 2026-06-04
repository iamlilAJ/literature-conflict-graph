# Method 30 — `precompute_atlas_overlap.py` operator wrapper

**Date:** 2026-05-31
**Method 30 of:** /loop "test different methods". Closes the deployment UX story for the Atlas filter.

## What ships

`scripts/precompute_atlas_overlap.py` — single command that does the 3-step Atlas overlap precompute end-to-end for a run directory.

### Usage

```bash
python scripts/precompute_atlas_overlap.py \
    --run artifacts/runs/<run-id> \
    --include-creator
```

That's it. The wrapper:

1. **Stage 1 (local).** Loads `hypotheses_scored.jsonl` and, with `--include-creator`, `creator_hypotheses.jsonl`. TF-IDF retrieves top-K Atlas bottleneck quotes per hyp from `artifacts/atlas_test/method3_atlas_quotes.jsonl`. Writes `<run>/atlas_judge_input.jsonl`.

2. **Stage 2 (remote).** scp's the judge input to `admin@8.208.118.99:/tmp/atlas_judge_input.jsonl`, kicks off `run_method3_judge.py` in a tmux session at 6 workers, polls every 30s for `ALL_DONE`, and pulls the output back to `<run>/atlas_judge_output.jsonl`.

3. **Stage 3 (local).** Reads the judge output, converts to sidecar format (`hypothesis_id`, `atlas_overlap`, the 5 Likert axes, `closest_quote`, `kind`), merges with any existing `<run>/atlas_overlap.jsonl` (existing records preserved unless overwritten by new ones for the same hyp_id). Prints the overlap distribution summary.

After this completes, query-time consumers — `aigraph_query.py --min-atlas-overlap 3` and the MCP `get_idea_report` — automatically use the new sidecar. For production, the operator scp's the resulting `atlas_overlap.jsonl` to the matching path on the aigraph server (the wrapper prints a reminder).

### Flags

- `--host` (default `admin@8.208.118.99`)
- `--remote-script` (default `/tmp/run_method3_judge.py`) — path to the Likert judge on the host
- `--quotes` (default `artifacts/atlas_test/method3_atlas_quotes.jsonl`) — Atlas bottleneck quotes corpus
- `--k-top` (default 5) — TF-IDF candidates per hyp passed to the judge
- `--workers` (default 6) — remote LLM concurrency (Kimi rate-limits ~6-8)
- `--include-creator` — also score creator_hypotheses.jsonl (recommended; the MCP defaults `kind=creator`)
- `--skip-judge` — stage 1 only; for ops that want to inspect the judge input before LLM calls

### Smoke test

```
$ python scripts/precompute_atlas_overlap.py \
      --run artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1 \
      --include-creator --skip-judge

[stage 1] retrieving top-5 Atlas quotes for 378 hyps (corpus: 1607 quotes)
[stage 1] wrote 378 judge inputs → atlas_judge_input.jsonl
--skip-judge set; not running stages 2-3
```

378 = 300 critic + 78 creator hyps from the 540p-thaw1 run. Both populations covered by the wrapper.

## How this fits the /loop story

After **18 iterations**, the deployment chain is:

```
fresh run completes
   ↓
operator: python scripts/precompute_atlas_overlap.py --run <run> --include-creator
   ↓ (locally, ~15-25 min Kimi run)
<run>/atlas_overlap.jsonl
   ↓ (operator: scp to ~/aigraph/<run>/ on production)
   ↓ (operator: tmux restart of aigraph server)
queries via MCP get_idea_report
   ↓ (in-process at query time)
matched candidates → drop overlap < 3 → MMR → deliver top-K
   ↓
OMC Stage 3 sees 0 leaks
```

Every step in this chain is now codified — either as a single script call or as a documented scp/restart procedure.

## Next operational improvements (deferred)

- **Auto-deploy step** in the wrapper: optional `--deploy` flag that scp's the sidecar to the host and triggers an aigraph restart. Keeps the wrapper local-only for now.
- **MCP arg exposure**: surface `min_atlas_overlap` as a per-call parameter in `get_idea_report` so OMC can opt out. Trivial follow-up; defaults stay `3`.
- **Wrapper for new runs in the OMC pipeline**: add a post-generation hook that calls the wrapper automatically.
