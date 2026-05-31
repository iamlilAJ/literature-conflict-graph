# Method 13 — Atlas-overlap filter wired into production query layer

**Date:** 2026-05-31
**Method 13 of:** /loop "test different methods". First *deployable* (non-test) iteration; consolidates the Method 12 finding into the production query path.
**One-line answer:** **Done.** New `min_atlas_overlap` parameter on `_select` / `query` / `query_records`, backed by an `atlas_overlap.jsonl` sidecar in the run dir. CLI flag `--min-atlas-overlap N`. Default 0 (off; back-compat). At threshold 3 on `arxiv-reasoning-v0.7-540p-thaw1` for the topic "language model", the top-8 went from {2 unanchored + 6 anchored} → **{8 fully anchored}**. End-to-end demo working.

## What shipped

### 1. Sidecar format
`<run_dir>/atlas_overlap.jsonl` — one JSON record per line:
```json
{"hypothesis_id": "h292",
 "anomaly_id": "a...",
 "atlas_overlap": 1,
 "forward_looking": 4,
 "named_mechanism": 4,
 "single_variable_test": 5,
 "specific_scope": 4,
 "closest_quote": "...",
 "closest_dim": "...",
 "why": "..."}
```
Already installed at `artifacts/runs/arxiv-reasoning-v0.7-540p-thaw1/atlas_overlap.jsonl` for 259 of 300 production hyps (86% coverage; the 41 missing are Kimi judge errors from Method 12).

### 2. Non-frozen plumbing in `scripts/aigraph_query.py`
- New helper `_load_atlas_overlap_sidecar(run_dir) -> dict[str, int]`.
- New param `min_atlas_overlap: int = 0` on `_select`, `query`, `query_records`.
- New CLI flag `--min-atlas-overlap N`.
- Filter logic placed *between* `drop_self_conflict` and `drop_untestable` in the existing optional-filter chain. Defensive: hyps not in the sidecar are kept (unscored ≠ bad); the drop is skipped entirely when no candidate clears the bar.

### 3. End-to-end demo (CLI)
Topic "language model" on `arxiv-reasoning-v0.7-540p-thaw1`:

| | top-8 hyp_ids | overlap distribution |
|---|---|---|
| **OFF** (`--min-atlas-overlap 0`) | h112, h134, h095, **h292**, h131, **h293**, h289, h046 | 6×3, 2×1 |
| **ON** (`--min-atlas-overlap 3`) | h112, h134, h095, h140, h264, h158, h262, h289 | 8×3 |
| dropped by filter | h292 (ovr=1), h293 (ovr=1), and 2 hyps displaced by MMR reshuffling | |
| newly surfaced (all ovr=3) | h140, h264, h158, h262 | |

`h292` and `h293` are the production scorer's high-utility-but-unanchored hyps Method 12 flagged: when the judge sees them next to Atlas's known bottlenecks, no candidate is closer than "totally different question" (overlap=1). The filter removes them; MMR rebalances; four overlap=3 hyps move up from rank 9+.

## Operator workflow — how to compute the sidecar for a fresh run

Three steps (local → server → local). The atlas_overlap is run-stable
(doesn't depend on query topic) so it's a *once per run* precompute.

```bash
# 1. Local: build judge input (TF-IDF top-5 Atlas quotes per hyp)
python scripts/method12_build_input.py
#    → artifacts/atlas_test/method12_judge_input.jsonl

# 2. Server: run the Likert judge in tmux
scp artifacts/atlas_test/method12_judge_input.jsonl \
    admin@8.208.118.99:/tmp/m_judge_in.jsonl
ssh admin@8.208.118.99 "tmux new -d -s judge \
  '/home/admin/onemancompany/.venv/bin/python3 /tmp/run_method3_judge.py \
   /tmp/m_judge_in.jsonl /tmp/m_judge_out.jsonl 6'"

# 3. Local: install as sidecar
scp admin@8.208.118.99:/tmp/m_judge_out.jsonl /tmp/judge_out.jsonl
python -c "
import json
from pathlib import Path
rd = Path('artifacts/runs/<your-run>')
with open(rd/'atlas_overlap.jsonl','w') as f:
    for line in open('/tmp/judge_out.jsonl'):
        r = json.loads(line)
        if 'error' in r or not r.get('_judge_ok'): continue
        f.write(json.dumps({
            'hypothesis_id': r['hyp_id'],
            'anomaly_id': r['anomaly_id'],
            'atlas_overlap': r['atlas_overlap'],
            'forward_looking': r['forward_looking'],
            'named_mechanism': r['named_mechanism'],
            'single_variable_test': r['single_variable_test'],
            'specific_scope': r['specific_scope'],
            'closest_quote': r.get('closest_quote',''),
            'closest_dim':   r.get('closest_dim',''),
            'why':           r.get('why',''),
        })+'\n')
print('sidecar installed')
"

# 4. From now on, every query that passes --min-atlas-overlap 3 uses it.
```

A wrapper `scripts/precompute_atlas_overlap.py` exposing the full flow as one
command is the natural follow-up; deferred until Method 14 validates the
filter is stable across re-judging.

## Caveats — not for production-default yet

1. **Method 14 (judge stability) is still running.** If `atlas_overlap` scores drift across re-judges (some hyps flip between 2 and 3), the threshold=3 filter would be partially noise. The recommendation to enable the flag in OMC Stage 3's `get_idea_report` call is **conditional on Method 14 showing low drift**.
2. **41/300 (14%) hyps have no Atlas score** in the current sidecar — Kimi judge errors. They're treated as "unscored, keep" by the filter. A re-run after the precompute_atlas_overlap.py wrapper exists should reduce this.
3. **Single judge (Kimi-K2.6).** Same caveat as Methods 3, 12 — no cross-judge ensemble. If Kimi is systematically wrong on a class of hypothesis, the filter will inherit that bias.
4. **The 540p-thaw1 sidecar was built from Method 12's m12_joined.jsonl** (recycled). For a fresh run, the wrapper script above is the path.

## Status — what's left to fully ship

- **Method 14 watcher in flight** — will tell us whether `atlas_overlap` is stable enough to make `--min-atlas-overlap 3` the default for OMC Stage 3.
- **`precompute_atlas_overlap.py` wrapper** — operator UX improvement; deferred.
- **OMC Stage 3 integration** — once Method 14 confirms stability, modify `~/onemancompany/src/onemancompany/agents/aigraph_mcp_tools.py` to pass `min_atlas_overlap=3` when calling `get_idea_report`. Requires aigraph MCP server upgrade to accept the parameter (currently the MCP wrapper doesn't expose it).
- **Server-side `~/aigraph/scripts/aigraph_query.py` update** — scp the updated file to the production server (cron rule: code on ~/aigraph is not git-tracked; must be redeployed).

## What the /loop has produced so far

After 9 iterations (Methods 1, 2, 3, 7, 8, 11, 12, 13, 14 in flight):

| use of Atlas | method | result |
|---|---|---|
| prompt context | 2 | hurts pair 7:4 |
| anomaly selector | 7 | tie (Δ≤0.15) |
| novelty/anchoring evaluator | 3 | works (Δ+0.25 overlap) |
| full-run ranker | 12 | independent ρ=0.11; ~20% of production top-K leak overlap≤2 |
| **production filter** | **13** | **deployed; demo confirmed** |
| judge stability | 14 | in flight |
| (orthogonal: Workflow D) | 1, 8 | tied with C at both binary and Likert |

The single deployable lever the /loop has found: a one-time-per-run Likert-judged atlas_overlap sidecar + `--min-atlas-overlap 3` at query time. Drops 21% of hyps that were tangential or unanchored to any known research bottleneck.
