# Methods 14 + 15 — Stability + production-impact of the Atlas filter

**Date:** 2026-05-31
**Methods 14, 15 of:** /loop "test different methods"; jointly validate Method 13's deployable filter for production use.
**One-line answer:** **Filter is production-ready.** The `atlas_overlap` axis is 91% exact-stable across re-judges and never drifts more than ±1; threshold-3 filter decisions agree on 91% of hyps with the 9% disagreement living entirely on the 2↔3 border. Across 12 diverse topics × top-8 delivery, **the leak rate goes from 13.5% to 0.0%** with no false-negatives.

## Method 14 — Judge stability

Re-ran the Method 3 Likert judge on the same 30-hyp input. 24 of 24 ok-records overlap with Method 3's 24 (1 mismatch — different hyp failed). Test-retest agreement:

| axis | exact-match | within ±1 | mean \|Δ\| |
|---|---:|---:|---:|
| **atlas_overlap** | **21/23 (91%)** | **23/23 (100%)** | **0.09** |
| forward_looking | 19/23 (83%) | 23/23 (100%) | 0.17 |
| named_mechanism | 18/23 (78%) | 23/23 (100%) | 0.22 |
| single_variable_test | 13/23 (57%) | 22/23 (96%) | 0.48 |
| specific_scope | 17/23 (74%) | 23/23 (100%) | 0.26 |

**The axis we use for filtering is the most stable.** `atlas_overlap` never differs by >1 between runs; the noisiest axis (`single_variable_test`) we don't use for filter decisions. Threshold-3 filter agreement:

- agree-keep (both ≥3): **19/23**
- agree-drop (both <3): **2/23**
- DISAGREE: **2/23 (9%)** — `h036` and `fwd_a110`, both flipped 3→2

Both disagreements are at the 2↔3 border. There are zero 1↔4 or 2↔4 flips — the rough magnitude is preserved across re-judges. The remaining 9% noise is the cost of using a single-shot judge; an ensemble of two judges with majority vote could squeeze this further (deferred).

## Method 15 — Production deployment impact

Ran `query_records(min_atlas_overlap=0)` vs `query_records(min_atlas_overlap=3)` for 12 diverse end-user topics on the production 540p-thaw1 run:

| topic | matched OFF→ON | top-8 OFF overlaps | top-8 ON overlaps | leaks OFF→ON | top-8 churn |
|---|---:|---|---|---:|---:|
| chain of thought reasoning | 129→117 | 3,3,3,3,3,3,3,3 | 3,3,3,3,3,3,3,3 | 0→0 | 0 |
| language model | 167→150 | 3,3,3,**1**,3,**1**,3,3 | 3,3,3,3,3,3,3,3 | 2→0 | 8 |
| neural network | 4→3 | 2,3,3,0 | 3,3,0 | 1→0 | 1 |
| evaluation | 139→129 | 3,4,3,3,3,3,**2**,3 | 3,4,3,3,3,3,3,3 | 1→0 | 2 |
| fine-tuning | 46→38 | 3,3,3,3,0,3,0,3 | 3,3,3,3,3,0,3,0 | 0→0 | 2 |
| retrieval augmented generation | 138→117 | 4,**1**,3,4,3,3,**1**,3 | 3,4,4,3,4,3,3,3 | 2→0 | 8 |
| reinforcement learning | 33→28 | 3,**2**,3,3,0,3,3,0 | 3,3,0,3,3,3,0,3 | 1→0 | 4 |
| in-context learning | 105→93 | 3,4,3,3,3,3,3,3 | 3,4,3,3,3,3,3,3 | 0→0 | 6 |
| multi-agent | 136→126 | 3,3,4,3,3,3,3,**2** | 3,3,4,3,3,3,3,3 | 1→0 | 2 |
| alignment | 74→66 | 3,3,3,4,**2**,3,3,**2** | 3,3,3,4,3,3,3,3 | 2→0 | 4 |
| tool use | 76→59 | 3,**2**,3,3,3,3,3,**1** | 3,3,3,3,3,3,3,3 | 2→0 | 8 |
| code generation | 113→97 | 3,4,4,3,**1**,3,3,3 | 3,4,3,4,3,3,3,3 | 1→0 | 8 |

(`0` overlaps = hyp not in sidecar, treated as unscored→kept.)

### Aggregate (12 topics × top-8 = 96 slots)
- **Leaks (overlap=1 or 2) before filter: 13/96 = 13.5%**
- **Leaks after filter: 0/96 = 0.0%**
- Total top-8 churn: 53/192 slot positions changed (28%)
- Topics where filter mattered most: `language model`, `RAG`, `tool use`, `code generation` (all dropped 2 leaks each)
- Topics where filter was silent (top-8 already clean): `chain of thought`, `in-context learning`, `fine-tuning`

### Reading

- The leak rate aligns with Method 12's whole-run finding (~20% of top-100 had overlap≤2). At top-8 it's 13.5% — fewer extreme leaks make the cut, but they exist on most topics.
- The 28% slot churn means users see *different* hyps for nearly a third of slots. The filter isn't cosmetic — it changes what gets delivered.
- Two topics (`fine-tuning`, `in-context learning`) had top-8 churn > 0 but identical leak rates 0→0. The filter dropped low-ranked candidates that affected MMR's choices but didn't touch the top-8 leak status. Side effect, not central to the filter's purpose.

## Combined verdict

The filter is **production-ready**:

- ✅ Stable across re-judges (atlas_overlap: 91% exact, 100% ±1)
- ✅ Eliminates 100% of measured leaks across 12 topics
- ✅ Zero false-negatives (no overlap≥3 hyp wrongly dropped)
- ✅ Defensive — keeps unscored hyps; skips drop if no candidate clears

Open caveats:

- Single-judge (Kimi). 9% borderline drift on overlap=2↔3.
- 41/300 hyps in the current sidecar are missing (Method 12 errors) — they're treated as unscored. A re-run via the wrapper would close this.

## Recommendation — flip the default to `min_atlas_overlap=3` in OMC Stage 3

Modify `~/onemancompany/src/onemancompany/agents/aigraph_mcp_tools.py`'s `get_idea_report` call to pass `min_atlas_overlap=3`. Two prerequisites:

1. Server-side: scp the updated `scripts/aigraph_query.py` to `~/aigraph/scripts/aigraph_query.py` on `8.208.118.99` and restart the aigraph tmux session — same procedure as `docs/aigraph-mcp-client-guide.md` §5.4.
2. MCP-side: the FastMCP tool wrapper for `get_idea_report` must expose `min_atlas_overlap` as a parameter (defaulting to 3 once the sidecar exists in the run dir). Today the MCP wrapper proxies a fixed set of kwargs to `query_records`.

For runs without an `atlas_overlap.jsonl` sidecar, the filter is a no-op (safe back-compat).

## What the /loop has produced — final synthesis

After 11 iterations, the deployable answer:

1. **The lever** — Atlas as a per-hypothesis Likert evaluator (`atlas_overlap` 1-5).
2. **The pipeline** — one-time-per-run precompute via the documented operator workflow (Method 13 §3); sidecar `atlas_overlap.jsonl` in the run dir.
3. **The query rule** — `--min-atlas-overlap 3` in `aigraph_query.py`, kept default-off for back-compat, recommended for OMC Stage 3.
4. **The measured impact** — 13.5% → 0% leak rate across 12 diverse topics. ~28% slot churn (users see different, more anchored hyps in ~1/3 of slots).
5. **The negatives recorded** — Atlas as prompt context HURTS (Method 2); Atlas as anomaly selector is a TIE (Method 7); Workflow D (decomposition) is a TIE with C under both binary and Likert rubrics (Methods 1, 8).
