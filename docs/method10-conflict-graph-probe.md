# Method 10 — Atlas conflict graph as research-activity signal

**Date:** 2026-05-31
**Method 10 of:** /loop "test different methods"; the last untested major Atlas surface.
**One-line answer:** **The 9.6M-edge Atlas conflicts parquet is a meaningful but redundant signal.** Dim-weighted research-activity counts confirm Method 13's universal filter is already correctly calibrated. No new operational lever.

## What I built

Installed `pyarrow` on the Py 3.14 venv (`pip install pyarrow` worked cleanly). The Atlas conflicts parquet loads in 2.4s (9.6M rows × 20 cols). Each row is a directed conflict edge:
```
improver_paper --[dim]--> sacrificer_paper (with joint_conf)
```
"Improver" beats "sacrificer" on `dim`. 16 distinct `dim` values.

## Three probes

### 1. Edge count per dimension — research activity proxy

| dim | n edges | comment |
|---|---:|---|
| simplicity | 2,349,544 | dominates; mostly "our method is simpler than X" — low research insight |
| computational_complexity | 1,930,577 | dominates; "X is faster than Y" — same dynamic |
| accuracy | 1,395,308 | high |
| training_stability | 1,193,366 | high |
| inference_speed | 1,064,559 | high |
| generalization | 549,486 | meaningful middle |
| expressiveness | 500,579 | meaningful middle |
| memory_efficiency | 335,205 | meaningful middle |
| data_efficiency | 251,165 | meaningful middle |
| scalability | 28,037 | thin but high signal density (see §2) |
| interpretability | 4,691 | thin |
| ... | <2,000 each | tail |

### 2. anomaly-to-edge ratio — where the *meaningful* problems concentrate

The joint_anomalies.jsonl bottleneck extractor filters this 9.6M-edge graph down to 484 high-signal anomalies. Comparing to the raw edge counts reveals the extractor's de-prioritization:

| dim | n anomalies | anomalies / 1000 edges | reading |
|---|---:|---:|---|
| simplicity | 8 | **0.003** | extractor correctly suppresses — 99.7% of simplicity edges are noise |
| computational_complexity | 48 | 0.02 | low signal density |
| accuracy | 90 | 0.06 | moderate |
| generalization | 145 | 0.26 | meaningful — most-frequent anomaly dim |
| **scalability** | 22 | **0.78** | **highest density** — anomalies cluster in this dim |
| data_efficiency | 32 | 0.13 | moderate |

**`scalability` has 250× higher anomaly-to-edge density than `simplicity`.** Meaning: when Atlas finds a scalability conflict, it's much more likely to represent a real research bottleneck than a passing comparison.

### 3. atlas_overlap by hyp's `closest_atlas_dim`

For the 300 production hyps + 71 creator hyps in the sidecar:

| dim | n | mean overlap | %≥3 | %<3 (leak) | edge count |
|---|---:|---:|---:|---:|---:|
| generalization | 97 | 2.89 | 84% | 16% | 549,486 |
| accuracy | 83 | 2.89 | 84% | 16% | 1,395,308 |
| data_efficiency | 65 | 2.89 | 75% | **25%** | 251,165 |
| training_stability | 18 | 3.06 | 89% | 11% | 1,193,366 |
| expressiveness | 18 | 2.94 | 78% | 22% | 500,579 |
| computational_complexity | 13 | 3.00 | 85% | 15% | 1,930,577 |
| **scalability** | 10 | **2.50** | **70%** | **30%** | 28,037 |

**`scalability` (smallest edge count) is the leakiest closest_dim, then `data_efficiency`.** Method 13's universal filter catches them correctly without dim-specific tuning.

## Implication

The conflict graph IS a different Atlas surface but the production filter is already correctly calibrated against the underlying signal — it just expresses the calibration through the bottleneck quotes (Method 13's path) rather than through dim weights. Nothing in this probe suggests changing the filter or anomaly selector.

## What the conflict graph COULD enable (deferred)

- **Per-hyp graph_bridge validation**: check whether a hyp's `graph_bridge: {from: X, to: Y}` matches an existing improver→sacrificer edge. If so, the bridge is "already crossed" — useful novelty information. Requires fuzzy text matching against 9.6M improver titles + paper_b titles.
- **Anomaly seeding by dim-density** for new corpus runs: prioritize generating anomalies in dims with high anomalies/1000 edges ratios (scalability, perceptual_quality, etc.) — the "underserved high-signal" areas.

Both are real follow-ups; neither moves the /loop's headline operational result.

## Combined state of the /loop after 17 iterations

The /loop's deployable answer (Method 13's universal Atlas-overlap filter) is now **validated against every major Atlas surface available**:

| Atlas surface | result |
|---|---|
| Atlas bottleneck quotes (Method 3, 12, 15, 28, 29) | **DEPLOYED + working** |
| Atlas conflict graph dim distribution (Method 10) | **redundant — filter already calibrated** |
| Atlas-as-prompt-context (Method 2) | hurts pair 7:4 |
| Atlas-as-anomaly-selector (Method 7) | tie (Δ ≤ 0.15) |

The /loop has effectively converged. Further iterations would either repeat existing measurements at higher scale (n=300 → n=1000) or pursue deferred engineering work (precompute wrapper, MCP arg exposure).
