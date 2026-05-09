# Intern-Atlas ↔ aigraph schema compatibility

A field-level audit of how
[`OpenRaiser/Intern-Atlas`](https://huggingface.co/datasets/OpenRaiser/Intern-Atlas)'s
4 parquet configs map to aigraph's `Paper` / `Claim` / stance-edge /
method-registry models. Establishes the contract for
`corpus.intern_atlas_loader` (planned in
[`docs/intern-atlas-pivot.md`](./intern-atlas-pivot.md) §5.2).

> Status: locked v0.1. Update if either schema changes (Atlas v2 or
> aigraph models.py changes).

---

## TL;DR

- `papers` ↔ `Paper`: 🟢 **high compat**. 9 fields direct-map, 14 are
  Atlas-only (worth keeping), 15 are aigraph-internal (Atlas can't
  fill). No conflicts.
- `paper_evolution_edges` ↔ stance edges: 🟡 **medium compat**. 5/7
  Atlas relation types map to aigraph stance, but Atlas has no
  `contradicts` — aigraph's differentiator stays.
- `paper_methods` ↔ `Claim`: 🟡 **low compat**. Atlas `paper_methods`
  is `(paper, method, relationship)` triples, much shallower than
  aigraph's `Claim`. **Do not use as Claim substitute**; use as
  canonical-method gazetteer / hint.
- `method_relations` (60 rows): 🟢 **drop-in vocabulary** for
  `canonical_method` field.

**Hard issue**: `paper_id` namespaces don't match. Atlas uses
`conf_NeurIPS_2018_0001`-style; aigraph uses `arxiv:NNNN.NNNNN` /
`openalex:W...`. Loader must keep Atlas paper_id verbatim and
populate alternate ID fields (arxiv_id, doi, openalex_id, s2_id)
from Atlas columns.

---

## 1. `papers` (28 cols, 4.2M rows) ↔ `Paper` (33 fields)

### 1.1 Direct map (8 fields, no transformation)

| Atlas column | aigraph Paper field | note |
|---|---|---|
| `title` | `title` | direct |
| `abstract` | `abstract` | direct (50% of Atlas rows have empty string here) |
| `year` | `year` | direct |
| `venue` | `venue` | raw venue string |
| `citation_count` | `cited_by_count` | rename only |
| `doi` | `doi` | direct |
| `openalex_id` | `openalex_id` | direct |
| `arxiv_id` | `arxiv_id_full` and `arxiv_id_base` | aigraph splits the version suffix; Atlas's `arxiv_id` is the base form (no `v1`/`v2`) |

### 1.2 Atlas-only fields aigraph should adopt (4 fields)

These are high-value enrichment — load explicitly into new optional
Paper fields:

| Atlas | new Paper field | rationale |
|---|---|---|
| `venue_canonical` | `venue_canonical: Optional[str]` | NeurIPS/ICML/ICLR canonical names — needed for cohort filter; using raw `venue` causes alias misses (e.g., "Advances in Neural...") |
| `venue_tier` | `venue_tier: Optional[str]` | tier1/tier2/tier3 — controlled-validation §3.1 stratification |
| `influential_citation_count` | `influential_citation_count: int = 0` | Semantic-Scholar's filtered-cite count. Strictly better than raw `cited_by_count` for the L1 metric — `controlled-validation-design.md §5.1` already plans to report both |
| `s2_id` | `s2_id: Optional[str]` | Semantic Scholar ID. Useful as 5th ID alias for cross-system lookups when arxiv_id/doi/openalex_id are missing |

### 1.3 Atlas-only fields → store via `extra="allow"` (10 fields)

Lower priority than §1.2 but still useful. With Paper using
`extra="allow"` (matching the Hypothesis Bug #1 fix pattern in PR #15),
these survive round-trip without schema bloat:

```
authors_json    fields_of_study    publication_date    is_open_access
pdf_url         paper_type         status              node_type
reference_count acl_id             dblp_id             pubmed_id
openreview_id
```

These are accessible via `getattr(paper, field, default)`. The loader
populates them from the Atlas row dict; downstream code reads them
when needed (e.g., `pdf_url` for full-text fetch decisions).

### 1.4 aigraph-only fields (Atlas can't fill)

The loader leaves these at their model defaults:

- **Full-text**: `text` (empty string default — Atlas only has abstract)
- **Citation graph derived**: `referenced_works`, `counts_by_year`
  (Atlas stores citations in `paper_evolution_edges`, not per-paper;
  recoverable via separate join)
- **Internal scoring**: `selection_score`, `selection_reason`,
  `retrieval_channel`, `paper_role`, `paper_role_score`,
  `paper_role_signals`, `academic_impact`, `recency_score`,
  `reasoning_relevance`, `role_weight`, `priority_score`
- **Corpus-build metadata**: `corpus_tag`, `seed_reason`, `sync_status`,
  `sync_attempt_count`, `first_seen_at`, `last_seen_at`,
  `last_attempted_at`, `completed_at`
- **Rule-extractor hint**: `structured_hint`

These are computed downstream by the existing aigraph pipeline; they
are NOT the loader's responsibility.

---

## 2. `paper_evolution_edges` (35 cols, 4.14M rows) ↔ stance edges

aigraph stance vocabulary (`citation_stance.py`):
```
extends      — B explicitly builds on A's framework / dataset / method
contradicts  — B reports a finding opposite to A's claim
contrasts    — B compares its approach to A's without directly contradicting
builds_on    — B uses A as foundation (cites for setup) without extending
mentions     — B references A in passing (related work boilerplate)
```

Atlas `evolution_relation` vocabulary:
```
extends         5.6%   ↔ extends                    1:1
improves        3.9%   ↔ extends (subtype)          high
adapts          1.7%   ↔ extends (subtype)          high
replaces        2.9%   ↔ extends or contrasts       medium
uses_component  28%    ↔ builds_on                  high
compares        58%    ↔ mentions or contrasts      medium-low (ambiguous)
background     (n/a)   ↔ mentions                   high
combines        ~0     (15 rows, near-empty)
```

### 2.1 Atlas does not have `contradicts`

Atlas's typology is strictly causal/derivational. Cross-paper
disagreement is not represented. **This is aigraph's structural
differentiator.** `detect_anomalies` produces 4 conflict-flavored
anomaly types (`impact_conflict`, `benchmark_inconsistency`,
`metric_mismatch`, `setting_mismatch`) and 1 replication-flavored
type (`replication_conflict`) — none of these have an Atlas analog.

### 2.2 Atlas edges carry richer per-edge content

Atlas attaches 4 JSON fields to each edge:

```
bottleneck_json    — what limitation in paper A drove paper B's improvement
mechanism_json     — how paper B's mechanism addresses A's bottleneck
impact_json        — improvement_dimensions / sacrifice_dimensions / tradeoff
components_json    — list of A's components reused by B
```

99.8% non-empty across 4M edges. This is content aigraph stores at the
**claim level** (mechanism/impact attached to individual claims, not
to cite-edges). Different granularity, similar information density.

### 2.3 Loader contract for edges

Two consumption modes:

**Mode A (preferred): Atlas as primary stance source**
- Map `extends` / `improves` / `adapts` → aigraph `extends`
- Map `replaces` → aigraph `extends` (with `notes` flag)
- Map `uses_component` → aigraph `builds_on`
- Map `compares` / `background` → aigraph `mentions`
- aigraph runs ONLY `contradicts` / conflict-type detection on top
  (since Atlas can't supply this)
- Saves ~58% × 4M × LLM-call cost vs running our stance classifier
  end-to-end

**Mode B (fallback): aigraph stance only**
- Skip Atlas edges, use existing `classify-stance` pipeline on the
  cohort papers
- Use Atlas edges only for sanity-check overlap analysis

Default is Mode A once `intern_atlas_edge_loader` is implemented.

---

## 3. `paper_methods` (11 cols, 797k rows) ↔ `Claim`

aigraph `Claim` is a **rich performance-claim record**: per-paper
extraction with method, model, task, dataset, metric, baseline,
result, magnitude_value, evidence_span, scope, conditions, etc.
Roughly 50 fields.

Atlas `paper_methods` is a **shallow `(paper, method, relationship)`
triple**:

| Atlas column | maps to | note |
|---|---|---|
| `paper_id` | (Paper.paper_id with format conversion) | Atlas namespace |
| `paper_title` / `paper_year` / `paper_venue` | (redundant — look up via Paper) | denormalized in Atlas |
| `method_id` | — | Atlas-internal method registry ID |
| `method_name` | (could populate Claim.canonical_method) | gazetteer-style |
| `method_description` | — | describes the method, not a claim about its performance |
| `method_level` | — | algorithm/technique/framework/paradigm — aigraph doesn't categorize |
| `relationship` | — | uses/proposes/etc. — aigraph doesn't track this |
| `confidence` | — | extraction confidence (informational) |
| `source` | — | seed / extracted / etc. (provenance) |

### 3.1 Verdict: NOT a Claim substitute

`paper_methods` is too shallow to substitute for `extract` /
`llm_extract` claim extraction. Specifically:
- No dataset / metric / baseline / result fields
- No magnitude or direction
- No evidence_span — can't pin a claim to a verbatim sentence
- No conditions or scope

### 3.2 Use cases for `paper_methods` in aigraph

- **Canonicalization gazetteer**: when running `canonical_method`
  resolution (`llm_extract` already canonicalizes), Atlas
  `(method_name, method_id)` pairs are a controlled vocabulary that
  reduces variance ("Vision Transformer" / "ViT" / "ViTs" all → `vit`).
- **Method-co-occurrence baseline**: per-paper method bag from Atlas
  is a cheap baseline against aigraph's claim-level method extraction.
- **Coverage check**: how many cohort papers have ≥ 1 Atlas method
  hit? If high, paper-method linking is "free"; if low, our claim
  extraction is the only signal.

### 3.3 Loader contract for paper_methods

Optional. Loader can attach Atlas methods to Paper via
`extra="allow"`:

```python
paper.atlas_methods = [
    {"method_id": "...", "method_name": "...", "method_level": "...",
     "relationship": "...", "confidence": 0.85},
    ...
]
```

Downstream uses these as gazetteer hints during claim extraction or
canonicalization. **Never** as the primary claim source.

---

## 4. `method_relations` (60 rows) — drop-in vocabulary

| Atlas column | use |
|---|---|
| `source_method_id`, `source_method_name`, `source_method_level` | source of relation |
| `target_method_id`, `target_method_name`, `target_method_level` | target of relation |
| `relation` | one of `variant_of`, `specializes`, `component_of`, `combines`, `inspired_by` |
| `evidence_paper_id`, `evidence_paper_title`, `confidence`, `source`, `created_at` | provenance |

60 hand-curated method-to-method edges (ViT specializes Transformer,
LoRA variant_of PEFT, etc). aigraph has no analog; this is a free
gazetteer.

### 4.1 Loader contract

Load once, expose as a module-level dict in
`corpus.intern_atlas_method_relations`:

```python
{
    ("vit", "transformer"): "specializes",
    ("lora", "peft"):       "variant_of",
    ...
}
```

Used downstream:
- `canonical_method` resolver: rewrite `"ViT"` → `"vit"` and
  `"Transformer"` → `"transformer"` consistently
- Anomaly detection: when comparing claims A and B, methods related
  via `variant_of` / `specializes` / `component_of` should NOT raise
  `setting_mismatch` (they're related variants by design)

---

## 5. The `paper_id` namespace problem

### 5.1 The mismatch

| | format | example |
|---|---|---|
| Atlas `papers.paper_id` | `conf_<VENUE>_<YEAR>_<SEQ>` | `conf_NeurIPS_2018_0001` |
| aigraph `Paper.paper_id` (from corpus) | `arxiv:<ID>v<VERSION>` | `arxiv:2206.10498v4` |
| aigraph `Paper.paper_id` (from openalex) | `openalex:W<ID>` | `openalex:W123456789` |
| aigraph `Paper.paper_id` (test fixtures) | bare string | `paper_smoke` |

Different namespaces. Cannot be merged on `paper_id` directly.

### 5.2 Atlas papers have at most 1 ID populated

Recon-confirmed (see `cohort_from_intern_atlas.py` notes): no Atlas
paper row carries 2+ of {arxiv_id, doi, openalex_id, s2_id}
simultaneously. So an alias map keyed on, say, arxiv_id will miss any
paper that's only doi-identified.

### 5.3 Loader strategy

The loader keeps Atlas's `paper_id` **verbatim** as `Paper.paper_id`
and populates the four ID fields (`arxiv_id_full`, `openalex_id`,
`doi`, `s2_id`) from the corresponding Atlas columns:

```python
def _atlas_row_to_paper(row: dict) -> Paper:
    return Paper(
        paper_id=row["paper_id"],                        # keep Atlas's "conf_..."
        title=row["title"],
        abstract=row["abstract"] or "",
        year=row["year"],
        venue=row["venue"] or "",
        cited_by_count=row.get("citation_count") or 0,
        doi=row.get("doi"),
        openalex_id=row.get("openalex_id"),
        arxiv_id_full=row.get("arxiv_id"),
        arxiv_id_base=row.get("arxiv_id"),  # Atlas already has base form
        s2_id=row.get("s2_id"),
        venue_canonical=row.get("venue_canonical"),
        venue_tier=row.get("venue_tier"),
        influential_citation_count=row.get("influential_citation_count") or 0,
        # extras (need extra="allow" on Paper)
        atlas_acl_id=row.get("acl_id"),
        atlas_dblp_id=row.get("dblp_id"),
        # ... etc for the §1.3 list
    )
```

### 5.4 Cross-namespace lookup

When aigraph code needs to look up an Atlas paper by an arxiv ID
(e.g., novelty_check returns an arxiv hit and we want to know if it's
in our corpus), provide a helper:

```python
def find_atlas_paper_by_id(papers: list[Paper], *, arxiv_id=None,
                          doi=None, openalex_id=None, s2_id=None) -> Optional[Paper]:
    """Return first paper matching any of the supplied IDs."""
```

The helper iterates and matches — O(n) per lookup, but our cohort is
≤ 5k papers so this is fine without an index. For larger corpora,
build a per-ID dict.

---

## 6. Required model changes

### 6.1 `Paper.model_config = ConfigDict(extra="allow")`

Mirrors PR #15's Hypothesis fix (Bug #1). Lets the loader inject the
10 §1.3 Atlas-only fields without schema bloat. Trade-off: typo'd
kwargs to `Paper(...)` are silently accepted.

Acceptable because Paper is constructed by controlled loaders
(corpus, fetch_arxiv, fetch_openalex, intern_atlas_loader) — not by
user input.

### 6.2 New explicit Paper fields (4)

```python
class Paper(LooseModel):
    model_config = ConfigDict(extra="allow")  # NEW

    # ... existing fields ...

    # NEW (Atlas-driven enrichment, set by intern_atlas_loader)
    venue_canonical: Optional[str] = None
    venue_tier: Optional[str] = None
    influential_citation_count: int = 0
    s2_id: Optional[str] = None
```

Default values keep all existing constructors working unchanged.

### 6.3 No changes to Claim, Anomaly, Hypothesis

These models are not modified by this loader. Atlas `paper_methods`
attaches to Paper via extras (§3.3); Atlas edges are written into the
typed graph at edge-load time, not into models.

---

## 7. Loader API surface

```python
# src/aigraph/corpus.py (additions)

def load_papers_from_intern_atlas(
    intern_atlas_dir: Path | str,
    *,
    venue_canonicals: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
    require_abstract: bool = False,
    min_id_count: int = 0,
) -> list[Paper]:
    """Read Atlas papers parquet, filter, and return Paper records.

    Args:
        intern_atlas_dir: HF download root (sharded layout)
            with data/papers/*.parquet inside.
        venue_canonicals: Optional venue allowlist (e.g.
            ["NeurIPS", "ICML", "ICLR"]).
        years: Optional year allowlist.
        require_abstract: Drop rows with empty/short abstract.
        min_id_count: Drop rows with fewer than this many populated
            IDs (arxiv_id / doi / openalex_id / s2_id).

    Returns:
        list of Paper. Empty list if no rows pass filter.
    """


def load_atlas_method_gazetteer(
    intern_atlas_dir: Path | str,
) -> dict[tuple[str, str], str]:
    """Read method_relations parquet, return canonical lookup.

    Returns: {(source_method_id, target_method_id): relation}
    """


def load_atlas_paper_methods_index(
    intern_atlas_dir: Path | str,
    *,
    paper_ids: Optional[set[str]] = None,
) -> dict[str, list[dict]]:
    """Read paper_methods parquet, return per-paper-id list of method
    associations. Optional paper_ids filter for memory savings.
    """


# Edge loader is a separate function (writes into graph.json):

def merge_atlas_edges_into_graph(
    intern_atlas_dir: Path | str,
    graph: nx.MultiDiGraph,
    *,
    paper_ids: set[str],
    relation_map: Optional[dict[str, str]] = None,  # default per §2
) -> dict[str, int]:
    """Read paper_evolution_edges, filter to (paper_a IN paper_ids AND
    paper_b IN paper_ids), map evolution_relation → aigraph stance,
    add to graph as cite-edges with stance attribute. Returns counters.
    """
```

Env switch (per `intern-atlas-pivot.md` §5.2):

```bash
AIGRAPH_USE_INTERN_ATLAS=1 \
AIGRAPH_INTERN_ATLAS_DIR=/path/to/data/intern_atlas \
python -m aigraph.cli ...
```

When the env is set, the existing fetch-arxiv / fetch-openalex CLI
paths short-circuit to the Atlas loader instead of hitting external
APIs.

---

## 8. What this doc does not specify

Out of scope for v0.1:

- **Schema versioning across Atlas releases**: Atlas v2 may rename
  fields, add columns, or change `evolution_relation` vocab. The
  loader should pin to a specific HF revision and surface a clear
  error on schema drift; the version pinning mechanism is a separate
  follow-up.
- **L3 manual-annotation tooling**: out of scope (separate doc).
- **Full migration of the OpenAlex/S2 fetch path**: kept as fallback;
  this doc only specifies the Atlas-first path.
- **Atlas `/api/eval` integration**: not in scope (decided 2026-05-09:
  do not depend on their API; consume their data only).

---

## 9. Open questions for v0.2

1. **Should aigraph adopt Atlas's `paper_id` format?** Pro: clean
   primary key everywhere, no namespace confusion. Con: every
   existing fixture, test, and persisted artifact (graph.json,
   anomalies.jsonl) uses the old `arxiv:` prefix. **Default**: keep
   the old format; aigraph and Atlas paper_ids coexist.

2. **Should `compares` map to `mentions` or `contrasts`?** It's 58%
   of Atlas edges and the highest-volume bucket. Treating them all as
   `mentions` underweights the Atlas signal; treating all as
   `contrasts` overcounts disagreement. **Default**: `mentions`, with
   a flag for downstream pipelines that want the conservative read.
   Revisit after a 100-edge sample of `compares` is hand-classified.

3. **How tightly should the Atlas paper_methods gazetteer feed back
   into our canonical_method extraction?** Hard binding (LLM gets
   forced to pick from gazetteer) vs soft hint (LLM sees gazetteer
   in prompt but can ignore). **Default**: soft hint, evaluate after
   first cohort run.
