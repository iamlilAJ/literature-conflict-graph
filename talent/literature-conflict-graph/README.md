# Literature Conflict Researcher (`literature-conflict-graph`)

A Stage 3 (Idea Generation) producer talent. Routes the OMC task through
[`iamlilAJ/literature-conflict-graph`](https://github.com/iamlilAJ/literature-conflict-graph)
(`aigraph`) — a tool that mines arXiv papers for typed claims, conflicts, and
hypotheses — then has Kimi synthesize the relevant ones into focused
research directions for the user's specific topic.

The talent is **opt-in**: default Stage 3 routing still goes to employee 00008
(generic LLM Idea Generator). Users select this talent per pipeline run via
the frontend Range Selector's per-stage agent picker.

## Setup (one-time)

The script handles cloning, venv, install, and the hire call:

```bash
bash scripts/setup-lcg.sh
```

What it does:
1. Clones `iamlilAJ/literature-conflict-graph` to `~/projects/literature-conflict-graph`
   (override with `LCG_REPO=<path>`), branch `stable/v0.7-runner-local`.
2. Builds a Python 3.12 venv at `<lcg-repo>/.venv` and installs `aigraph[real]`.
3. Runs a smoke test against the bundled 540-paper run.
4. If the OMC backend is reachable on `:8001`, hires the talent via
   `/api/candidates/hire-from-cv` (creates employee 00015).

Re-runnable. `--skip-clone` skips git, `--no-hire` skips the API call.

## Usage

After setup:
1. Open the OMC UI.
2. Submit a research topic.
3. **Click the ▾ button under Stage 3** in the Range Selector and pick
   "Literature Conflict Researcher".
4. Launch.

Producer wall: ~60–90s. Critic gate: ~40–80s. Total Stage 3: 1.5–3 min.

The deliverable is written to two places:
- `<project>/iterations/iter_*/stage3_idea_generator.md` — workspace file for the UI.
- `pipeline_state.yaml::stage_results['3']` — the engine's source of truth.

## How it works (brief)

```
launch.sh receives OMC_TASK_DESCRIPTION_FILE  (Stage 1+2 context + boilerplate)
   │
   ├─ aigraph_query.py            # 0 LLM, ~30ms
   │     run-dir = artifacts/runs/arxiv-reasoning-v0.7-540p
   │     topic   = full task text (bag-of-words token match)
   │     k       = 10 (override LCG_K)
   │     → topic-filtered hypothesis markdown
   │
   ├─ Kimi-K2.6 chat (1 call, ~50–150s, ~30K input tokens)
   │     system  = methodology-advisor persona + the hypothesis pool
   │     user    = the OMC task description
   │     → 3 research directions citing 3 hypothesis IDs
   │
   ├─ filter attached markdown to advisor-cited sections only
   ├─ write deliverable to $OMC_PROJECT_DIR/stage3_idea_generator.md
   └─ stdout: { output, model, input_tokens, output_tokens }   # OMC contract
```

## Modes (`LCG_MODE`)

| Mode  | When | LLM calls | Wall  | Output |
|-------|------|-----------|-------|--------|
| `chat` (default) | LLM key present | 1 | 60–150s | Advisor synthesis + 3 cited hypothesis sections |
| `topic` | no LLM key, or set explicitly | 0 | ~170ms | Raw topic-filtered hypothesis dump |

Auto-fallback: if `chat` is selected but no LLM key is in env, drops to `topic`.

## Configuration (env)

All optional — sane defaults assume `setup-lcg.sh` ran:

| Var | Default | Purpose |
|---|---|---|
| `LCG_REPO` | `~/projects/literature-conflict-graph` | Where the aigraph repo is cloned |
| `LCG_RUN_DIR` | `<repo>/artifacts/runs/arxiv-reasoning-v0.7-540p` | Pre-computed aigraph run to query |
| `LCG_PYTHON` | `<repo>/.venv/bin/python3` | Python in aigraph's venv |
| `LCG_K` | `10` | Top-K hypothesis candidates (10 = balanced; 5 = focused; 20 = broad) |
| `LCG_MODE` | `chat` | `chat` or `topic` (see above) |
| `LCG_TEMPERATURE` | `0.3` | Kimi temperature for the chat call |
| `LCG_MAX_TOKENS` | `32000` | Reasoning models eat budget; ≥16K recommended |
| `LCG_REASONING_EFFORT` | `low` | `none` / `low` / `medium` / `high` (LiteLLM Kimi accepts only these) |

OMC's own env vars (`CUSTOM_API_KEY`, `DEFAULT_API_BASE_URL`, `DEFAULT_LLM_MODEL`)
are inherited from the parent process via `SubprocessExecutor` and used as the
LLM credentials. No separate config needed.

## Failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| `ERROR: aigraph run dir missing: <path>` | aigraph not cloned, or wrong branch (no `artifacts/runs/arxiv-reasoning-v0.7-540p`) | `bash scripts/setup-lcg.sh` |
| `ERROR: aigraph python missing` | venv not built | `bash scripts/setup-lcg.sh --skip-clone` |
| `ERROR: OMC_TASK_DESCRIPTION_FILE missing` | direct invocation outside OMC | always run via OMC; for manual smoke-test: `OMC_TASK_DESCRIPTION_FILE=<file> bash launch.sh <employee_dir>` |
| Advisor section is empty / just a title | Kimi reasoning ate `max_tokens` | bump `LCG_MAX_TOKENS=48000` or set `LCG_REASONING_EFFORT=none` |
| Critic REJECTs with "off-topic" | corpus topic mismatch — the 540p corpus is reasoning/agent/RAG-heavy, not math/code-specific | use a topic that matches; or build a corpus aligned to your domain |
| Critic REJECTs with "task already completed" | OMC critic hallucinating from polluted task history | use a fresh project (`POST /api/ceo/task` with new topic) |

For more, see the OMC startup log at `/tmp/omc-startup.log` and grep for `[lcg]`.

## Updating

If the lcg repo changes (`stable/v0.7-runner-local` advances):
```bash
bash scripts/setup-lcg.sh --skip-clone   # re-runs `git pull` + venv refresh
```

If `launch.sh` in this directory changes after the talent is hired:
```bash
# the live employee dir copy doesn't auto-sync — refresh it manually:
cp src/onemancompany/talent_market/talents/literature-conflict-graph/launch.sh \
   .onemancompany/company/human_resource/employees/00015/launch.sh
```

(Or fire+rehire 00015 to re-run `copy_talent_assets` cleanly.)

## Topics that PASS reliably (corpus-aligned)

The bundled 540p run is "LLM reasoning (arxiv 2023+)" topic-tagged but content
leans toward RAG / agent / web / multimodal / long-context QA. Topics that
match these corpus areas tend to PASS critic gate first try; topics outside
the corpus may need a domain-specific corpus to land cleanly.

Worked first try (0.88 confidence):
- `agentic reasoning evaluation metric divergence`

Likely to work:
- `RAG benchmark inconsistencies on factual QA`
- `long-context QA with retrieval-augmented agents`
- `agent on multi-step reasoning evaluation`

May need a domain-specific corpus first:
- math-only or code-only topics — the 540p has only ~6 hypotheses each.
