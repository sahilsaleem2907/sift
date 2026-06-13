# Sift Review Engine Redesign

Goal: raise the **quality** of Sift's PR reviews — find more severe/meaningful bugs,
surface architectural insights (not just linter noise), and produce strong reviews
across a wide range of models (Claude Opus 4.8, deepseek-coder, llama 3.1, Haiku,
Gemini Flash/Pro) by degrading gracefully.

## Core idea: two orthogonal knobs

| Knob | Set by | Controls |
|------|--------|----------|
| **Effort** (`low`/`balanced`/`high`) | user (`SIFT_REVIEW_EFFORT`) | *how hard we try*: number of passes, retrieval depth, critic granularity |
| **Capability** (auto-detected) | LiteLLM model introspection | *how we talk to the model*: structured output vs JSON, context budget, reasoning, whether the agentic loop is possible |

Effort raises the pass count; capability detection keeps each call within the model's
limits. This is why a cheap-but-large-context cloud model (Haiku/Flash) can run `high`
effort, and a tiny local llama running `high` won't overflow context.

## Locked decisions

- **Effort default**: `balanced` (critic + holistic on, no agentic).
- **Pass models**: `SIFT_REVIEW_MODEL` (+ `SIFT_REVIEW_MODEL_KEY`, `SIFT_REVIEW_MODEL_BASE_URL`)
  lets critic/holistic passes run on a separate/cheaper model. Defaults to `LLM_MODEL`.
- **Agentic retrieval**: only at `high` *and* when the model supports tool-calling;
  deterministic pre-built context otherwise.
- **Critic granularity**: skipped at `low`, batched-per-file at `balanced`, per-finding at `high`.
- **Legacy single-pass code** (`review()`, `SYSTEM_PROMPT`, old `review_file` path): **removed**;
  the pipeline replaces it fully.
- **Eval**: lightweight offline golden-set harness; start synthetic, add curated real PRs later.
- **Severity labels**: keep the existing 5-tier
  (`bug`/`security`/`warning`/`suggestion`/`informational`), **derived** from a new
  `impact × certainty` rubric so the summary/badge/block-policy code keeps working.

## Pass pipeline

```
1. Context build (retrieval.py)      deterministic; +semantic diff/callers at balanced; +callees/agentic at high
2. Candidate generation (per file)   passes/candidates.py   — runs at every effort
3. Critic / verification             passes/critic.py       — balanced (batched) / high (per-finding)
4. Holistic PR pass                  passes/holistic.py     — balanced+
5. Severity rubric + noise gate      passes/severity.py     — always (deterministic)
6. Summarize + post                  (existing, lightly adapted)
```

| Stage | low | balanced | high |
|-------|-----|----------|------|
| 1 Context | changed lines + window | + semantic fn before/after, callers | + callees, vector neighbors, agentic loop (if fn-calling) |
| 2 Candidates | single cheap call | + reasoning if supported | richer prompt + reasoning |
| 3 Critic | skip (rule dedupe) | batched per file | per-finding deep re-read |
| 4 Holistic | skip | one PR-level pass | richer / multi-step |
| 5 Severity gate | yes | yes | yes |
| 6 Summarize+post | yes | yes | yes |

## Target module layout

```
src/intelligence/
  capability.py        # NEW
  effort.py            # NEW
  schema.py            # NEW  Finding (impact, certainty, category, origin)
  prompts.py           # NEW  centralized prompt templates
  llm_client.py        # SLIMMED to _call_llm + structured-output + parsing/formatting
  retrieval.py         # NEW  deterministic context + agentic loop
  passes/
    __init__.py
    pipeline.py        # NEW  orchestrates passes per EffortPlan
    candidates.py      # NEW
    critic.py          # NEW
    holistic.py        # NEW
    severity.py        # NEW
eval/                  # NEW  golden-set harness
```

`src/core/review_engine.py` keeps all GitHub/static-tool/routing/caching machinery and
only swaps the per-file `review_file` call + `summarize_review` for `passes/pipeline.py`.

## Phases (each independently shippable + measurable)

1. **[Phase 1 — Foundation](phase-1-foundation.md)**: schema, capability, effort, prompts,
   slim llm_client, behavior-neutral pipeline, eval harness.
2. **[Phase 2 — Critic + Severity](phase-2-critic-severity.md)**: verification pass +
   impact×certainty rubric + noise gate (biggest quality win).
3. **[Phase 3 — Holistic](phase-3-holistic.md)**: PR-level architectural discovery pass.
4. **[Phase 4 — Retrieval + Agentic](phase-4-retrieval-agentic.md)**: semantic diff,
   callee resolution, bounded agentic context loop for high effort.

Each phase ends with golden-set numbers (precision / recall / noise-rate) per model+effort.
