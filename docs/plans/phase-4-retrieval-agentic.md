# Phase 4 — Deeper context + bounded agentic retrieval

**Objective:** give the model the *right* context. Deterministic enrichment for all
efforts (semantic before/after diffs, callee resolution), plus a bounded agentic
retrieval loop at `high` effort for models that support tool-calling — letting strong
models (Opus, deepseek) pull exactly the code they need to confirm/deny an issue.

**Exit criterion:** on golden cases that require off-diff context (e.g. a bug only
visible by reading a called helper), `high` effort with a tool-calling model recovers
findings that `balanced` misses, without raising noise-rate on self-contained cases.

---

## 4.1 Deterministic context — `src/intelligence/retrieval.py` (NEW)

Centralize context assembly currently inlined in `review_engine._process_file`
(file_context ranges, semgrep/codeql/linter blocks, AST, caller graph, vector snippets).
Add depth tiers driven by `EffortPlan.context_depth`:

- **depth 0** (low): today's behavior — changed lines + a small surrounding window.
- **depth 1** (balanced): + **semantic before/after** of each changed function. We have
  the new-file functions from `extract_modified_functions`; reconstruct the old function
  body from the diff (the `-` lines / pre-image) so the model sees what *changed*
  semantically, not just `+`/`-` lines.
- **depth 2** (high): + **callee resolution** — for functions called by the changed code
  that are defined in files present in the PR (or fetchable), include their signatures/
  bodies. Reuse `import_analyzer` machinery, but resolve the *callee* direction (current
  graph resolves callers/importers; add the inverse).

All context packing is **capability-bounded**: `retrieval.py` takes
`cap.context_window` and trims (drop lowest-value blocks first: vector snippets →
callees → callers → semantic diff → core diff which is never dropped).

## 4.2 Agentic loop — `retrieval.py` + `passes/`

Only when `plan.enable_agentic and cap.supports_function_calling`:

```python
TOOLS = [get_file(path), get_function(path, name), find_references(symbol)]

async def agentic_review(file_input, plan, cap) -> list[Finding]:
    # bounded loop: model may call tools up to SIFT_AGENTIC_MAX_STEPS times,
    # then must emit findings. Each tool result is appended to the context.
```

- Tools are backed by the already-fetched `path_to_content` and the GitHub client
  (`get_file_content`) for files outside the diff; cache reads.
- Hard cap: `config.SIFT_AGENTIC_MAX_STEPS` (default 4) and a total token budget from
  `cap`. Abort to deterministic context on tool errors or budget exhaustion.
- Falls back automatically to depth-2 deterministic context when the model lacks
  tool-calling — this is the "yes, with fallback" decision.

## 4.3 Wiring
- `candidates.py` and per-finding `critic.py` both accept the richer context; the critic
  at `high` can trigger a *targeted* `get_function`/`find_references` to confirm a
  specific claim (cheaper than a full agentic candidate loop).
- Keep deterministic context the default path everywhere; agentic is an opt-in branch
  gated by effort + capability.

## 4.4 Tests + eval
- `tests/test_retrieval.py` — semantic before/after reconstruction from a diff; callee
  resolution from a 2-file PR; context trimming honors a small `context_window`.
- `tests/test_agentic.py` — mock tool-calling model issues `get_function`, receives the
  body, then emits a finding; loop respects `SIFT_AGENTIC_MAX_STEPS`; non-tool model
  falls back to deterministic.
- `eval/cases/` "needs-off-diff-context" fixtures (real PRs are ideal here — add curated
  ones). Compare `high`+tool-calling vs `balanced`.

## Risks / notes
- Agentic loops add latency and unpredictability — the step cap + token budget + clean
  deterministic fallback are mandatory.
- `find_references` across a repo is expensive; restrict to PR files + direct imports
  initially, expand only if eval shows value.
- Token blow-up is the main failure mode; the capability-bounded trimming in 4.1 is the
  safeguard and must be unit-tested with tiny windows.
