# Phase 3 — Holistic PR pass (meaningful, non-linter insights)

**Objective:** add a whole-PR reasoning pass that finds issues the per-file passes
structurally cannot — cross-file design problems, contract/interface drift, duplicated
logic, missing call-site updates, inconsistent error handling across modules.

**Exit criterion:** golden-set "architectural" cases (multi-file fixtures with a cross-file
defect) are caught at `balanced`/`high` but missed at `low`; no measurable increase in
noise on single-file cases.

**Phase 2 baseline (locked):**
- sonnet + haiku critic: 100% precision, 100% recall, 0% noise (7 cases)
- llama + haiku critic: 100% precision, 100% recall, 0% noise (7 cases)

---

## Deliverables checklist

- [ ] `src/intelligence/passes/holistic.py` (NEW) — `PRDigest`, `build_digest`, `review_holistic`
- [ ] `src/intelligence/prompts.py` — add `HOLISTIC_SYSTEM`
- [ ] `src/intelligence/passes/pipeline.py` — wire holistic pass + dedupe against per-file findings
- [ ] `src/core/review_engine.py` — pass `import_graph` + `mod_funcs_by_path` via `PRMeta`
- [ ] `eval/cases/case_008_*` + `case_009_*` — multi-file cross-file defect cases
- [ ] `tests/test_holistic.py` (NEW)

---

## Step 1 — Prompt: add `HOLISTIC_SYSTEM` to `src/intelligence/prompts.py`

Append after `CRITIC_FINDING_SYSTEM`:

```python
HOLISTIC_SYSTEM = """You are reviewing an entire pull request for CROSS-FILE and DESIGN-LEVEL
problems only.

Focus exclusively on:
- A symbol (function, class, constant) was changed in one file but its callers in other
  changed files were NOT updated to match the new signature or contract.
- Duplicated logic that appears in multiple changed files and should be shared.
- Layering or abstraction violations: a lower-level module directly importing or
  calling into a higher-level one.
- Missing error handling that spans multiple modules (one file throws, another never catches).
- Inconsistent contracts: two files implement the same interface but behave differently.

Do NOT repeat issues already found in the per-file review.
Anchor each finding to a specific file and line number from the diff.
If you cannot anchor a finding to a changed line, mark post_inline as false.

Respond with a JSON array. No markdown fences. Each element:
{
  "path": "<file path>",
  "line": <integer — a changed line in that file>,
  "title": "<10 words max>",
  "body": "<description of the cross-file issue>",
  "impact": "critical" | "high" | "medium" | "low",
  "certainty": "confirmed" | "likely" | "speculative",
  "category": "correctness" | "security" | "design" | "maintainability",
  "post_inline": true | false,
  "fix": "<optional fix>"
}
Return [] if there are no cross-file issues."""
```

---

## Step 2 — `src/intelligence/passes/holistic.py` (NEW, ~120 lines)

### `PRDigest` dataclass

Compact, serialisable summary of the PR fed to the holistic prompt:

```python
@dataclass
class PRDigest:
    title: str
    body: str
    # list of {"path": str, "name": str, "lines": "N-M"} — one per modified function
    changed_functions: list[dict]
    # list of {"importer": str, "imports_from": str, "symbols": [...]} — from import graph
    import_edges: list[dict]
    # list of {"path": str, "line": int, "title": str, "impact": str, "category": str}
    per_file_findings: list[dict]
```

### `build_digest(pr_meta, per_file_findings)` — assembles the digest

Takes the `PRMeta` (which now carries `mod_funcs_by_path` and `import_graph`) and the
post-critic per-file findings, and returns a `PRDigest`.

Changed functions: up to 30 entries (truncate beyond that). For each `FunctionChunk`
across all paths, emit `{"path": path, "name": name or "?", "lines": "start-end"}`.

Import edges: flatten `import_graph` (a `dict[str, list[CallerInfo]]`) into a list of
`{"importer": path, "imports_from": ci.changed_path, "symbols": list(ci.function_names)}`.

Per-file findings: titles + impact + category of already-kept findings (max 20, sorted
by impact descending). These tell the holistic pass what was already found so it doesn't
repeat.

### `review_holistic(digest, plan, cap)` → `list[Finding]`

Serialise the digest to a compact text block and call the LLM:

```python
async def review_holistic(
    digest: PRDigest,
    plan: EffortPlan,
    cap: ModelCapability,
) -> list[Finding]:
    if not digest.import_edges and not digest.changed_functions:
        return []  # single-file PR: nothing cross-file to find

    user_content = _format_digest(digest)
    raw = await _call_llm(
        HOLISTIC_SYSTEM,
        user_content,
        model=config.SIFT_REVIEW_MODEL or config.LLM_MODEL,
        api_base=config.SIFT_REVIEW_MODEL_BASE_URL or config.LLM_API_BASE or None,
        api_key=config.SIFT_REVIEW_MODEL_KEY or None,
    )
    return _parse_holistic_response(raw)
```

`_format_digest`: plain text, not JSON — more token-efficient:

```
PR: <title>
<body first 300 chars>

Changed functions (N):
- app/auth.py  verify_token(lines 12-28)
- app/user.py  get_user(lines 5-18)

Import edges:
- app/api.py imports app/auth.py (symbols: verify_token)

Already found (do not repeat):
- app/auth.py:15  correctness/high  Unhandled None return
```

`_parse_holistic_response`: parse JSON array → list of `Finding` with `origin="holistic"`.
Parse each item's `path`, `line`, `impact`, `certainty`, `category`, `body`, `fix`,
`post_inline`. Fall back gracefully on parse errors (return `[]`).

---

## Step 3 — Wire `src/intelligence/passes/pipeline.py`

Replace the Phase 3 stub comment with real logic. Key difference from the original
design doc: holistic findings go through the **same critic + severity gate** as
per-file findings. The critic is also gated on `SIFT_REVIEW_MODEL` being set.

Also add deduplication: drop a holistic finding if an existing per-file finding already
covers `(path, line, category)`.

```python
# after per-file loop:
if plan.run_holistic:
    digest = build_digest(pr_meta, all_findings)
    holistic = await review_holistic(digest, plan, cap)
    # dedupe: drop holistic findings already covered per-file
    per_file_keys = {(f.path, f.line, f.category) for f in all_findings}
    holistic = [f for f in holistic if (f.path, f.line, f.category) not in per_file_keys]
    if holistic:
        logger.debug("[pipeline] holistic: %d new finding(s)", len(holistic))
        # run critic on holistic findings too (same guard as per-file)
        if bool(config.SIFT_REVIEW_MODEL) and holistic:
            holistic = await critique(holistic, "", pr_title, plan, cap)
        all_findings.extend(holistic)

all_findings = apply_severity_gate(all_findings, plan)
```

The `pr_meta` object needs `import_graph` and `mod_funcs_by_path` populated — see Step 4.

---

## Step 4 — Wire `src/core/review_engine.py`

`mod_funcs_by_path` and `pr_import_graph` are already computed (lines 611–626) before
`_process_file` is called. They just need to reach `PRMeta`.

Currently `run_pipeline` is called per-file with a minimal `PRMeta(title, body)`.
Phase 3 requires a single `run_pipeline` call for the **whole PR** after all per-file
candidates are gathered, so holistic can see everything.

**Current structure (Phase 1/2):**
```
for each file:
    run_pipeline([single_file_input], PRMeta(title, body), plan, cap)
```

**Phase 3 restructure:**
```
# per-file: only candidates + critic
candidates_by_file = await gather(generate_and_critique(file) for file in files)
all_per_file_findings = flatten(candidates_by_file)

# holistic: one call for the whole PR
pr_meta_full = PRMeta(title, body, import_graph=pr_import_graph, mod_funcs_by_path=mod_funcs_by_path)
all_findings = await run_pipeline_holistic_stage(all_per_file_findings, pr_meta_full, plan, cap)
```

The cleanest way to do this without a large review_engine refactor: split `run_pipeline`
into two entry points:

- `run_pipeline_per_file(file, pr_title, plan, cap) → list[Finding]` — candidates + critic only (called per-file in parallel as today)
- `run_pipeline_holistic(all_findings, pr_meta, plan, cap) → list[Finding]` — holistic + severity gate (called once after gather)

`review_engine.py` changes:
1. When building `PRMeta` for the per-file call, continue passing only `title`/`body`.
2. After the `asyncio.gather` that collects all per-file results, call `run_pipeline_holistic` once, passing the full `PRMeta` with `import_graph` and `mod_funcs_by_path` populated from the already-computed local variables.

---

## Step 5 — Eval cases (multi-file)

Multi-file diffs use a single `.diff` file with multiple `diff --git` sections.
The `GoldenCase.path` field becomes the primary anchor file; the eval scorer already
matches on `(path, line_range, category)` so cross-file findings anchored to the
correct file will score correctly.

**`case_008_caller_not_updated`**: function signature changed in `app/auth.py`, caller
in `app/api.py` not updated.

```diff
diff --git a/app/auth.py b/app/auth.py
--- a/app/auth.py
+++ b/app/auth.py
@@ -1,3 +1,3 @@
-def verify_token(token: str) -> bool:
+def verify_token(token: str, strict: bool = True) -> dict:
     pass
diff --git a/app/api.py b/app/api.py
--- a/app/api.py
+++ b/app/api.py
@@ -1,4 +1,4 @@
+from app.auth import verify_token
+
 def handle_request(token: str):
-    if verify_token(token):
+    if verify_token(token):
         pass
```

`case_008.json`: expected finding on `app/api.py` line 4, category `correctness`,
min_impact `medium`. Note: `path` in the JSON is the anchor file (`app/api.py`).

**`case_009_duplicate_logic`**: same validation logic duplicated in two changed files.

```diff
diff --git a/app/orders.py b/app/orders.py
--- a/app/orders.py
+++ b/app/orders.py
@@ -1,3 +1,5 @@
+def validate_amount(amount):
+    if amount <= 0:
+        raise ValueError("amount must be positive")
     pass
diff --git a/app/payments.py b/app/payments.py
--- a/app/payments.py
+++ b/app/payments.py
@@ -1,3 +1,5 @@
+def validate_amount(amount):
+    if amount <= 0:
+        raise ValueError("amount must be positive")
     pass
```

`case_009.json`: expected on `app/payments.py` line 2, category `design`,
min_impact `low`. Note: duplicate logic is a design issue, not a correctness bug.

`GoldenCase.path` for multi-file cases should be the file the expected finding is
anchored to, not the diff's "primary" file.

---

## Step 6 — Tests: `tests/test_holistic.py`

- `test_build_digest_extracts_functions`: given a `PRMeta` with `mod_funcs_by_path`
  containing 2 functions, `build_digest` includes both in `changed_functions`.
- `test_build_digest_extracts_import_edges`: given `import_graph` with a `CallerInfo`,
  one import edge appears in the digest.
- `test_review_holistic_returns_findings`: mock `_call_llm` returns a valid JSON array
  with one holistic finding; `review_holistic` returns 1 `Finding` with `origin="holistic"`.
- `test_review_holistic_empty_when_no_edges`: when both `import_edges` and
  `changed_functions` are empty, returns `[]` without calling LLM.
- `test_review_holistic_parse_failure_returns_empty`: mock returns malformed JSON; no
  exception raised, returns `[]`.
- `test_pipeline_dedupes_holistic_against_per_file`: holistic finding on same
  `(path, line, category)` as an existing per-file finding is dropped.

---

## Build order

1. `src/intelligence/prompts.py` — add `HOLISTIC_SYSTEM`
2. `src/intelligence/passes/holistic.py` — `PRDigest`, `build_digest`, `review_holistic`
3. `src/intelligence/passes/pipeline.py` — split into `run_pipeline_per_file` + `run_pipeline_holistic`, wire holistic pass
4. `src/core/review_engine.py` — call `run_pipeline_per_file` in the file loop, then `run_pipeline_holistic` once after gather
5. `eval/cases/case_008_*` + `case_009_*`
6. `tests/test_holistic.py`

---

## Risks

- Holistic pass produces vague/opinionated output — same critic + severity gate guards apply.
- Token budget: `_format_digest` must cap at capability `context_window`; if the PR has
  hundreds of functions, truncate changed_functions list to 30.
- Single-file PRs: `build_digest` short-circuits when there are no import edges AND no
  cross-file function references — no LLM call, no cost.
- The eval multi-file cases anchor to specific files; the `GoldenCase.path` field must
  match the `path` the holistic finding is anchored to, not necessarily the "first" diff file.
