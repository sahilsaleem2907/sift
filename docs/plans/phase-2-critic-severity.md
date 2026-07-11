# Phase 2 — Critic pass + Severity rubric (the precision equalizer)

**Objective:** add the verification/critic pass and replace self-reported `confidence`
with a real `impact × certainty` rubric + noise gate. This is the single biggest quality
lever and the main mechanism that lets weaker models approach strong-model precision —
models judge a concrete claim against code far better than they generate claims unaided.

**Exit criterion:** on the golden set, `balanced` shows **higher precision and lower
noise-rate** than Phase 1 at equal recall, across at least Opus, a mid model
(deepseek/Haiku), and a weak model (llama 3.1). No regression in recall of `critical`/
`high` impact findings.

**Phase 1 baseline (locked):**
- claude-sonnet-4.6 balanced: Precision 80%, Recall 100%, Noise 0%
- llama-3.1-8b balanced: Precision 67%, Recall 100%, Noise 17%

---

## Deliverables checklist

- [ ] `src/intelligence/passes/critic.py` (NEW)
- [ ] `src/intelligence/passes/severity.py` (NEW)
- [ ] `src/intelligence/prompts.py` — add `CRITIC_SYSTEM` (batched) and `CRITIC_FINDING_SYSTEM` (per-finding)
- [ ] `src/intelligence/passes/pipeline.py` — wire critic + severity gate
- [ ] `eval/cases/case_006_false_positive_type_annotation.py.diff` + `.json` (NEW bait case)
- [ ] `eval/cases/case_007_false_positive_correct_guard.py.diff` + `.json` (NEW bait case)
- [ ] `tests/test_critic.py` (NEW)
- [ ] `tests/test_severity.py` (NEW)

---

## Step 1 — Prompts: add critic templates to `src/intelligence/prompts.py`

Add two new constants after `SUMMARIZE_SYSTEM`:

**`CRITIC_BATCHED_SYSTEM`** — used at `balanced` effort (all file candidates in one call):

```python
CRITIC_BATCHED_SYSTEM = """You are a strict second-pass code reviewer. You are given a list
of proposed findings and the actual code diff they were found in.

For each finding, decide:
1. KEEP — it is a real, reproducible problem directly visible in the changed code.
2. DROP — it is speculative, a style nit, cannot be confirmed from the diff, or is
   unrelated to the PR's stated intent.

Rules:
- Bias toward KEEP for impact "critical" or "high" — only drop if the claim is clearly wrong.
- Always DROP if the finding is about code that was NOT changed (pre-existing issues).
- Always DROP exact duplicates or near-duplicates of another finding on the same line.
- You may re-rate impact and certainty. Use the same scale:
  impact: critical | high | medium | low | trivial
  certainty: confirmed | likely | speculative

Respond with a JSON array. One object per input finding, in the same order:
{
  "index": <0-based integer matching the input>,
  "verdict": "keep" | "drop",
  "impact": "<re-rated or unchanged>",
  "certainty": "<re-rated or unchanged>",
  "reason": "<one sentence, required>"
}
No markdown fences. No prose outside the array."""
```

**`CRITIC_FINDING_SYSTEM`** — used at `high` effort (one call per finding, deeper):

```python
CRITIC_FINDING_SYSTEM = """You are a strict second-pass code reviewer. You are given a
single proposed finding and the code context it was found in.

Decide: is this a REAL, MEANINGFUL problem that a developer should act on?

Rules:
- KEEP if the issue is directly confirmable from the code shown.
- DROP if it is a style nit, speculative, unrelated to what changed, or already handled.
- Bias heavily toward KEEP for security and correctness bugs; allow more DROP for
  maintainability and style.
- Re-rate impact and certainty honestly.

Respond with a single JSON object (no array, no markdown fences):
{
  "verdict": "keep" | "drop",
  "impact": "critical" | "high" | "medium" | "low" | "trivial",
  "certainty": "confirmed" | "likely" | "speculative",
  "reason": "<one sentence>"
}"""
```

---

## Step 2 — `src/intelligence/passes/critic.py` (NEW, ~130 lines)

```python
"""Pass 2: critic / verification pass."""
import asyncio, json, logging
from typing import Any, Optional

from src import config
from src.intelligence.capability import ModelCapability, review_capability
from src.intelligence.effort import EffortPlan
from src.intelligence.llm_client import _call_llm, _extract_json_array
from src.intelligence.prompts import CRITIC_BATCHED_SYSTEM, CRITIC_FINDING_SYSTEM
from src.intelligence.schema import Certainty, Finding, Impact

logger = logging.getLogger(__name__)

_IMPACT_VALUES = [i.value for i in Impact]
_CERTAINTY_VALUES = [c.value for c in Certainty]
```

### `rule_dedupe(findings)` — used at `low` effort

Cheap O(n²) pass: drop any finding whose `(path, line)` is identical to a higher-impact
finding already in the list. No LLM call.

```python
def rule_dedupe(findings: list[Finding]) -> list[Finding]:
    seen: dict[tuple[str, int], Finding] = {}
    for f in findings:
        key = (f.path, f.line)
        if key not in seen or _impact_rank(f.impact) < _impact_rank(seen[key].impact):
            seen[key] = f
    return list(seen.values())
```

### `_apply_verdict(finding, verdict_obj)` — updates a Finding in-place

Parses the LLM's `{"verdict", "impact", "certainty", "reason"}` dict and returns
a new `Finding` with updated fields (or `None` if verdict is `drop`).

### `critique_batched(findings, diff, pr_title, cap)` — one LLM call per file

Serializes all findings as a numbered list, calls `CRITIC_BATCHED_SYSTEM`, parses the
JSON array of verdicts, and filters/updates findings accordingly.

```python
async def critique_batched(
    findings: list[Finding],
    diff: str,
    pr_title: str,
    cap: ModelCapability,
) -> list[Finding]:
    if not findings:
        return []

    items = "\n".join(
        f'[{i}] line={f.line} impact={f.impact.value} certainty={f.certainty.value}\n'
        f'     title: {f.title or "(see body)"}\n'
        f'     body: {f.body[:300]}'
        for i, f in enumerate(findings)
    )
    user_content = f"PR title: {pr_title}\n\nDiff:\n{diff}\n\nProposed findings:\n{items}"

    raw = await _call_llm(
        CRITIC_BATCHED_SYSTEM,
        user_content,
        model=config.SIFT_REVIEW_MODEL or config.LLM_MODEL,
        api_base=config.SIFT_REVIEW_MODEL_BASE_URL or config.LLM_API_BASE or None,
        api_key=config.SIFT_REVIEW_MODEL_KEY or None,
    )

    verdicts = _extract_json_array(raw) or []
    verdict_map = {int(v["index"]): v for v in verdicts if isinstance(v, dict) and "index" in v}

    kept = []
    for i, f in enumerate(findings):
        v = verdict_map.get(i)
        if v is None:
            kept.append(f)  # no verdict = keep (safe default)
            continue
        if (v.get("verdict") or "keep").lower() == "drop":
            logger.debug("[critic] DROP line=%d reason=%s", f.line, v.get("reason", ""))
            continue
        updated = _apply_verdict(f, v)
        kept.append(updated)
        logger.debug("[critic] KEEP line=%d impact=%s certainty=%s", updated.line, updated.impact, updated.certainty)

    return kept
```

### `critique_per_finding(findings, diff, pr_title, cap)` — one LLM call per finding

Same logic but calls `CRITIC_FINDING_SYSTEM` once per finding. Respects
`SIFT_LLM_REQUEST_DELAY` between calls.

### Main entry point

```python
async def critique(
    findings: list[Finding],
    diff: str,
    pr_title: str,
    plan: EffortPlan,
    cap: ModelCapability,
) -> list[Finding]:
    if plan.critic_per_finding:
        return await critique_per_finding(findings, diff, pr_title, cap)
    return await critique_batched(findings, diff, pr_title, cap)
```

---

## Step 3 — `src/intelligence/passes/severity.py` (NEW, ~60 lines)

```python
"""Pass 4 (Phase 2): severity rubric + noise gate."""
from src.intelligence.effort import EffortPlan
from src.intelligence.schema import Certainty, Finding, Impact

_GATE_RULES = [
    # (condition, action)
    # 1. Trivial impact — always drop
    (lambda f: f.impact == Impact.TRIVIAL, "drop"),
    # 2. Speculative + low impact — drop
    (lambda f: f.certainty == Certainty.SPECULATIVE and f.impact == Impact.LOW, "drop"),
    # 3. Speculative + medium impact — downgrade to informational (keep as inline)
    # (handled in body prefix logic, not a drop)
]
```

**`apply_severity_gate(findings, plan)`:**

```python
def apply_severity_gate(findings: list[Finding], plan: EffortPlan) -> list[Finding]:
    out = []
    for f in findings:
        if f.impact == Impact.TRIVIAL:
            continue
        if f.certainty == Certainty.SPECULATIVE and f.impact == Impact.LOW:
            continue
        if f.certainty == Certainty.SPECULATIVE and f.impact == Impact.CRITICAL:
            # Keep but mark as unverified
            object.__setattr__(f, "body", "[Unverified] " + f.body) if hasattr(f, "__dataclass_fields__") else None
            # Finding is mutable dataclass, can assign directly:
            f = Finding(
                **{**f.__dict__,
                   "body": "[Unverified — needs manual check] " + f.body}
            )
        out.append(f)
    return out
```

Note: `Finding` is a regular (mutable) dataclass so field assignment works directly.

---

## Step 4 — Wire `src/intelligence/passes/pipeline.py`

Replace the three stub comment lines and `_ = pr_meta, plan, cap` with real logic.

The `diff` needed by the critic is available on each `FileReviewInput.file_diff`.
Group findings back by file for the batched critic call:

```python
async def run_pipeline(
    files: list[FileReviewInput],
    pr_meta: PRMeta,
    plan: EffortPlan,
    cap: ModelCapability,
) -> list[Finding]:
    from src.intelligence.passes.critic import critique, rule_dedupe
    from src.intelligence.passes.severity import apply_severity_gate

    all_findings: list[Finding] = []

    for f in files:
        candidates = await generate_candidates(f.file_diff, f.path, f.pr_context)
        logger.debug("[pipeline] %s: %d candidate(s)", f.path, len(candidates))

        if plan.run_critic and candidates:
            pr_title = pr_meta.title or ""
            candidates = await critique(candidates, f.file_diff, pr_title, plan, cap)
            logger.debug("[pipeline] %s: %d after critic", f.path, len(candidates))
        else:
            candidates = rule_dedupe(candidates)

        all_findings.extend(candidates)

    # Phase 3: all_findings += await review_holistic(digest, plan, cap)

    all_findings = apply_severity_gate(all_findings, plan)
    return all_findings
```

Remove `_ = pr_meta, plan, cap` — they are now used.

---

## Step 5 — Two new false-positive bait eval cases

**`case_006_false_positive_type_annotation.py.diff`** — correct code that adds a type annotation:

```diff
diff --git a/app/utils.py b/app/utils.py
--- a/app/utils.py
+++ b/app/utils.py
@@ -1,3 +1,3 @@
-def get_name(user):
+def get_name(user: dict) -> str:
     return user["name"]
```

`case_006.json`: `expected: []`, `false_positives: [2]` — model must stay quiet.

**`case_007_false_positive_correct_guard.py.diff`** — correct None guard added:

```diff
diff --git a/app/service.py b/app/service.py
--- a/app/service.py
+++ b/app/service.py
@@ -1,4 +1,5 @@
 def process(item):
+    if item is None:
+        return
     print(item.value)
```

`case_007.json`: `expected: []`, `false_positives: [2, 3]` — this is a fix, not a bug.

Updated score target: **7 cases, Precision 90%+, Recall 100%, Noise 0%** at balanced.

---

## Step 6 — Tests

### `tests/test_critic.py`

- `test_batched_keeps_real_bug`: mock `_call_llm` returns `[{"index": 0, "verdict": "keep", ...}]`; assert finding survives.
- `test_batched_drops_false_positive`: mock returns `"drop"` for a trivial finding; assert it is removed.
- `test_batched_missing_verdict_keeps`: when LLM returns empty list, all findings kept (safe default).
- `test_per_finding_drops`: mock per-finding call returns `"drop"`; assert finding removed.
- `test_rule_dedupe_keeps_higher_impact`: two findings on same line, different impact; lower-impact dropped.
- `test_critique_empty_input_returns_empty`: no candidates → no LLM call, empty list returned.

### `tests/test_severity.py`

- `test_trivial_dropped`: `Impact.TRIVIAL` finding is removed regardless of certainty.
- `test_speculative_low_dropped`: `SPECULATIVE + LOW` is removed.
- `test_speculative_medium_kept`: `SPECULATIVE + MEDIUM` is kept (downgrade handled by body prefix, not drop).
- `test_speculative_critical_kept_as_unverified`: body is prefixed with `[Unverified...]`.
- `test_confirmed_high_kept`: `CONFIRMED + HIGH` passes through unchanged.

---

## Build order

1. `src/intelligence/prompts.py` — add `CRITIC_BATCHED_SYSTEM`, `CRITIC_FINDING_SYSTEM`
2. `src/intelligence/passes/critic.py` — full implementation
3. `src/intelligence/passes/severity.py` — full implementation
4. `src/intelligence/passes/pipeline.py` — wire both passes
5. `eval/cases/case_006.*` + `case_007.*` — new bait cases
6. `tests/test_critic.py` + `tests/test_severity.py`

---

## Cost and concurrency notes

- Batched critic: +1 LLM call per file (same semaphore as candidate generation).
- Per-finding critic: +N calls per file; each call respects `SIFT_LLM_REQUEST_DELAY`.
- Critic uses `SIFT_REVIEW_MODEL` if set; falls back to `LLM_MODEL`. Document in README.
- Log `[critic]` prefix with keep/drop count per file at DEBUG level.

---

## Risks

- Critic too aggressive on real bugs — `_extract_json_array` parse failure defaults to KEEP (safe). Prompt biases KEEP for critical/high.
- Weak model as critic hurts precision — `SIFT_REVIEW_MODEL` lets you point critic at a stronger model independently of the generation model.
