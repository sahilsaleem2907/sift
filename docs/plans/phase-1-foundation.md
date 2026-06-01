# Phase 1 — Foundation (behavior-neutral pipeline + eval harness)

**Goal:** introduce all new scaffolding modules and rewire the review pipeline through
them — *without changing any review output*. Every later phase then has a clean,
independently testable seam to work against, and the eval harness makes every future
change measurable.

**Behavioral contract:** with `SIFT_REVIEW_EFFORT=low` (or the default `balanced` before
Phase 2 features are wired in) the exact same LLM call is made and the exact same
comment bodies are posted as today. This is verified by the pipeline passthrough test.

**Deliverables checklist:**
- [ ] `src/config.py` — 4 new env vars
- [ ] `.env.example` — document them
- [ ] `src/intelligence/effort.py` (NEW)
- [ ] `src/intelligence/capability.py` (NEW)
- [ ] `src/intelligence/schema.py` (NEW)
- [ ] `src/intelligence/prompts.py` (NEW)
- [ ] `src/intelligence/llm_client.py` — slimmed, `_call_llm` made multi-model
- [ ] `src/intelligence/passes/__init__.py` (NEW, empty)
- [ ] `src/intelligence/passes/candidates.py` (NEW)
- [ ] `src/intelligence/passes/pipeline.py` (NEW)
- [ ] `src/core/review_engine.py` — swap `review_file` for `run_pipeline`
- [ ] `eval/` directory + 5 golden cases + scorer
- [ ] `tests/` — 4 unit test files

---

## Step 1 — `src/config.py`

Append after the existing `SIFT_STATUS_CONTEXT` block:

```python
# Review engine effort: low | balanced | high  (default: balanced)
SIFT_REVIEW_EFFORT = (os.environ.get("SIFT_REVIEW_EFFORT") or "balanced").strip().lower()

# Optional separate model for critic / holistic passes.
# Defaults to LLM_MODEL when unset.  Useful to run generation on a local model
# and verification on a cheap cloud model, or vice-versa.
SIFT_REVIEW_MODEL = os.environ.get("SIFT_REVIEW_MODEL") or None
SIFT_REVIEW_MODEL_KEY = os.environ.get("SIFT_REVIEW_MODEL_KEY") or None
_review_base = (os.environ.get("SIFT_REVIEW_MODEL_BASE_URL") or "").strip()
SIFT_REVIEW_MODEL_BASE_URL = _review_base.rstrip("/") if _review_base else None

# JSON object to hard-override capability detection for unknown / self-hosted models.
# Example: {"context_window":32768,"supports_function_calling":true,"supports_reasoning":false}
SIFT_CAPABILITY_OVERRIDE = os.environ.get("SIFT_CAPABILITY_OVERRIDE") or None

# Max tool-call steps in the high-effort agentic retrieval loop (Phase 4).
SIFT_AGENTIC_MAX_STEPS = int(os.environ.get("SIFT_AGENTIC_MAX_STEPS") or "4")
```

In `validate_required()` add:

```python
_valid_efforts = ("low", "balanced", "high")
if SIFT_REVIEW_EFFORT not in _valid_efforts:
    _log.warning(
        "SIFT_REVIEW_EFFORT=%r is not one of %s; falling back to 'balanced'.",
        SIFT_REVIEW_EFFORT, _valid_efforts,
    )
    # NOTE: caller must re-read SIFT_REVIEW_EFFORT through effort.resolve_effort()
    # which applies the same fallback — do not mutate the module-level var here.
```

## Step 2 — `.env.example`

Append a new section at the end:

```
# Review engine effort and secondary model
# SIFT_REVIEW_EFFORT=balanced          # low | balanced | high (default: balanced)
# SIFT_REVIEW_MODEL=anthropic/claude-haiku-4-5   # separate model for critic/holistic passes
# SIFT_REVIEW_MODEL_KEY=sk-ant-...
# SIFT_REVIEW_MODEL_BASE_URL=
# SIFT_CAPABILITY_OVERRIDE=           # JSON: {"context_window":32768,"supports_function_calling":true,"supports_reasoning":false}
# SIFT_AGENTIC_MAX_STEPS=4
```

---

## Step 3 — `src/intelligence/effort.py` (NEW, ~50 lines)

```python
"""Effort levels and per-level execution plans for the review pipeline."""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src import config

logger = logging.getLogger(__name__)

_VALID_EFFORTS = ("low", "balanced", "high")


class EffortLevel(str, Enum):
    LOW = "low"
    BALANCED = "balanced"
    HIGH = "high"


@dataclass(frozen=True)
class EffortPlan:
    level: EffortLevel
    run_critic: bool           # Phase 2+
    critic_per_finding: bool   # True=per-finding (high), False=batched (balanced)
    run_holistic: bool         # Phase 3+
    enable_agentic: bool       # Phase 4+; also gated by ModelCapability
    context_depth: int         # 0=window-only, 1=+semantic/callers, 2=+callees
    request_reasoning: bool    # ask model for extended thinking if supported


_PLANS: dict[EffortLevel, EffortPlan] = {
    EffortLevel.LOW: EffortPlan(
        level=EffortLevel.LOW,
        run_critic=False,
        critic_per_finding=False,
        run_holistic=False,
        enable_agentic=False,
        context_depth=0,
        request_reasoning=False,
    ),
    EffortLevel.BALANCED: EffortPlan(
        level=EffortLevel.BALANCED,
        run_critic=True,
        critic_per_finding=False,
        run_holistic=True,
        enable_agentic=False,
        context_depth=1,
        request_reasoning=True,
    ),
    EffortLevel.HIGH: EffortPlan(
        level=EffortLevel.HIGH,
        run_critic=True,
        critic_per_finding=True,
        run_holistic=True,
        enable_agentic=True,
        context_depth=2,
        request_reasoning=True,
    ),
}


def plan_for(level: EffortLevel) -> EffortPlan:
    return _PLANS[level]


def resolve_effort() -> EffortLevel:
    """Read SIFT_REVIEW_EFFORT from config; fall back to BALANCED on invalid input."""
    raw = (config.SIFT_REVIEW_EFFORT or "balanced").strip().lower()
    try:
        return EffortLevel(raw)
    except ValueError:
        logger.warning(
            "SIFT_REVIEW_EFFORT=%r is invalid; using 'balanced'. Valid values: %s",
            raw, list(_VALID_EFFORTS),
        )
        return EffortLevel.BALANCED


def current_plan() -> EffortPlan:
    """Convenience: resolve effort and return its plan."""
    return plan_for(resolve_effort())
```

---

## Step 4 — `src/intelligence/capability.py` (NEW, ~80 lines)

```python
"""Model capability detection for adapting calls and context budgets."""
import json
import logging
from dataclasses import dataclass
from typing import Optional

import litellm

from src import config

logger = logging.getLogger(__name__)

_CACHE: dict[str, "ModelCapability"] = {}

_CONSERVATIVE_DEFAULTS = dict(
    context_window=8192,
    max_output_tokens=2048,
    supports_function_calling=False,
    supports_reasoning=False,
)

# Known reasoning models that LiteLLM may not flag automatically.
_REASONING_MODEL_SUBSTRINGS = (
    "o1", "o3", "thinking", "reasoning", "r1",
    "claude-opus-4", "claude-3-7",
)


@dataclass(frozen=True)
class ModelCapability:
    context_window: int
    max_output_tokens: int
    supports_function_calling: bool
    supports_reasoning: bool


def _from_override(raw: Optional[str]) -> Optional[ModelCapability]:
    if not raw:
        return None
    try:
        d = json.loads(raw)
        return ModelCapability(
            context_window=int(d.get("context_window", _CONSERVATIVE_DEFAULTS["context_window"])),
            max_output_tokens=int(d.get("max_output_tokens", _CONSERVATIVE_DEFAULTS["max_output_tokens"])),
            supports_function_calling=bool(d.get("supports_function_calling", False)),
            supports_reasoning=bool(d.get("supports_reasoning", False)),
        )
    except Exception as exc:
        logger.warning("SIFT_CAPABILITY_OVERRIDE is not valid JSON (%s); ignoring.", exc)
        return None


def _detect_reasoning(model: str) -> bool:
    m = model.lower()
    return any(s in m for s in _REASONING_MODEL_SUBSTRINGS)


def detect(model: str) -> ModelCapability:
    """Return capability for a model string. Results are cached per model string."""
    if model in _CACHE:
        return _CACHE[model]

    # 1. SIFT_CAPABILITY_OVERRIDE wins if present.
    override = _from_override(config.SIFT_CAPABILITY_OVERRIDE)
    if override is not None:
        _CACHE[model] = override
        return override

    # 2. LiteLLM introspection — swallow all errors, unknown models are fine.
    ctx = _CONSERVATIVE_DEFAULTS["context_window"]
    max_out = _CONSERVATIVE_DEFAULTS["max_output_tokens"]
    fn_calling = False
    try:
        info = litellm.get_model_info(model)
        ctx = int(info.get("max_input_tokens") or info.get("max_tokens") or ctx)
        max_out = int(info.get("max_output_tokens") or max_out)
    except Exception:
        pass
    try:
        fn_calling = bool(litellm.supports_function_calling(model=model))
    except Exception:
        pass

    cap = ModelCapability(
        context_window=ctx,
        max_output_tokens=max_out,
        supports_function_calling=fn_calling,
        supports_reasoning=_detect_reasoning(model),
    )
    _CACHE[model] = cap
    logger.debug(
        "[capability] model=%s ctx=%d max_out=%d fn_calling=%s reasoning=%s",
        model, cap.context_window, cap.max_output_tokens,
        cap.supports_function_calling, cap.supports_reasoning,
    )
    return cap


def primary_capability() -> ModelCapability:
    """Capability for the primary LLM_MODEL."""
    return detect(config.LLM_MODEL)


def review_capability() -> ModelCapability:
    """Capability for SIFT_REVIEW_MODEL (critic/holistic); falls back to primary."""
    return detect(config.SIFT_REVIEW_MODEL or config.LLM_MODEL)
```

---

## Step 5 — `src/intelligence/schema.py` (NEW, ~120 lines)

This is the single source of truth for what a finding is. The `to_comment_dict` method
is the adapter that lets the rest of `review_engine.py` remain untouched.

```python
"""Core data types for review findings."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Impact(str, Enum):
    CRITICAL = "critical"   # data loss, crash, auth bypass
    HIGH = "high"           # correctness bug, wrong logic
    MEDIUM = "medium"       # perf, resource leak
    LOW = "low"             # maintainability, unclear code
    TRIVIAL = "trivial"     # pure style / nit


class Certainty(str, Enum):
    CONFIRMED = "confirmed"     # directly verifiable from code
    LIKELY = "likely"           # high confidence, minor assumption
    SPECULATIVE = "speculative" # cannot confirm without more context


# Valid category values (open string to allow future extension without Enum churn)
CATEGORIES = frozenset({
    "correctness", "security", "perf", "resource",
    "design", "maintainability", "style",
})


@dataclass
class Finding:
    path: str
    line: int
    title: str
    body: str
    impact: Impact
    certainty: Certainty
    category: str              # one of CATEGORIES
    origin: str                # llm | semgrep | codeql | linter | holistic
    fix: Optional[str] = None
    post_inline: bool = True

    def severity(self) -> str:
        """Derive the legacy 5-tier severity label from impact × certainty."""
        return derive_severity(self.impact, self.certainty, self.category)

    def to_comment_dict(self) -> dict:
        """Adapter: produce the {path, line, body, post_inline} dict review_engine expects."""
        return {
            "path": self.path,
            "line": self.line,
            "body": self.body,
            "post_inline": self.post_inline,
        }


def derive_severity(impact: Impact, certainty: Certainty, category: str = "") -> str:
    """Map impact × certainty → bug/security/warning/suggestion/informational.

    Rules (evaluated top-to-bottom, first match wins):
    1. security category + impact >= HIGH                → security
    2. impact CRITICAL                                   → bug
    3. impact HIGH + certainty != SPECULATIVE            → bug
    4. impact HIGH + certainty == SPECULATIVE            → warning
    5. impact MEDIUM                                     → warning
    6. impact LOW                                        → suggestion
    7. impact TRIVIAL OR certainty == SPECULATIVE        → informational
    """
    if category == "security" and impact in (Impact.CRITICAL, Impact.HIGH):
        return "security"
    if impact == Impact.CRITICAL:
        return "bug"
    if impact == Impact.HIGH:
        return "bug" if certainty != Certainty.SPECULATIVE else "warning"
    if impact == Impact.MEDIUM:
        return "warning"
    if impact == Impact.LOW:
        return "suggestion"
    return "informational"


# ── Legacy JSON → Finding adapter ──────────────────────────────────────────
# Maps the old {severity, confidence} the LLM emits today onto Impact/Certainty
# so Phase-1 output is identical to pre-refactor.

_OLD_SEVERITY_TO_IMPACT: dict[str, Impact] = {
    "bug":           Impact.HIGH,
    "security":      Impact.HIGH,
    "warning":       Impact.MEDIUM,
    "suggestion":    Impact.LOW,
    "informational": Impact.LOW,
}
_OLD_SEVERITY_TO_CATEGORY: dict[str, str] = {
    "bug":           "correctness",
    "security":      "security",
    "warning":       "correctness",
    "suggestion":    "maintainability",
    "informational": "maintainability",
}


def confidence_to_certainty(confidence: int) -> Certainty:
    if confidence >= 8:
        return Certainty.CONFIRMED
    if confidence >= 7:
        return Certainty.LIKELY
    return Certainty.SPECULATIVE


def from_legacy_item(item: dict, path: str, body: str) -> Finding:
    """Build a Finding from the existing LLM JSON output and already-formatted body."""
    old_sev = (item.get("severity") or "suggestion").lower()
    try:
        confidence = int(item.get("confidence", 7))
    except (TypeError, ValueError):
        confidence = 7

    return Finding(
        path=path,
        line=int(item["line"]),
        title=(item.get("title") or "").strip() or "Issue",
        body=body,
        impact=_OLD_SEVERITY_TO_IMPACT.get(old_sev, Impact.LOW),
        certainty=confidence_to_certainty(confidence),
        category=_OLD_SEVERITY_TO_CATEGORY.get(old_sev, "maintainability"),
        origin="llm",
        fix=(item.get("fix") or None),
        post_inline=True,
    )
```

---

## Step 6 — `src/intelligence/prompts.py` (NEW, ~30 lines)

Move the two prompt constants here verbatim. Add a trivial `render` helper for later
phases (no-op templating in Phase 1).

```python
"""Centralized prompt templates for the review pipeline."""
from string import Template
from typing import Any

# Imported and used by passes/candidates.py
REVIEW_FILE_SYSTEM = """You are a code reviewer focused on correctness. Your job is to find real bugs and issues.
...(exact existing text from llm_client.py lines 19-57)...
"""

SUMMARIZE_SYSTEM = """You are reviewing aggregated inline PR review findings. ..."""


def render(template: str, **kwargs: Any) -> str:
    """Simple $key substitution for prompt templates. No-op when no kwargs."""
    if not kwargs:
        return template
    return Template(template).safe_substitute(**kwargs)
```

Keep the actual text byte-for-byte identical to what is in `llm_client.py` today so the
LLM sees the same system prompt during Phase 1.

---

## Step 7 — Slim `src/intelligence/llm_client.py`

**What to remove:**
- `SYSTEM_PROMPT` constant (lines 15–17) — only used by `review()`.
- `REVIEW_FILE_SYSTEM` and `SUMMARIZE_SYSTEM` constants — now live in `prompts.py`;
  replace with `from src.intelligence.prompts import REVIEW_FILE_SYSTEM, SUMMARIZE_SYSTEM`.
- `review()` function at the bottom (lines 843–857) — the legacy whole-diff call. Confirm
  it is only referenced within this file by grepping — it is not imported anywhere else.

**What to change in `_call_llm`:**
Add optional `model`, `api_base`, `api_key` kwargs so later passes can target
`SIFT_REVIEW_MODEL` without duplicating the call boilerplate:

```python
async def _call_llm(
    system: str,
    user_content: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    response = await acompletion(
        model=model or config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        api_base=api_base or config.LLM_API_BASE or None,
        api_key=api_key or None,
        timeout=120.0,
    )
    return (response.choices[0].message.content or "").strip()
```

Everything else — `_extract_json_array`, `_parse_review_file_response`, all badge/
formatting helpers, `extract_comment_severity_and_title`, `_build_structured_summary`,
`summarize_review`, `review_file`, `_annotate_diff_with_line_numbers`, and all `_format_*`
helpers — stays **untouched** in Phase 1. The `review_file` function is still exported
and internally used by `passes/candidates.py` (see Step 8).

---

## Step 8 — `src/intelligence/passes/` (NEW)

### `src/intelligence/passes/__init__.py`
Empty file.

### `src/intelligence/passes/candidates.py` (~60 lines)

This is a thin adapter that calls the **existing** `review_file` function from
`llm_client.py` and converts its output to `list[Finding]`.

```python
"""Pass 1: per-file candidate generation.

Phase 1 behaviour: delegates directly to the existing review_file() call in
llm_client.py and wraps the output in Finding objects using from_legacy_item().
"""
import logging
from typing import Any, Optional

from src.intelligence import llm_client
from src.intelligence.schema import Finding, from_legacy_item, _format_structured_comment_body

logger = logging.getLogger(__name__)


async def generate_candidates(
    file_diff: str,
    path: str,
    pr_context: Optional[dict[str, Any]] = None,
) -> list[Finding]:
    """Call the LLM for a single file and return findings as Finding objects.

    In Phase 1 this is a direct pass-through to review_file().
    """
    raw_comments = await llm_client.review_file(file_diff, path, pr_context)
    # review_file already returns {line, body, post_inline} dicts with formatted bodies.
    # We reconstruct a minimal Finding from each. The body is already formatted with
    # badges, so impact/certainty are derived conservatively and body is preserved as-is.
    findings: list[Finding] = []
    for c in raw_comments:
        findings.append(Finding(
            path=path,
            line=c["line"],
            title="",        # already embedded in body badge; title is extracted later
            body=c["body"],
            impact=_infer_impact_from_body(c["body"]),
            certainty=_infer_certainty_from_body(c["body"]),
            category="correctness",
            origin="llm",
            fix=None,
            post_inline=c.get("post_inline", True),
        ))
    return findings


def _infer_impact_from_body(body: str) -> Any:
    """Back-infer impact from the badge already present in the body."""
    from src.intelligence.schema import Impact
    low = body.lower()
    if "bug" in low or "security" in low:
        return Impact.HIGH
    if "warning" in low:
        return Impact.MEDIUM
    return Impact.LOW


def _infer_certainty_from_body(body: str) -> Any:
    from src.intelligence.schema import Certainty
    if "informational" in body.lower():
        return Certainty.SPECULATIVE
    return Certainty.LIKELY
```

### `src/intelligence/passes/pipeline.py` (~70 lines)

```python
"""Review pipeline orchestrator.

Phase 1: single-pass (candidates only).  Critic, holistic, and severity gate are
stubs that pass data through unchanged, ready for Phases 2 and 3 to fill in.
"""
import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.intelligence.capability import ModelCapability
from src.intelligence.effort import EffortPlan
from src.intelligence.schema import Finding
from src.intelligence.passes.candidates import generate_candidates

logger = logging.getLogger(__name__)


@dataclass
class FileReviewInput:
    path: str
    file_diff: str
    pr_context: dict[str, Any]  # exactly what _process_file builds today


@dataclass
class PRMeta:
    title: str
    body: str
    # Populated in Phase 3:
    import_graph: Optional[dict] = None
    mod_funcs_by_path: Optional[dict] = None


async def run_pipeline(
    files: list[FileReviewInput],
    pr_meta: PRMeta,
    plan: EffortPlan,
    cap: ModelCapability,
) -> list[Finding]:
    """Run all enabled passes and return the final list of findings.

    Phase 1: only pass 1 (candidates) runs.  Subsequent passes are no-ops.
    """
    all_findings: list[Finding] = []

    for f in files:
        candidates = await generate_candidates(f.file_diff, f.path, f.pr_context)
        logger.debug("[pipeline] %s: %d candidate(s)", f.path, len(candidates))
        all_findings.extend(candidates)

    # Phase 2 will insert: all_findings = await critique(all_findings, plan, cap)
    # Phase 3 will insert: all_findings += await review_holistic(digest, plan, cap)
    # Phase 2 will insert: all_findings = apply_severity_gate(all_findings, plan)

    return all_findings
```

---

## Step 9 — Wire `src/core/review_engine.py`

**Only two changes** inside `_process_file` and the gather below it.

### 9a. Add imports at the top of the file

```python
from src.intelligence.effort import current_plan
from src.intelligence.capability import primary_capability
from src.intelligence.passes.pipeline import FileReviewInput, PRMeta, run_pipeline
```

Remove:
```python
from src.intelligence.llm_client import review_file, summarize_review
```
Replace with:
```python
from src.intelligence.llm_client import summarize_review
```
(`review_file` is now called only from `passes/candidates.py`.)

### 9b. Before the `path_diff_lists` loop, resolve plan + capability once

After the vector-upsert-queue setup (around line 592), add:

```python
_effort_plan = current_plan()
_model_cap = primary_capability()
logger.debug("[pipeline] effort=%s ctx_window=%d fn_calling=%s",
             _effort_plan.level, _model_cap.context_window,
             _model_cap.supports_function_calling)
```

### 9c. Replace the `_process_file` body (lines 814–832) with a pipeline call

**Remove** (the inner block under `async with _review_sem:`):

```python
file_comments: List[Dict[str, Any]] = []
async with _review_sem:
    try:
        comments = await review_file(file_diff, path0, file_pr_context)
        for c in comments:
            for path, _ in path_diff_list:
                file_comments.append({
                    "path": path,
                    "line": c["line"],
                    "body": c["body"],
                    "post_inline": c.get("post_inline", True),
                })
    except Exception as e:
        logger.warning("review_file failed for %s: %s", path0, e)
    if config.SIFT_LLM_REQUEST_DELAY > 0:
        await asyncio.sleep(config.SIFT_LLM_REQUEST_DELAY)
return file_comments
```

**Replace with:**

```python
file_input = FileReviewInput(
    path=path0,
    file_diff=file_diff,
    pr_context=file_pr_context,
)
file_comments: List[Dict[str, Any]] = []
async with _review_sem:
    try:
        findings = await run_pipeline(
            [file_input],
            PRMeta(
                title=pr_context.get("title") or "" if pr_context else "",
                body=pr_context.get("body") or "" if pr_context else "",
            ),
            _effort_plan,
            _model_cap,
        )
        for finding in findings:
            for path, _ in path_diff_list:
                file_comments.append({
                    "path": path,
                    "line": finding.line,
                    "body": finding.body,
                    "post_inline": finding.post_inline,
                })
    except Exception as e:
        logger.warning("review_file failed for %s: %s", path0, e)
    if config.SIFT_LLM_REQUEST_DELAY > 0:
        await asyncio.sleep(config.SIFT_LLM_REQUEST_DELAY)
return file_comments
```

Nothing downstream of the return changes — `_merge_comments_by_line`, `summarize_review`,
`create_pull_request_review` and `create_comment` all see the same dict shape they always did.

---

## Step 10 — Eval harness `eval/`

```
eval/
  cases/
    case_001_null_deref.py.diff
    case_001.json
    case_002_sql_injection.py.diff
    case_002.json
    case_003_breaking_signature.py.diff
    case_003.json
    case_004_resource_leak.py.diff
    case_004.json
    case_005_false_positive_bait.py.diff
    case_005.json
  schema.py
  run_eval.py
  README.md
```

### Golden case format (`eval/cases/case_NNN.json`)

```json
{
  "id": "001_null_deref",
  "description": "Function returns None and caller dereferences without check",
  "path": "app/user.py",
  "diff_file": "case_001_null_deref.py.diff",
  "expected": [
    {
      "line_range": [12, 14],
      "category": "correctness",
      "min_impact": "high",
      "note": "None dereference on return value of get_user()"
    }
  ],
  "false_positives": []
}
```

- `line_range` — match if the finding's `line` falls within this range (±2 is too loose;
  explicit range is more honest).
- `min_impact` — findings at this impact or above count as a hit.
- `false_positives` — findings at these lines are *noise* regardless of impact.

### Diff files (`eval/cases/*.py.diff`)

Hand-crafted minimal unified diffs. Example for case 001:

```diff
--- a/app/user.py
+++ b/app/user.py
@@ -8,6 +8,10 @@ def get_user(user_id: int):
     return None

+def process_request(user_id: int):
+    user = get_user(user_id)
+    print(user.name)   # potential None dereference
+    return user.email
```

### `eval/schema.py`

```python
from dataclasses import dataclass, field
from typing import Optional
import json, pathlib

@dataclass
class ExpectedFinding:
    line_range: tuple[int, int]
    category: str
    min_impact: str
    note: str = ""

@dataclass
class GoldenCase:
    id: str
    description: str
    path: str
    diff_text: str
    expected: list[ExpectedFinding]
    false_positive_lines: list[int] = field(default_factory=list)

    @classmethod
    def load(cls, json_path: pathlib.Path) -> "GoldenCase":
        d = json.loads(json_path.read_text())
        diff_file = json_path.parent / d["diff_file"]
        expected = [ExpectedFinding(**e) for e in d["expected"]]
        return cls(
            id=d["id"], description=d["description"],
            path=d["path"], diff_text=diff_file.read_text(),
            expected=expected,
            false_positive_lines=[fp["line"] for fp in d.get("false_positives", [])],
        )
```

### `eval/run_eval.py`

```python
"""Offline golden-set scorer.

Usage:
    python -m eval.run_eval --model ollama/llama3.2 --effort low
    python -m eval.run_eval --model anthropic/claude-opus-4-8 --effort balanced
"""
import argparse, asyncio, os, sys, pathlib, json
from eval.schema import GoldenCase
from src import config
from src.intelligence.effort import EffortLevel, plan_for
from src.intelligence.capability import detect
from src.intelligence.passes.pipeline import FileReviewInput, PRMeta, run_pipeline
from src.intelligence.schema import Impact

_IMPACT_RANK = {i.value: n for n, i in enumerate(Impact)}
CASES_DIR = pathlib.Path(__file__).parent / "cases"


def _is_hit(finding, expected, min_impact_rank: int) -> bool:
    lo, hi = expected.line_range
    return (
        lo <= finding.line <= hi
        and finding.category == expected.category
        and _IMPACT_RANK.get(finding.impact.value, 9) <= min_impact_rank
    )


async def score_case(case: GoldenCase, plan, cap) -> dict:
    inp = FileReviewInput(path=case.path, file_diff=case.diff_text,
                          pr_context={"title": case.description, "body": ""})
    findings = await run_pipeline([inp], PRMeta(title=case.description, body=""), plan, cap)

    hits, misses, noise = 0, 0, 0
    matched_findings = set()
    for exp in case.expected:
        min_rank = _IMPACT_RANK.get(exp.min_impact, 9)
        matched = [f for f in findings if _is_hit(f, exp, min_rank)]
        if matched:
            hits += 1
            matched_findings.update(id(f) for f in matched)
        else:
            misses += 1

    for f in findings:
        if id(f) not in matched_findings and f.line not in case.false_positive_lines:
            noise += 1

    return {"case": case.id, "hits": hits, "misses": misses,
            "noise": noise, "findings": len(findings)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "ollama/llama3.2"))
    parser.add_argument("--effort", default="balanced", choices=["low", "balanced", "high"])
    parser.add_argument("--case", default=None, help="Run a single case by ID")
    args = parser.parse_args()

    # Inject model into config for this run.
    config.LLM_MODEL = args.model
    plan = plan_for(EffortLevel(args.effort))
    cap = detect(args.model)

    cases = [GoldenCase.load(p) for p in sorted(CASES_DIR.glob("*.json"))]
    if args.case:
        cases = [c for c in cases if c.id == args.case]
        if not cases:
            print(f"No case found with id={args.case!r}"); sys.exit(1)

    results = await asyncio.gather(*[score_case(c, plan, cap) for c in cases])

    total_expected = sum(r["hits"] + r["misses"] for r in results)
    total_hits = sum(r["hits"] for r in results)
    total_noise = sum(r["noise"] for r in results)
    total_findings = sum(r["findings"] for r in results)

    precision = total_hits / total_findings if total_findings else 0.0
    recall = total_hits / total_expected if total_expected else 0.0
    noise_rate = total_noise / total_findings if total_findings else 0.0

    print(f"\nModel: {args.model}  Effort: {args.effort}")
    print(f"Precision: {precision:.0%}  Recall: {recall:.0%}  Noise-rate: {noise_rate:.0%}")
    print(f"  ({total_hits}/{total_expected} expected hits, {total_noise} noise findings)\n")
    for r in results:
        status = "✓" if r["misses"] == 0 else "✗"
        print(f"  {status} {r['case']}: hits={r['hits']} misses={r['misses']} noise={r['noise']}")

asyncio.run(main())
```

---

## Step 11 — Tests `tests/`

### `tests/test_schema.py`
- `test_derive_severity_security_high` — `(HIGH, CONFIRMED, "security")` → `"security"`.
- `test_derive_severity_critical_speculative` — `(CRITICAL, SPECULATIVE, "correctness")` → `"bug"` (critical overrides certainty).
- `test_derive_severity_high_speculative` — `(HIGH, SPECULATIVE, "correctness")` → `"warning"`.
- `test_derive_severity_trivial` — any certainty → `"informational"`.
- `test_legacy_mapping_round_trip` — for each old severity value, `from_legacy_item` + `finding.severity()` should return the same legacy label (modulo the security special case).
- `test_to_comment_dict` — `Finding.to_comment_dict()` has exactly `path`, `line`, `body`, `post_inline` keys.

### `tests/test_effort.py`
- `test_plan_for_low` — `run_critic=False`, `run_holistic=False`, `context_depth=0`.
- `test_plan_for_balanced` — `run_critic=True`, `critic_per_finding=False`, `run_holistic=True`.
- `test_plan_for_high` — `critic_per_finding=True`, `enable_agentic=True`, `context_depth=2`.
- `test_resolve_effort_valid` — `"high"` → `EffortLevel.HIGH`.
- `test_resolve_effort_invalid` — bad string → `EffortLevel.BALANCED` (no crash).

### `tests/test_capability.py`
- `test_conservative_fallback` — unknown model string → `context_window=8192`, `supports_function_calling=False`.
- `test_override_wins` — valid JSON in `SIFT_CAPABILITY_OVERRIDE` overrides LiteLLM.
- `test_invalid_override_does_not_crash` — garbage JSON → falls back to LiteLLM/defaults.
- `test_reasoning_detection` — model name containing `"claude-opus-4"` → `supports_reasoning=True`.
- `test_caching` — `detect()` called twice with same model string hits cache (mock `litellm.get_model_info` called only once).

### `tests/test_pipeline_phase1.py`
Behavioral contract test.

```python
async def test_pipeline_matches_old_review_file(monkeypatch):
    """run_pipeline with LOW effort must produce the same bodies as review_file did."""
    diff = "<a minimal test diff>"
    path = "app/test.py"
    pr_ctx = {"title": "test", "body": ""}

    # Capture what the old path would have returned.
    old_comments = await review_file(diff, path, pr_ctx)

    # Run through the new pipeline.
    plan = plan_for(EffortLevel.LOW)
    cap = ModelCapability(8192, 2048, False, False)
    findings = await run_pipeline(
        [FileReviewInput(path, diff, pr_ctx)],
        PRMeta("test", ""), plan, cap,
    )

    assert [f.to_comment_dict() for f in findings] == [
        {"path": path, "line": c["line"], "body": c["body"],
         "post_inline": c.get("post_inline", True)}
        for c in old_comments
    ]
```

Mock the LLM call so the test is deterministic (return a fixed JSON payload).

---

## Build order (within Phase 1)

This ordering avoids forward-reference import issues:

1. `config.py` additions + `.env.example`
2. `schema.py` (no src imports)
3. `effort.py` (imports config)
4. `capability.py` (imports config, litellm)
5. `prompts.py` (no src imports)
6. `llm_client.py` slimming (imports prompts)
7. `passes/__init__.py`, `passes/candidates.py` (imports llm_client, schema)
8. `passes/pipeline.py` (imports candidates, schema, effort, capability)
9. `review_engine.py` wiring (imports pipeline, effort, capability)
10. `eval/schema.py`, `eval/run_eval.py`, `eval/cases/*.json + *.diff`
11. `tests/`

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Legacy-mapping table produces different severity labels → changed badge text | test_legacy_mapping_round_trip catches any divergence |
| `review_file` still imported via candidates.py — circular if restructured | candidates.py imports `llm_client.review_file` directly; no cycle |
| `review()` removal breaks something not caught by grep | Confirmed: only `llm_client.py` line 855 calls it internally; no external import |
| `_call_llm` signature change breaks `summarize_review` (which also calls it) | `model/api_base/api_key` are Optional with defaults = current config values; fully backward-compatible |
| LiteLLM `get_model_info` raises for self-hosted models | All LiteLLM calls in `capability.py` are in try/except; conservative defaults are returned |
