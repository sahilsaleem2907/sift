"""Context-window budgeting: diff chunking, block trimming, overflow handling."""
import re
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest

from sift import config
from sift.intelligence.ast.diff_ast import build_diff_ast, filter_ast_to_range
from sift.intelligence.ast.function_extract import FunctionChunk
from sift.intelligence.budget import (
    Block,
    cap_tool_result,
    count_tokens,
    drop_tool_turns,
    fit_blocks,
    input_budget,
    is_new_file_diff,
    overflow_shrink_factor,
    split_diff,
)
from sift.intelligence.capability import ModelCapability
from sift.intelligence.effort import EffortLevel, plan_for
from sift.intelligence.llm_client import _annotate_diff_with_line_numbers, _format_ast_diff, review_file
from sift.intelligence.passes.pipeline import FileReviewInput, PRMeta, run_pipeline_per_file
from sift.intelligence.rate_limit import permanent_failure_reason
from sift.intelligence.schema import Certainty, Finding, Impact

OLLAMA_OVERFLOW = (
    "litellm.BadRequestError: OpenAIException - The prompt is too long: 147021, "
    "model maximum context length: 131072 (ref: d2845610)"
)


def _overflow_error() -> litellm.BadRequestError:
    return litellm.BadRequestError(
        message="OpenAIException - The prompt is too long: 147021, model maximum context length: 131072",
        model="openai/x",
        llm_provider="openai",
    )


def _source(n_funcs: int = 40) -> str:
    """A synthetic Python file: n_funcs functions of 8 lines, separated by blank lines."""
    parts = []
    for i in range(n_funcs):
        parts.append(
            f"def handler_{i}(request, payload):\n"
            f"    value = payload.get('field_{i}')\n"
            f"    if value is None:\n"
            f"        return {{'status': 'missing', 'index': {i}}}\n"
            f"    result = transform(value, scale={i})\n"
            f"    log('handled request {i}', result)\n"
            f"    return {{'status': 'ok', 'result': result}}\n"
        )
    return "\n".join(parts)


def _new_file_diff(path: str, src: str) -> str:
    lines = src.splitlines()
    return (
        f"diff --git a/{path} b/{path}\nnew file mode 100644\n--- /dev/null\n+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n" + "\n".join("+" + ln for ln in lines)
    )


def _func_ranges(src: str) -> list[tuple[int, int]]:
    lines = src.splitlines()
    starts = [i + 1 for i, ln in enumerate(lines) if ln.startswith("def ")]
    return [(s, s + 6) for s in starts]


def _marks(diff: str, path: str = "x.py") -> list[str]:
    return re.findall(r"\[L(\d+)\]", _annotate_diff_with_line_numbers(diff, path))


# ---- split_diff ----


def test_split_new_file_single_hunk_preserves_line_numbers_and_cuts_at_function_ends():
    src = _source()
    diff = _new_file_diff("x.py", src)
    ranges = _func_ranges(src)
    chunks = split_diff(diff, count_tokens(diff) // 4, func_ranges=ranges)

    assert len(chunks) >= 4
    # Every chunk keeps the file header; line markers match the unsplit diff exactly once.
    assert all(c.text.startswith("diff --git a/x.py b/x.py") for c in chunks)
    assert [m for c in chunks for m in _marks(c.text)] == _marks(diff)
    # Chunk ranges are contiguous and non-overlapping.
    for a, b in zip(chunks, chunks[1:]):
        assert b.new_start == a.new_end + 1
    # Cuts land on function ends (or the blank line right after one).
    func_ends = {e for _s, e in ranges}
    assert all(c.new_end in func_ends or c.new_end - 1 in func_ends for c in chunks[:-1])


def test_split_multi_hunk_diff_packs_whole_hunks():
    hunks = []
    for k in range(6):
        start = 1 + k * 100
        body = "\n".join(f"+    added_{k}_{j} = compute({j})" for j in range(30))
        hunks.append(f"@@ -{start},2 +{start},32 @@ def f{k}():\n context_{k}\n{body}\n context_end_{k}")
    diff = "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n" + "\n".join(hunks)
    per_hunk = count_tokens(hunks[0])
    chunks = split_diff(diff, int(per_hunk * 2.5))

    assert 2 <= len(chunks) <= 3
    # No hunk was split: every original header survives verbatim in some chunk.
    for k in range(6):
        assert sum(f"+{1 + k * 100},32 @@" in c.text for c in chunks) == 1
    assert [m for c in chunks for m in _marks(c.text, "m.py")] == _marks(diff, "m.py")


def test_split_small_diff_is_one_chunk():
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,3 @@\n x = 1\n+y = 2\n z = 3"
    chunks = split_diff(diff, 10_000)
    assert len(chunks) == 1
    assert _marks(chunks[0].text, "a.py") == ["2"]


def test_is_new_file_diff():
    assert is_new_file_diff(_new_file_diff("n.py", "a = 1\nb = 2"))
    assert is_new_file_diff("@@ -0,0 +1,2 @@\n+a\n+b")
    assert not is_new_file_diff("@@ -1,2 +1,3 @@\n a\n+b\n c")
    assert not is_new_file_diff("")


# ---- blocks / budget ----


def test_fit_blocks_applies_reductions_in_rank_order_and_keeps_unreducible():
    big = "word " * 2000
    blocks = [
        Block("keep", "important " * 50),
        Block("file", big, [(5, "file:reduced", "ranges " * 100), (6, "file", "")]),
        Block("snippets", big, [(0, "snippets", "")]),
        Block("ast", big, [(1, "ast", "")]),
    ]
    kept, applied = fit_blocks(blocks, budget=3000)
    assert applied == ["snippets", "ast"]
    assert [b.name for b in kept] == ["keep", "file"]

    kept, applied = fit_blocks(blocks, budget=400)
    assert applied == ["snippets", "ast", "file:reduced"]
    assert [b.name for b in kept] == ["keep", "file"]

    kept, applied = fit_blocks(blocks, budget=0)
    assert applied == ["snippets", "ast", "file:reduced", "file"]
    assert [b.name for b in kept] == ["keep"]


def test_input_budget_reserves_output_and_headroom():
    assert input_budget(131072, reasoning=True) == int(131072 * config.SIFT_CONTEXT_SAFETY_FACTOR) - 16000
    assert input_budget(131072, reasoning=False) == int(131072 * config.SIFT_CONTEXT_SAFETY_FACTOR) - 4096
    assert input_budget(4096, reasoning=True) == 2048  # floor


def test_overflow_shrink_factor_parses_provider_messages():
    assert overflow_shrink_factor(RuntimeError(OLLAMA_OVERFLOW)) == pytest.approx(131072 / 147021 * 0.9)
    assert overflow_shrink_factor(_overflow_error()) == pytest.approx(131072 / 147021 * 0.9)
    openai_msg = "This model's maximum context length is 8192 tokens. However, your messages resulted in 9000 tokens."
    assert overflow_shrink_factor(RuntimeError(openai_msg)) == pytest.approx(8192 / 9000 * 0.9)
    assert overflow_shrink_factor(RuntimeError("429 too many concurrent requests")) is None


def test_permanent_failure_reason_classifies_errors():
    assert permanent_failure_reason(_overflow_error()) == "prompt exceeds the model's context window"
    bad = litellm.BadRequestError(message="invalid tool schema", model="m", llm_provider="openai")
    assert permanent_failure_reason(bad) == "provider rejected the request (400)"
    rl = litellm.RateLimitError(message="slow down", llm_provider="openai", model="m")
    assert permanent_failure_reason(rl) is None
    assert permanent_failure_reason(RuntimeError("429 too many concurrent requests")) is None


def test_cap_tool_result_and_drop_tool_turns():
    assert cap_tool_result("short") == "short"
    huge = "token " * 20_000
    capped = cap_tool_result(huge, max_tokens=1000)
    assert count_tokens(capped) < 1100 and "tool result truncated" in capped

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "diff"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_file", "arguments": "{}"}}]},
        {"role": "tool", "content": huge},
        {"role": "assistant", "tool_calls": [{"function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "content": "small"},
    ]
    # Always pops the latest turn, then keeps popping while over budget.
    assert drop_tool_turns(messages, budget=100) == 2
    assert [m["role"] for m in messages] == ["system", "user"]


# ---- compact AST ----


def test_ast_for_new_file_is_outermost_named_nodes_only():
    src = _source(30)
    diff = _new_file_diff("x.py", src)
    ast = build_diff_ast("x.py", src, diff)
    text = _format_ast_diff(ast)
    # One node per top-level function, not one per token at every depth (was ~18x the source).
    assert len(ast["nodes"]) == 30
    assert all(n["type"] == "function_definition" for n in ast["nodes"])
    assert len(text) < 2 * len(src)

    scoped = filter_ast_to_range(ast, 1, 16)
    assert [n["start_line"] for n in scoped["nodes"]] == [1, 9]
    assert filter_ast_to_range(ast, 10_000, 10_001) is None


# ---- review_file ----


@pytest.mark.asyncio
async def test_review_file_reduces_blocks_in_order_and_keeps_diff():
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,2 +1,3 @@\n x = 1\n+y = 2\n z = 3"
    content = "\n".join(f"line_{i} = {i}" for i in range(400))
    snippet = MagicMock(path="b.py", start_line=1, end_line=2, func_name="g", chunk_text="filler " * 3000)
    ctx = {
        "title": "Add y",
        "body": "Why: " + "context " * 2000,
        "runtime_target": "Python 3.12",
        "file_context": {"path": "a.py", "ranges": [(1, 3)], "content": content},
        "semantic_before_after": "before/after " * 500,
        "similar_snippets": [snippet],
    }
    call = AsyncMock(return_value="[]")
    with mock.patch("sift.intelligence.llm_client._call_llm", new=call):
        await review_file(diff, "a.py", ctx, context_window=16000)
    user = call.await_args.args[1]
    # snippets → before/after → full file (downgraded to its changed ranges) went, in that order;
    # the PR body (a later reduction) was not needed.
    assert "Similar code elsewhere" not in user
    assert "Semantic before/after" not in user
    assert "Full file (read-only" not in user
    assert "Surrounding context (read-only" in user and "Lines 1–3:" in user
    assert "description truncated" not in user
    assert "Target runtime for this file: Python 3.12" in user
    assert "y = 2" in user and "[L2]" in user
    assert count_tokens(call.await_args.args[0] + user) <= input_budget(16000, False)

    # Without a window nothing is reduced (unchanged behavior for callers that don't pass one).
    with mock.patch("sift.intelligence.llm_client._call_llm", new=call):
        await review_file(diff, "a.py", ctx)
    user = call.await_args.args[1]
    assert "Similar code elsewhere" in user and "Full file (read-only" in user


@pytest.mark.asyncio
async def test_review_file_skips_duplicate_blocks_for_new_files():
    src = "a = 1\nb = 2\n"
    diff = _new_file_diff("n.py", src)
    ctx = {
        "file_context": {"path": "n.py", "ranges": [(1, 2)], "content": src},
        "semantic_before_after": "--- old: f (n.py) ---\n(new or no removed lines)",
    }
    call = AsyncMock(return_value="[]")
    with mock.patch("sift.intelligence.llm_client._call_llm", new=call):
        await review_file(diff, "n.py", ctx, context_window=131072)
    user = call.await_args.args[1]
    assert "Full file (read-only" not in user
    assert "Semantic before/after" not in user
    assert "+a = 1" in user


# ---- pipeline chunking ----


def _finding(line: int, path: str = "big.py") -> Finding:
    return Finding(
        path=path, line=line, title=f"issue {line}", body=f"issue at {line}",
        impact=Impact.HIGH, certainty=Certainty.LIKELY, category="correctness", origin="llm",
    )


def _mod_funcs(path: str, src: str) -> list[FunctionChunk]:
    lines = src.splitlines()
    return [
        FunctionChunk(path=path, name=f"handler_{k}", start_line=s, end_line=e,
                      text="\n".join(lines[s - 1:e]), content_hash=str(k))
        for k, (s, e) in enumerate(_func_ranges(src))
    ]


@pytest.mark.asyncio
async def test_pipeline_chunks_oversized_diff_with_scoped_context():
    src = _source(120)
    path = "big.py"
    diff = _new_file_diff(path, src)
    n_lines = len(src.splitlines())
    plan = plan_for(EffortLevel.LOW)
    cap = ModelCapability(8000, 4096, False, False)
    pr_ctx = {
        "title": "Add handlers",
        "semgrep_findings": [
            {"line": 3, "message": "early", "check_id": "r1", "severity": "WARNING"},
            {"line": n_lines - 2, "message": "late", "check_id": "r2", "severity": "WARNING"},
        ],
    }
    seen: list[tuple[str, dict]] = []

    async def _gen(chunk_diff, p, ctx, reasoning_effort=None, context_window=None):
        seen.append((chunk_diff, ctx))
        lines = [int(m) for m in _marks(chunk_diff, p)]
        # One finding inside the chunk, one on the first line of the file (only chunk 0 owns it).
        return [_finding(lines[len(lines) // 2]), _finding(1)]

    meta = PRMeta(title="t", body="", mod_funcs_by_path={path: _mod_funcs(path, src)})
    with (
        mock.patch("sift.intelligence.passes.pipeline.generate_candidates", new=_gen),
        mock.patch("sift.intelligence.passes.critic.critique", new=AsyncMock(side_effect=lambda c, *a, **k: c)),
    ):
        findings = await run_pipeline_per_file(FileReviewInput(path, diff, pr_ctx), "t", plan, cap, meta)

    assert len(seen) > 1
    budget = input_budget(min(cap.context_window, 8000), False)
    for chunk_diff, _ctx in seen:
        assert count_tokens(chunk_diff) < budget
    assert [m for chunk_diff, _ in seen for m in _marks(chunk_diff, path)] == _marks(diff, path)
    # Static findings are scoped: the early one only in the first chunk, the late one only in the last.
    assert [f["line"] for f in seen[0][1]["semgrep_findings"]] == [3]
    assert [f["line"] for f in seen[-1][1]["semgrep_findings"]] == [n_lines - 2]
    # Line 1 is kept once (chunk 0), the out-of-range copies from later chunks are dropped.
    assert sorted(f.line for f in findings).count(1) == 1
    assert len(findings) == len(seen) + 1


@pytest.mark.asyncio
async def test_pipeline_replans_once_on_provider_overflow():
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,2 @@\n x = 1\n+y = 2"
    plan = plan_for(EffortLevel.LOW)
    cap = ModelCapability(131072, 4096, False, False)
    windows: list[int] = []

    async def _gen(chunk_diff, p, ctx, reasoning_effort=None, context_window=None):
        windows.append(context_window)
        if len(windows) == 1:
            raise _overflow_error()
        return []

    with (
        mock.patch("sift.intelligence.passes.pipeline.generate_candidates", new=_gen),
        mock.patch("sift.intelligence.capability.review_capability", return_value=cap),
    ):
        await run_pipeline_per_file(FileReviewInput("a.py", diff, {}), "t", plan, cap, None)
    assert len(windows) == 2
    assert windows[1] == int(131072 * min(0.9, 131072 / 147021 * 0.9))

    async def _always(*a, **k):
        raise _overflow_error()

    with (
        mock.patch("sift.intelligence.passes.pipeline.generate_candidates", new=_always),
        mock.patch("sift.intelligence.capability.review_capability", return_value=cap),
        pytest.raises(litellm.BadRequestError),
    ):
        await run_pipeline_per_file(FileReviewInput("a.py", diff, {}), "t", plan, cap, None)


# ---- tool loops ----


@pytest.mark.asyncio
async def test_agentic_overflow_mid_loop_drops_turns_and_requests_final():
    from tests.test_agentic import _mock_emit_message, _mock_message, _mock_tool_call, _resp
    from sift.intelligence.passes.agentic import agentic_review

    plan = plan_for(EffortLevel.HIGH)
    cap = ModelCapability(128_000, 4096, True, False)
    file_input = FileReviewInput(
        path="app/a.py",
        file_diff="diff --git a/app/a.py\n+++ b/app/a.py\n@@ -1 +1,2 @@\n+x\n",
        pr_context={},
    )
    tool_step = _resp(_mock_message(tool_calls=[_mock_tool_call("tc1", "get_file", '{"path": "app/a.py"}')]))
    final = _resp(_mock_emit_message([{"line": 1, "severity": "bug", "title": "t", "body": "b", "confidence": 9}]))
    sent: list[int] = []

    async def _llm(**kwargs):
        sent.append(len(kwargs["messages"]))
        if len(sent) == 1:
            return tool_step
        if len(sent) == 2:
            raise _overflow_error()
        return final

    with mock.patch("sift.intelligence.passes.agentic.acompletion", new=_llm):
        findings = await agentic_review(file_input, plan, cap, {"app/a.py": "x=1\n"})

    assert [f.line for f in findings] == [1]
    # Step 2 carried the tool turn (4 messages); the final call went out without it.
    assert sent[1] == 4 and sent[2] <= 3
