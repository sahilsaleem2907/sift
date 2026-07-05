"""Tests for JSON extraction from reasoning-model output.

Regression coverage for the parser bug that shipped silently through PRs
#236–#238: the candidates prompt instructs the model to emit a
<reasoning>...</reasoning> block before the JSON array, and the stray brackets
in that prose (e.g. "[L10]") defeated the old first-bracket-to-last-bracket
parser, producing zero findings with no visible error.
"""
import pytest

from src.intelligence.llm_client import (
    _balanced_array_end,
    _build_structured_summary,
    _extract_json_array,
    _parse_review_file_response,
    _strip_thinking_blocks,
    parse_with_repair,
    summarize_review,
)


# The exact shape of the PR #238 model output that silently failed: a
# <reasoning> block containing "[L10]" / "[L25]" line refs, then the array.
PR_238_RAW = """<reasoning>
The PR introduces several issues in `src/extension.ts`:

1. **Line 10 [L10]**: A typo in the namespace `vscde` instead of `vscode` will
   cause a runtime error. This breaks the command registration.
2. **Line 10 [L10]**: Missing parentheses `()` in the `registerCommand` call.
3. **Line 11 [L11]**: The logic is inverted. `!vscode.workspace.workspaceFolders`
   then accesses `workspaceFolders[0]`, which will be undefined.
4. **Line 25 [L25]**: A hardcoded GitHub token is present.

The most critical issues are the typo, syntax error, and logic error.
</reasoning>

[
  {"line": 10, "severity": "bug", "title": "Typo in namespace", "body": "Use 'vscode' not 'vscde'.", "confidence": 10},
  {"line": 11, "severity": "bug", "title": "Inverted condition", "body": "Logic is inverted.", "confidence": 9},
  {"line": 25, "severity": "security", "title": "Hardcoded GitHub token", "body": "Remove the token.", "confidence": 10}
]"""


# ---- _strip_thinking_blocks ----

def test_strip_removes_reasoning_block():
    out = _strip_thinking_blocks("<reasoning>foo [L1] bar</reasoning>\n[{}]")
    assert "reasoning" not in out
    assert out.strip().startswith("[")


def test_strip_removes_think_block():
    out = _strip_thinking_blocks("<think>deciding...</think>[1,2]")
    assert out == "[1,2]"


def test_strip_removes_thinking_block_case_insensitive():
    out = _strip_thinking_blocks("<Thinking>x</Thinking>[1]")
    assert out == "[1]"


def test_strip_noop_when_no_blocks():
    assert _strip_thinking_blocks("[1,2,3]") == "[1,2,3]"


# ---- _balanced_array_end ----

def test_balanced_end_simple():
    s = "[1, 2, 3]"
    assert _balanced_array_end(s, 0) == len(s)


def test_balanced_end_nested():
    s = "[[1], [2, [3]]]"
    assert _balanced_array_end(s, 0) == len(s)


def test_balanced_end_ignores_brackets_in_strings():
    s = '["a]b", "c"]'
    assert _balanced_array_end(s, 0) == len(s)


def test_balanced_end_unterminated():
    assert _balanced_array_end("[1, 2", 0) == -1


# ---- _extract_json_array ----

def test_extract_pr238_reasoning_payload():
    arr = _extract_json_array(PR_238_RAW)
    assert arr is not None
    assert len(arr) == 3
    assert {f["line"] for f in arr} == {10, 11, 25}


def test_extract_with_think_block():
    arr = _extract_json_array('<think>hmm [a]</think>\n[{"line": 5}]')
    assert arr == [{"line": 5}]


def test_extract_markdown_fenced():
    arr = _extract_json_array('Here you go:\n```json\n[{"line": 1}]\n```')
    assert arr == [{"line": 1}]


def test_extract_skips_stray_prose_brackets():
    # A "[note]" that is valid-ish prose but not a JSON list must be skipped,
    # and the real array after it returned.
    raw = 'Findings [note: 3 issues] below:\n[{"line": 7}]'
    arr = _extract_json_array(raw)
    assert arr == [{"line": 7}]


def test_extract_returns_none_on_no_array():
    assert _extract_json_array("No issues found in this diff.") is None


def test_extract_returns_none_on_empty():
    assert _extract_json_array("") is None


def test_extract_empty_array():
    assert _extract_json_array("<reasoning>nothing</reasoning>\n[]") == []


# ---- full seam: _parse_review_file_response ----

def test_parse_review_full_seam_pr238():
    """The PR #238 payload run through the full candidates parser yields findings."""
    out = _parse_review_file_response(PR_238_RAW, "src/extension.ts")
    assert len(out) == 3
    lines = sorted(c["line"] for c in out)
    assert lines == [10, 11, 25]
    # Security finding should carry the SECURITY badge in its body.
    sec = next(c for c in out if c["line"] == 25)
    assert "SECURITY" in sec["body"]


def test_parse_review_genuinely_empty_returns_empty():
    assert _parse_review_file_response("<reasoning>looks fine</reasoning>\n[]", "f.ts") == []


# ---- unclosed <reasoning> (truncated / never-emitted-array output) ----

def test_strip_unclosed_reasoning_to_eof():
    # Model wrote reasoning then stopped before the closing tag + JSON array.
    assert _strip_thinking_blocks("<reasoning>\nfound a bug on [L5]") == ""


def test_strip_keeps_json_before_unclosed_reasoning():
    out = _strip_thinking_blocks('[{"line": 5}]\n<reasoning>\ntrailing junk')
    assert '"line": 5' in out
    assert "trailing junk" not in out


# ---- parse_with_repair ----

@pytest.mark.asyncio
async def test_parse_with_repair_recovers_prose_only_output():
    """Flash's failure mode: reasoning written, JSON array never emitted."""
    raw = "<reasoning>\nThe change on [L5] dereferences x which may be None.\n"  # unclosed, no array
    calls = []

    async def recall(prompt: str) -> str:
        calls.append(prompt)
        return '[{"line": 5, "severity": "bug", "title": "None deref", "body": "x may be None", "confidence": 9}]'

    out = await parse_with_repair(raw, "f.py", recall)
    assert len(out) == 1
    assert out[0]["line"] == 5
    assert len(calls) == 1
    assert "JSON array" in calls[0]


@pytest.mark.asyncio
async def test_parse_with_repair_skips_genuine_empty():
    """A correct [] means the model complied — do NOT waste a repair call."""
    raw = "<reasoning>\nNothing wrong here.\n</reasoning>\n[]"
    called = False

    async def recall(prompt: str) -> str:
        nonlocal called
        called = True
        return "[]"

    out = await parse_with_repair(raw, "f.py", recall)
    assert out == []
    assert called is False


@pytest.mark.asyncio
async def test_parse_with_repair_no_repair_when_already_parsed():
    raw = PR_238_RAW  # has a valid array already
    called = False

    async def recall(prompt: str) -> str:
        nonlocal called
        called = True
        return "[]"

    out = await parse_with_repair(raw, "src/extension.ts", recall)
    assert len(out) == 3
    assert called is False


# ---- off-diff routing into the summary ----

_OFF_DIFF = [
    {
        "path": "src/flusher.py",
        "line": 130,
        "body": "![BUG](https://img.shields.io/badge/BUG-AA0000?style=for-the-badge) SpawnProcess isinstance check\n\nThe isinstance check is always false for spawned processes.",
    }
]


def test_summary_renders_off_diff_section():
    out = _build_structured_summary([], _OFF_DIFF)
    assert "Findings not on changed lines" in out
    assert "src/flusher.py" in out
    assert "SpawnProcess isinstance check" in out
    assert "found no issues" not in out


def test_summary_off_diff_only_is_not_empty_state():
    """Regression: off-diff-only reviews must not fall back to 'no issues'."""
    out = _build_structured_summary([], _OFF_DIFF)
    assert out.strip() != "Sifted through the code and found no issues."


def test_summary_no_findings_at_all_is_empty_state():
    assert _build_structured_summary([], []) == "Sifted through the code and found no issues."


@pytest.mark.asyncio
async def test_summarize_review_appends_off_diff_to_inline():
    inline = [{"path": "a.py", "line": 3, "body": "![WARNING](https://img.shields.io/badge/WARNING-B8860B?style=for-the-badge) Something\n\nbody"}]
    out = await summarize_review(inline, _OFF_DIFF)
    assert "Sift Review" in out                       # main table present
    assert "Findings not on changed lines" in out     # off-diff section appended
    assert "SpawnProcess isinstance check" in out
