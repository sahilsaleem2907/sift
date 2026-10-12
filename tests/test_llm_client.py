"""Tests for JSON extraction from reasoning-model output.

Regression coverage for the parser bug that shipped silently through PRs
#236–#238: the candidates prompt instructs the model to emit a
<reasoning>...</reasoning> block before the JSON array, and the stray brackets
in that prose (e.g. "[L10]") defeated the old first-bracket-to-last-bracket
parser, producing zero findings with no visible error.
"""
from sift.intelligence.llm_client import (
    _balanced_array_end,
    _extract_json_array,
    _parse_review_file_response,
    _strip_thinking_blocks,
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


# ---- structured fields carried through the parser (PR #24 severity fix) ----

def test_parse_json_carries_severity_title_confidence():
    out = _parse_review_file_response(PR_238_RAW, "src/extension.ts")
    by_line = {c["line"]: c for c in out}
    assert by_line[10]["severity"] == "bug"
    assert by_line[10]["title"] == "Typo in namespace"
    assert by_line[10]["confidence"] == 10
    assert by_line[25]["severity"] == "security"
    assert by_line[11]["confidence"] == 9


def test_parse_tab_format_carries_severity():
    raw = "[L12]\twarning\tPossible race\tTwo writers touch the same key."
    out = _parse_review_file_response(raw, "x.py")
    assert len(out) == 1
    assert out[0]["severity"] == "warning"
    assert out[0]["title"] == "Possible race"
    assert out[0]["confidence"] == 7


def test_parse_freeform_has_no_structured_label():
    raw = "Line 8: this loop never terminates when items is empty."
    out = _parse_review_file_response(raw, "x.py")
    assert len(out) == 1
    assert out[0]["severity"] is None
    assert out[0]["confidence"] == 7


# ---- _call_llm: api_key / num_retries passthrough on the LiteLLM path ----

import asyncio
from unittest import mock

from sift.intelligence import llm_client


def _fake_response(text: str = "ok"):
    resp = mock.MagicMock()
    resp.choices = [mock.MagicMock(message=mock.MagicMock(content=text))]
    return resp


def _run_call_llm(cfg_overrides, **call_kwargs):
    fake = mock.AsyncMock(return_value=_fake_response())
    with mock.patch.object(llm_client, "acompletion", new=fake), \
         mock.patch.multiple(llm_client.config, **cfg_overrides):
        asyncio.run(llm_client._call_llm("sys", "user", **call_kwargs))
    return fake.call_args.kwargs


_GROQ_CFG = {
    "LLM_MODEL": "groq/openai/gpt-oss-120b",
    "LLM_API_BASE": None,
    "LLM_API_KEY": "gsk_test",
    "SIFT_LLM_NUM_RETRIES": 4,
}


def test_call_llm_passes_llm_api_key_and_retries_without_base():
    kw = _run_call_llm(_GROQ_CFG)
    assert kw["model"] == "groq/openai/gpt-oss-120b"
    assert kw["api_key"] == "gsk_test"
    assert kw["api_base"] is None
    assert kw["num_retries"] == 4


def test_call_llm_critic_on_primary_model_inherits_key():
    # Critic/holistic pass model=SIFT_REVIEW_MODEL or LLM_MODEL and api_key=None.
    kw = _run_call_llm(_GROQ_CFG, model="groq/openai/gpt-oss-120b", api_key=None)
    assert kw["api_key"] == "gsk_test"


def test_call_llm_critic_on_other_provider_does_not_get_primary_key():
    kw = _run_call_llm(_GROQ_CFG, model="anthropic/claude-haiku-4-5", api_key=None)
    assert kw["api_key"] is None


def test_call_llm_explicit_key_wins():
    kw = _run_call_llm(_GROQ_CFG, model="anthropic/claude-haiku-4-5", api_key="sk-ant-x")
    assert kw["api_key"] == "sk-ant-x"
