"""Regression tests for the PR #21 review-quality fixes.

Covers:
- A4: _format_file_context renders the full file under the line cap and falls
  back to changed-range rendering above it.
- A1 safety net: the deterministic secret scanner flags literal credential
  values but never secret *references* (${{ secrets.X }}, env lookups).
"""
from src import config
from src.core.secret_scan import scan_diff_for_secrets
from src.core.semgrep_runner import _semgrep_handles_language
from src.intelligence.llm_client import _format_file_context


# ---- A4: full-file vs range rendering ----

def test_format_file_context_renders_full_file_under_cap():
    content = "\n".join(f"line{i}" for i in range(1, 11))
    block = _format_file_context(
        {"path": "small.py", "content": content, "ranges": [(3, 4)]}
    )
    assert block.startswith("Full file")
    # Every line is present, not just the changed range.
    assert "line1" in block and "line10" in block


def test_format_file_context_falls_back_to_ranges_above_cap(monkeypatch):
    monkeypatch.setattr(config, "SIFT_FULL_FILE_RENDER_MAX_LINES", 5)
    content = "\n".join(f"line{i}" for i in range(1, 21))  # 20 lines > cap
    block = _format_file_context(
        {"path": "big.py", "content": content, "ranges": [(2, 3)]}
    )
    assert block.startswith("Surrounding context")
    assert "line2" in block and "line3" in block
    # Lines outside the changed range are not rendered.
    assert "line19" not in block


def test_format_file_context_empty_content_returns_empty():
    assert _format_file_context({"path": "x.py", "content": "", "ranges": [(1, 2)]}) == ""


# ---- A1 safety net: literal secrets flagged, references ignored ----

def _added_diff(line: str) -> str:
    return "@@ -0,0 +1 @@\n+" + line + "\n"


def test_secret_scanner_flags_literal_value():
    diff = _added_diff('API_KEY = "sk-abcdefghijklmnopqrstuvwxyz0123"')
    findings = scan_diff_for_secrets(diff)
    assert findings, "literal secret should be flagged by the deterministic scanner"


def test_secret_scanner_ignores_actions_reference():
    diff = _added_diff("SIFT_API_KEY: ${{ secrets.SIFT_API_KEY }}")
    assert scan_diff_for_secrets(diff) == []


def test_secret_scanner_ignores_env_reference():
    diff = _added_diff('API_KEY = os.environ["SIFT_API_KEY"]')
    assert scan_diff_for_secrets(diff) == []


# ---- PR #22: semgrep parse errors gated by language capability ----

def test_semgrep_parse_error_kept_for_reliable_languages():
    assert _semgrep_handles_language("src/core/foo.py")
    assert _semgrep_handles_language("src/app.ts")
    assert _semgrep_handles_language("pkg/main.go")


def test_semgrep_parse_error_dropped_for_workflow_and_shell():
    # GitHub Actions workflow YAML (${{ }} templating breaks semgrep's parser)
    assert not _semgrep_handles_language(".github/workflows/sift-review.yml")
    assert not _semgrep_handles_language(".github/workflows/pr-feedback.yaml")
    # Shell: semgrep's bash parser is weak and chokes on templated run-steps
    assert not _semgrep_handles_language("scripts/install-linters.sh")
    # Unknown / extensionless
    assert not _semgrep_handles_language("Dockerfile")
