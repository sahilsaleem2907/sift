"""Unit tests for file classification, test detection, and tool routing."""
import pytest

from sift.core.analysis_routing import (
    FileType,
    RiskLevel,
    classify_file_type,
    get_tools_for_file,
    is_test_path,
)


@pytest.mark.parametrize(
    "path",
    [
        # Directory-segment conventions
        "tests/test_foo.py",
        "app/tests/helpers.py",
        "src/__tests__/button.tsx",
        "src/__mocks__/api.ts",
        "spec/models/user_spec.rb",
        "pkg/handler/testdata/golden.json",
        "internal/fixtures/sample.yaml",
        "e2e/checkout.spec.ts",
        # Java/Kotlin source root
        "src/test/java/com/acme/FooTest.java",
        "app/src/androidTest/java/com/acme/UiTest.kt",
        # Filename conventions per language
        "app/test_login.py",
        "app/login_test.py",
        "conftest.py",
        "pkg/server/server_test.go",
        "web/components/Button.test.tsx",
        "web/components/Button.spec.js",
        "src/main/java/com/acme/FooTest.java",
        "src/UserTests.kt",
        "models/user_spec.rb",
        "models/user_test.rb",
        "src/CalculatorTest.cs",
        "src/Http/ClientTest.php",
    ],
)
def test_is_test_path_positive(path: str) -> None:
    assert is_test_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        # "test" as a substring of a word — must NOT match
        "src/latest_config.py",
        "app/contest/main.go",
        "lib/attestation.ts",
        "src/greatest.rb",
        "protester.java",
        # Ordinary production files
        "sift/core/review_engine.py",
        "src/components/Button.tsx",
        "pkg/server/server.go",
        "README.md",
    ],
)
def test_is_test_path_negative(path: str) -> None:
    assert is_test_path(path) is False


def test_get_tools_for_file_drops_deep_static_for_tests() -> None:
    # A CRITICAL-risk code file normally gets linter+semgrep+codeql...
    prod = get_tools_for_file(FileType.CODE, RiskLevel.CRITICAL, "auth/login.py")
    assert prod == frozenset({"linter", "semgrep", "codeql"})
    # ...but as a test file it keeps only the linter.
    test = get_tools_for_file(
        FileType.CODE, RiskLevel.CRITICAL, "tests/test_login.py", is_test=True
    )
    assert test == frozenset({"linter"})


def test_get_tools_for_file_test_config_drops_semgrep() -> None:
    # Config MEDIUM+ normally gets semgrep; a test fixture config gets nothing.
    prod = get_tools_for_file(FileType.CONFIG, RiskLevel.MEDIUM, "config/app.yaml")
    assert "semgrep" in prod
    test = get_tools_for_file(
        FileType.CONFIG, RiskLevel.MEDIUM, "tests/fixtures/app.yaml", is_test=True
    )
    assert "semgrep" not in test and "codeql" not in test


def test_get_tools_for_file_low_risk_test_keeps_linter() -> None:
    assert get_tools_for_file(
        FileType.CODE, RiskLevel.LOW, "tests/test_util.py", is_test=True
    ) == frozenset({"linter"})


def test_classify_still_treats_test_files_as_code() -> None:
    # is_test is an orthogonal signal; classification is unchanged.
    assert classify_file_type("tests/test_login.py") == FileType.CODE
    assert classify_file_type("pkg/server/server_test.go") == FileType.CODE
