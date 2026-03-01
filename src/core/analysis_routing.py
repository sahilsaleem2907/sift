"""File type classification, risk scoring, and tool routing for smart analysis.

Routes PR files to linter / semgrep / codeql based on file type and risk level
to reduce analysis time while preserving security coverage on high-risk code.
"""
import re
from enum import Enum
from pathlib import Path
from typing import FrozenSet

# --- File type classification (path-only) ------------------------------------

# Code: extensions used by linter_runner and codeql_runner
_CODE_EXTENSIONS = frozenset({
    ".py", ".pyw",
    ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
    ".go", ".java", ".kt", ".kts",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hh",
    ".cs", ".rb", ".swift", ".php", ".scala", ".rs", ".hs", ".lua",
    ".sh", ".bash", ".zsh", ".ps1",
})
_CONFIG_EXTENSIONS = frozenset({".yml", ".yaml", ".env", ".json"})
_INFRA_PATTERNS = (
    "Dockerfile",
    ".dockerignore",
    ".gitignore",
    ".tf",
    ".tfvars",
)
_DOCS_EXTENSIONS = frozenset({".md", ".txt", ".rst", ".adoc"})
_ASSET_EXTENSIONS = frozenset({
    ".png", ".svg", ".jpg", ".jpeg", ".gif", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".webm", ".mp3", ".wav",
})


class FileType(str, Enum):
    CODE = "code"
    CONFIG = "config"
    INFRASTRUCTURE = "infrastructure"
    DOCUMENTATION = "documentation"
    ASSETS = "assets"


def classify_file_type(path: str) -> FileType:
    """Classify file type from path only (no content)."""
    p = path.replace("\\", "/").lower().strip()
    name = Path(p).name.lower()

    # Check infra by filename/extension (case-insensitive)
    for pattern in _INFRA_PATTERNS:
        pl = pattern.lower()
        if name == pl or name.endswith(pl) or p.lower().endswith(pl):
            return FileType.INFRASTRUCTURE

    ext = Path(p).suffix.lower()
    if ext in _CODE_EXTENSIONS:
        return FileType.CODE
    if ext in _CONFIG_EXTENSIONS:
        return FileType.CONFIG
    if ext in _DOCS_EXTENSIONS:
        return FileType.DOCUMENTATION
    if ext in _ASSET_EXTENSIONS:
        return FileType.ASSETS

    # No extension or unknown: treat as code if it looks like code, else docs
    if not ext and "/" in p:
        return FileType.DOCUMENTATION
    return FileType.CODE  # default unknown to code so we don't skip by mistake


# --- Risk scoring (path + content) --------------------------------------------

_PATH_SENSITIVE = re.compile(
    r"(?:^|/)(?:auth|login|payment|admin|security)(?:/|$)",
    re.IGNORECASE,
)
_SIZE_1000 = 1000
_SIZE_500 = 500
_POINTS_PATH = 30
_POINTS_SIZE_1000 = 20
_POINTS_SIZE_500 = 10
_POINTS_DB = 15
_POINTS_API = 20
_POINTS_SECURITY_KW = 25

# Word-boundary style keywords (\\b not reliable for all; use simple substring with boundaries)
_DB_PATTERN = re.compile(
    r"\b(execute|cursor|query|sql|database|orm|raw\s*\()",
    re.IGNORECASE,
)
_API_PATTERN = re.compile(
    r"\b(api|endpoint|route|request|flask|fastapi|express|app\.get|app\.post|router\.)",
    re.IGNORECASE,
)
_SECURITY_PATTERN = re.compile(
    r"\b(password|token|crypto|secret|credential)\b",
    re.IGNORECASE,
)


def score_risk(path: str, content: str, file_type: FileType) -> int:
    """Compute risk score from path and content. Only code/config/infra are scored."""
    total, _ = score_risk_with_breakdown(path, content, file_type)
    return total


def score_risk_with_breakdown(
    path: str, content: str, file_type: FileType
) -> tuple[int, dict[str, int]]:
    """Compute risk score and return (total, breakdown) for logging why a risk level was chosen.

    breakdown keys: path, size, database, api, security (each 0 or the points added).
    """
    breakdown: dict[str, int] = {
        "path": 0,
        "size": 0,
        "database": 0,
        "api": 0,
        "security": 0,
    }
    p = path.replace("\\", "/")

    # Path-based
    if _PATH_SENSITIVE.search(p):
        breakdown["path"] = _POINTS_PATH

    # File size (only meaningful for text content)
    if file_type in (FileType.CODE, FileType.CONFIG, FileType.INFRASTRUCTURE):
        try:
            lines = len(content.splitlines()) if content else 0
            if lines > _SIZE_1000:
                breakdown["size"] = _POINTS_SIZE_1000
            elif lines > _SIZE_500:
                breakdown["size"] = _POINTS_SIZE_500
        except Exception:
            pass

    # Content heuristics (only for code and config)
    if file_type in (FileType.CODE, FileType.CONFIG):
        sample = content[:50000] if content else ""
        if _DB_PATTERN.search(sample):
            breakdown["database"] = _POINTS_DB
        if _API_PATTERN.search(sample):
            breakdown["api"] = _POINTS_API
        if _SECURITY_PATTERN.search(sample):
            breakdown["security"] = _POINTS_SECURITY_KW

    total = sum(breakdown.values())
    return total, breakdown


# --- Risk level buckets -------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"       # 0-19
    MEDIUM = "medium"  # 20-39
    HIGH = "high"     # 40-59
    CRITICAL = "critical"  # 60+


def risk_level(score: int) -> RiskLevel:
    """Map raw risk score to level."""
    if score >= 60:
        return RiskLevel.CRITICAL
    if score >= 40:
        return RiskLevel.HIGH
    if score >= 20:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# --- Routing matrix ----------------------------------------------------------

def get_tools_for_file(file_type: FileType, risk: RiskLevel) -> FrozenSet[str]:
    """Return which tools to run for this file. Values: 'linter', 'semgrep', 'codeql'."""
    if file_type == FileType.DOCUMENTATION or file_type == FileType.ASSETS:
        return frozenset()

    if file_type == FileType.CODE:
        if risk == RiskLevel.LOW:
            return frozenset({"linter"})
        if risk in (RiskLevel.MEDIUM, RiskLevel.HIGH):
            return frozenset({"linter", "semgrep"})
        # CRITICAL
        return frozenset({"linter", "semgrep", "codeql"})

    if file_type == FileType.CONFIG:
        if risk in (RiskLevel.LOW, RiskLevel.MEDIUM):
            return frozenset()
        return frozenset({"semgrep"})

    if file_type == FileType.INFRASTRUCTURE:
        return frozenset({"semgrep"})

    return frozenset()
