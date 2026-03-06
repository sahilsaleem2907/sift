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

# Tiered path sensitivity: high (+30), medium (+15), framework (+10)
# Take the highest matching tier (path score is at most one of these)
_PATH_HIGH = re.compile(
    r"(?:^|/)(?:auth|login|payment|admin|security|crypto|keys|secrets|oauth|session|billing|checkout)(?:/|$)",
    re.IGNORECASE,
)
_PATH_MEDIUM = re.compile(
    r"(?:^|/)(?:middleware|webhook|migration|permissions|rbac|acl|gateway|proxy|cert)(?:/|$)|(?:^|/)api/v\d",
    re.IGNORECASE,
)
_PATH_FRAMEWORK = re.compile(
    r"(?:^|/)(?:routes|controllers|handlers|views)(?:/|$)",
    re.IGNORECASE,
)
_SIZE_1000 = 1000
_SIZE_500 = 500
_POINTS_PATH = 30  # high-sensitivity paths
_POINTS_PATH_MEDIUM = 15  # medium-sensitivity paths
_POINTS_PATH_FRAMEWORK = 10  # framework entry points (routes, controllers, etc.)
_POINTS_SIZE_1000 = 10
_POINTS_SIZE_500 = 5
_POINTS_DB = 15
_POINTS_API = 20
_POINTS_SECURITY_KW = 25
_POINTS_DANGEROUS_OPS = 20
_POINTS_CRYPTO = 15

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
_DANGEROUS_OPS_PATTERN = re.compile(
    r"\b(eval|exec|subprocess|os\.system|os\.popen|pickle\.loads|yaml\.load\s*\(|innerHTML|dangerouslySetInnerHTML|Function\s*\(|__import__|compile\s*\(|deserialize)\b",
    re.IGNORECASE,
)
_CRYPTO_PATTERN = re.compile(
    r"\b(md5|sha1|ECB|DES|verify\s*=\s*False|CERTIFICATE_VERIFY_FAILED|ssl\._create_unverified_context)\b",
    re.IGNORECASE,
)


# Diff complexity scoring (tiebreaker/nudge, not category jump)
_DIFF_ADDED_11_50 = 3
_DIFF_ADDED_51_200 = 5
_DIFF_ADDED_200_PLUS = 8
_DIFF_NEW_FILE = 5
_DIFF_DELETION_HEAVY = -5
_DIFF_DELETION_RATIO = 0.8


def score_diff_complexity(file_diff: str) -> tuple[int, dict[str, int]]:
    """Score diff complexity. Returns (points, breakdown) for tiebreaker/nudge.

    Added lines: 1-10=0, 11-50=+3, 51-200=+5, 200+=+8.
    New file: +5. Deletion ratio >80%: -5.
    """
    breakdown: dict[str, int] = {"diff_added": 0, "diff_new_file": 0, "diff_deletion": 0}
    if not file_diff or not file_diff.strip():
        return 0, breakdown

    added = 0
    deleted = 0
    is_new_file = "new file mode" in file_diff[:200].lower()

    for line in file_diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if len(line) >= 1:
            if line[0] == "+":
                added += 1
            elif line[0] == "-":
                deleted += 1

    if is_new_file:
        breakdown["diff_new_file"] = _DIFF_NEW_FILE

    total_changed = added + deleted
    if total_changed > 0 and deleted / total_changed >= _DIFF_DELETION_RATIO:
        breakdown["diff_deletion"] = _DIFF_DELETION_HEAVY

    if added <= 10:
        pass
    elif added <= 50:
        breakdown["diff_added"] = _DIFF_ADDED_11_50
    elif added <= 200:
        breakdown["diff_added"] = _DIFF_ADDED_51_200
    else:
        breakdown["diff_added"] = _DIFF_ADDED_200_PLUS

    return sum(breakdown.values()), breakdown


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
        "dangerous_ops": 0,
        "crypto": 0,
    }
    p = path.replace("\\", "/")

    # Path-based (tiered: take highest matching tier)
    if _PATH_HIGH.search(p):
        breakdown["path"] = _POINTS_PATH  # 30
    elif _PATH_MEDIUM.search(p):
        breakdown["path"] = _POINTS_PATH_MEDIUM  # 15
    elif _PATH_FRAMEWORK.search(p):
        breakdown["path"] = _POINTS_PATH_FRAMEWORK  # 10

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
        if _DANGEROUS_OPS_PATTERN.search(sample):
            breakdown["dangerous_ops"] = _POINTS_DANGEROUS_OPS
        if _CRYPTO_PATTERN.search(sample):
            breakdown["crypto"] = _POINTS_CRYPTO

    total = sum(breakdown.values())
    return total, breakdown


def score_risk_combined(
    path: str, content: str, file_type: FileType, file_diff: str
) -> tuple[int, dict[str, int]]:
    """Combine path/content risk score with diff complexity. Merges breakdowns."""
    base_total, base_breakdown = score_risk_with_breakdown(path, content, file_type)
    diff_total, diff_breakdown = score_diff_complexity(file_diff)
    merged = {**base_breakdown, **diff_breakdown}
    return base_total + diff_total, merged


# --- Risk level buckets -------------------------------------------------------

class RiskLevel(str, Enum):
    LOW = "low"       # 0-14
    MEDIUM = "medium"  # 15-34
    HIGH = "high"     # 35-54
    CRITICAL = "critical"  # 55+


def risk_level(score: int) -> RiskLevel:
    """Map raw risk score to level."""
    if score >= 55:
        return RiskLevel.CRITICAL
    if score >= 35:
        return RiskLevel.HIGH
    if score >= 15:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# --- Routing matrix ----------------------------------------------------------

def get_tools_for_file(
    file_type: FileType, risk: RiskLevel, path: str = ""
) -> FrozenSet[str]:
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
        # .env (not .env.example) always gets Semgrep for secret scanning
        p = path.replace("\\", "/").lower()
        if ".env.example" not in p and (p.endswith(".env") or "/.env" in p):
            return frozenset({"semgrep"})
        # Config LOW: no tools. Config MEDIUM+: Semgrep
        if risk == RiskLevel.LOW:
            return frozenset()
        return frozenset({"semgrep"})

    if file_type == FileType.INFRASTRUCTURE:
        return frozenset({"semgrep"})

    return frozenset()
