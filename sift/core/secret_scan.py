"""Built-in regex secret scanner.

Scans added lines of a unified diff for hardcoded secrets and returns
semgrep-shaped finding dicts so they flow through promote_static_findings
(critic_exempt, CRITICAL impact, security category).

Complements real Semgrep: fires even when Semgrep is not installed or when
the ``--config auto`` ruleset does not cover secrets.
"""
from __future__ import annotations

import re
from typing import Optional

_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("github-pat",              re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("github-fine-grained-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b")),
    ("openai-key",              re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("aws-access-key",          re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack-token",             re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-api-key",          re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("private-key",             re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
]

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def scan_diff_for_secrets(file_diff: str) -> list[dict]:
    """Return semgrep-shaped finding dicts for any secrets found in added lines.

    Each dict has the same shape as a Semgrep result (``check_id``, ``line``,
    ``severity``, ``message``) so it flows through ``promote_static_findings``
    unchanged — the ``builtin.hardcoded-secret.*`` rule-id prefix is recognised
    by ``_is_secret_rule`` in ``static_promote.py``, ensuring ``Impact.CRITICAL``
    and ``critic_exempt=True`` on every run.
    """
    if not file_diff or not file_diff.strip():
        return []

    findings: list[dict] = []
    current_new_line: Optional[int] = None

    for line in file_diff.splitlines():
        m = _HUNK_RE.match(line)
        if m:
            current_new_line = int(m.group(1))
            continue
        if current_new_line is None or not line:
            continue

        prefix = line[0]
        if prefix == " ":
            current_new_line += 1
        elif prefix == "-" and not line.startswith("---"):
            pass  # removed line: new-file counter unchanged
        elif prefix == "+" and not line.startswith("+++"):
            content = line[1:]
            for suffix, pattern in _SECRET_PATTERNS:
                match = pattern.search(content)
                if match:
                    secret = match.group(0)
                    redacted = secret[:6] + "…" if len(secret) > 6 else secret
                    findings.append({
                        "check_id": f"builtin.hardcoded-secret.{suffix}",
                        "line": current_new_line,
                        "severity": "ERROR",
                        "message": (
                            f"Hardcoded secret detected ({suffix}): '{redacted}'. "
                            f"Move to an environment variable or secret manager and rotate this value."
                        ),
                    })
            current_new_line += 1

    return findings
