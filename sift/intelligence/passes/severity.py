"""Pass 4 (Phase 2): severity rubric + noise gate."""
import re
from dataclasses import replace

from sift.intelligence.effort import EffortPlan
from sift.intelligence.schema import Certainty, Finding, Impact, derive_severity

_UNVERIFIED_PREFIX = "[Unverified — needs manual check] "

_SHIELD_BADGE_RE = re.compile(
    r"!\[[^\]]+\]\(https://img\.shields\.io/badge/[^)]+\)",
    re.IGNORECASE,
)


def apply_severity_gate(findings: list[Finding], plan: EffortPlan) -> list[Finding]:
    """Filter and adjust findings using impact × certainty rules."""
    _ = plan
    out: list[Finding] = []
    for f in findings:
        if f.critic_exempt:
            # Static-tool findings are pre-confirmed; skip all noise filtering
            out.append(f)
            continue
        if f.impact == Impact.TRIVIAL:
            continue
        if f.certainty == Certainty.SPECULATIVE and f.impact == Impact.LOW:
            continue
        if (
            f.certainty == Certainty.SPECULATIVE
            and f.impact == Impact.CRITICAL
            and not f.body.startswith(_UNVERIFIED_PREFIX)
        ):
            f = Finding(
                path=f.path,
                line=f.line,
                title=f.title,
                body=_UNVERIFIED_PREFIX + f.body,
                impact=f.impact,
                certainty=f.certainty,
                category=f.category,
                origin=f.origin,
                fix=f.fix,
                post_inline=f.post_inline,
            )
        out.append(f)
    return out


def apply_final_severity_labels(findings: list[Finding]) -> list[Finding]:
    """Re-render each body's severity badge from final impact × certainty.

    The badge is baked into the body when the candidate is first formatted, from
    the LLM's original severity label. The critic re-rates impact/certainty
    afterwards, and downstream consumers (block policy, summary counts) read
    severity back out of the badge text — so without this pass, critic
    downgrades never reach the output. Idempotent; safe to apply twice.
    """
    from sift.intelligence.llm_client import _SEV_BADGE_BY_KEY

    out: list[Finding] = []
    for f in findings:
        if f.critic_exempt:
            # Static-tool findings keep their pre-rendered severity untouched
            out.append(f)
            continue
        badge = _SEV_BADGE_BY_KEY[derive_severity(f.impact, f.certainty, f.category)]
        body = f.body or ""
        if _SHIELD_BADGE_RE.search(body):
            new_body = _SHIELD_BADGE_RE.sub(lambda _: badge, body, count=1)
        elif f.title:
            new_body = f"{badge} {f.title}\n\n{body}"
        else:
            new_body = f"{badge}\n\n{body}" if body else badge
        out.append(f if new_body == body else replace(f, body=new_body))
    return out
