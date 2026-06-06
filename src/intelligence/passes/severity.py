"""Pass 4 (Phase 2): severity rubric + noise gate."""
from src.intelligence.effort import EffortPlan
from src.intelligence.schema import Certainty, Finding, Impact

_UNVERIFIED_PREFIX = "[Unverified — needs manual check] "


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
