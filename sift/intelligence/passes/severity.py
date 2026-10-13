# SPDX-License-Identifier: AGPL-3.0-only
"""Pass 4 (Phase 2): severity rubric + noise gate."""
import logging
import re
from dataclasses import replace
from typing import Optional

from sift.intelligence.effort import EffortPlan
from sift.intelligence.schema import Certainty, Finding, Impact, derive_severity

logger = logging.getLogger(__name__)

_UNVERIFIED_PREFIX = "[Unverified — needs manual check] "

_SHIELD_BADGE_RE = re.compile(
    r"!\[[^\]]+\]\(https://img\.shields\.io/badge/[^)]+\)",
    re.IGNORECASE,
)

# Subjective categories require CONFIRMED certainty to avoid noise.
_CONFIRMED_ONLY_CATEGORIES = frozenset({"design", "maintainability", "style"})

# Epistemic-hedge / self-doubt signals: the finding itself expresses uncertainty about whether
# its premise or claim actually holds (NOT a logical trigger like "if x is null -> crash").
# The generator/critic often mis-rate these as confident, so we key on the body TEXT, not the
# certainty field. Matched findings are demoted (HIGH/CRITICAL kept as unverified) or dropped
# (LOW/MEDIUM) in apply_severity_gate. Conservative by design; tune from the [noise] logs.
_HEDGE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # explicit inability to verify from the available context
        r"cannot be (?:confirmed|verified|determined|inspected|established)",
        r"can(?:'|no)?t be (?:confirmed|verified)",
        r"not (?:be )?(?:fully )?confirmed(?: from| here| in the diff)",
        r"not visible in the diff",
        r"not shown (?:here|in the diff)",
        r"unable to (?:confirm|verify|determine)",
        r"without (?:seeing|access to)",
        r"depends (?:entirely )?on how",
        # premise-uncertainty conditionals (epistemic "if", not a logical bug trigger)
        r"if [^.\n]{0,80}\b(?:user[- ]controlled|attacker[- ](?:controlled|supplied"
        r"|influenced)|untrusted|tainted|tampered|malicious)",
        r"if the (?:project|codebase|code)(?: ever)? migrates",
        r"if [^.\n]{0,60}\b(?:has|have)(?: any)? gaps",
        # self-negation / residual-doubt tails
        r"\bnot a (?:real )?bug\b",
        r"\bno bug here\b",
        r"this is not a bug",
        r"no (?:concrete|real|actual) (?:issue|scenario|bug|consequence|impact)",
        r"no runtime consequence",
        r"(?:is|are) actually correct",
        r"\bself-consistent\b",
        r"works but (?:is )?fragile",
        r"correct only if",
    )
]


def _is_hedged_or_self_doubting(body: str) -> Optional[str]:
    """Return the matched hedge phrase if the finding hedges/doubts its own claim, else None.

    Matches EPISTEMIC uncertainty (doubt about whether the premise/claim holds) or explicit
    self-negation — NOT logical conditionals ("if x is null -> crash"), which are legitimate
    bug triggers. Keyed on body text because the certainty field is unreliable for this class.
    """
    if not body:
        return None
    for pat in _HEDGE_PATTERNS:
        m = pat.search(body)
        if m:
            return m.group(0)
    return None


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
        # Hedged / self-doubting findings: the generator/critic can mis-rate these as
        # confident, so key on the body text. HIGH/CRITICAL are kept but demoted to
        # SPECULATIVE (CRITICAL then gets the [Unverified] prefix below; HIGH renders as a
        # warning) to protect recall on genuinely-severe cautious findings; LOW/MEDIUM are
        # dropped as speculative noise.
        hedge = _is_hedged_or_self_doubting(f.body)
        if hedge:
            if f.impact in (Impact.CRITICAL, Impact.HIGH):
                if f.certainty != Certainty.SPECULATIVE:
                    logger.debug(
                        "[noise] demote hedged (kept, impact=%s) line=%d phrase=%r",
                        f.impact.name, f.line, hedge,
                    )
                    f = replace(f, certainty=Certainty.SPECULATIVE)
            else:
                logger.debug(
                    "[noise] drop hedged (impact=%s) line=%d phrase=%r",
                    f.impact.name, f.line, hedge,
                )
                continue
        if f.category in _CONFIRMED_ONLY_CATEGORIES and f.certainty != Certainty.CONFIRMED:
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
