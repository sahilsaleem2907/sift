"""Basic quality scoring (0-100) from feedback events. Defined formula; read-only."""
import logging
from typing import List

from src.storage.database import get_feedback_events_for_pr, get_feedback_events_for_review
from src.storage.models import FeedbackEvent

logger = logging.getLogger(__name__)

# Defined weights (documented in plan and docstring)
BASELINE = 50
REACTION_PLUS_ONE = 15
REACTION_MINUS_ONE = -15
REACTION_CONFUSED = -10
REACTION_POSITIVE = 10   # heart, hooray, rocket
REACTION_MILD = 5       # laugh, eyes
COMMAND_HELPFUL = 20
COMMAND_NOT_HELPFUL = -20


def _event_points(event: FeedbackEvent) -> int:
    """Return the score contribution for one feedback event. Uses defined numbers."""
    if event.event_type == "command":
        if event.command == "helpful":
            return COMMAND_HELPFUL
        if event.command == "not_helpful":
            return COMMAND_NOT_HELPFUL
        return 0
    if event.event_type == "reaction" and event.reaction_content:
        c = event.reaction_content
        if c == "+1":
            return REACTION_PLUS_ONE
        if c == "-1":
            return REACTION_MINUS_ONE
        if c == "confused":
            return REACTION_CONFUSED
        if c in ("heart", "hooray", "rocket"):
            return REACTION_POSITIVE
        if c in ("laugh", "eyes"):
            return REACTION_MILD
    return 0


def compute_quality_score(review_id: int) -> int:
    """
    Compute quality score 0-100 for a review from its feedback events.

    Formula: baseline 50; +1 → +15, -1 → -15, confused → -10;
    heart/hooray/rocket → +10, laugh/eyes → +5;
    helpful → +20, not_helpful → -20. Sum then clamp to [0, 100].
    """
    events = get_feedback_events_for_review(review_id)
    return _score_from_events(events)


def compute_quality_score_for_pr(repo: str, pr_number: int) -> int:
    """Compute quality score 0-100 for a repo+pr from feedback events (when review_id not used)."""
    events = get_feedback_events_for_pr(repo, pr_number)
    return _score_from_events(events)


def _score_from_events(events: List[FeedbackEvent]) -> int:
    """Sum contributions from events and clamp to [0, 100]. Only reaction and command events count."""
    total = BASELINE
    for ev in events:
        if ev.event_type not in ("reaction", "command"):
            continue
        total += _event_points(ev)
    return max(0, min(100, total))
