"""Effort levels and per-level execution plans for the review pipeline."""
import logging
from dataclasses import dataclass
from enum import Enum

from src import config

logger = logging.getLogger(__name__)

_VALID_EFFORTS = ("low", "balanced", "high")


class EffortLevel(str, Enum):
    LOW = "low"
    BALANCED = "balanced"
    HIGH = "high"


@dataclass(frozen=True)
class EffortPlan:
    level: EffortLevel
    run_critic: bool
    critic_per_finding: bool
    run_holistic: bool
    enable_agentic: bool
    context_depth: int
    request_reasoning: bool


_PLANS: dict[EffortLevel, EffortPlan] = {
    EffortLevel.LOW: EffortPlan(
        level=EffortLevel.LOW,
        run_critic=False,
        critic_per_finding=False,
        run_holistic=False,
        enable_agentic=False,
        context_depth=0,
        request_reasoning=False,
    ),
    EffortLevel.BALANCED: EffortPlan(
        level=EffortLevel.BALANCED,
        run_critic=True,
        critic_per_finding=False,
        run_holistic=True,
        enable_agentic=False,
        context_depth=1,
        request_reasoning=True,
    ),
    EffortLevel.HIGH: EffortPlan(
        level=EffortLevel.HIGH,
        run_critic=True,
        critic_per_finding=True,
        run_holistic=True,
        enable_agentic=True,
        context_depth=2,
        request_reasoning=True,
    ),
}


def plan_for(level: EffortLevel) -> EffortPlan:
    return _PLANS[level]


def resolve_effort() -> EffortLevel:
    """Read SIFT_REVIEW_EFFORT from config; fall back to BALANCED on invalid input."""
    raw = (config.SIFT_REVIEW_EFFORT or "balanced").strip().lower()
    try:
        return EffortLevel(raw)
    except ValueError:
        logger.warning(
            "SIFT_REVIEW_EFFORT=%r is invalid; using 'balanced'. Valid values: %s",
            raw,
            list(_VALID_EFFORTS),
        )
        return EffortLevel.BALANCED


def current_plan() -> EffortPlan:
    """Convenience: resolve effort and return its plan."""
    return plan_for(resolve_effort())
