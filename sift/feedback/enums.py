"""Enums for feedback events; store values as strings in DB and validate on write."""
from enum import Enum


class FeedbackEventType(str, Enum):
    reaction = "reaction"
    command = "command"
    comment = "comment"
    pr_closed = "pr_closed"
    pr_merged = "pr_merged"


class FeedbackSource(str, Enum):
    webhook = "webhook"
    api = "api"


class FeedbackCommand(str, Enum):
    helpful = "helpful"
    not_helpful = "not_helpful"


class ReactionContent(str, Enum):
    """GitHub reaction content values (match API)."""
    PLUS_ONE = "+1"
    MINUS_ONE = "-1"
    LAUGH = "laugh"
    CONFUSED = "confused"
    HEART = "heart"
    HOORAY = "hooray"
    ROCKET = "rocket"
    EYES = "eyes"
