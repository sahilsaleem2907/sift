"""SQLAlchemy models."""
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for models."""
    pass


class Review(Base):
    """Stored review for a PR."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    pr_number: Mapped[int] = mapped_column(Integer, nullable=False)
    installation_id: Mapped[int] = mapped_column(Integer, nullable=False)
    review_body: Mapped[str] = mapped_column(Text, nullable=False)
    comment_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class FeedbackEvent(Base):
    """Single feedback event (reaction or /feedback command)."""

    __tablename__ = "feedback_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    repo: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    pr_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    comment_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    review_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("reviews.id"), nullable=True
    )
    actor: Mapped[str] = mapped_column(String(256), nullable=False)
    reaction_content: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    command: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index(
            "uq_feedback_reaction_comment_actor",
            "comment_id",
            "reaction_content",
            "actor",
            unique=True,
            postgresql_where=text("event_type = 'reaction'"),
        ),
    )
