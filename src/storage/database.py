"""Database connection and session handling."""
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy import create_engine, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from src import config
from src.storage.models import Base, FeedbackEvent, Review, ReviewFile, ToolResultCache

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(config.DATABASE_URL or "", pool_pre_ping=True)
    return _engine


def _get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_get_engine(),
            expire_on_commit=False,
        )
    return _SessionLocal


def init_db() -> None:
    """Create tables if they do not exist. Add missing columns to existing tables (one-off migration)."""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("DB tables created or already exist")

    if config.VECTOR_DB_ENABLED:
        from src.storage.vector_store import init_vector_db
        init_vector_db()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope for a single operation."""
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def store_review(
    repo: str,
    pr_number: int,
    installation_id: int,
    review_body: str,
    comment_id: Optional[int] = None,
    paths: Optional[List[str]] = None,
) -> Optional[int]:
    """Insert a review row. Truncate body if needed. Optionally store paths for feedback loop.
    Returns review_id if paths were provided (needed for store_review_files), else None."""
    max_body = 65535
    body = review_body if len(review_body) <= max_body else review_body[: max_body - 3] + "..."
    review_id = None
    with session_scope() as session:
        review = Review(
            repo=repo,
            pr_number=pr_number,
            installation_id=installation_id,
            review_body=body,
            comment_id=comment_id,
        )
        session.add(review)
        session.flush()  # get review.id
        review_id = review.id
        if paths:
            for p in paths[:500]:  # cap to avoid huge inserts
                session.add(ReviewFile(review_id=review_id, path=p))
    logger.info("Stored review for %s PR #%s", repo, pr_number)
    return review_id


def store_pr_closed_event(repo: str, pr_number: int, merged: bool) -> None:
    """Record that a PR was closed or merged for analytics. Uses FeedbackEventType.pr_merged or pr_closed."""
    from src.feedback.enums import FeedbackEventType, FeedbackSource
    event_type = FeedbackEventType.pr_merged if merged else FeedbackEventType.pr_closed
    store_feedback_event(
        event_type=event_type.value,
        repo=repo,
        pr_number=pr_number,
        actor="webhook",
        source=FeedbackSource.webhook.value,
        comment_id=None,
        review_id=None,
        reaction_content=None,
        command=None,
    )
    logger.info("Stored %s event for %s PR #%s", event_type.value, repo, pr_number)


def store_feedback_event(
    event_type: str,
    repo: str,
    pr_number: int,
    actor: str,
    source: str,
    comment_id: Optional[int] = None,
    review_id: Optional[int] = None,
    reaction_content: Optional[str] = None,
    command: Optional[str] = None,
) -> None:
    """Insert one feedback event. For reactions, caller must ensure dedup (check-then-insert or rely on unique constraint)."""
    with session_scope() as session:
        session.add(
            FeedbackEvent(
                event_type=event_type,
                repo=repo,
                pr_number=pr_number,
                actor=actor,
                source=source,
                comment_id=comment_id,
                review_id=review_id,
                reaction_content=reaction_content,
                command=command,
            )
        )


def _reaction_exists(
    session: Session,
    comment_id: int,
    reaction_content: str,
    actor: str,
) -> bool:
    """Check if we already have this reaction event (dedup: check-then-insert)."""
    stmt = select(FeedbackEvent.id).where(
        FeedbackEvent.event_type == "reaction",
        FeedbackEvent.comment_id == comment_id,
        FeedbackEvent.reaction_content == reaction_content,
        FeedbackEvent.actor == actor,
    ).limit(1)
    return session.execute(stmt).scalar() is not None


def store_reaction_event_if_new(
    repo: str,
    pr_number: int,
    comment_id: int,
    actor: str,
    reaction_content: str,
    review_id: Optional[int] = None,
    is_inline_comment: bool = False,
) -> bool:
    """Store a reaction feedback event only if not already present. Returns True if stored. Dedup: check-then-insert; on race, unique constraint may raise IntegrityError - we catch and return False."""
    try:
        with session_scope() as session:
            if _reaction_exists(session, comment_id, reaction_content, actor):
                return False
            session.add(
                FeedbackEvent(
                    event_type="reaction",
                    repo=repo,
                    pr_number=pr_number,
                    actor=actor,
                    source="api",
                    comment_id=comment_id,
                    review_id=review_id,
                    reaction_content=reaction_content,
                    command=None,
                    is_inline_comment=is_inline_comment,
                )
            )
        return True
    except IntegrityError:
        return False


def get_feedback_events_for_review(review_id: int) -> List[FeedbackEvent]:
    """Return feedback events for a review (for scoring)."""
    with session_scope() as session:
        stmt = select(FeedbackEvent).where(FeedbackEvent.review_id == review_id).order_by(FeedbackEvent.created_at)
        return list(session.execute(stmt).scalars().all())


def get_feedback_events_for_pr(repo: str, pr_number: int) -> List[FeedbackEvent]:
    """Return feedback events for a repo+pr (for scoring when review_id not used)."""
    with session_scope() as session:
        stmt = (
            select(FeedbackEvent)
            .where(FeedbackEvent.repo == repo, FeedbackEvent.pr_number == pr_number)
            .order_by(FeedbackEvent.created_at)
        )
        return list(session.execute(stmt).scalars().all())


def get_review_ids_for_path_pattern(repo: str, path_prefix: str) -> List[int]:
    """Return distinct review_ids for reviews that touched files under path_prefix.
    path_prefix is a directory prefix, e.g. 'src/auth' matches 'src/auth/login.py'."""
    if not path_prefix:
        return []
    prefix_slash = path_prefix.rstrip("/") + "/"
    with session_scope() as session:
        stmt = (
            select(ReviewFile.review_id)
            .join(Review, ReviewFile.review_id == Review.id)
            .where(Review.repo == repo)
            .where(
                or_(
                    ReviewFile.path == path_prefix.rstrip("/"),
                    ReviewFile.path.like(prefix_slash + "%"),
                )
            )
            .distinct()
        )
        rows = session.execute(stmt).scalars().all()
        return list(rows)


def get_avg_quality_score_for_path_pattern(repo: str, path_prefix: str) -> Optional[float]:
    """Average quality score (0-100) for past reviews that touched files under path_prefix.
    Returns None if no such reviews exist."""
    review_ids = get_review_ids_for_path_pattern(repo, path_prefix)
    if not review_ids:
        return None
    from src.feedback.scorer import compute_quality_score

    scores = [compute_quality_score(rid) for rid in review_ids]
    return sum(scores) / len(scores)


def get_review_by_repo_pr(repo: str, pr_number: int) -> Optional[tuple[int, Optional[int]]]:
    """Get the most recent review for this repo+pr. Returns (review_id, comment_id) or None.
    Returns plain values so callers (e.g. background tasks) do not hold a detached ORM instance."""
    with session_scope() as session:
        stmt = (
            select(Review.id, Review.comment_id)
            .where(Review.repo == repo, Review.pr_number == pr_number)
            .order_by(Review.created_at.desc())
            .limit(1)
        )
        row = session.execute(stmt).one_or_none()
        if row is None:
            return None
        return (row[0], row[1])


def get_tool_cache_hits(keys: List[str], ttl_hours: int) -> Dict[str, List[Any]]:
    """Batch lookup: return {cache_key: findings} for non-expired keys. findings are parsed from JSON."""
    if not keys:
        return {}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
    with session_scope() as session:
        stmt = (
            select(ToolResultCache.cache_key, ToolResultCache.findings_json)
            .where(ToolResultCache.cache_key.in_(keys))
            .where(ToolResultCache.created_at >= cutoff)
        )
        rows = session.execute(stmt).all()
    out: Dict[str, List[Any]] = {}
    for cache_key, findings_json in rows:
        try:
            out[cache_key] = json.loads(findings_json)
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def store_tool_cache(entries: List[Dict[str, Any]]) -> None:
    """Batch upsert cache rows. Each entry: cache_key, tool, findings_json (str)."""
    if not entries:
        return
    now = datetime.now(timezone.utc)
    with session_scope() as session:
        stmt = pg_insert(ToolResultCache).values(
            [
                {
                    "cache_key": e["cache_key"],
                    "tool": e["tool"],
                    "findings_json": e["findings_json"],
                    "created_at": now,
                }
                for e in entries
            ]
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["cache_key"],
            set_={
                "findings_json": stmt.excluded.findings_json,
                "created_at": stmt.excluded.created_at,
            },
        )
        session.execute(stmt)
