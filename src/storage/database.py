"""Database connection and session handling."""
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy import case, create_engine, func, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError, ProgrammingError
from sqlalchemy.orm import Session, sessionmaker

from src import config
from src.storage.models import Base, FeedbackEvent, Review, ReviewComment, ReviewFile, ToolResultCache

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


def get_repo_feedback_summary(repo: str) -> Dict[str, int]:
    """Aggregate feedback signals for a repo (reactions + /feedback commands).

    Positive reactions: +1, heart, hooray, rocket (inline vs summary via is_inline_comment).
    Negative reactions: -1, confused.
    Commands: helpful / not_helpful (repo-wide, not split by inline).
    """
    positive_reactions = frozenset({"+1", "heart", "hooray", "rocket"})
    negative_reactions = frozenset({"-1", "confused"})

    inline_positive = 0
    inline_negative = 0
    summary_positive = 0
    summary_negative = 0
    helpful_commands = 0
    not_helpful_commands = 0

    with session_scope() as session:
        stmt = (
            select(FeedbackEvent)
            .where(
                FeedbackEvent.repo == repo,
                FeedbackEvent.event_type.in_(("reaction", "command")),
            )
            .order_by(FeedbackEvent.created_at)
        )
        rows = list(session.execute(stmt).scalars().all())

    for ev in rows:
        if ev.event_type == "command":
            if ev.command == "helpful":
                helpful_commands += 1
            elif ev.command == "not_helpful":
                not_helpful_commands += 1
            continue
        if ev.event_type != "reaction" or not ev.reaction_content:
            continue
        c = ev.reaction_content
        is_inline = bool(ev.is_inline_comment)
        if c in positive_reactions:
            if is_inline:
                inline_positive += 1
            else:
                summary_positive += 1
        elif c in negative_reactions:
            if is_inline:
                inline_negative += 1
            else:
                summary_negative += 1

    total_events = (
        inline_positive
        + inline_negative
        + summary_positive
        + summary_negative
        + helpful_commands
        + not_helpful_commands
    )

    return {
        "inline_positive": inline_positive,
        "inline_negative": inline_negative,
        "summary_positive": summary_positive,
        "summary_negative": summary_negative,
        "helpful_commands": helpful_commands,
        "not_helpful_commands": not_helpful_commands,
        "total_events": total_events,
    }


def upsert_review_comment(
    comment_id: int,
    review_id: Optional[int],
    repo: str,
    severity: str,
    title: Optional[str],
) -> None:
    """Insert or refresh inline comment metadata (severity/title) on each reaction sync."""
    sev = (severity or "suggestion").lower()[:32]
    if sev not in ("bug", "security", "warning", "suggestion"):
        sev = "suggestion"
    t = title
    if t and len(t) > 256:
        t = t[:253] + "..."
    ins = pg_insert(ReviewComment).values(
        comment_id=comment_id,
        review_id=review_id,
        repo=repo,
        severity=sev,
        title=t,
    )
    stmt = ins.on_conflict_do_update(
        index_elements=["comment_id"],
        set_={
            "severity": ins.excluded.severity,
            "title": ins.excluded.title,
            "review_id": ins.excluded.review_id,
            "repo": ins.excluded.repo,
        },
    )
    with session_scope() as session:
        session.execute(stmt)


_MIN_SEVERITY_REACTIONS = 3


def get_severity_feedback_summary(repo: str) -> Dict[str, Dict[str, int]]:
    """Net reaction signal per severity for inline comments with known labels.

    Joins feedback_events (inline reactions) to review_comments. Only includes
    severities with at least _MIN_SEVERITY_REACTIONS total counted reactions.
    """
    positive_reactions = frozenset({"+1", "heart", "hooray", "rocket"})
    negative_reactions = frozenset({"-1", "confused"})

    with session_scope() as session:
        stmt = (
            select(FeedbackEvent.reaction_content, ReviewComment.severity)
            .join(ReviewComment, FeedbackEvent.comment_id == ReviewComment.comment_id)
            .where(
                FeedbackEvent.repo == repo,
                FeedbackEvent.event_type == "reaction",
                FeedbackEvent.is_inline_comment.is_(True),
                FeedbackEvent.comment_id.isnot(None),
            )
        )
        rows = session.execute(stmt).all()

    buckets: Dict[str, Dict[str, int]] = {}
    for reaction_content, severity in rows:
        if not reaction_content or not severity:
            continue
        c = reaction_content
        if c in positive_reactions:
            key = "positive"
        elif c in negative_reactions:
            key = "negative"
        else:
            continue
        sev = severity.lower()
        if sev not in ("bug", "security", "warning", "suggestion"):
            sev = "suggestion"
        if sev not in buckets:
            buckets[sev] = {"positive": 0, "negative": 0}
        buckets[sev][key] += 1

    out: Dict[str, Dict[str, int]] = {}
    for sev, counts in buckets.items():
        pos = counts["positive"]
        neg = counts["negative"]
        total = pos + neg
        if total < _MIN_SEVERITY_REACTIONS:
            continue
        net = pos - neg
        out[sev] = {"positive": pos, "negative": neg, "net": net}
    return out


def get_repo_feedback_comment_examples(repo: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Inline comments with stored severity/title plus aggregated reaction counts.

    Ordered by total reactions (then by |net|) so the LLM sees the strongest signals first.
    """
    positive_reactions = ("+1", "heart", "hooray", "rocket")
    negative_reactions = ("-1", "confused")

    reaction_agg = (
        select(
            FeedbackEvent.comment_id.label("comment_id"),
            func.sum(
                case((FeedbackEvent.reaction_content.in_(positive_reactions), 1), else_=0)
            ).label("positive"),
            func.sum(
                case((FeedbackEvent.reaction_content.in_(negative_reactions), 1), else_=0)
            ).label("negative"),
        )
        .where(
            FeedbackEvent.repo == repo,
            FeedbackEvent.event_type == "reaction",
            FeedbackEvent.is_inline_comment.is_(True),
            FeedbackEvent.comment_id.isnot(None),
        )
        .group_by(FeedbackEvent.comment_id)
    ).subquery()

    total_rx = reaction_agg.c.positive + reaction_agg.c.negative
    net_expr = reaction_agg.c.positive - reaction_agg.c.negative

    stmt = (
        select(
            ReviewComment.severity,
            ReviewComment.title,
            reaction_agg.c.positive,
            reaction_agg.c.negative,
        )
        .join(reaction_agg, ReviewComment.comment_id == reaction_agg.c.comment_id)
        .where(ReviewComment.repo == repo)
        .where(total_rx > 0)
        .order_by(total_rx.desc(), func.abs(net_expr).desc())
        .limit(limit)
    )

    out: List[Dict[str, Any]] = []
    with session_scope() as session:
        for row in session.execute(stmt).all():
            sev, title, pos, neg = row[0], row[1], int(row[2] or 0), int(row[3] or 0)
            out.append({
                "severity": sev or "suggestion",
                "title": title,
                "positive": pos,
                "negative": neg,
                "net": pos - neg,
            })
    return out


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
    """Get the most recent review for this repo+pr.

    Returns (db_review_id, summary_comment_id) or None, where summary_comment_id is the GitHub
    issue-comment id created for the PR summary ("Conversation" tab).

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
