"""Database connection and session handling."""
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src import config
from src.storage.models import Base, Review

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
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return _SessionLocal


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=_get_engine())
    logger.info("DB tables created or already exist")


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


def store_review(repo: str, pr_number: int, installation_id: int, review_body: str) -> None:
    """Insert a review row. Truncate body if needed to fit DB (e.g. 64k)."""
    max_body = 65535  # conservative text limit
    body = review_body if len(review_body) <= max_body else review_body[: max_body - 3] + "..."
    with session_scope() as session:
        session.add(
            Review(
                repo=repo,
                pr_number=pr_number,
                installation_id=installation_id,
                review_body=body,
            )
        )
    logger.info("Stored review for %s PR #%s", repo, pr_number)
