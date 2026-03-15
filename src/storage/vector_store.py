"""pgvector-backed vector store for code chunk similarity search."""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Set

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from src import config

from src.intelligence.ast.function_extract import FunctionChunk
from src.storage.models import Base

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CodeChunk(Base):
    """Stored function-level code chunk with its embedding vector."""

    __tablename__ = "code_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    func_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(config.EMBEDDING_DIMENSION), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint("repo_id", "content_hash", name="uq_code_chunks_repo_hash"),
        Index("ix_code_chunks_repo_path", "repo_id", "path"),
    )


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimilarMatch:
    path: str
    func_name: Optional[str]
    start_line: int
    end_line: int
    chunk_text: str
    content_hash: str
    score: float


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_vector_db() -> None:
    """Enable pgvector extension and create the code_chunks table + HNSW index.

    Safe to call multiple times (IF NOT EXISTS).
    Must be called *after* the engine is available (i.e. after init_db or as part of it).
    If the embedding column exists without dimensions (legacy schema), drops and recreates the table.
    """
    from src.storage.database import _get_engine

    logger.debug("[Vector] init_vector_db: enabling pgvector and creating code_chunks table + HNSW index")
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(bind=engine, tables=[CodeChunk.__table__])

    def _create_hnsw_index(conn):
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_code_chunks_embedding_hnsw "
            "ON code_chunks USING hnsw (embedding vector_cosine_ops)"
        ))
        conn.commit()

    try:
        with engine.connect() as conn:
            _create_hnsw_index(conn)
    except Exception as e:
        err_msg = str(e).lower()
        if "column does not have dimensions" in err_msg or "dimensions" in err_msg:
            logger.warning(
                "[Vector] code_chunks.embedding has no dimensions (legacy schema). "
                "Dropping and recreating table with vector(%d).",
                config.EMBEDDING_DIMENSION,
            )
            with engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS code_chunks CASCADE"))
                conn.commit()
            Base.metadata.create_all(bind=engine, tables=[CodeChunk.__table__])
            with engine.connect() as conn:
                _create_hnsw_index(conn)
        else:
            raise

    logger.info("Vector DB initialized (code_chunks table + HNSW index)")


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    repo_id: str,
    chunks: List[FunctionChunk],
    embeddings: List[List[float]],
) -> None:
    """Insert or update code chunks with their embeddings.

    On conflict (repo_id, content_hash) updates path, lines, embedding, and timestamp.
    """
    if not chunks:
        logger.debug("[Vector upsert] repo_id=%s: no chunks, skip", repo_id)
        return

    logger.debug(
        "[Vector upsert] repo_id=%s: upserting %d chunk(s) paths=%s",
        repo_id, len(chunks), list({c.path for c in chunks}),
    )
    from src.storage.database import session_scope

    with session_scope() as session:
        for chunk, emb in zip(chunks, embeddings):
            existing = (
                session.query(CodeChunk)
                .filter(
                    CodeChunk.repo_id == repo_id,
                    CodeChunk.content_hash == chunk.content_hash,
                )
                .first()
            )
            if existing:
                logger.debug(
                    "[Vector upsert]   update path=%s lines %d-%d hash=%s...",
                    chunk.path, chunk.start_line, chunk.end_line, chunk.content_hash[:12],
                )
                existing.path = chunk.path
                existing.func_name = chunk.name
                existing.start_line = chunk.start_line
                existing.end_line = chunk.end_line
                existing.chunk_text = chunk.text
                existing.embedding = emb
                existing.updated_at = datetime.now(timezone.utc)
            else:
                logger.debug(
                    "[Vector upsert]   insert path=%s lines %d-%d func=%s hash=%s...",
                    chunk.path, chunk.start_line, chunk.end_line, chunk.name, chunk.content_hash[:12],
                )
                session.add(CodeChunk(
                    repo_id=repo_id,
                    path=chunk.path,
                    func_name=chunk.name,
                    content_hash=chunk.content_hash,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    chunk_text=chunk.text,
                    embedding=emb,
                ))

    logger.debug("[Vector upsert] repo_id=%s: completed %d chunk(s)", repo_id, len(chunks))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_similar(
    repo_id: str,
    query_embedding: List[float],
    exclude_hashes: Set[str],
    exclude_path: Optional[str],
    top_k: int,
) -> List[SimilarMatch]:
    """Find the top-k most similar code chunks by cosine distance.

    Excludes chunks whose content_hash is in exclude_hashes (self-match prevention)
    and optionally excludes chunks from exclude_path (same-file exclusion).
    """
    from src.storage.database import session_scope

    logger.debug(
        "[Vector search] repo_id=%s top_k=%d exclude_hashes=%d exclude_path=%s embedding_len=%d",
        repo_id, top_k, len(exclude_hashes), exclude_path, len(query_embedding),
    )
    with session_scope() as session:
        q = (
            session.query(
                CodeChunk,
                CodeChunk.embedding.cosine_distance(query_embedding).label("distance"),
            )
            .filter(CodeChunk.repo_id == repo_id)
        )

        if exclude_hashes:
            q = q.filter(CodeChunk.content_hash.notin_(exclude_hashes))

        if exclude_path:
            q = q.filter(CodeChunk.path != exclude_path)

        rows = q.order_by("distance").limit(top_k).all()

        results = []
        for chunk, distance in rows:
            score = 1.0 - distance
            results.append(SimilarMatch(
                path=chunk.path,
                func_name=chunk.func_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                chunk_text=chunk.chunk_text,
                content_hash=chunk.content_hash,
                score=score,
            ))
            logger.debug(
                "[Vector search]   match: path=%s lines %d-%d func=%s score=%.4f",
                chunk.path, chunk.start_line, chunk.end_line, chunk.func_name, score,
            )

    logger.debug(
        "[Vector search] repo_id=%s: returned %d result(s) (top_k=%d, excluded %d hash(es), exclude_path=%s)",
        repo_id, len(results), top_k, len(exclude_hashes), exclude_path,
    )
    return results


__all__ = [
    "CodeChunk",
    "SimilarMatch",
    "init_vector_db",
    "upsert_chunks",
    "search_similar",
]
