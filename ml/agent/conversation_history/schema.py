"""Database schema definitions and SQLAlchemy setup for chat history storage."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ml.config.settings import settings
from ml.config.logger import get_logger
from .models import Base

logger = get_logger(__name__)

# Thread pool executor for running synchronous database operations
_executor = ThreadPoolExecutor(max_workers=2)

# Global engine and session factory
_engine = None
_SessionLocal = None


def get_engine():
    """
    Get or create SQLAlchemy engine.

    Returns
    -------
    Engine
        SQLAlchemy engine instance.
    """
    global _engine
    if _engine is None:
        # Build connection string for SQLAlchemy
        conn_string = settings.get_postgres_connection_string(async_driver=False)
        _engine = create_engine(
            conn_string,
            poolclass=StaticPool,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("Created SQLAlchemy engine")
    return _engine


def get_session_factory() -> sessionmaker:
    """
    Get or create SQLAlchemy session factory.

    Returns
    -------
    sessionmaker
        SQLAlchemy session factory.
    """
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Created SQLAlchemy session factory")
    return _SessionLocal


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Get database session context manager.

    Yields
    ------
    Session
        SQLAlchemy database session.

    Examples
    --------
    >>> with get_db() as db:
    ...     # Use db session
    ...     pass
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _create_schema_sync() -> None:
    """Create the database schema synchronously using SQLAlchemy."""
    logger.info("Ensuring database schema exists...")
    engine = get_engine()

    # Ensure pgvector extension exists
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception as e:
        # Log warning but continue, as we might not have permissions or it might exist
        logger.warning(f"Could not create pgvector extension (might require superuser): {e}")

    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ensured successfully")


async def ensure_schema_exists() -> None:
    """
    Ensure database tables exist, creating them if necessary.

    This function creates the chat_sessions and chat_messages tables
    if they don't already exist. Runs synchronous SQLAlchemy operations
    in a thread pool executor to avoid blocking the event loop.

    Raises
    ------
    Exception
        If database connection or table creation fails.
    """
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _create_schema_sync)
    except Exception as e:
        logger.error(f"Failed to ensure database schema: {e}")
        raise
