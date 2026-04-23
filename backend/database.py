"""
AI Courtroom v2.0 — Async Database Configuration.

Provides:
  - async SQLAlchemy engine (PostgreSQL via asyncpg, fallback to SQLite via aiosqlite)
  - async session factory
  - get_db() FastAPI dependency
  - init_db() startup initialiser
  - lifespan() context manager for FastAPI
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from backend.models import Base

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logger = logging.getLogger("courtroom.db")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_db_path = os.path.join(BASE_DIR, "courtroom.db").replace("\\", "/")

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite+aiosqlite:///{default_db_path}",
)

# asyncpg does not accept "postgresql://" — it must be "postgresql+asyncpg://"
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    # pool_size / max_overflow only apply to non-SQLite engines
    **({} if "sqlite" in DATABASE_URL else {"pool_size": 10, "max_overflow": 20}),
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session and ensure it is closed afterwards."""
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Create all tables that don't yet exist."""
    logger.info("Initialising database tables …")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready.")


# ---------------------------------------------------------------------------
# Health check helper
# ---------------------------------------------------------------------------

async def check_db_connection() -> bool:
    """Return True if the DB is reachable."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("DB health-check failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# FastAPI lifespan (used by main.py)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):  # noqa: ANN001
    """Startup / shutdown lifecycle hook for FastAPI."""
    await init_db()
    logger.info("AI Courtroom v2.0 backend started.")
    yield
    await engine.dispose()
    logger.info("AI Courtroom v2.0 backend shut down.")
