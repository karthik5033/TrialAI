"""
AI Courtroom v2.0 — FastAPI Application Entry Point.

Wires up:
  - async lifespan (DB init on startup, engine dispose on shutdown)
  - CORS middleware for Next.js frontend at localhost:3000
  - /api/upload router
  - /api/health endpoint with real DB connectivity check
"""

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import lifespan, check_db_connection

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-24s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("courtroom.main")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Courtroom v2.0",
    description=(
        "Real-time AI bias detection, adversarial courtroom simulation, "
        "and automated remediation platform. Powered by Fairlearn, SHAP, "
        "AIF360, DiCE-ML, and Claude claude-sonnet-4-20250514."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Next.js dev server and common local origins
# ---------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
from backend.routers.upload import router as upload_router  # noqa: E402
from backend.routers.analysis import router as analysis_router  # noqa: E402
from backend.routers.courtroom import router as courtroom_router  # noqa: E402
from backend.routers.remediation import router as remediation_router  # noqa: E402
from backend.routers.reports import router as reports_router  # noqa: E402
from backend.routers.sessions import router as sessions_router  # noqa: E402
from backend.routers.explain import router as explain_router  # noqa: E402

app.include_router(upload_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(courtroom_router, prefix="/api")
app.include_router(remediation_router, prefix="/api")
app.include_router(reports_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(explain_router, prefix="/api")


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["system"])
async def health():
    """
    Verify the backend is alive and the database is reachable.
    Returns real connectivity status, not a hardcoded value.
    """
    db_ok = await check_db_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "db": "connected" if db_ok else "unreachable",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0",
    }


# ---------------------------------------------------------------------------
# Uvicorn runner (python -m backend.main)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )
