"""
AI Courtroom v2.0 — Remediation Router.

POST /api/remediation/run/{session_id}
    Apply a mitigation strategy, retrain, compare before/after metrics.
    Persists RemediationRun to DB and saves the mitigated model to disk.

GET  /api/remediation/{session_id}
    Retrieve persisted remediation results for a session.

GET  /api/remediation/{session_id}/download
    Download the mitigated model file.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models import (
    AnalysisSession,
    RemediationRun,
    RemediationStatus,
    SessionStatus,
)
from backend.services.bias_engine import load_model
from backend.services.remediation import run_remediation, STRATEGIES

logger = logging.getLogger("courtroom.remediation_router")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))

router = APIRouter(prefix="/remediation", tags=["remediation"])


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class RunRemediationRequest(BaseModel):
    target_column: str = Field(..., description="Target/label column name.")
    sensitive_attributes: list[str] = Field(
        ..., min_length=1, description="Protected attribute column names."
    )
    strategy: str = Field(
        default="reweighing",
        description="Mitigation strategy: reweighing | threshold_adjustment | fairness_constraint",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  POST /api/remediation/run/{session_id}
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/run/{session_id}")
async def run_remediation_endpoint(
    session_id: str,
    body: RunRemediationRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Apply a bias mitigation strategy, retrain the model, and compare
    before/after fairness metrics.
    """
    # ── 1. Validate ─────────────────────────────────────────────────────────
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format.")

    session = await db.get(AnalysisSession, sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    if session.status != SessionStatus.complete:
        raise HTTPException(
            status_code=422,
            detail=f"Session status is '{session.status.value}'. Run analysis first.",
        )

    if body.strategy not in STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown strategy '{body.strategy}'. "
                   f"Available: {list(STRATEGIES.keys())}",
        )

    # ── 2. Load files ──────────────────────────────────────────────────────
    session_dir = UPLOAD_DIR / session_id
    dataset_path = session_dir / "dataset.csv"
    model_path = session_dir / "model.pkl"

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk.")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found on disk.")

    try:
        df = pd.read_csv(str(dataset_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {exc}")

    if body.target_column not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Target column '{body.target_column}' not found. "
                   f"Available: {list(df.columns)}",
        )

    try:
        model = load_model(str(model_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")

    # ── 3. Create pending DB record ─────────────────────────────────────────
    run_row = RemediationRun(
        session_id=sid,
        strategy_used=body.strategy,
        status=RemediationStatus.running,
    )
    db.add(run_row)
    await db.commit()
    await db.refresh(run_row)

    # ── 4. Run remediation ──────────────────────────────────────────────────
    try:
        result = run_remediation(
            model=model,
            df=df,
            target_column=body.target_column,
            sensitive_attrs=body.sensitive_attributes,
            strategy=body.strategy,
        )
    except Exception as exc:
        run_row.status = RemediationStatus.failed
        await db.commit()
        logger.exception("Remediation failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Remediation failed: {exc}")

    # ── 5. Save mitigated model to disk ─────────────────────────────────────
    mitigated_model_path = session_dir / "mitigated_model.pkl"
    try:
        joblib.dump(result["mitigated_model"], str(mitigated_model_path))
    except Exception as exc:
        logger.error("Failed to save mitigated model: %s", exc)

    # ── 6. Update DB record ─────────────────────────────────────────────────
    run_row.original_dir = result["original_dir"]
    run_row.mitigated_dir = result["mitigated_dir"]
    run_row.original_accuracy = result["original_accuracy"]
    run_row.mitigated_accuracy = result["mitigated_accuracy"]
    run_row.script_diff = result["script_diff"]
    run_row.status = RemediationStatus.complete
    await db.commit()
    await db.refresh(run_row)

    logger.info(
        "Remediation complete: session=%s strategy=%s DIR %.4f→%.4f acc %.4f→%.4f",
        session_id, body.strategy,
        result["original_dir"], result["mitigated_dir"],
        result["original_accuracy"], result["mitigated_accuracy"],
    )

    # ── 7. Return result (exclude the model object) ─────────────────────────
    return {
        "session_id": session_id,
        "remediation_id": str(run_row.id),
        "strategy": result["strategy"],
        "model_type": result["model_type"],
        "original_accuracy": result["original_accuracy"],
        "mitigated_accuracy": result["mitigated_accuracy"],
        "original_dir": result["original_dir"],
        "mitigated_dir": result["mitigated_dir"],
        "improvements": result["improvements"],
        "script_diff": result["script_diff"],
        "all_passed": result["all_passed"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /api/remediation/{session_id}
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}")
async def get_remediation(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve the most recent remediation run for a session."""
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format.")

    stmt = (
        select(RemediationRun)
        .where(RemediationRun.session_id == sid)
        .order_by(RemediationRun.created_at.desc())
    )
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if run is None:
        return {
            "session_id": session_id,
            "status": "no_remediation",
            "message": "No remediation has been run for this session.",
        }

    return {
        "session_id": session_id,
        "remediation_id": str(run.id),
        "strategy": run.strategy_used,
        "status": run.status.value if hasattr(run.status, "value") else str(run.status),
        "original_dir": run.original_dir,
        "mitigated_dir": run.mitigated_dir,
        "original_accuracy": run.original_accuracy,
        "mitigated_accuracy": run.mitigated_accuracy,
        "script_diff": run.script_diff,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /api/remediation/{session_id}/download
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}/download")
async def download_mitigated_model(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Download the mitigated model .pkl file."""
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format.")

    mitigated_path = UPLOAD_DIR / session_id / "mitigated_model.pkl"
    if not mitigated_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No mitigated model found. Run remediation first.",
        )

    return FileResponse(
        str(mitigated_path),
        filename="mitigated_model.pkl",
        media_type="application/octet-stream",
    )
