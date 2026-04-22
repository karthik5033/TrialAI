"""
AI Courtroom v2.0 — Bias Analysis Router.

POST /api/analysis/run/{session_id}
    Trigger a full Fairlearn + SHAP analysis on the uploaded dataset & model.
    Accepts JSON body: { target_column: str, sensitive_attributes: [str] }
    Persists BiasResult rows.  Updates AnalysisSession status.

GET  /api/analysis/{session_id}
    Retrieve the analysis results (metrics, SHAP, proxies, verdict) for a
    completed session.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models import (
    AnalysisSession,
    BiasResult,
    BiasSeverity,
    SessionStatus,
)
from backend.services.bias_engine import load_model, run_full_analysis

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("courtroom.analysis")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))

router = APIRouter(prefix="/analysis", tags=["analysis"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class RunAnalysisRequest(BaseModel):
    target_column: str = Field(
        ..., description="Name of the target/label column in the CSV."
    )
    sensitive_attributes: list[str] = Field(
        ...,
        min_length=1,
        description="List of column names to treat as protected attributes.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  POST /api/analysis/run/{session_id}
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/run/{session_id}")
async def run_analysis(
    session_id: str,
    body: RunAnalysisRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Run the full bias analysis pipeline on the uploaded data + model.
    Computes Fairlearn metrics, SHAP feature importance, and proxy
    detection.  Persists every metric as a BiasResult row.
    """
    # ── 1. Validate session ─────────────────────────────────────────────────
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format.")

    session = await db.get(AnalysisSession, sid)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    if session.model_filename is None:
        raise HTTPException(
            status_code=422,
            detail="No model has been uploaded for this session. "
                   "Upload a model first via POST /api/upload/model.",
        )

    # ── 2. Locate files on disk ─────────────────────────────────────────────
    session_dir = UPLOAD_DIR / session_id
    dataset_path = session_dir / "dataset.csv"
    model_path = session_dir / "model.pkl"

    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk.")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found on disk.")

    # ── 3. Mark session as running ──────────────────────────────────────────
    session.status = SessionStatus.running
    await db.commit()

    # ── 4. Load data + model ────────────────────────────────────────────────
    try:
        df = pd.read_csv(str(dataset_path))
    except Exception as exc:
        session.status = SessionStatus.failed
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {exc}")

    if body.target_column not in df.columns:
        session.status = SessionStatus.failed
        await db.commit()
        raise HTTPException(
            status_code=422,
            detail=f"Target column '{body.target_column}' not found. "
                   f"Available: {list(df.columns)}",
        )

    try:
        model = load_model(str(model_path))
    except Exception as exc:
        session.status = SessionStatus.failed
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")

    # ── 5. Run analysis ─────────────────────────────────────────────────────
    try:
        result = run_full_analysis(
            model=model,
            df=df,
            target_column=body.target_column,
            sensitive_attrs=body.sensitive_attributes,
        )
    except Exception as exc:
        session.status = SessionStatus.failed
        await db.commit()
        logger.exception("Analysis failed for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    # ── 6. Persist BiasResult rows ──────────────────────────────────────────
    for metric in result["bias_metrics"]:
        severity_value = metric["severity"]
        # Map string to BiasSeverity enum (handle "pass" → pass_)
        if severity_value == "pass":
            sev_enum = BiasSeverity.pass_
        elif severity_value == "warning":
            sev_enum = BiasSeverity.warning
        else:
            sev_enum = BiasSeverity.critical

        bias_row = BiasResult(
            session_id=sid,
            protected_attribute=metric["protected_attribute"],
            metric_name=metric["metric_name"],
            metric_value=metric["metric_value"],
            threshold=metric["threshold"],
            passed=metric["passed"],
            severity=sev_enum,
            group_breakdown=metric["group_breakdown"],
        )
        db.add(bias_row)

    # ── 7. Update session status ────────────────────────────────────────────
    session.status = SessionStatus.complete
    session.row_count = result["row_count"]
    session.feature_count = result["feature_count"]
    await db.commit()

    logger.info(
        "Analysis complete: session=%s verdict=%s accuracy=%s metrics=%d",
        session_id,
        result["verdict"],
        result["accuracy"],
        len(result["bias_metrics"]),
    )

    # ── 8. Return full result ───────────────────────────────────────────────
    return {
        "session_id": session_id,
        "status": "complete",
        "accuracy": result["accuracy"],
        "model_type": result["model_type"],
        "row_count": result["row_count"],
        "feature_count": result["feature_count"],
        "target_column": result["target_column"],
        "sensitive_attributes": result["sensitive_attributes"],
        "primary_protected_attribute": result["primary_protected_attribute"],
        "verdict": result["verdict"],
        "bias_score": result["bias_score"],
        "bias_metrics": result["bias_metrics"],
        "shap_values": result["shap_values"],
        "proxy_features": result["proxy_features"],
        "demographic_breakdown": result["demographic_breakdown"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  POST /api/full-analysis (Old UI Compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/full-analysis", tags=["upload", "analysis"])
async def full_analysis(
    file: UploadFile = File(...),
    model_file: UploadFile = File(...),
    target_column: str = Form(...),
    sensitive_attributes: str = Form(...),
    training_script: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Monolithic endpoint for the old UI: uploads files and runs analysis in one go.
    """
    import io
    
    # 1. Create Session
    session_id = uuid.uuid4()
    session = AnalysisSession(
        id=session_id,
        dataset_filename=file.filename,
        model_filename=model_file.filename,
        script_filename=training_script.filename if training_script else None,
        status=SessionStatus.running,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # 2. Save Files
    session_dir = UPLOAD_DIR / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    
    csv_bytes = await file.read()
    (session_dir / "dataset.csv").write_bytes(csv_bytes)
    
    model_bytes = await model_file.read()
    (session_dir / "model.pkl").write_bytes(model_bytes)
    
    if training_script:
        script_bytes = await training_script.read()
        (session_dir / "script.py").write_bytes(script_bytes)

    # 3. Run Analysis
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
        model = load_model(str(session_dir / "model.pkl"))
        
        sensitive_list = [s.strip() for s in sensitive_attributes.split(",")]
        
        result = run_full_analysis(
            model=model,
            df=df,
            target_column=target_column,
            sensitive_attrs=sensitive_list,
        )
        
        # Persist results
        for metric in result["bias_metrics"]:
            severity_value = metric["severity"]
            if severity_value == "pass":
                sev_enum = BiasSeverity.pass_
            elif severity_value == "warning":
                sev_enum = BiasSeverity.warning
            else:
                sev_enum = BiasSeverity.critical

            bias_row = BiasResult(
                session_id=session_id,
                protected_attribute=metric["protected_attribute"],
                metric_name=metric["metric_name"],
                metric_value=metric["metric_value"],
                threshold=metric["threshold"],
                passed=metric["passed"],
                severity=sev_enum,
                group_breakdown=metric["group_breakdown"],
            )
            db.add(bias_row)

        session.status = SessionStatus.complete
        session.row_count = result["row_count"]
        session.feature_count = result["feature_count"]
        await db.commit()
        
        return {
            "session_id": str(session_id),
            "status": "complete",
            **result
        }
        
    except Exception as exc:
        session.status = SessionStatus.failed
        await db.commit()
        logger.exception("Full analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /api/analysis/{session_id}
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}")
async def get_analysis(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve previously computed analysis results for a session.
    Returns metrics from the DB plus session metadata.
    """
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format.")

    # Eager-load bias_results
    stmt = (
        select(AnalysisSession)
        .where(AnalysisSession.id == sid)
        .options(selectinload(AnalysisSession.bias_results))
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    if session.status == SessionStatus.pending:
        return {
            "session_id": session_id,
            "status": "pending",
            "message": "Analysis has not been started yet. "
                       "Call POST /api/analysis/run/{session_id} first.",
        }

    if session.status == SessionStatus.running:
        return {
            "session_id": session_id,
            "status": "running",
            "message": "Analysis is currently in progress.",
        }

    if session.status == SessionStatus.failed:
        return {
            "session_id": session_id,
            "status": "failed",
            "message": "Analysis failed. Check server logs for details.",
        }

    # ── Build response from persisted BiasResult rows ───────────────────────
    bias_metrics = []
    for br in session.bias_results:
        severity_str = br.severity.value if hasattr(br.severity, "value") else str(br.severity)
        # Map "pass_" back to "pass" for the API response
        if severity_str == "pass_" or severity_str == "pass":
            severity_str = "pass"

        bias_metrics.append({
            "id": str(br.id),
            "protected_attribute": br.protected_attribute,
            "metric_name": br.metric_name,
            "metric_value": br.metric_value,
            "threshold": br.threshold,
            "passed": br.passed,
            "severity": severity_str,
            "group_breakdown": br.group_breakdown,
            "created_at": br.created_at.isoformat() if br.created_at else None,
        })

    return {
        "session_id": session_id,
        "status": "complete",
        "dataset_filename": session.dataset_filename,
        "model_filename": session.model_filename,
        "row_count": session.row_count,
        "feature_count": session.feature_count,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "bias_metrics": bias_metrics,
    }
