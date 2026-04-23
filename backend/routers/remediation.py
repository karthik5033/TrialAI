"""
AI Courtroom v2.0 — Remediation Router.

POST /api/remediation/run/{session_id}
    Starts remediation as a background task. Returns run_id immediately.
    Ollama (qwen2.5-coder) is tried first; falls back to Groq API on timeout/failure.

GET  /api/remediation/status/{run_id}
    Poll the status of a background remediation run.

GET  /api/remediation/{session_id}
    Retrieve persisted remediation results for a session.

GET  /api/remediation/{session_id}/download
    Download the mitigated model file.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models import (
    AnalysisSession,
    RemediationRun,
    RemediationStatus,
    SessionStatus,
)
from backend.services.bias_engine import load_model
from backend.services.remediation_pipeline import run_local_remediation
from backend.services.ollama_client import is_ollama_available, list_ollama_models

# Validation set for strategies
STRATEGIES = {"reweighing", "threshold_adjustment", "fairness_constraint"}

logger = logging.getLogger("courtroom.remediation_router")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))

router = APIRouter(prefix="/remediation", tags=["remediation"])

# ---------------------------------------------------------------------------
# In-memory job store for background tasks
# ---------------------------------------------------------------------------
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _set_job(run_id: str, data: dict) -> None:
    with _jobs_lock:
        _jobs[run_id] = data


def _get_job(run_id: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(run_id)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class RunRemediationRequest(BaseModel):
    strategy: str = Field(default="auto", description="Mitigation strategy or 'auto'.")
    target_column: str = Field(..., description="Target/label column name.")
    sensitive_attributes: list[str] = Field(..., description="Protected attribute columns.")


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_remediation_worker(
    run_id: str,
    session_id: str,
    model: Any,
    df: pd.DataFrame,
    target_column: str,
    sensitive_attrs: list[str],
    strategy: str,
    session_dir: Path,
    script_content: str | None,
) -> None:
    """Executes remediation in a background thread. Updates _jobs[run_id]."""
    try:
        _set_job(run_id, {**_get_job(run_id), "status": "running", "step": "Starting Ollama…"})

        ollama_ok = is_ollama_available()
        logger.info("Ollama available: %s", ollama_ok)

        if ollama_ok:
            _set_job(run_id, {**_get_job(run_id), "step": "Running local qwen2.5-coder pipeline…"})

        if strategy in ["auto", "optimize"]:
            _set_job(run_id, {**_get_job(run_id), "step": "Optimizing tradeoffs (testing 3 strategies)…"})
            import shutil
            
            best_result = None
            best_score = -float('inf')
            
            for cand in ["reweighing", "threshold_adjustment", "fairness_constraint"]:
                logger.info("Optimizing: Trying strategy %s", cand)
                try:
                    res = run_local_remediation(
                        model=model,
                        df=df,
                        target_column=target_column,
                        sensitive_attrs=sensitive_attrs,
                        session_dir=session_dir,
                        strategy=cand,
                        script_content=script_content,
                        timeout=180,
                    )
                    
                    # Score the result: combine accuracy and DIR closeness to 1
                    acc = res["mitigated_accuracy"]
                    dir_val = res["mitigated_dir"]
                    
                    # Compute score (penalize DIR deviation from 1.0)
                    dir_score = 1.0 - abs(1.0 - dir_val)
                    score = (acc * 0.4) + (dir_score * 0.6)
                    
                    logger.info("Strategy %s score: %.4f (Acc: %.4f, DIR: %.4f)", cand, score, acc, dir_val)
                    
                    if score > best_score:
                        best_score = score
                        best_result = res
                        # Backup the best artifacts
                        if (session_dir / "mitigated_model.pkl").exists():
                            shutil.copy(session_dir / "mitigated_model.pkl", session_dir / "best_model.pkl")
                        if (session_dir / "mitigated_script.py").exists():
                            shutil.copy(session_dir / "mitigated_script.py", session_dir / "best_script.py")
                            
                except Exception as e:
                    logger.error("Strategy %s failed during optimization: %s", cand, e)
            
            if best_result is None:
                raise ValueError("All optimization strategies failed.")
                
            # Restore the best artifacts
            if (session_dir / "best_model.pkl").exists():
                shutil.copy(session_dir / "best_model.pkl", session_dir / "mitigated_model.pkl")
            if (session_dir / "best_script.py").exists():
                shutil.copy(session_dir / "best_script.py", session_dir / "mitigated_script.py")
                
            result = best_result
        else:
            result = run_local_remediation(
                model=model,
                df=df,
                target_column=target_column,
                sensitive_attrs=sensitive_attrs,
                session_dir=session_dir,
                strategy=strategy,
                script_content=script_content,
                timeout=180,
            )

        # Save mitigated model
        mitigated_model_path = session_dir / "mitigated_model.pkl"
        if not mitigated_model_path.exists() and result.get("mitigated_model"):
            try:
                joblib.dump(result["mitigated_model"], str(mitigated_model_path))
            except Exception as exc:
                logger.error("Failed to save mitigated model: %s", exc)

        _set_job(run_id, {
            **_get_job(run_id),
            "status": "complete",
            "step": "Done",
            "result": {
                "session_id": session_id,
                "remediation_id": run_id,
                "strategy": result["strategy"],
                "model_type": result["model_type"],
                "original_accuracy": result["original_accuracy"],
                "mitigated_accuracy": result["mitigated_accuracy"],
                "original_dir": result["original_dir"],
                "mitigated_dir": result["mitigated_dir"],
                "improvements": result["improvements"],
                "before": {m["metric_name"]: m["original_value"] for m in result.get("improvements", [])},
                "after": {m["metric_name"]: m["mitigated_value"] for m in result.get("improvements", [])},
                "improvement": {
                    m["metric_name"]: (
                        0.0 if not m["original_value"] else (
                            (m["mitigated_value"] - m["original_value"]) / abs(m["original_value"]) * 100
                            if "difference" not in m["metric_name"] else 
                            (m["original_value"] - m["mitigated_value"]) / abs(m["original_value"]) * 100
                        )
                    )
                    for m in result.get("improvements", [])
                },
                "new_accuracy": result.get("mitigated_accuracy"),
                "script_diff": result.get("script_diff", ""),
                "modified_script": result.get("modified_script", ""),
                "original_script": result.get("original_script", ""),
                "patch_applied": result.get("patch_applied", False),
                "patch_description": result.get("patch_description", ""),
                "llm_explanation": result.get("llm_explanation"),
                "reevaluation_report": result.get("reevaluation_report"),
                "strategy_info": result.get("strategy_info", {}),
                "script_execution": result.get("script_execution", {}),
                "all_passed": result.get("all_passed", False),
                "ollama_used": ollama_ok,
            }
        })
        logger.info("Background remediation complete: run_id=%s strategy=%s", run_id, result["strategy"])

    except Exception as exc:
        logger.exception("Background remediation failed: run_id=%s", run_id)
        _set_job(run_id, {
            **(_get_job(run_id) or {}),
            "status": "failed",
            "step": "Failed",
            "error": str(exc),
        })


# ═══════════════════════════════════════════════════════════════════════════════
#  POST /api/remediation/run/{session_id}  — starts background job
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/run/{session_id}")
async def run_remediation_endpoint(
    session_id: str,
    body: RunRemediationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Starts remediation as a background task. Returns run_id immediately (fast).
    Poll /api/remediation/status/{run_id} to get progress and results.
    """
    # ── 1. Validate session_id ─────────────────────────────────────────────
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

    effective_strategy = body.strategy if body.strategy in STRATEGIES else "reweighing"

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

    # ── 3. Load optional script ────────────────────────────────────────────
    script_path = session_dir / "script.py"
    script_content = None
    if script_path.exists():
        try:
            script_content = script_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to read script.py: %s", exc)

    # ── 4. Create DB record ────────────────────────────────────────────────
    run_row = RemediationRun(
        session_id=sid,
        strategy_used=effective_strategy,
        status=RemediationStatus.running,
    )
    db.add(run_row)
    await db.commit()
    await db.refresh(run_row)
    run_id = str(run_row.id)

    # ── 5. Register job in memory ──────────────────────────────────────────
    _set_job(run_id, {
        "status": "pending",
        "step": "Queued",
        "session_id": session_id,
        "result": None,
        "error": None,
    })

    # ── 6. Launch background thread ────────────────────────────────────────
    t = threading.Thread(
        target=_run_remediation_worker,
        args=(run_id, session_id, model, df, body.target_column,
              body.sensitive_attributes, effective_strategy, session_dir, script_content),
        daemon=True,
    )
    t.start()
    logger.info("Remediation background thread started: run_id=%s session=%s", run_id, session_id)

    # Return immediately — frontend polls /status/{run_id}
    return {
        "run_id": run_id,
        "session_id": session_id,
        "status": "pending",
        "message": "Remediation started. Poll /api/remediation/status/{run_id} for progress.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /api/remediation/status/{run_id}  — poll job progress
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/status/{run_id}")
async def get_remediation_status(run_id: str):
    """Poll the status of a background remediation run."""
    job = _get_job(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    return {
        "run_id": run_id,
        "status": job["status"],        # pending | running | complete | failed
        "step": job.get("step", ""),
        "error": job.get("error"),
        "result": job.get("result"),    # populated when status == complete
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
        "llm_explanation": run.llm_explanation,
        "reevaluation_report": run.reevaluation_report,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GET /api/remediation/{session_id}/download/{download_type}
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}/download/{download_type}")
async def download_mitigated_artifacts(
    session_id: str,
    download_type: str,
    db: AsyncSession = Depends(get_db),
):
    """Download the mitigated model .pkl file or the mitigated script .py file."""
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format.")

    if download_type not in ["model", "script"]:
        raise HTTPException(status_code=400, detail="Invalid download type. Must be 'model' or 'script'.")

    filename = "mitigated_model.pkl" if download_type == "model" else "mitigated_script.py"
    file_path = UPLOAD_DIR / session_id / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No {download_type} found. Run remediation first.",
        )

    return FileResponse(
        str(file_path),
        filename=filename,
        media_type="application/octet-stream" if download_type == "model" else "text/x-python",
    )
