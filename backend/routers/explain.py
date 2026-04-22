"""
AI Courtroom v2.0 — Explainability Router.

POST /api/explain/{session_id}/replay
POST /api/explain/{session_id}/counterfactual
POST /api/explain/{session_id}/narrative
"""

import logging
import os
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models import AnalysisSession
from backend.services.bias_engine import load_model
from backend.services.explainability import (
    get_single_prediction_shap,
    generate_counterfactuals,
    generate_llm_narrative
)

logger = logging.getLogger("courtroom.explain_router")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))

router = APIRouter(prefix="/explain", tags=["explainability"])


class ExplainRequest(BaseModel):
    row_index: int
    target_column: str
    sensitive_attributes: list[str]

class NarrativeRequest(BaseModel):
    shap_data: dict
    mode: str = "manager"


async def _load_session_data(session_id: str, db: AsyncSession):
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format.")

    session = await db.get(AnalysisSession, sid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    session_dir = UPLOAD_DIR / session_id
    dataset_path = session_dir / "dataset.csv"
    model_path = session_dir / "model.pkl"

    if not dataset_path.exists() or not model_path.exists():
        raise HTTPException(status_code=404, detail="Dataset or model file missing.")

    df = pd.read_csv(str(dataset_path))
    model = load_model(str(model_path))

    return df, model


@router.post("/{session_id}/replay")
async def replay_decision(
    session_id: str,
    body: ExplainRequest,
    db: AsyncSession = Depends(get_db),
):
    """Decision Replay: SHAP waterfall for a specific prediction."""
    df, model = await _load_session_data(session_id, db)
    try:
        res = get_single_prediction_shap(
            model=model,
            df=df,
            target_column=body.target_column,
            sensitive_attrs=body.sensitive_attributes,
            row_index=body.row_index
        )
        return res
    except Exception as e:
        logger.exception("Replay failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/counterfactual")
async def counterfactuals(
    session_id: str,
    body: ExplainRequest,
    db: AsyncSession = Depends(get_db),
):
    """Counterfactuals: DiCE-ML minimum flip path."""
    df, model = await _load_session_data(session_id, db)
    try:
        res = generate_counterfactuals(
            model=model,
            df=df,
            target_column=body.target_column,
            sensitive_attrs=body.sensitive_attributes,
            row_index=body.row_index
        )
        return res
    except Exception as e:
        logger.exception("Counterfactuals failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/narrative")
async def narrative(
    session_id: str,
    body: NarrativeRequest,
):
    """LLM Explainability Narrative."""
    try:
        text = generate_llm_narrative(body.shap_data, body.mode)
        return {"narrative": text, "mode": body.mode}
    except Exception as e:
        logger.exception("Narrative failed")
        raise HTTPException(status_code=500, detail=str(e))
