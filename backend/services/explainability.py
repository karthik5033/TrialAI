"""
AI Courtroom v2.0 — Phase 5: Explainability Engine.

Implements:
1. Decision Replay: Single-prediction SHAP breakdown.
2. Counterfactuals: DiCE-ML to find the minimum changes needed to flip a prediction.
3. LLM Narratives: Claude translates SHAP values into Technical, Manager, or Legal explanations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backend.services.bias_engine import prepare_dataset, match_features
from backend.services.courtroom import _get_client, MODEL

logger = logging.getLogger("courtroom.explainability")


# ═══════════════════════════════════════════════════════════════════════════════
#  Decision Replay (SHAP for a single instance)
# ═══════════════════════════════════════════════════════════════════════════════

def get_single_prediction_shap(
    model: Any,
    df: pd.DataFrame,
    target_column: str,
    sensitive_attrs: list[str],
    row_index: int,
) -> dict:
    """
    Compute SHAP values for a single specific row.
    Returns the base value, prediction, and feature contributions.
    """
    import shap
    X, _, feature_cols, _, _ = prepare_dataset(df, target_column, sensitive_attrs)
    X, feature_cols = match_features(model, X, feature_cols)

    if row_index < 0 or row_index >= len(X):
        raise ValueError(f"Invalid row_index {row_index}. Max is {len(X)-1}.")

    instance = X.iloc[[row_index]]
    
    # Try TreeExplainer, then Linear, then Kernel
    explainer = None
    shap_values = None
    expected_value = 0.5
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(instance)
        expected_value = explainer.expected_value
    except Exception:
        try:
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(instance)
            expected_value = explainer.expected_value
        except Exception:
            try:
                # KernelExplainer is slow, use a small background
                background = shap.sample(X, 50)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                # For classification, taking the proba of class 1
                shap_values = explainer.shap_values(instance)[1]
                expected_value = explainer.expected_value[1]
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")
                raise ValueError("Could not compute SHAP values for this model.")

    # Handle multi-class / binary SHAP output formats
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]  # Class 1
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]
    else:
        # If it's 2D array, we might need just the first row
        if len(shap_vals := np.array(shap_values).shape) > 1 and shap_vals[0] == 1:
             shap_vals = np.array(shap_values)[0]
        else:
             shap_vals = shap_values
            
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[0]) if len(expected_value) > 0 else 0.5

    # Make sure shap_vals is 1D
    if len(np.array(shap_vals).shape) > 1:
        shap_vals = np.array(shap_vals)[0]

    contributions = []
    for i, col in enumerate(feature_cols):
        val = float(instance[col].iloc[0])
        sv = float(shap_vals[i])
        contributions.append({
            "feature": col,
            "value": val,
            "contribution": sv
        })
        
    # Sort by absolute contribution magnitude
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    
    prediction = int(model.predict(instance)[0])
    proba = model.predict_proba(instance)[0][1] if hasattr(model, "predict_proba") else float(prediction)

    return {
        "row_index": row_index,
        "prediction": prediction,
        "probability": float(proba),
        "base_value": float(expected_value),
        "contributions": contributions
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Counterfactuals (DiCE)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_counterfactuals(
    model: Any,
    df: pd.DataFrame,
    target_column: str,
    sensitive_attrs: list[str],
    row_index: int,
    total_CFs: int = 3,
) -> dict:
    """
    Use DiCE-ML to find counterfactuals (how to flip the prediction).
    """
    import dice_ml
    from dice_ml.utils import helpers

    # We need the full dataframe including target for DiCE Data object
    X, y, feature_cols, raw_sensitive, _ = prepare_dataset(df, target_column, sensitive_attrs)
    
    # Reconstruct a clean dataframe for DiCE
    dice_df = X.copy()
    dice_df[target_column] = y
    
    if row_index < 0 or row_index >= len(X):
        raise ValueError(f"Invalid row_index {row_index}.")

    instance = X.iloc[[row_index]]
    original_prediction = int(model.predict(instance)[0])
    desired_class = 1 if original_prediction == 0 else 0

    # DiCE setup
    continuous_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col]) and len(X[col].unique()) > 10]
    
    try:
        d = dice_ml.Data(
            dataframe=dice_df,
            continuous_features=continuous_features,
            outcome_name=target_column
        )
        
        # DiCE expects a specific model interface
        m = dice_ml.Model(model=model, backend="sklearn")
        
        # Use random method as it's the most robust for arbitrary sklearn models
        exp = dice_ml.Dice(d, m, method="random")
        
        dice_exp = exp.generate_counterfactuals(
            instance, 
            total_CFs=total_CFs, 
            desired_class=desired_class,
            features_to_vary="all"
        )
        
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        if cf_df is None or len(cf_df) == 0:
            return {"original_prediction": original_prediction, "desired_class": desired_class, "counterfactuals": []}
            
        cfs = []
        for i in range(len(cf_df)):
            changes = {}
            for col in feature_cols:
                orig_val = float(instance[col].iloc[0])
                new_val = float(cf_df[col].iloc[i])
                if abs(orig_val - new_val) > 1e-5:
                    changes[col] = {"from": orig_val, "to": new_val}
            
            cfs.append({
                "cf_index": i,
                "changes": changes
            })
            
        return {
            "original_prediction": original_prediction,
            "desired_class": desired_class,
            "counterfactuals": cfs
        }
        
    except Exception as e:
        logger.error(f"DiCE counterfactual generation failed: {e}")
        raise ValueError(f"Counterfactual generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM Narratives (Explainability)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_llm_narrative(
    shap_data: dict,
    mode: str,  # "technical", "manager", "legal"
) -> str:
    """
    Call Claude to translate SHAP values into a human-readable narrative.
    """
    client = _get_client()

    contributions_text = "\n".join(
        f"- {c['feature']}: value={c['value']}, impact={c['contribution']:.4f}"
        for c in shap_data["contributions"][:10]  # Top 10 features
    )

    system_prompts = {
        "technical": "You are a Data Scientist explaining a model prediction to another engineer. Focus on raw SHAP values, log-odds, base values, and feature interactions.",
        "manager": "You are an AI Product Manager explaining a prediction to a business stakeholder. Focus on business impact, which user attributes drove the decision, and what it means practically. Avoid heavy math.",
        "legal": "You are an AI Compliance Officer. Explain the prediction focusing on fairness, potential disparate impact, reliance on proxy variables, and alignment with anti-discrimination principles."
    }

    prompt = f"""Explain this specific model prediction.

Prediction output: {shap_data['prediction']} (Probability: {shap_data['probability']:.4f})
Base expected value: {shap_data['base_value']:.4f}

Top driving features (SHAP values):
{contributions_text}

Provide a 2-3 paragraph narrative explanation in the requested persona."""

    logger.info("Calling Claude for %s narrative…", mode)
    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=system_prompts.get(mode, system_prompts["manager"]),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    
    return response.content[0].text.strip()
