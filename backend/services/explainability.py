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
from backend.services.llm import get_llm_client

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
    X, _, feature_cols, _, _, _X_raw = prepare_dataset(df, target_column, sensitive_attrs)
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
    X, y, feature_cols, raw_sensitive, _, X_raw = prepare_dataset(df, target_column, sensitive_attrs)
    
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
    counterfactual_data: dict | None = None,
    sensitive_attributes: list[str] | None = None,
) -> dict:
    """
    Call Claude to translate SHAP values into a structured JSON narrative.
    """
    import json
    client = _get_client()

    system_prompt = """You are the Explainability Engine inside an AI accountability platform called “AI Courtroom”.

You receive REAL mathematical evidence about a SINGLE model prediction:
- Input row: raw feature values for one individual case
- Prediction: predicted class/score and probability
- SHAP results: base value, per-feature contribution, sorted by absolute impact
- Counterfactuals: DiCE-ML “minimum change” scenarios that flip or significantly change the prediction
- Fairness context: protected attributes for this individual and any relevant group-level bias metrics

Your job is to:
1) Explain WHY this prediction happened, grounded ONLY in the evidence provided.
2) Highlight WHICH features mattered most and HOW they pushed the decision up or down.
3) Describe WHAT would need to change in this individual’s features to flip or materially change the outcome, using the counterfactuals.
4) Adapt tone, level of detail, and framing to the requested audience MODE: “technical”, “manager”, or “legal”.

STRICT RULES:
- Never invent metrics, numbers, or features that are not present in the input JSON.
- If something is missing (e.g., no counterfactuals), explicitly say so instead of guessing.
- Be concise: aim for around 3–6 short paragraphs plus bullet points where helpful.
- Always clearly name the top 3 contributing features and explain their effect direction (pushing towards or away from the final outcome).
- If protected attributes are among the top contributors, clearly flag this and describe the potential fairness concern without making legal conclusions you cannot support from the data.
- Output MUST be valid JSON in the exact schema specified by the user message."""

    mode_snippets = {
        "technical": """MODE: technical
Audience: ML engineers and data scientists.
Style: precise, compact, and technical, but still readable.
Focus: SHAP math, feature contributions, decision boundary intuition, probability behavior, and any trade-offs between accuracy and fairness directly visible in this single case.""",

        "manager": """MODE: manager
Audience: product managers and business leaders.
Style: plain English, no equations, 5th–8th grade reading level.
Focus: top 3 drivers of the outcome, how favorable/unfavorable this decision is, and 2–3 practical actions or policy tweaks this suggests.
Avoid ML jargon; talk in terms of customers, risk, and fairness.""",

        "legal": """MODE: legal
Audience: legal, risk, and compliance teams.
Style: formal, cautious, and documentation-friendly.
Focus: transparency of the decision, role of any protected attributes, alignment with principles of non-discrimination and explainability (e.g., under the EU AI Act), and what should be documented for audit.
Do NOT make definitive legal conclusions; instead, describe factual observations about this prediction."""
    }

    payload = {
        "prediction": {
            "predicted_label": shap_data.get("prediction"),
            "probability": shap_data.get("probability")
        },
        "shap": {
            "base_value": shap_data.get("base_value"),
            "top_features": shap_data.get("contributions", [])[:10]
        },
        "counterfactuals": counterfactual_data.get("counterfactuals", []) if counterfactual_data else [],
        "fairness_context": {
            "protected_attributes_flagged": sensitive_attributes or []
        }
    }

    user_prompt = f"""You are generating an explainability report for a SINGLE prediction.

{mode_snippets.get(mode, mode_snippets['manager'])}

Here is the evidence in JSON format:

```json
{json.dumps(payload, indent=2)}
```

Return a SINGLE JSON object with this schema:

```json
{{
  "mode": "{mode}",
  "headline": "One-sentence headline summary in plain language",
  "verdict": "Very short phrase about how favorable or unfavorable the outcome is for this individual",
  "why_summary": "2–4 sentence plain-language explanation of why the model made this prediction, grounded in SHAP.",
  "top_factors": [
    {{
      "feature": "feature_name",
      "original_value": "as string",
      "direction": "increases_risk | decreases_risk | pushes_towards_approval | pushes_towards_rejection",
      "relative_importance": "high | medium | low",
      "commentary": "1–2 sentence explanation of how this feature influenced the decision."
    }}
  ],
  "counterfactual_insights": {{
    "summary": "2–3 sentence explanation of what would need to change to flip or significantly change the outcome, based ONLY on provided counterfactuals.",
    "examples": [
      {{
        "description": "Short description of the scenario in human terms.",
        "changed_features": {{ "feature": "from → to" }},
        "new_prediction": "label / score from payload"
      }}
    ]
  }},
  "fairness_and_risk": {{
    "uses_protected_attributes": "yes | no | unknown",
    "protected_attribute_impact": "If any protected attributes are among top contributors, describe their role. Otherwise, say they are not among the main drivers.",
    "notes": "Optional short note tying to group-level fairness metrics if provided."
  }},
  "recommended_actions": [
    "Short bullet-style sentence tailored to the MODE, suggesting next steps or cautions."
  ]
}}
```

Generate the JSON object now, following the schema exactly and without adding extra top-level keys."""

    logger.info("Calling LLM for %s narrative…", mode)
    llm = get_llm_client()
    raw = llm.chat(
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=1500,
        temperature=0.3,
    )
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM JSON: %s\nRaw: %s", exc, raw[:500])
        return {"error": "Failed to generate structured explainability report.", "raw_text": raw}
