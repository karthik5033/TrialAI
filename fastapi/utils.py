"""Utility functions for bias analysis, SHAP, Fairlearn, and Gemini API calls."""

import os
import json
import numpy as np
import pandas as pd
import joblib
import pickle
import shap
import httpx
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model_file(file_path: str):
    """Load a sklearn-compatible model from .pkl or .joblib."""
    # Try joblib first (most common for sklearn), then pickle as fallback
    try:
        return joblib.load(file_path)
    except Exception:
        with open(file_path, "rb") as f:
            return pickle.load(f)


# ─── Data Preparation ────────────────────────────────────────────────────────

def prepare_dataset(df: pd.DataFrame, target_column: str, sensitive_attrs: list[str]):
    """Prepare dataset: encode categoricals, separate X/y, preserve sensitive values."""
    df = df.dropna().copy()

    # Preserve raw sensitive values before encoding
    raw_sensitive = {}
    for attr in sensitive_attrs:
        col_match = next((c for c in df.columns if c.lower() == attr.lower()), None)
        if col_match:
            raw_sensitive[col_match] = df[col_match].astype(str).values

    # Separate features and target
    feature_cols = [c for c in df.columns if c != target_column]
    X = df[feature_cols].copy()
    y = df[target_column].values

    # Encode categoricals
    encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Encode target if categorical
    if df[target_column].dtype == "object" or df[target_column].dtype.name == "category":
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
        encoders["__target__"] = le_target

    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, y, feature_cols, raw_sensitive, encoders


# ─── Fairlearn Metrics ───────────────────────────────────────────────────────

def compute_fairlearn_metrics(y_true, y_pred, sensitive_features):
    """Compute real fairness metrics using Fairlearn."""
    sensitive = np.array(sensitive_features, dtype=str)

    # Bin if too many unique values
    try:
        if len(np.unique(sensitive)) > 10:
            numeric_vals = pd.to_numeric(sensitive, errors="coerce")
            if not np.isnan(numeric_vals).all():
                sensitive = pd.qcut(numeric_vals, q=4, duplicates="drop").astype(str).values
    except Exception:
        pass

    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)

    # Selection rates per group
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "accuracy": accuracy_score,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    group_rates = mf.by_group["selection_rate"]
    di_ratio = (group_rates.min() / group_rates.max()) if group_rates.max() > 0 else 1.0

    # FPR/FNR per group
    try:
        mf_rates = MetricFrame(
            metrics={
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )
        fpr_by_group = mf_rates.by_group["false_positive_rate"].to_dict()
        fnr_by_group = mf_rates.by_group["false_negative_rate"].to_dict()
    except Exception:
        fpr_by_group = {}
        fnr_by_group = {}

    return {
        "demographic_parity_difference": round(abs(dp_diff), 4),
        "equalized_odds_difference": round(abs(eo_diff), 4),
        "disparate_impact_ratio": round(di_ratio, 4),
        "selection_rates": {str(k): round(v, 4) for k, v in group_rates.to_dict().items()},
        "accuracy_by_group": {str(k): round(v, 4) for k, v in mf.by_group["accuracy"].to_dict().items()},
        "fpr_by_group": {str(k): round(v, 4) for k, v in fpr_by_group.items()},
        "fnr_by_group": {str(k): round(v, 4) for k, v in fnr_by_group.items()},
        # Legacy format for frontend compatibility
        "demographic_parity": round(max(0, 1 - abs(dp_diff)), 4),
        "equal_opportunity": round(max(0, 1 - abs(eo_diff)), 4),
        "disparate_impact": round(di_ratio, 4),
    }


# ─── SHAP Analysis ───────────────────────────────────────────────────────────

def compute_shap_values(model, X: pd.DataFrame, feature_names: list, max_samples=300):
    """Compute real SHAP values. Auto-detects explainer type."""
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    sample_size = min(max_samples, len(X_arr))
    X_sample = X_arr[np.random.choice(len(X_arr), sample_size, replace=False)]

    model_name = type(model).__name__
    tree_types = {
        "RandomForestClassifier", "RandomForestRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "XGBClassifier", "XGBRegressor",
        "LGBMClassifier", "LGBMRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "AdaBoostClassifier",
    }
    linear_types = {
        "LogisticRegression", "LinearRegression", "Ridge", "Lasso",
        "SGDClassifier", "SGDRegressor", "ElasticNet",
    }

    try:
        if model_name in tree_types or hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_sample)
        elif model_name in linear_types:
            bg = X_sample[:min(50, len(X_sample))]
            explainer = shap.LinearExplainer(model, bg)
            sv = explainer.shap_values(X_sample)
        else:
            bg = shap.sample(pd.DataFrame(X_sample), min(50, len(X_sample)))
            predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            explainer = shap.KernelExplainer(predict_fn, bg)
            sv = explainer.shap_values(X_sample, nsamples=100)
    except Exception as e:
        print(f"[SHAP] Primary explainer failed ({e}), using KernelExplainer fallback")
        bg = shap.sample(pd.DataFrame(X_sample), min(30, len(X_sample)))
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        explainer = shap.KernelExplainer(predict_fn, bg)
        sv = explainer.shap_values(X_sample, nsamples=50)

    # Handle multi-class / multi-output SHAP values
    if isinstance(sv, list):
        sv = sv[-1]  # Take positive/last class
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]  # (samples, features, classes) → take last class
    if sv.ndim != 2:
        sv = sv.reshape(len(X_sample), -1)

    mean_abs = np.abs(sv).mean(axis=0)
    max_val = float(mean_abs.max()) if mean_abs.max() > 0 else 1.0
    normalized = mean_abs / max_val

    results = []
    for i, col in enumerate(feature_names[:len(normalized)]):
        results.append({
            "feature": col.replace("_", " ").title(),
            "importance": round(float(normalized[i]), 4),
            "raw_shap": round(float(mean_abs[i]), 6),
        })

    results.sort(key=lambda x: x["importance"], reverse=True)
    return results[:10]


# ─── Proxy Variable Detection ────────────────────────────────────────────────

def detect_proxy_features(df: pd.DataFrame, sensitive_col: str, feature_cols: list):
    """Detect features highly correlated with the sensitive attribute."""
    proxies = []
    try:
        # Only include columns that exist in df
        cols_to_use = [c for c in feature_cols if c in df.columns and c != sensitive_col]
        if sensitive_col not in df.columns:
            return proxies

        df_enc = df[cols_to_use + [sensitive_col]].copy()
        for col in df_enc.select_dtypes(include=["object", "category", "string"]).columns:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
        df_enc = df_enc.apply(pd.to_numeric, errors="coerce").dropna()

        if len(df_enc) < 5 or sensitive_col not in df_enc.columns:
            return proxies

        for col in cols_to_use:
            if col in df_enc.columns:
                corr = df_enc[col].corr(df_enc[sensitive_col])
                if not np.isnan(corr) and abs(corr) > 0.25:
                    proxies.append({
                        "feature": col,
                        "correlation": round(abs(corr), 4),
                        "corr_with": sensitive_col,
                    })
    except Exception as e:
        print(f"[PROXY] Detection error: {e}")

    proxies.sort(key=lambda x: x["correlation"], reverse=True)
    return proxies


# ─── Gemini API ──────────────────────────────────────────────────────────────

async def call_groq(prompt: str) -> str:
    """Call Groq API (Llama 3.3 70B) as a fallback."""
    import os
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not configured for fallback.")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4000
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def call_gemini(prompt: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
    """Call Gemini API, fallback to Groq on failure (like 429 Rate Limit)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192},
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=body, params={"key": api_key})
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"[LLM] Gemini failed ({e}). Falling back to Groq...")
        return await call_groq(prompt)


async def analyze_code_with_llm(script_content: str, api_key: str) -> dict:
    """Use Gemini to analyze training code for bias-prone patterns."""
    prompt = f"""You are an ML fairness expert. Analyze this training script for potential sources of bias.

TRAINING SCRIPT:
```python
{script_content}
```

Return a JSON object (no markdown fences) with:
{{
  "issues": ["list of specific bias concerns found in the code"],
  "risk_level": "HIGH" or "MODERATE" or "LOW",
  "recommendations": ["list of specific mitigation recommendations"],
  "uses_sensitive_features": true/false,
  "has_class_balancing": true/false,
  "model_type_detected": "name of the model class used"
}}"""

    try:
        text = await call_gemini(prompt, api_key)
        # Extract JSON
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        return {
            "issues": [f"Code analysis failed: {str(e)}"],
            "risk_level": "UNKNOWN",
            "recommendations": [],
            "uses_sensitive_features": False,
            "has_class_balancing": False,
            "model_type_detected": "Unknown",
        }


def generate_fallback_training_script(
    model_type: str, dataset_path: str, target_column: str,
    sensitive_attrs: list, output_model_path: str
) -> str:
    """Generate a baseline training script when the user didn't upload one.
    Uses the same model class as the uploaded model so retrain stays consistent."""
    # Map common model names to import + constructor
    model_imports = {
        "LogisticRegression": ("from sklearn.linear_model import LogisticRegression", "LogisticRegression(max_iter=1000, random_state=42)"),
        "RandomForestClassifier": ("from sklearn.ensemble import RandomForestClassifier", "RandomForestClassifier(n_estimators=100, random_state=42)"),
        "GradientBoostingClassifier": ("from sklearn.ensemble import GradientBoostingClassifier", "GradientBoostingClassifier(n_estimators=100, random_state=42)"),
        "DecisionTreeClassifier": ("from sklearn.tree import DecisionTreeClassifier", "DecisionTreeClassifier(random_state=42)"),
        "ExtraTreesClassifier": ("from sklearn.ensemble import ExtraTreesClassifier", "ExtraTreesClassifier(n_estimators=100, random_state=42)"),
        "AdaBoostClassifier": ("from sklearn.ensemble import AdaBoostClassifier", "AdaBoostClassifier(n_estimators=100, random_state=42)"),
        "SVC": ("from sklearn.svm import SVC", "SVC(probability=True, random_state=42)"),
        "KNeighborsClassifier": ("from sklearn.neighbors import KNeighborsClassifier", "KNeighborsClassifier()"),
    }
    import_line, constructor = model_imports.get(
        model_type,
        ("from sklearn.ensemble import RandomForestClassifier", "RandomForestClassifier(n_estimators=100, random_state=42)")
    )

    return f'''"""Auto-generated training script for bias mitigation."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
{import_line}

# Load data
df = pd.read_csv("{dataset_path}")
df = df.dropna()

# Encode categoricals
for col in df.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Separate features and target
X = df.drop(columns=["{target_column}"])
y = df["{target_column}"]

# Train model
model = {constructor}
model.fit(X, y)

accuracy = accuracy_score(y, model.predict(X))
print(json.dumps({{"accuracy": round(accuracy, 4), "status": "success"}}))

# Save
joblib.dump(model, "{output_model_path}")
'''


async def generate_mitigated_code(
    script_content: str, analysis_results: dict, dataset_path: str,
    target_column: str, sensitive_attrs: list, output_model_path: str, api_key: str
) -> str:
    """Use Gemini to rewrite training script with fairness constraints."""
    prompt = f"""You are an ML fairness expert. Rewrite this training script to mitigate the detected bias.

ORIGINAL TRAINING SCRIPT:
```python
{script_content}
```

BIAS ANALYSIS RESULTS:
{json.dumps(analysis_results, indent=2)}

REQUIREMENTS FOR THE MODIFIED SCRIPT:
1. Read the dataset from: "{dataset_path}"
2. Target column: "{target_column}"
3. Sensitive attributes: {sensitive_attrs}
4. Apply these mitigation techniques where appropriate:
   - Sample reweighting using sklearn.utils.class_weight.compute_sample_weight
   - Remove or reduce influence of proxy features correlated with sensitive attributes
   - If possible, use fairlearn.reductions.ExponentiatedGradient with DemographicParity constraint
5. Save the retrained model to: "{output_model_path}" using joblib.dump()
6. Print a JSON line to stdout with format: {{"accuracy": 0.XX, "status": "success"}}
7. The script must be fully self-contained and runnable with `python script.py`
8. Keep the same model type/algorithm as the original when possible
9. Import all necessary packages at the top
10. Handle errors gracefully

Return ONLY the modified Python script, no explanations or markdown fences."""

    text = await call_gemini(prompt, api_key)
    # Strip markdown fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text
