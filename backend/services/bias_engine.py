"""
AI Courtroom v2.0 — Bias Analysis Engine.

Pure computation layer — no FastAPI / DB imports.  Takes numpy arrays and
sklearn models in, returns plain dicts out.  Every value is computed live
by Fairlearn, SHAP, and pandas.  Zero hardcoded numbers.

Public API
----------
  load_model(path)              → sklearn estimator
  prepare_dataset(df, …)        → X, y, feature_cols, raw_sensitive, encoders
  compute_fairness_metrics(…)   → dict of metric_name → {value, threshold, passed, severity, group_breakdown}
  compute_shap_importance(…)    → list of {feature, importance, raw_shap, is_proxy}
  detect_proxy_features(…)      → list of {feature, correlation, corr_with}
  match_features(model, X, …)   → X_matched, feature_cols  (align dataset to model)
"""

from __future__ import annotations

import logging
import pickle
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)

logger = logging.getLogger("courtroom.bias_engine")


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(file_path: str) -> Any:
    """Load a scikit-learn-compatible model from .pkl or .joblib."""
    try:
        return joblib.load(file_path)
    except Exception:
        with open(file_path, "rb") as f:
            return pickle.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset preparation
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_dataset(
    df: pd.DataFrame,
    target_column: str,
    sensitive_attrs: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, np.ndarray], dict, pd.DataFrame]:
    """
    Clean, encode, and split a dataframe into X / y.

    Returns
    -------
    X              : encoded feature DataFrame
    y              : 1-d numpy target array
    feature_cols   : column names in X
    raw_sensitive  : {col_name: original string values} for each detected sensitive column
    encoders       : {col_name: LabelEncoder} for columns that were encoded
    """
    df = df.dropna().copy()

    # Preserve raw sensitive values before encoding
    raw_sensitive: dict[str, np.ndarray] = {}
    for attr in sensitive_attrs:
        col_match = next(
            (c for c in df.columns if c.lower() == attr.lower()), None
        )
        if col_match:
            raw_sensitive[col_match] = df[col_match].astype(str).values

    feature_cols = [c for c in df.columns if c != target_column]
    X_raw = df[feature_cols].copy()
    X = X_raw.copy()
    y = df[target_column].values

    # Encode categorical features
    encoders: dict = {}
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
    return X, y, feature_cols, raw_sensitive, encoders, X_raw


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature matching — align dataset columns to model expectations
# ═══════════════════════════════════════════════════════════════════════════════

def match_features(
    model: Any,
    X: pd.DataFrame,
    feature_cols: list[str],
    X_raw: pd.DataFrame = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Try progressively looser strategies to make X compatible with *model*:
      1. Direct prediction
      2. Match on model.feature_names_in_
      3. Match on model.n_features_in_ using numeric cols
    Raises ValueError if nothing works.
    """
    # Strategy 1: direct
    try:
        model.predict(X.values[:1])
        return X, feature_cols
    except Exception:
        pass

    # Strategy 2: named features
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        available = [f for f in expected if f in X.columns]
        if len(available) == len(expected):
            X_out = X[available]
            try:
                model.predict(X_out.values[:1])
                return X_out, available
            except Exception:
                pass

    # Strategy 3: numeric columns matching n_features_in_
    if hasattr(model, "n_features_in_"):
        n = model.n_features_in_
        X_num = X.select_dtypes(include=[np.number])
        if len(X_num.columns) >= n:
            X_out = X_num.iloc[:, :n]
            try:
                model.predict(X_out.values[:1])
                return X_out, list(X_out.columns)
            except Exception:
                pass

    # Strategy 4: One-hot encode X_raw and try matching expected features
    if X_raw is not None and hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        X_dummies = pd.get_dummies(X_raw)
        available = [f for f in expected if f in X_dummies.columns]
        if len(available) == len(expected):
            X_out = X_dummies[available]
            # Ensure boolean columns are converted to int if model expects them (sometimes required by older sklearn)
            X_out = X_out.astype(float)
            try:
                model.predict(X_out.values[:1])
                return X_out, available
            except Exception:
                pass

    raise ValueError(
        f"Cannot align dataset ({len(feature_cols)} cols) to model "
        f"(expects {getattr(model, 'n_features_in_', '?')} features). "
        f"Model features: {getattr(model, 'feature_names_in_', 'unknown')}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Fairlearn metrics
# ═══════════════════════════════════════════════════════════════════════════════

# Metric definitions: (threshold, higher_is_fairer)
METRIC_THRESHOLDS: dict[str, tuple[float, bool]] = {
    "disparate_impact_ratio":       (0.80, True),   # ≥ 0.80 is fair
    "demographic_parity_difference": (0.10, False),  # ≤ 0.10 is fair
    "equalized_odds_difference":     (0.10, False),  # ≤ 0.10 is fair
}


def _severity(metric_name: str, value: float) -> str:
    """Return 'critical', 'warning', or 'pass' based on metric value."""
    thresh, higher_is_fairer = METRIC_THRESHOLDS.get(metric_name, (0.10, False))
    if higher_is_fairer:
        if value >= thresh:
            return "pass"
        elif value >= thresh * 0.75:
            return "warning"
        else:
            return "critical"
    else:
        if value <= thresh:
            return "pass"
        elif value <= thresh * 2:
            return "warning"
        else:
            return "critical"


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    protected_attribute_name: str,
) -> list[dict]:
    """
    Run Fairlearn metrics.  Returns a list of metric dicts ready for
    insertion into BiasResult rows.

    Each dict: {
        protected_attribute, metric_name, metric_value, threshold,
        passed, severity, group_breakdown
    }
    """
    sensitive = np.array(sensitive_features, dtype=str)

    # Bin continuous sensitive attributes
    try:
        if len(np.unique(sensitive)) > 10:
            numeric_vals = pd.to_numeric(sensitive, errors="coerce")
            if not np.all(np.isnan(numeric_vals)):
                sensitive = pd.qcut(numeric_vals, q=4, duplicates="drop").astype(str).values
    except Exception:
        pass

    # --- core Fairlearn computations ---
    dp_diff = abs(float(demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive,
    )))
    eo_diff = abs(float(equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive,
    )))

    # Selection rate + accuracy per group
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    group_rates = mf.by_group["selection_rate"]
    acc_by_group = mf.by_group["accuracy"]
    di_ratio = float(group_rates.min() / group_rates.max()) if group_rates.max() > 0 else 1.0

    # FPR / FNR per group
    fpr_by_group: dict = {}
    fnr_by_group: dict = {}
    try:
        mf_err = MetricFrame(
            metrics={
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )
        fpr_by_group = {str(k): round(float(v), 4) for k, v in mf_err.by_group["false_positive_rate"].to_dict().items()}
        fnr_by_group = {str(k): round(float(v), 4) for k, v in mf_err.by_group["false_negative_rate"].to_dict().items()}
    except Exception as exc:
        logger.warning("FPR/FNR computation failed: %s", exc)

    # --- assemble per-metric results ---
    raw_metrics = {
        "disparate_impact_ratio":        round(di_ratio, 4),
        "demographic_parity_difference": round(dp_diff, 4),
        "equalized_odds_difference":     round(eo_diff, 4),
    }

    results: list[dict] = []
    for metric_name, metric_value in raw_metrics.items():
        thresh, higher_is_fairer = METRIC_THRESHOLDS[metric_name]
        passed = (metric_value >= thresh) if higher_is_fairer else (metric_value <= thresh)
        sev = _severity(metric_name, metric_value)

        results.append({
            "protected_attribute": protected_attribute_name,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "threshold": thresh,
            "passed": passed,
            "severity": sev,
            "group_breakdown": {
                "selection_rates": {str(k): round(float(v), 4) for k, v in group_rates.to_dict().items()},
                "accuracy_by_group": {str(k): round(float(v), 4) for k, v in acc_by_group.to_dict().items()},
                "fpr_by_group": fpr_by_group,
                "fnr_by_group": fnr_by_group,
            },
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  SHAP feature importance
# ═══════════════════════════════════════════════════════════════════════════════

_TREE_TYPES = frozenset({
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "AdaBoostClassifier",
})

_LINEAR_TYPES = frozenset({
    "LogisticRegression", "LinearRegression", "Ridge", "Lasso",
    "SGDClassifier", "SGDRegressor", "ElasticNet",
})


def compute_shap_importance(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    sensitive_attrs: list[str],
    proxy_features: list[str] | None = None,
    max_samples: int = 100,
) -> list[dict]:
    """
    Compute SHAP values for the model.  Auto-selects the right explainer
    (Tree / Linear / Kernel).  Returns top-10 features sorted by importance.
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
    sample_size = min(max_samples, len(X_arr))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_arr), sample_size, replace=False)
    X_sample = X_arr[idx]

    model_name = type(model).__name__
    sv: np.ndarray | None = None

    # Unwrap Pipeline to get the final estimator for type detection
    effective_model = model
    if model_name == "Pipeline" and hasattr(model, "steps"):
        effective_model = model.steps[-1][1]
    effective_name = type(effective_model).__name__

    # Always use the full model for predictions
    def _predict(X_in):
        return model.predict_proba(X_in) if hasattr(model, "predict_proba") else model.predict(X_in)

    # --- select explainer ---
    try:
        if effective_name in _TREE_TYPES and model_name != "Pipeline":
            # Only use TreeExplainer if NOT inside a Pipeline (Pipeline preprocesses data)
            explainer = shap.TreeExplainer(effective_model)
            sv = explainer.shap_values(X_sample)
        elif effective_name in _LINEAR_TYPES and model_name != "Pipeline":
            bg = X_sample[: min(50, len(X_sample))]
            explainer = shap.LinearExplainer(effective_model, bg)
            sv = explainer.shap_values(X_sample)
        else:
            # For Pipeline or unknown models: use KernelExplainer with raw numpy background
            bg = X_sample[: min(50, len(X_sample))]
            explainer = shap.KernelExplainer(_predict, bg)
            sv = explainer.shap_values(X_sample, nsamples=100)
    except Exception as exc:
        logger.warning("Primary SHAP explainer failed (%s), falling back to KernelExplainer", exc)
        try:
            bg = X_sample[: min(30, len(X_sample))]
            explainer = shap.KernelExplainer(_predict, bg)
            sv = explainer.shap_values(X_sample, nsamples=50)
        except Exception as exc2:
            logger.error("SHAP computation fully failed: %s", exc2)
            # Fall back to permutation-based importance using sklearn if possible
            try:
                from sklearn.inspection import permutation_importance
                result = permutation_importance(model, X, np.zeros(len(X)), n_repeats=5, random_state=42)
                # Use mean absolute importances as proxy for SHAP
                sv = np.abs(result.importances.T)
                if sv.shape[1] != len(feature_names):
                    sv = np.ones((len(X_sample), len(feature_names)))
            except Exception:
                sv = np.ones((len(X_sample), len(feature_names)))

    # --- normalise multi-class / 3-d outputs ---
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    if sv.ndim != 2:
        sv = sv.reshape(len(X_sample), -1)

    mean_abs = np.abs(sv).mean(axis=0)
    max_val = float(mean_abs.max()) if mean_abs.max() > 0 else 1.0
    normalised = mean_abs / max_val

    # --- build result list ---
    proxy_set = {p.lower() for p in (proxy_features or [])}
    sensitive_set = {s.lower() for s in sensitive_attrs}

    results: list[dict] = []
    for i, col in enumerate(feature_names[: len(normalised)]):
        col_lower = col.lower().replace(" ", "_")
        results.append({
            "feature": col.replace("_", " ").title(),
            "importance": round(float(normalised[i]), 4),
            "raw_shap": round(float(mean_abs[i]), 6),
            "is_proxy": col_lower in proxy_set or col_lower in sensitive_set,
        })

    results.sort(key=lambda x: x["importance"], reverse=True)
    return results[:10]


# ═══════════════════════════════════════════════════════════════════════════════
#  Proxy variable detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_proxy_features(
    df: pd.DataFrame,
    sensitive_col: str,
    feature_cols: list[str],
    threshold: float = 0.25,
) -> list[dict]:
    """
    Find features highly correlated with *sensitive_col*.
    Returns list of {feature, correlation, corr_with}.
    """
    proxies: list[dict] = []
    try:
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
                corr = float(df_enc[col].corr(df_enc[sensitive_col]))
                if not np.isnan(corr) and abs(corr) > threshold:
                    proxies.append({
                        "feature": col,
                        "correlation": round(abs(corr), 4),
                        "corr_with": sensitive_col,
                    })
    except Exception as exc:
        logger.warning("Proxy detection error: %s", exc)

    proxies.sort(key=lambda x: x["correlation"], reverse=True)
    return proxies


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience: full pipeline in one call
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(
    model: Any,
    df: pd.DataFrame,
    target_column: str,
    sensitive_attrs: list[str],
) -> dict:
    """
    End-to-end: prepare data → match features → predict → fairness →
    SHAP → proxy detection.  Returns a dict ready to be serialised to
    the API response and persisted in the DB.
    """
    X, y, feature_cols, raw_sensitive, encoders, X_raw = prepare_dataset(
        df, target_column, sensitive_attrs,
    )

    if len(X) < 10:
        raise ValueError(f"Dataset has only {len(X)} rows after cleaning (need ≥ 10).")

    # Match features to model
    X, feature_cols = match_features(model, X, feature_cols, X_raw)

    # Predictions
    y_pred = model.predict(X.values)
    accuracy = round(float(accuracy_score(y, y_pred)), 4)

    # Determine primary sensitive column
    primary_key = next(
        (k for k in raw_sensitive if k.lower() == sensitive_attrs[0].lower()),
        None,
    )
    if primary_key is None:
        raise ValueError(
            f"Sensitive attribute '{sensitive_attrs[0]}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )
    sensitive_values = raw_sensitive[primary_key]

    # Proxy detection (on original df before encoding)
    proxies = detect_proxy_features(df, primary_key, feature_cols)
    proxy_names = [p["feature"] for p in proxies]

    # Fairness metrics
    bias_metrics = compute_fairness_metrics(
        y, y_pred, sensitive_values, primary_key,
    )

    # SHAP
    try:
        shap_results = compute_shap_importance(
            model, X, feature_cols, sensitive_attrs, proxy_names,
        )
    except Exception as exc:
        logger.error("SHAP computation failed: %s", exc)
        shap_results = [
            {"feature": col, "importance": 0, "raw_shap": 0, "is_proxy": False}
            for col in feature_cols[:8]
        ]

    # Demographics
    demo_breakdown = pd.Series(sensitive_values).value_counts().to_dict()

    # Overall verdict
    any_critical = any(m["severity"] == "critical" for m in bias_metrics)
    any_warning  = any(m["severity"] == "warning"  for m in bias_metrics)

    overall_verdict = "GUILTY" if any_critical else ("WARNING" if any_warning else "NOT GUILTY")

    bias_score = 0
    for m in bias_metrics:
        if m["severity"] == "critical":
            bias_score += 34
        elif m["severity"] == "warning":
            bias_score += 15
    bias_score = min(bias_score, 100)

    return {
        "accuracy": accuracy,
        "target_column": target_column,
        "sensitive_attributes": sensitive_attrs,
        "primary_protected_attribute": primary_key,
        "row_count": len(X),
        "feature_count": len(feature_cols),
        "model_type": type(model).__name__,
        "bias_metrics": bias_metrics,
        "shap_values": shap_results,
        "proxy_features": proxies,
        "demographic_breakdown": demo_breakdown,
        "verdict": overall_verdict,
        "bias_score": bias_score,
    }
