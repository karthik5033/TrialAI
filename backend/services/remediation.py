"""
AI Courtroom v2.0 — Bias Remediation Engine.

Applies real mitigation strategies to retrain the model and re-evaluate
fairness metrics.  Three strategies:

  1. reweighing       — compute sample weights inversely proportional to
                        group × label frequency, retrain with sample_weight
  2. threshold_adjustment — find per-group classification thresholds that
                        equalise selection rates (post-processing)
  3. fairness_constraint — use Fairlearn ExponentiatedGradient with a
                        DemographicParity constraint (in-processing)

Every function runs real sklearn / Fairlearn code on the actual data.
"""

from __future__ import annotations

import difflib
import logging
import textwrap
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from backend.services.bias_engine import (
    compute_fairness_metrics,
    compute_shap_importance,
    detect_proxy_features,
    prepare_dataset,
    match_features,
)

logger = logging.getLogger("courtroom.remediation")


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 1: Reweighing
# ═══════════════════════════════════════════════════════════════════════════════

def _reweigh(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    sensitive: np.ndarray,
) -> Any:
    """
    Clone the model, compute per-sample weights based on the joint
    distribution of (group, label), and retrain.
    """
    new_model = clone(model)

    # Build combined group string for weight computation
    combined = np.array([f"{s}_{l}" for s, l in zip(sensitive, y)])
    weights = compute_sample_weight(class_weight="balanced", y=combined)

    try:
        new_model.fit(X.values, y, sample_weight=weights)
    except TypeError:
        # Model doesn't support sample_weight — train without
        logger.warning(
            "%s does not support sample_weight; training unweighted.",
            type(model).__name__,
        )
        new_model.fit(X.values, y)

    return new_model


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 2: Threshold Adjustment (post-processing)
# ═══════════════════════════════════════════════════════════════════════════════

class _ThresholdAdjustedModel:
    """Wraps a model and applies per-group thresholds to predict_proba."""

    def __init__(self, base_model: Any, thresholds: dict[str, float], sensitive: np.ndarray):
        self.base_model = base_model
        self.thresholds = thresholds
        self._sensitive = sensitive
        # Copy attributes sklearn expects
        if hasattr(base_model, "n_features_in_"):
            self.n_features_in_ = base_model.n_features_in_
        if hasattr(base_model, "feature_names_in_"):
            self.feature_names_in_ = base_model.feature_names_in_
        if hasattr(base_model, "classes_"):
            self.classes_ = base_model.classes_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.base_model, "predict_proba"):
            return self.base_model.predict(X)

        proba = self.base_model.predict_proba(X)[:, 1]
        preds = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            group = str(self._sensitive[i]) if i < len(self._sensitive) else "default"
            thresh = self.thresholds.get(group, 0.5)
            preds[i] = 1 if proba[i] >= thresh else 0
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict_proba(X)


def _threshold_adjust(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    sensitive: np.ndarray,
) -> Any:
    """
    Find per-group thresholds that equalise selection rates.
    Uses a grid search over threshold values per group.
    """
    if not hasattr(model, "predict_proba"):
        logger.warning("Model lacks predict_proba — falling back to reweighing.")
        return _reweigh(model, X, y, sensitive)

    proba = model.predict_proba(X.values)[:, 1]
    groups = np.unique(sensitive)

    # Target selection rate = overall mean
    target_rate = float(np.mean(y))

    thresholds: dict[str, float] = {}
    for group in groups:
        mask = sensitive == group
        group_proba = proba[mask]
        # Find threshold that gives selection rate closest to target
        best_thresh = 0.5
        best_diff = float("inf")
        for t in np.arange(0.1, 0.9, 0.02):
            sel_rate = float(np.mean(group_proba >= t))
            diff = abs(sel_rate - target_rate)
            if diff < best_diff:
                best_diff = diff
                best_thresh = float(t)
        thresholds[str(group)] = round(best_thresh, 2)

    logger.info("Threshold adjustment thresholds: %s", thresholds)
    return _ThresholdAdjustedModel(model, thresholds, sensitive)


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 3: Fairness Constraint (ExponentiatedGradient)
# ═══════════════════════════════════════════════════════════════════════════════

def _fairness_constraint(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    sensitive: np.ndarray,
) -> Any:
    """
    Use Fairlearn ExponentiatedGradient with DemographicParity constraint.
    """
    try:
        from fairlearn.reductions import (
            DemographicParity,
            ExponentiatedGradient,
        )
    except ImportError:
        logger.warning("fairlearn.reductions not available — falling back to reweighing.")
        return _reweigh(model, X, y, sensitive)

    base = clone(model)
    constraint = DemographicParity()

    mitigator = ExponentiatedGradient(
        estimator=base,
        constraints=constraint,
        max_iter=50,
    )

    try:
        mitigator.fit(X.values, y, sensitive_features=sensitive)
        return mitigator
    except Exception as exc:
        logger.warning("ExponentiatedGradient failed (%s) — falling back to reweighing.", exc)
        return _reweigh(model, X, y, sensitive)


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    "reweighing": _reweigh,
    "threshold_adjustment": _threshold_adjust,
    "fairness_constraint": _fairness_constraint,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Script diff generation
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_script_diff(strategy: str, model_type: str, target_col: str, sensitive_attrs: list[str]) -> str:
    """Generate a unified diff showing what changed conceptually."""
    original = textwrap.dedent(f"""\
        # Original training script
        import pandas as pd
        from sklearn.metrics import accuracy_score
        import joblib

        df = pd.read_csv("dataset.csv")
        X = df.drop(columns=["{target_col}"])
        y = df["{target_col}"]

        from sklearn.ensemble import RandomForestClassifier
        model = {model_type}()
        model.fit(X, y)

        accuracy = accuracy_score(y, model.predict(X))
        joblib.dump(model, "model.pkl")
    """)

    if strategy == "reweighing":
        modified = textwrap.dedent(f"""\
            # Mitigated training script (reweighing)
            import pandas as pd
            import numpy as np
            from sklearn.metrics import accuracy_score
            from sklearn.utils.class_weight import compute_sample_weight
            import joblib

            df = pd.read_csv("dataset.csv")
            X = df.drop(columns=["{target_col}"])
            y = df["{target_col}"]
            sensitive = df["{sensitive_attrs[0]}"].astype(str).values

            # Compute fairness-aware sample weights
            combined = np.array([f"{{s}}_{{l}}" for s, l in zip(sensitive, y)])
            weights = compute_sample_weight(class_weight="balanced", y=combined)

            from sklearn.ensemble import RandomForestClassifier
            model = {model_type}()
            model.fit(X, y, sample_weight=weights)

            accuracy = accuracy_score(y, model.predict(X))
            joblib.dump(model, "mitigated_model.pkl")
        """)
    elif strategy == "threshold_adjustment":
        modified = textwrap.dedent(f"""\
            # Mitigated training script (threshold adjustment)
            import pandas as pd
            import numpy as np
            from sklearn.metrics import accuracy_score
            import joblib

            df = pd.read_csv("dataset.csv")
            X = df.drop(columns=["{target_col}"])
            y = df["{target_col}"]
            sensitive = df["{sensitive_attrs[0]}"].astype(str).values

            from sklearn.ensemble import RandomForestClassifier
            model = {model_type}()
            model.fit(X, y)

            # Post-processing: per-group threshold adjustment
            proba = model.predict_proba(X)[:, 1]
            target_rate = np.mean(y)
            for group in np.unique(sensitive):
                mask = sensitive == group
                # Find threshold that equalises selection rate
                best_t = 0.5
                for t in np.arange(0.1, 0.9, 0.02):
                    if abs(np.mean(proba[mask] >= t) - target_rate) < abs(np.mean(proba[mask] >= best_t) - target_rate):
                        best_t = t
                print(f"Group {{group}}: threshold = {{best_t:.2f}}")

            joblib.dump(model, "mitigated_model.pkl")
        """)
    else:  # fairness_constraint
        modified = textwrap.dedent(f"""\
            # Mitigated training script (fairness constraint)
            import pandas as pd
            from sklearn.metrics import accuracy_score
            from fairlearn.reductions import ExponentiatedGradient, DemographicParity
            import joblib

            df = pd.read_csv("dataset.csv")
            X = df.drop(columns=["{target_col}"])
            y = df["{target_col}"]
            sensitive = df["{sensitive_attrs[0]}"].astype(str).values

            from sklearn.ensemble import RandomForestClassifier
            base_model = {model_type}()

            # Apply Fairlearn ExponentiatedGradient with DemographicParity
            mitigator = ExponentiatedGradient(
                estimator=base_model,
                constraints=DemographicParity(),
                max_iter=50,
            )
            mitigator.fit(X, y, sensitive_features=sensitive)

            accuracy = accuracy_score(y, mitigator.predict(X))
            joblib.dump(mitigator, "mitigated_model.pkl")
        """)

    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="original_train.py",
        tofile="mitigated_train.py",
    ))
    return "".join(diff_lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Full remediation pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_remediation(
    model: Any,
    df: pd.DataFrame,
    target_column: str,
    sensitive_attrs: list[str],
    strategy: str,
) -> dict:
    """
    End-to-end remediation: prepare → original metrics → mitigate → new
    metrics → diff.

    Returns dict with original/mitigated metrics, accuracy, strategy info.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES.keys())}")

    # ── Prepare data ────────────────────────────────────────────────────────
    X, y, feature_cols, raw_sensitive, encoders = prepare_dataset(df, target_column, sensitive_attrs)

    if len(X) < 10:
        raise ValueError(f"Dataset has only {len(X)} rows after cleaning.")

    X, feature_cols = match_features(model, X, feature_cols)

    primary_key = next(
        (k for k in raw_sensitive if k.lower() == sensitive_attrs[0].lower()), None,
    )
    if primary_key is None:
        raise ValueError(f"Sensitive attribute '{sensitive_attrs[0]}' not found.")
    sensitive_values = raw_sensitive[primary_key]

    # ── Original metrics ────────────────────────────────────────────────────
    y_pred_original = model.predict(X.values)
    original_accuracy = round(float(accuracy_score(y, y_pred_original)), 4)
    original_metrics = compute_fairness_metrics(y, y_pred_original, sensitive_values, primary_key)

    # Extract original DIR
    original_dir = next(
        (m["metric_value"] for m in original_metrics if m["metric_name"] == "disparate_impact_ratio"),
        0.0,
    )

    # ── Apply mitigation ────────────────────────────────────────────────────
    logger.info("Applying strategy '%s' to %s…", strategy, type(model).__name__)
    mitigate_fn = STRATEGIES[strategy]
    mitigated_model = mitigate_fn(model, X, y, sensitive_values)

    # ── Mitigated metrics ───────────────────────────────────────────────────
    y_pred_mitigated = mitigated_model.predict(X.values)
    mitigated_accuracy = round(float(accuracy_score(y, y_pred_mitigated)), 4)
    mitigated_metrics = compute_fairness_metrics(y, y_pred_mitigated, sensitive_values, primary_key)

    mitigated_dir = next(
        (m["metric_value"] for m in mitigated_metrics if m["metric_name"] == "disparate_impact_ratio"),
        0.0,
    )

    # ── Compute improvements ────────────────────────────────────────────────
    improvements: list[dict] = []
    for orig in original_metrics:
        mit = next((m for m in mitigated_metrics if m["metric_name"] == orig["metric_name"]), None)
        if mit:
            improvements.append({
                "metric_name": orig["metric_name"],
                "original_value": orig["metric_value"],
                "mitigated_value": mit["metric_value"],
                "threshold": orig["threshold"],
                "original_passed": orig["passed"],
                "mitigated_passed": mit["passed"],
                "original_severity": orig["severity"],
                "mitigated_severity": mit["severity"],
            })

    # ── Script diff ─────────────────────────────────────────────────────────
    script_diff = _generate_script_diff(strategy, type(model).__name__, target_column, sensitive_attrs)

    # ── Save mitigated model ────────────────────────────────────────────────
    # (will be saved by the router to the session directory)

    return {
        "strategy": strategy,
        "model_type": type(model).__name__,
        "original_accuracy": original_accuracy,
        "mitigated_accuracy": mitigated_accuracy,
        "original_dir": round(original_dir, 4),
        "mitigated_dir": round(mitigated_dir, 4),
        "original_metrics": original_metrics,
        "mitigated_metrics": mitigated_metrics,
        "improvements": improvements,
        "script_diff": script_diff,
        "mitigated_model": mitigated_model,
        "all_passed": all(m["mitigated_passed"] for m in improvements),
    }
