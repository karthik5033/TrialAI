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
import json
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
from backend.services.llm import get_llm_client

logger = logging.getLogger("courtroom.remediation")

# ═══════════════════════════════════════════════════════════════════════════════
#  Active Remediation Loop v2.0 — Two-Stage LLM Prompts
# ═══════════════════════════════════════════════════════════════════════════════

STAGE1_SYSTEM_PROMPT = """You are a senior ML fairness auditor inside the \"AI Courtroom\" platform.
Your job is Stage 1: Script Analysis.  Read the training script and the fairness
metrics and identify every bias-inducing pattern.

Return ONLY a JSON object — no markdown, no extra text — with this schema:
{
  "bias_patterns": [
    {
      "location": "line number or function name",
      "pattern": "short description of the bias-inducing pattern",
      "severity": "critical | warning | info"
    }
  ],
  "recommended_strategy": "reweighing | fairlearn_demographic_parity | threshold_adjustment",
  "protected_columns_used": ["list of columns from the script that are protected or proxy"],
  "model_fit_location": "the line or expression where model.fit() is called",
  "summary": "2-3 sentence overall assessment"
}"""

STAGE2_SYSTEM_PROMPT = """You are a senior ML fairness engineer in the \"AI Courtroom\" platform.
You are in Stage 2: Code Modification of the Active Remediation Loop.

Goal:
- Apply the PLAN produced in Stage 1 to the TRAINING_SCRIPT.
- Inject the requested bias mitigation while preserving the public interface.
- Output both a unified diff and a human-readable explanation for auditors.

Rules:
- Preserve the script's entrypoint, function signatures, and output paths.
- Implement the mitigation in the simplest place that affects model training.
- Prefer adding small, clearly commented blocks over large refactors.
- Use recognised techniques for the chosen STRATEGY:
  - "reweighing": compute per-instance weights by group and pass as sample_weight.
  - "fairlearn_demographic_parity": wrap with fairlearn ExponentiatedGradient + DemographicParity.
  - "threshold_adjustment": adjust per-group decision thresholds post-training.
- Do NOT invent new external dependencies beyond standard Python, sklearn, numpy, pandas, fairlearn.

Return ONLY a JSON object — no markdown fences, no prose outside the JSON — with this schema:
{
  "diff": "UNIFIED DIFF PATCH starting with --- original.py / +++ modified.py",
  "modified_script": "Full modified script as plain text",
  "change_log": [
    {
      "category": "data_preprocessing | model_training | evaluation | thresholding | other",
      "summary": "1-2 sentence summary of this change.",
      "risk_tradeoff": "short description of expected impact on fairness vs accuracy."
    }
  ],
  "fairness_expectations": {
    "expected_effect": "short paragraph on how fairness metrics should change.",
    "unchanged_aspects": "what is intentionally left unchanged."
  }
}"""


def _run_stage1_analysis(
    script_content: str,
    metrics: list[dict],
    protected_attrs: list[str],
    proxies: list[dict],
) -> dict:
    """Stage 1: call LLM to analyse the training script for bias patterns."""
    llm = get_llm_client()

    metrics_text = "\n".join(
        f"- {m['metric_name']}: {m['metric_value']:.4f} (threshold={m['threshold']}, passed={m['passed']})"
        for m in metrics
    )
    proxy_text = "\n".join(
        f"- {p['feature']} correlates with {p['corr_with']} (r={p['correlation']:.4f})"
        for p in proxies
    ) if proxies else "None detected."

    prompt = f"""Analyse this training script.

PROTECTED ATTRIBUTES: {", ".join(protected_attrs)}
PROXY FEATURES:
{proxy_text}

FAIRNESS METRICS:
{metrics_text}

TRAINING SCRIPT:
```python
{script_content}
```

Return ONLY the JSON object."""

    logger.info("Stage 1: calling LLM for script analysis…")
    raw = llm.chat(
        system=STAGE1_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.1,
        force_local=True,
    )
    # Strip markdown fences if present
    if raw.strip().startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Stage 1 JSON parse failed; returning raw text as summary.")
        return {"summary": raw, "bias_patterns": [], "recommended_strategy": "reweighing"}


def _run_stage2_modification(
    script_content: str,
    stage1_json: dict,
    strategy: str,
    metrics: list[dict],
    protected_attrs: list[str],
) -> dict:
    """Stage 2: call LLM to rewrite the script with bias mitigation."""
    llm = get_llm_client()

    metrics_text = "\n".join(
        f"- {m['metric_name']}: {m['metric_value']:.4f} (threshold={m['threshold']}, passed={m['passed']})"
        for m in metrics
    )

    prompt = f"""You are in Stage 2: Code Modification.

TRAINING_SCRIPT (original, unmodified):
```python
{script_content}
```

STAGE1_ANALYSIS (JSON from previous call):
```json
{json.dumps(stage1_json, indent=2)}
```

STRATEGY: "{strategy}"
PROTECTED ATTRIBUTES: {", ".join(protected_attrs)}

FAIRNESS_METRICS:
{metrics_text}

Apply the plan.  Return ONLY the JSON object."""

    logger.info("Stage 2: calling LLM for code modification (strategy=%s)…", strategy)
    raw = llm.chat(
        system=STAGE2_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.15,
        force_local=True,
    )
    # Strip markdown fences
    if raw.strip().startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Stage 2 JSON parse failed; extracting what we can.")
        # Try to find JSON within the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        return {"diff": "", "modified_script": "", "change_log": [], "fairness_expectations": {"expected_effect": raw[:500], "unchanged_aspects": ""}, "_raw": raw}


def generate_llm_mitigation(
    script_content: str,
    strategy: str,
    metrics: list[dict],
    protected_attrs: list[str],
    proxies: list[dict],
) -> dict:
    """
    Two-stage Active Remediation Loop:
      Stage 1 → analyse script for bias patterns
      Stage 2 → rewrite script with mitigation
    Returns {script_diff, modified_script, change_log, fairness_expectations, stage1_analysis}.
    """
    # Map frontend strategy names to prompt-level names
    strategy_map = {
        "reweighing": "reweighing",
        "threshold_adjustment": "threshold_adjustment",
        "fairness_constraint": "fairlearn_demographic_parity",
    }
    llm_strategy = strategy_map.get(strategy, strategy)

    # ── Stage 1 ─────────────────────────────────────────────────────────────
    stage1 = _run_stage1_analysis(script_content, metrics, protected_attrs, proxies)
    logger.info("Stage 1 complete: %d bias patterns found.", len(stage1.get("bias_patterns", [])))

    # ── Stage 2 ─────────────────────────────────────────────────────────────
    stage2 = _run_stage2_modification(script_content, stage1, llm_strategy, metrics, protected_attrs)
    logger.info("Stage 2 complete: diff length=%d chars.", len(stage2.get("diff", "")))

    return {
        "script_diff": stage2.get("diff", ""),
        "modified_script": stage2.get("modified_script", ""),
        "change_log": stage2.get("change_log", []),
        "fairness_expectations": stage2.get("fairness_expectations", {}),
        "stage1_analysis": stage1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 5: Re-Evaluation — Multi-Audience Impact Report
# ═══════════════════════════════════════════════════════════════════════════════

STAGE5_SYSTEM_PROMPT = """You are a senior AI auditor writing the final re-evaluation report.
You compare BEFORE vs AFTER fairness and accuracy metrics.

Rules:
- Be honest: if a metric worsened, say so directly.
- Base everything strictly on the BEFORE/AFTER numbers provided.
- Provide 3 separate summaries for different audiences.

Return ONLY a JSON object with this schema:
{
  "headline": "One-sentence summary of how fairness and performance changed.",
  "technical_summary": "3-5 sentences for ML engineers: metric deltas, model behaviour.",
  "manager_summary": "3-5 sentences for business stakeholders: risk, customer impact, trade-offs.",
  "legal_summary": "3-5 sentences for compliance/legal: non-discrimination risk, documentation.",
  "key_numbers": [
    {
      "metric": "metric name",
      "before": 0.0,
      "after": 0.0,
      "comment": "whether this change is positive or negative."
    }
  ]
}"""


def _run_stage5_reevaluation(
    model_type: str,
    strategy: str,
    original_metrics: list[dict],
    mitigated_metrics: list[dict],
    original_accuracy: float,
    mitigated_accuracy: float,
    change_log: list[dict] | None = None,
) -> dict:
    """Stage 5: call LLM to produce a multi-audience impact report."""
    llm = get_llm_client()

    model_card = {
        "model_type": model_type,
        "strategy_applied": strategy,
    }
    before_json = {
        "accuracy": original_accuracy,
        "fairness_metrics": [
            {"name": m["metric_name"], "value": m["metric_value"],
             "threshold": m["threshold"], "passed": m["passed"], "severity": m["severity"]}
            for m in original_metrics
        ],
    }
    after_json = {
        "accuracy": mitigated_accuracy,
        "fairness_metrics": [
            {"name": m["metric_name"], "value": m["metric_value"],
             "threshold": m["threshold"], "passed": m["passed"], "severity": m["severity"]}
            for m in mitigated_metrics
        ],
    }
    mitigation_summary = change_log or [{"summary": f"{strategy} applied to the model."}]

    prompt = f"""Compare BEFORE vs AFTER and produce the re-evaluation report.

MODEL_CARD:
```json
{json.dumps(model_card, indent=2)}
```

BEFORE_METRICS:
```json
{json.dumps(before_json, indent=2)}
```

AFTER_METRICS:
```json
{json.dumps(after_json, indent=2)}
```

MITIGATION_SUMMARY:
```json
{json.dumps(mitigation_summary, indent=2)}
```

Return ONLY the JSON object."""

    logger.info("Stage 5: calling LLM for re-evaluation report…")
    raw = llm.chat(
        system=STAGE5_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.2,
    )
    # Strip markdown fences if present
    if raw.strip().startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning("Stage 5 JSON parse failed.")
        return {
            "headline": "Re-evaluation completed but report generation failed.",
            "technical_summary": raw[:500],
            "manager_summary": "",
            "legal_summary": "",
            "key_numbers": [],
        }


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
    Handles sklearn Pipeline objects by using the stepname__sample_weight syntax.
    """
    new_model = clone(model)

    # Build combined group string for weight computation
    combined = np.array([f"{s}_{l}" for s, l in zip(sensitive, y)])
    weights = compute_sample_weight(class_weight="balanced", y=combined)

    X_fit = X.values if isinstance(X, pd.DataFrame) else X

    # For sklearn Pipeline, sample_weight must be passed as step__sample_weight
    fit_params: dict = {}
    model_name = type(new_model).__name__
    if model_name == "Pipeline" and hasattr(new_model, "steps"):
        # Find the final estimator step name
        last_step_name, last_estimator = new_model.steps[-1]
        last_est_name = type(last_estimator).__name__

        # Check if the final estimator supports sample_weight
        import inspect
        try:
            sig = inspect.signature(last_estimator.fit)
            if "sample_weight" in sig.parameters:
                fit_params[f"{last_step_name}__sample_weight"] = weights
                logger.info(
                    "Pipeline reweighing: passing %s__sample_weight to step '%s' (%s)",
                    last_step_name, last_step_name, last_est_name
                )
            else:
                logger.warning(
                    "Pipeline final step '%s' (%s) does not support sample_weight — training unweighted.",
                    last_step_name, last_est_name
                )
        except Exception:
            logger.warning("Could not inspect Pipeline step signature — training unweighted.")
    else:
        fit_params["sample_weight"] = weights

    try:
        new_model.fit(X_fit, y, **fit_params)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "%s sample_weight fit failed (%s) — retrying without weights.",
            model_name, exc
        )
        new_model.fit(X_fit, y)

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
    script_content: str | None = None,
) -> dict:
    """
    End-to-end remediation: prepare → original metrics → mitigate → new
    metrics → diff.

    If script_content is provided, uses LLM to generate the diff and explanation.
    Returns dict with original/mitigated metrics, accuracy, strategy info.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES.keys())}")

    # ── Prepare data ────────────────────────────────────────────────────────
    X, y, feature_cols, raw_sensitive, encoders, X_raw = prepare_dataset(df, target_column, sensitive_attrs)

    if len(X) < 10:
        raise ValueError(f"Dataset has only {len(X)} rows after cleaning.")

    X, feature_cols = match_features(model, X, feature_cols, X_raw)

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

    # ── Script diff & LLM Analysis ──────────────────────────────────────────
    llm_explanation = None
    modified_script = None
    change_log: list[dict] = []
    if script_content:
        proxies = detect_proxy_features(df, primary_key, feature_cols)
        llm_result = generate_llm_mitigation(
            script_content=script_content,
            strategy=strategy,
            metrics=original_metrics,
            protected_attrs=sensitive_attrs,
            proxies=proxies,
        )
        script_diff = llm_result["script_diff"]
        modified_script = llm_result.get("modified_script", "")
        change_log = llm_result.get("change_log", [])
        llm_explanation = {
            "stage1_analysis": llm_result.get("stage1_analysis", {}),
            "change_log": change_log,
            "fairness_expectations": llm_result.get("fairness_expectations", {}),
        }
    else:
        script_diff = _generate_script_diff(strategy, type(model).__name__, target_column, sensitive_attrs)

    # ── Stage 5: Re-Evaluation Report ───────────────────────────────────────
    reevaluation_report = None
    try:
        reevaluation_report = _run_stage5_reevaluation(
            model_type=type(model).__name__,
            strategy=strategy,
            original_metrics=original_metrics,
            mitigated_metrics=mitigated_metrics,
            original_accuracy=original_accuracy,
            mitigated_accuracy=mitigated_accuracy,
            change_log=change_log if change_log else None,
        )
        logger.info("Stage 5 complete: %s", reevaluation_report.get("headline", "(no headline)"))
    except Exception as exc:
        logger.warning("Stage 5 re-evaluation failed: %s", exc)

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
        "modified_script": modified_script,
        "llm_explanation": llm_explanation,
        "reevaluation_report": reevaluation_report,
        "mitigated_model": mitigated_model,
        "all_passed": all(m["mitigated_passed"] for m in improvements),
    }
