"""
Local Remediation Pipeline (Ollama-powered)
============================================
Fully self-contained remediation system using:
  - Ollama (local LLM) for strategy selection & explanation
  - Deterministic patch templates for code modification
  - subprocess runner for safe script execution
  - Real fairness re-evaluation (no mocked metrics)

Flow:
  1. Read original script (or auto-generate one)
  2. Call Ollama → pick strategy
  3. Apply deterministic patch to script
  4. Save patched script to session directory
  5. Execute patched script in subprocess
  6. Load output_model.pkl
  7. Re-compute fairness metrics
  8. Call Ollama → generate explanation
  9. Return full before/after comparison

Usage:
  from backend.services.remediation_pipeline import run_local_remediation

  result = run_local_remediation(
      model=original_model,
      df=dataframe,
      target_column="Survived",
      sensitive_attrs=["Sex"],
      session_dir=Path("uploads/abc123"),
      strategy="auto",  # let Ollama decide, or pass specific strategy
  )
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from backend.services.bias_engine import (
    compute_fairness_metrics,
    match_features,
    prepare_dataset,
)
from backend.services.ollama_client import is_ollama_available
from backend.services.remediation_llm import (
    select_strategy_with_ollama,
    generate_explanation_with_ollama,
)
from backend.services.remediation_patch import (
    PATCH_FUNCTIONS,
    generate_auto_script,
)
from backend.services.runner import run_script, load_output_model

logger = logging.getLogger("courtroom.remediation_pipeline")

VALID_STRATEGIES = {"reweighing", "threshold_adjustment", "fairness_constraint"}


# ─────────────────────────────────────────────────────────────────────────────
# Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_local_remediation(
    model: Any,
    df: pd.DataFrame,
    target_column: str,
    sensitive_attrs: list[str],
    session_dir: Path,
    strategy: str = "auto",
    script_content: Optional[str] = None,
    timeout: int = 60,
) -> dict:
    """
    Run the full local remediation pipeline.

    Args:
        model:            Original trained model.
        df:               Full dataset as a DataFrame.
        target_column:    Name of the label column.
        sensitive_attrs:  List of protected attribute column names.
        session_dir:      Directory where files are stored (dataset.csv lives here).
        strategy:         "auto" to let Ollama pick, or specify directly.
        script_content:   Optional uploaded training script. If None, auto-generate.
        timeout:          Subprocess timeout in seconds.

    Returns:
        Full result dict with before/after metrics, scripts, explanation.

    Raises:
        ValueError, RuntimeError on critical failures.
    """
    logger.info("=" * 60)
    logger.info("LOCAL REMEDIATION PIPELINE STARTED")
    logger.info("Session dir: %s | Strategy: %s", session_dir, strategy)
    logger.info("=" * 60)

    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise ValueError(f"Session directory does not exist: {session_dir}")

    # ── Step 1: Prepare dataset ──────────────────────────────────────────────
    logger.info("[Step 1] Preparing dataset...")

    X, y, feature_cols, raw_sensitive, _, X_raw = prepare_dataset(df, target_column, sensitive_attrs)

    if not raw_sensitive:
        raise ValueError(f"No sensitive attributes found: {sensitive_attrs}")
    primary_key = list(raw_sensitive.keys())[0]

    sensitive_values = raw_sensitive[primary_key]

    # ── Align features to model's expected input ─────────────────────────────
    X, feature_cols = match_features(model, X, feature_cols, X_raw)

    logger.info("[Step 1] Dataset ready: %d rows, %d features, sensitive=%s", len(df), X.shape[1], primary_key)

    # ── Step 2: Compute original metrics ────────────────────────────────────
    logger.info("[Step 2] Computing original fairness metrics...")

    y_pred_original = model.predict(X.values)
    original_accuracy = round(float(accuracy_score(y, y_pred_original)), 4)
    original_metrics = compute_fairness_metrics(y, y_pred_original, sensitive_values, primary_key)

    original_dir = next(
        (m["metric_value"] for m in original_metrics if "disparate_impact" in m["metric_name"]),
        0.0,
    )

    logger.info("[Step 2] Original accuracy=%.4f DIR=%.4f", original_accuracy, original_dir)
    for m in original_metrics:
        logger.info("  [Step 2] %s = %.4f (passed=%s)", m["metric_name"], m["metric_value"], m["passed"])

    # ── Step 3: Ollama strategy selection ────────────────────────────────────
    logger.info("[Step 3] Selecting mitigation strategy...")

    strategy_info = {"strategy": strategy, "reason": "Manually specified.", "source": "manual"}
    if strategy == "auto":
        strategy_info = select_strategy_with_ollama(original_metrics, sensitive_attrs)
        strategy = strategy_info["strategy"]

    if strategy not in VALID_STRATEGIES:
        logger.warning("[Step 3] Invalid strategy '%s' — defaulting to reweighing", strategy)
        strategy = "reweighing"

    logger.info("[Step 3] Strategy selected: %s (source=%s)", strategy, strategy_info["source"])
    logger.info("[Step 3] Reason: %s", strategy_info.get("reason", ""))

    # ── Step 4: Prepare script ───────────────────────────────────────────────
    logger.info("[Step 4] Preparing training script...")

    if script_content:
        original_script = script_content
        logger.info("[Step 4] Using uploaded script (%d chars)", len(original_script))
    else:
        # Use model's actual feature list if available — prevents column count mismatch
        _mf = getattr(model, "feature_names_in_", None)
        model_features = list(_mf) if _mf is not None else feature_cols
        original_script = generate_auto_script(
            target_column=target_column,
            sensitive_attr=primary_key,
            model_type=type(model).__name__,
            strategy=strategy,
            feature_names=model_features,
        )
        logger.info("[Step 4] Auto-generated training script with %d features: %s", len(model_features), model_features)

    # Save original script
    original_script_path = session_dir / "original_script.py"
    original_script_path.write_text(original_script, encoding="utf-8")

    # ── Step 5: Apply deterministic patch ───────────────────────────────────
    logger.info("[Step 5] Applying %s patch...", strategy)

    patcher = PATCH_FUNCTIONS.get(strategy, PATCH_FUNCTIONS["reweighing"])
    patch_result = patcher(original_script, sensitive_attr=sensitive_attrs[0] if sensitive_attrs else None)

    patched_script = patch_result["patched_script"]
    patch_applied = patch_result["patch_applied"]
    patch_description = patch_result["description"]

    logger.info("[Step 5] Patch applied: %s | %s", patch_applied, patch_description)

    # Save patched script to session directory
    patched_script_path = session_dir / "mitigated_script.py"
    patched_script_path.write_text(patched_script, encoding="utf-8")
    logger.info("[Step 5] Patched script saved to: %s", patched_script_path)

    # Also copy dataset.csv to session dir (already there, but make sure)
    dataset_in_session = session_dir / "dataset.csv"

    # ── Step 6: Execute patched script ──────────────────────────────────────
    logger.info("[Step 6] Executing patched training script...")

    run_result = run_script(
        script_path=patched_script_path,
        working_dir=session_dir,
        timeout=timeout,
    )

    logger.info("[Step 6] Script execution: success=%s returncode=%d", run_result["success"], run_result["returncode"])
    if run_result["stdout"]:
        logger.info("[Step 6] Script stdout:\n%s", run_result["stdout"][-1000:])
    if run_result["stderr"] and not run_result["success"]:
        logger.error("[Step 6] Script stderr:\n%s", run_result["stderr"][-1000:])

    # ── Step 7: Load retrained model ─────────────────────────────────────────
    retrained_model = None
    script_execution_success = run_result["success"]

    if script_execution_success:
        logger.info("[Step 7] Loading retrained model from output_model.pkl...")
        try:
            retrained_model = load_output_model(session_dir)
            logger.info("[Step 7] ✓ Retrained model loaded: %s", type(retrained_model).__name__)
        except Exception as exc:
            logger.error("[Step 7] Failed to load retrained model: %s", exc)
            script_execution_success = False

    # ── Step 8: Compute mitigated metrics ────────────────────────────────────
    logger.info("[Step 8] Computing mitigated fairness metrics...")

    if retrained_model is not None:
        mitigated_model = retrained_model
        mitigated_source = "retrained"
    else:
        # Fall back to in-memory mitigation using remediation.py strategies
        logger.warning(
            "[Step 8] Script execution failed — falling back to in-memory mitigation"
        )
        from backend.services.remediation import STRATEGIES
        mitigate_fn = STRATEGIES.get(strategy, STRATEGIES["reweighing"])
        from sklearn.base import clone
        mitigated_model = mitigate_fn(model, X, y, sensitive_values)
        mitigated_source = "in_memory_fallback"

    # Save mitigated model to session directory
    mitigated_model_path = session_dir / "mitigated_model.pkl"
    try:
        joblib.dump(mitigated_model, str(mitigated_model_path))
        logger.info("[Step 8] ✓ Mitigated model saved to: %s", mitigated_model_path)
    except Exception as exc:
        logger.error("[Step 8] Failed to save mitigated model: %s", exc)

    y_pred_mitigated = mitigated_model.predict(X.values)
    mitigated_accuracy = round(float(accuracy_score(y, y_pred_mitigated)), 4)
    mitigated_metrics = compute_fairness_metrics(y, y_pred_mitigated, sensitive_values, primary_key)

    mitigated_dir = next(
        (m["metric_value"] for m in mitigated_metrics if "disparate_impact" in m["metric_name"]),
        0.0,
    )

    logger.info("[Step 8] Mitigated accuracy=%.4f DIR=%.4f (source=%s)", mitigated_accuracy, mitigated_dir, mitigated_source)

    # ── Step 9: Build improvements summary ───────────────────────────────────
    logger.info("[Step 9] Building improvements comparison...")

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
                "improved": mit["metric_value"] > orig["metric_value"],
            })
            logger.info(
                "[Step 9]  %s: %.4f → %.4f (%s)",
                orig["metric_name"],
                orig["metric_value"],
                mit["metric_value"],
                "✓ improved" if mit["metric_value"] > orig["metric_value"] else "↓ unchanged"
            )

    all_passed = all(m["mitigated_passed"] for m in improvements)
    logger.info("[Step 9] All fairness checks passed: %s", all_passed)

    # ── Step 10: Ollama explanation ──────────────────────────────────────────
    logger.info("[Step 10] Generating explanation with Ollama...")

    explanation = generate_explanation_with_ollama(
        strategy=strategy,
        patch_description=patch_description,
        original_metrics=original_metrics,
        mitigated_metrics=mitigated_metrics,
        original_accuracy=original_accuracy,
        mitigated_accuracy=mitigated_accuracy,
    )

    logger.info("[Step 10] Explanation: %s", explanation[:200])

    # ── Build final result ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("LOCAL REMEDIATION PIPELINE COMPLETE")
    logger.info("Strategy: %s | Source: %s | All passed: %s", strategy, mitigated_source, all_passed)
    logger.info("=" * 60)

    return {
        # Core results
        "strategy": strategy,
        "model_type": type(model).__name__,
        "original_accuracy": original_accuracy,
        "mitigated_accuracy": mitigated_accuracy,
        "original_dir": round(original_dir, 4),
        "mitigated_dir": round(mitigated_dir, 4),
        "original_metrics": original_metrics,
        "mitigated_metrics": mitigated_metrics,
        "improvements": improvements,
        "all_passed": all_passed,

        # Script info
        "original_script": original_script,
        "modified_script": patched_script,
        "script_diff": _compute_diff(original_script, patched_script),
        "patch_applied": patch_applied,
        "patch_description": patch_description,

        # Execution info
        "script_execution": {
            "success": run_result["success"],
            "returncode": run_result["returncode"],
            "stdout": run_result["stdout"][-2000:] if run_result["stdout"] else "",
            "stderr": run_result["stderr"][-1000:] if run_result["stderr"] else "",
            "model_source": mitigated_source,
        },

        # LLM info
        "strategy_info": strategy_info,
        "llm_explanation": explanation,

        # Reevaluation report (structured)
        "reevaluation_report": _build_reevaluation_report(
            strategy, original_metrics, mitigated_metrics, original_accuracy, mitigated_accuracy
        ),

        # Model objects for router to save
        "mitigated_model": mitigated_model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_diff(original: str, patched: str) -> str:
    """Return a unified diff between original and patched script."""
    import difflib
    orig_lines = original.splitlines(keepends=True)
    patch_lines = patched.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        orig_lines, patch_lines,
        fromfile="original_script.py",
        tofile="mitigated_script.py",
        lineterm="",
    ))
    return "".join(diff)


def _build_reevaluation_report(
    strategy: str,
    original_metrics: list[dict],
    mitigated_metrics: list[dict],
    original_accuracy: float,
    mitigated_accuracy: float,
) -> dict:
    """Build a structured reevaluation report without LLM."""
    key_numbers = []
    for orig in original_metrics:
        mit = next((m for m in mitigated_metrics if m["metric_name"] == orig["metric_name"]), None)
        if mit:
            delta = mit["metric_value"] - orig["metric_value"]
            key_numbers.append({
                "metric": orig["metric_name"],
                "before": round(orig["metric_value"], 4),
                "after": round(mit["metric_value"], 4),
                "comment": f"{'↑ Improved' if delta > 0 else '↓ Worsened' if delta < 0 else '→ Unchanged'} by {abs(delta):.4f}",
            })

    acc_delta = mitigated_accuracy - original_accuracy
    passed_count = sum(1 for m in mitigated_metrics if m.get("passed"))
    total = len(mitigated_metrics)

    return {
        "headline": (
            f"Bias mitigation via '{strategy}' {'improved' if acc_delta >= 0 else 'slightly reduced'} "
            f"accuracy ({original_accuracy:.4f}→{mitigated_accuracy:.4f}). "
            f"{passed_count}/{total} fairness metrics now pass."
        ),
        "technical_summary": (
            f"Applied {strategy} strategy. Accuracy delta: {acc_delta:+.4f}. "
            f"Fairness metrics passing: {passed_count}/{total}."
        ),
        "manager_summary": (
            f"The AI model was retrained with fairness constraints. "
            f"{passed_count} out of {total} bias checks now pass their thresholds."
        ),
        "legal_summary": (
            f"Mitigation strategy '{strategy}' applied to reduce disparate impact. "
            f"Model was retrained and re-evaluated against {total} fairness criteria."
        ),
        "key_numbers": key_numbers,
    }
