"""
Ollama Strategy Selector & LLM Explanation Generator
=====================================================
Uses the local Ollama LLM ONLY for two safe tasks:
  1. Choosing which mitigation strategy to apply (JSON output).
  2. Generating a plain-English explanation of what was changed.

The LLM NEVER writes or modifies any Python code.
"""

from __future__ import annotations

import json
import logging
import os
import requests
from typing import Optional

from backend.services.ollama_client import call_ollama_json, call_ollama, is_ollama_available

logger = logging.getLogger("courtroom.remediation_llm")

VALID_STRATEGIES = {"reweighing", "threshold_adjustment", "fairness_constraint"}
DEFAULT_STRATEGY = "reweighing"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"


# ─────────────────────────────────────────────────────────────────────────────
# Groq API fallback helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(system: str, prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    """Call Groq API as fallback when Ollama is unavailable or times out."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_groq_json(system: str, prompt: str, fallback: dict) -> dict:
    """Call Groq and parse JSON response."""
    try:
        raw = _call_groq(system, prompt, temperature=0.1, max_tokens=256)
        text = raw.strip()
        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.strip().endswith("```"):
                text = text.strip().rsplit("```", 1)[0].strip()
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception as exc:
        logger.warning("Groq JSON fallback failed: %s", exc)
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Selection
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_SYSTEM = """You are an AI fairness expert. Your ONLY job is to pick one mitigation strategy.

Available strategies:
- reweighing: Best when disparate impact ratio < 0.8. Rebalances sample weights during training.
- threshold_adjustment: Best for post-processing. Adjusts decision threshold per group.  
- fairness_constraint: Best for severe bias. Uses fairlearn ExponentiatedGradient.

Return ONLY a JSON object with no other text:
{"strategy": "<one of: reweighing | threshold_adjustment | fairness_constraint>", "reason": "<one sentence>"}"""


def select_strategy_with_ollama(
    fairness_metrics: list[dict],
    sensitive_attrs: list[str],
) -> dict:
    """
    Ask Ollama to pick the best mitigation strategy based on fairness metrics.

    Returns:
        {"strategy": str, "reason": str, "source": "ollama" | "fallback"}
    """
    if not is_ollama_available():
        logger.warning("Ollama not available — trying Groq fallback")
        return _select_strategy_via_groq(fairness_metrics, sensitive_attrs)

    # Format metrics summary for the prompt
    failed_metrics = [m for m in fairness_metrics if not m.get("passed", True)]
    metrics_lines = []
    for m in fairness_metrics:
        status = "FAILED" if not m.get("passed", True) else "passed"
        metrics_lines.append(
            f"  - {m['metric_name']}: {m['metric_value']:.4f} "
            f"(threshold={m['threshold']}, {status})"
        )

    prompt = f"""Choose a fairness mitigation strategy for a model with these metrics:

Protected attributes: {', '.join(sensitive_attrs)}
Total metrics evaluated: {len(fairness_metrics)}
Failed metrics: {len(failed_metrics)}

Metrics:
{chr(10).join(metrics_lines)}

Return ONLY the JSON object as instructed."""

    logger.info("Asking Ollama to select mitigation strategy...")

    try:
        result = call_ollama_json(
            prompt=prompt,
            system=STRATEGY_SYSTEM,
            fallback=None,
        )
        if result is None:
            raise RuntimeError("Ollama returned empty result")
    except Exception as exc:
        logger.warning("Ollama strategy selection failed (%s) — falling back to Groq", exc)
        return _select_strategy_via_groq(fairness_metrics, sensitive_attrs)

    strategy = result.get("strategy", DEFAULT_STRATEGY).strip().lower()

    if strategy not in VALID_STRATEGIES:
        logger.warning(
            "LLM returned invalid strategy '%s' — falling back to '%s'",
            strategy, DEFAULT_STRATEGY
        )
        strategy = DEFAULT_STRATEGY

    logger.info("Strategy selected: %s | reason: %s", strategy, result.get("reason", ""))

    return {
        "strategy": strategy,
        "reason": result.get("reason", "Strategy selected by local LLM."),
        "source": "ollama",
    }


def _select_strategy_via_groq(
    fairness_metrics: list[dict],
    sensitive_attrs: list[str],
) -> dict:
    """Groq API fallback for strategy selection."""
    failed_metrics = [m for m in fairness_metrics if not m.get("passed", True)]
    metrics_lines = [
        f"  - {m['metric_name']}: {m['metric_value']:.4f} (threshold={m['threshold']}, "
        f"{'FAILED' if not m.get('passed', True) else 'passed'})"
        for m in fairness_metrics
    ]
    prompt = (
        f"Choose a fairness mitigation strategy. Protected attributes: {', '.join(sensitive_attrs)}. "
        f"Failed metrics: {len(failed_metrics)}/{len(fairness_metrics)}.\n"
        + "\n".join(metrics_lines)
        + "\nReturn ONLY the JSON object."
    )
    result = _call_groq_json(STRATEGY_SYSTEM, prompt, fallback={"strategy": DEFAULT_STRATEGY, "reason": "Groq fallback"})
    strategy = result.get("strategy", DEFAULT_STRATEGY).strip().lower()
    if strategy not in VALID_STRATEGIES:
        strategy = DEFAULT_STRATEGY
    logger.info("Groq selected strategy: %s", strategy)
    return {"strategy": strategy, "reason": result.get("reason", "Selected via Groq fallback."), "source": "groq"}


# ─────────────────────────────────────────────────────────────────────────────
# Explanation Generator
# ─────────────────────────────────────────────────────────────────────────────

EXPLANATION_SYSTEM = """You are a clear, concise AI fairness explainer.
Given a strategy name and before/after metrics, write a plain-English explanation.
Keep it to 3-4 sentences. Be specific about what numbers changed and why."""


def generate_explanation_with_ollama(
    strategy: str,
    patch_description: str,
    original_metrics: list[dict],
    mitigated_metrics: list[dict],
    original_accuracy: float,
    mitigated_accuracy: float,
) -> str:
    """
    Ask Ollama to explain what happened during remediation.
    Returns a plain-English explanation string.
    Falls back to a template if Ollama is unavailable.
    """
    if not is_ollama_available():
        logger.warning("Ollama not available — trying Groq for explanation")
        return _explain_via_groq(
            strategy, original_metrics, mitigated_metrics, original_accuracy, mitigated_accuracy
        ) or _template_explanation(
            strategy, original_metrics, mitigated_metrics, original_accuracy, mitigated_accuracy
        )

    def fmt_metrics(metrics: list[dict]) -> str:
        return "\n".join(
            f"  - {m['metric_name']}: {m['metric_value']:.4f} "
            f"({'passed' if m.get('passed') else 'FAILED'})"
            for m in metrics
        )

    prompt = f"""A bias mitigation was applied to an ML model. Explain what happened.

Strategy applied: {strategy}
What was changed: {patch_description}

BEFORE metrics (accuracy={original_accuracy:.4f}):
{fmt_metrics(original_metrics)}

AFTER metrics (accuracy={mitigated_accuracy:.4f}):
{fmt_metrics(mitigated_metrics)}

Write a plain English explanation in 3-4 sentences for a technical audience."""

    try:
        explanation = call_ollama(prompt=prompt, system=EXPLANATION_SYSTEM, temperature=0.4)
        if not explanation:
            raise RuntimeError("Empty response from Ollama")
        logger.info("Ollama explanation generated (%d chars)", len(explanation))
        return explanation.strip()
    except Exception as exc:
        logger.warning("Ollama explanation failed: %s — trying Groq", exc)
        return _explain_via_groq(
            strategy, original_metrics, mitigated_metrics, original_accuracy, mitigated_accuracy, prompt
        ) or _template_explanation(
            strategy, original_metrics, mitigated_metrics, original_accuracy, mitigated_accuracy
        )


def _template_explanation(
    strategy: str,
    original_metrics: list[dict],
    mitigated_metrics: list[dict],
    original_accuracy: float,
    mitigated_accuracy: float,
) -> str:
    """Fallback template explanation when Ollama is unavailable."""
    orig_dir = next(
        (m["metric_value"] for m in original_metrics if "disparate_impact" in m["metric_name"]),
        None
    )
    mit_dir = next(
        (m["metric_value"] for m in mitigated_metrics if "disparate_impact" in m["metric_name"]),
        None
    )

    lines = [
        f"Applied '{strategy}' mitigation strategy to the model.",
        f"Model accuracy changed from {original_accuracy:.4f} to {mitigated_accuracy:.4f} "
        f"({'improved' if mitigated_accuracy >= original_accuracy else 'slight reduction'}).",
    ]

    if orig_dir is not None and mit_dir is not None:
        direction = "improved" if mit_dir > orig_dir else "did not improve"
        lines.append(
            f"Disparate Impact Ratio {direction} from {orig_dir:.4f} to {mit_dir:.4f}."
        )

    passed_after = sum(1 for m in mitigated_metrics if m.get("passed"))
    total = len(mitigated_metrics)
    lines.append(f"{passed_after}/{total} fairness metrics now pass their thresholds.")

    return " ".join(lines)


def _explain_via_groq(
    strategy: str,
    original_metrics: list[dict],
    mitigated_metrics: list[dict],
    original_accuracy: float,
    mitigated_accuracy: float,
    prompt: str = None,
) -> str | None:
    """Groq API fallback for explanation."""
    if prompt is None:
        def fmt_metrics(metrics: list[dict]) -> str:
            return "\n".join(
                f"  - {m['metric_name']}: {m['metric_value']:.4f} "
                f"({'passed' if m.get('passed') else 'FAILED'})"
                for m in metrics
            )
        prompt = f"""A bias mitigation was applied to an ML model. Explain what happened.

Strategy applied: {strategy}

BEFORE metrics (accuracy={original_accuracy:.4f}):
{fmt_metrics(original_metrics)}

AFTER metrics (accuracy={mitigated_accuracy:.4f}):
{fmt_metrics(mitigated_metrics)}

Write a plain English explanation in 3-4 sentences for a technical audience."""

    try:
        explanation = _call_groq(EXPLANATION_SYSTEM, prompt, temperature=0.3)
        logger.info("Groq explanation generated (%d chars)", len(explanation))
        return explanation.strip()
    except Exception as exc:
        logger.warning("Groq explanation fallback failed: %s", exc)
        return None
