"""
Ollama Local LLM Client
=======================
Calls the local Ollama server at http://localhost:11434.
No external API keys required.
"""

from __future__ import annotations

import json
import logging
import requests
from typing import Optional

import os

logger = logging.getLogger("courtroom.ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest")


def is_ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: Optional[str] = None,
    temperature: float = 0.3,
    timeout: int = 45,
) -> str:
    """
    Call the local Ollama API and return the generated text.

    Args:
        prompt: The user prompt to send.
        model: Ollama model name (default: llama3).
        system: Optional system instruction prepended to the prompt.
        temperature: Sampling temperature (lower = more deterministic).
        timeout: Request timeout in seconds.

    Returns:
        Generated text string.

    Raises:
        RuntimeError: If Ollama is unavailable or returns an error.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"

    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }

    logger.info("Calling Ollama model=%s prompt_len=%d", model, len(full_prompt))

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434. "
            "Please ensure Ollama is installed and running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama request timed out after {timeout}s.")

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    try:
        data = resp.json()
        text = data.get("response", "").strip()
        logger.info("Ollama response len=%d", len(text))
        return text
    except Exception as exc:
        raise RuntimeError(f"Failed to parse Ollama response: {exc}")


def call_ollama_json(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: Optional[str] = None,
    fallback: Optional[dict] = None,
) -> dict:
    """
    Call Ollama expecting a JSON response.
    Strips markdown fences and parses JSON.
    Returns fallback dict on parse failure (never raises on JSON errors).
    """
    raw = call_ollama(prompt=prompt, model=model, system=system)

    # Strip markdown code fences
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        text = "\n".join(lines[1:])
        if text.strip().endswith("```"):
            text = text.strip().rsplit("```", 1)[0].strip()

    # Try to extract JSON object from anywhere in the response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Ollama JSON parse failed: %s | raw: %s", exc, raw[:300])
        return fallback or {}


def list_ollama_models() -> list[str]:
    """Return list of locally available Ollama model names."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []
