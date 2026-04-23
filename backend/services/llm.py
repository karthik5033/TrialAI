"""
AI Courtroom v2.0 — Universal LLM Client.

Supports Anthropic, OpenAI (NVIDIA, Groq, Together), and OpenRouter.
Automatically selects the best available provider based on env keys.
"""

import os
import logging
from typing import Any, List, Dict

# Providers
import anthropic
import httpx

logger = logging.getLogger("courtroom.llm")

class LLMClient:
    def __init__(self):
        self.provider = None
        self.client = None
        self.model = None
        
        # 0. Check Ollama (Local) first ONLY if it's the only one or strictly forced globally
        # We will instead use force_local in chat() to route to Ollama selectively.

        # 1. Check Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_key != "your_key_here":
            self.provider = "anthropic"
            self.client = anthropic.Anthropic(api_key=anthropic_key)
            self.model = "claude-3-5-sonnet-20240620"
            logger.info("LLM Client initialized with Anthropic (Claude 3.5 Sonnet)")
            return

        # 2. Check OpenRouter (Universal Fallback)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            self.provider = "openrouter"
            self.api_key = openrouter_key
            self.model = "anthropic/claude-3.5-sonnet"
            logger.info("LLM Client initialized with OpenRouter (Claude 3.5 Sonnet)")
            return

        # 3. Check NVIDIA NIM (OpenAI Compatible)
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        if nvidia_key:
            self.provider = "nvidia"
            self.api_key = nvidia_key
            self.model = "meta/llama-3.1-70b-instruct"
            self.base_url = "https://integrate.api.nvidia.com/v1"
            logger.info("LLM Client initialized with NVIDIA NIM (Llama 3.1 70B)")
            return

        # 4. Check Groq (OpenAI Compatible)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            self.provider = "groq"
            self.api_key = groq_key
            self.model = "llama-3.1-70b-versatile"
            self.base_url = "https://api.groq.com/openai/v1"
            logger.info("LLM Client initialized with Groq (Llama 3.1 70B)")
            return

        logger.error("No LLM API keys found in environment!")

    def chat(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7, force_local: bool = False) -> str:
        if not self.provider and not force_local:
            raise RuntimeError("LLM Client not initialized. Please set an API key in backend/.env")

        provider = "ollama" if force_local else self.provider
        
        if provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                temperature=temperature
            )
            return resp.content[0].text

        # OpenAI-Compatible providers (OpenRouter, NVIDIA, Groq, Ollama)
        else:
            api_key = "dummy" if provider == "ollama" else self.api_key
            model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:latest") if provider == "ollama" else self.model
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Combine system prompt for OpenAI style if needed
            combined_messages = []
            if system:
                combined_messages.append({"role": "system", "content": system})
            combined_messages.extend(messages)
            
            payload = {
                "model": model,
                "messages": combined_messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            url = ""
            if provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers["HTTP-Referer"] = "https://github.com/karthik5033/TrialAI"
                headers["X-Title"] = "AI Courtroom v2.0"
            elif provider == "nvidia":
                url = self.base_url + "/chat/completions"
            elif provider == "groq":
                url = self.base_url + "/chat/completions"
            elif provider == "ollama":
                url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1") + "/chat/completions"

            timeout = 120.0 if provider == "ollama" else 60.0
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    logger.error(f"LLM API Error ({provider}): {resp.text}")
                    raise RuntimeError(f"LLM API Error: {resp.status_code}")
                
                data = resp.json()
                return data["choices"][0]["message"]["content"]

# Singleton instance
_global_client = None

def get_llm_client() -> LLMClient:
    global _global_client
    if _global_client is None:
        _global_client = LLMClient()
    return _global_client
