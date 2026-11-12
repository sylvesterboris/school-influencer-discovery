"""Small Groq client (OpenAI-compatible endpoint) that returns parsed JSON."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def chat_json(self, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> dict[str, Any]: ...


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object even if the model wrapped it in code fences or prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


class GroqClient:
    def __init__(self, api_key: str, model: str, fallback_model: str | None = None, timeout: int = 60):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.calls_made = 0
        self.tokens_used = 0

    def _call(self, model: str, system: str, user: str, max_tokens: int, temperature: float) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(4):
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 429:
                # Free tier rate limit: honour retry-after when given.
                wait = float(resp.headers.get("retry-after", 2 * (attempt + 1)))
                time.sleep(min(wait, 30))
                continue
            if resp.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code >= 400:
                raise LLMError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")
            body = resp.json()
            self.calls_made += 1
            self.tokens_used += int(body.get("usage", {}).get("total_tokens", 0))
            content = body["choices"][0]["message"]["content"]
            try:
                return extract_json(content)
            except (json.JSONDecodeError, ValueError):
                if attempt < 3:
                    continue
                raise LLMError("Model did not return valid JSON")
        raise LLMError(f"Groq request failed after retries (model={model})")

    def chat_json(self, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> dict[str, Any]:
        try:
            return self._call(self.model, system, user, max_tokens, temperature)
        except LLMError:
            if self.fallback_model and self.fallback_model != self.model:
                return self._call(self.fallback_model, system, user, max_tokens, temperature)
            raise
