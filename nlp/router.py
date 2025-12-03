"""LLM routing and payload normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from nlp.intent import Intent


@dataclass
class RouterDecision:
    model: str
    payload: Dict[str, Any]


class LLMRouter:
    def __init__(self, default_system_prompt: Optional[str] = None) -> None:
        self.default_system_prompt = default_system_prompt or "You are Jarvis, a helpful assistant."

    def normalize_request(self, intent: Intent, message: str, tools: Optional[list] = None) -> RouterDecision:
        model = intent.model
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.default_system_prompt},
                {"role": "user", "content": message},
            ],
        }
        if tools:
            payload["tools"] = tools
        return RouterDecision(model=model, payload=payload)
