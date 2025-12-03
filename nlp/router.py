"""LLM routing and payload normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from nlp.intent import Intent
from tools.registry import ToolRegistry


@dataclass
class RouterDecision:
    model: str
    payload: Dict[str, Any]


class LLMRouter:
    def __init__(self, default_system_prompt: Optional[str] = None, tool_registry: Optional[ToolRegistry] = None) -> None:
        self.default_system_prompt = default_system_prompt or (
            "You are Jarvis, a helpful assistant. Maintain privacy (never leak secrets/voiceprints). "
            "Be cost-aware: use concise Haiku responses for simple tasks; use Sonnet for complex reasoning. "
            "Use tools only when they clearly help the user."
        )
        self.tool_registry = tool_registry

    def normalize_request(self, intent: Intent, message: str, tools: Optional[list] = None) -> RouterDecision:
        model = intent.model
        tool_defs: Optional[List[Dict[str, Any]]] = None
        if tools:
            tool_defs = self._tool_definitions(tools)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.default_system_prompt},
                {"role": "user", "content": message},
            ],
        }
        if tool_defs:
            payload["tools"] = tool_defs
            payload["tool_choice"] = "auto"
        if intent.confidence < 0.5:
            payload["messages"].append(
                {
                    "role": "assistant",
                    "content": "I'm not fully confident about your intent. Could you clarify what you want?",
                }
            )
            payload["needs_clarification"] = True
        return RouterDecision(model=model, payload=payload)

    def _tool_definitions(self, tools: list) -> List[Dict[str, Any]]:
        definitions: List[Dict[str, Any]] = []
        schemas = self.tool_registry.describe() if self.tool_registry else {}
        for name in tools:
            definitions.append(
                {
                    "name": name,
                    "description": schemas.get(name, {}).get("description", f"Invoke tool {name}"),
                    "input_schema": schemas.get(name, {}).get("input_schema", {"type": "object", "properties": {}}),
                }
            )
        return definitions
