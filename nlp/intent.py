"""Intent classification and complexity heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


INTENT_LABELS = ["chat", "email", "call", "blog", "tool-needed"]


@dataclass
class Intent:
    label: str
    complexity: str
    model: str


class IntentClassifier:
    def classify(self, text: str) -> Intent:
        lowered = text.lower()
        label = "chat"
        if any(keyword in lowered for keyword in ["email", "inbox", "draft"]):
            label = "email"
        elif any(keyword in lowered for keyword in ["call", "dial", "ring"]):
            label = "call"
        elif any(keyword in lowered for keyword in ["blog", "publish", "post"]):
            label = "blog"
        elif any(keyword in lowered for keyword in ["run", "execute", "tool"]):
            label = "tool-needed"

        complexity = self._heuristic_complexity(lowered)
        model = "haiku" if complexity == "simple" else "sonnet"
        return Intent(label=label, complexity=complexity, model=model)

    def _heuristic_complexity(self, text: str) -> str:
        token_count = len(text.split())
        if token_count > 40 or any(word in text for word in ["summarize", "analyze", "compose"]):
            return "complex"
        return "simple"
