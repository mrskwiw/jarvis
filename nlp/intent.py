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
    confidence: float


class IntentClassifier:
    def classify(self, text: str) -> Intent:
        lowered = text.lower()
        label = "chat"
        confidence = 0.5
        if any(keyword in lowered for keyword in ["email", "inbox", "draft"]):
            label = "email"
            confidence = 0.8
        elif any(keyword in lowered for keyword in ["call", "dial", "ring"]):
            label = "call"
            confidence = 0.8
        elif any(keyword in lowered for keyword in ["blog", "publish", "post"]):
            label = "blog"
            confidence = 0.75
        elif any(keyword in lowered for keyword in ["run", "execute", "tool"]):
            label = "tool-needed"
            confidence = 0.7

        complexity = self._heuristic_complexity(lowered)
        model = "haiku" if complexity == "simple" else "sonnet"
        # Lower confidence for short/ambiguous chat.
        if label == "chat" and len(lowered.split()) < 4:
            confidence = 0.4
        return Intent(label=label, complexity=complexity, model=model, confidence=confidence)

    def _heuristic_complexity(self, text: str) -> str:
        token_count = len(text.split())
        if token_count > 40 or any(word in text for word in ["summarize", "analyze", "compose"]):
            return "complex"
        return "simple"
