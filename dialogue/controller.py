"""Conversation controller maintaining rolling context."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from nlp.intent import Intent
from nlp.router import LLMRouter
from observability.logging import RedactingLogger


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class ConversationController:
    router: LLMRouter
    max_history: int = 6
    logger: RedactingLogger = field(default_factory=lambda: RedactingLogger(__name__))
    history: Deque[ConversationTurn] = field(default_factory=deque)

    def record_turn(self, role: str, content: str) -> None:
        self.history.append(ConversationTurn(role=role, content=content))
        while len(self.history) > self.max_history:
            self.history.popleft()

    def build_context(self) -> List[dict]:
        return [{"role": turn.role, "content": turn.content} for turn in self.history]

    def respond(self, intent: Intent, message: str, tools: Optional[list] = None) -> dict:
        self.record_turn("user", message)
        context_messages = [{"role": "system", "content": "Recent conversation"}] + self.build_context()
        decision = self.router.normalize_request(intent, message, tools=tools)
        payload = {**decision.payload, "context": context_messages}
        self.logger.info("Routed message to %s", decision.model)
        return payload

    def summarize_task(self, result: str) -> str:
        summary = f"Summary: {result}"
        self.record_turn("assistant", summary)
        return summary
