"""Call adapter stub."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CallSession:
    to: str
    status: str


class CallService:
    def __init__(self, provider: str = "twilio") -> None:
        self.provider = provider

    def place_call(self, to: str, message: str) -> CallSession:
        return CallSession(to=to, status="queued")

    def receive_call(self, from_number: str) -> CallSession:
        return CallSession(to=from_number, status="received")
