"""Minimal email adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class EmailMessage:
    subject: str
    sender: str
    body: str


class EmailService:
    def __init__(self, imap_url: str = "", smtp_url: str = "") -> None:
        self.imap_url = imap_url
        self.smtp_url = smtp_url

    def list_inbox(self) -> List[EmailMessage]:
        # Placeholder for IMAP list
        return []

    def read_message(self, message_id: str) -> EmailMessage:
        # Placeholder for IMAP fetch
        return EmailMessage(subject="", sender="", body="")

    def draft_reply(self, to: str, subject: str, body: str) -> EmailMessage:
        return EmailMessage(subject=subject, sender=to, body=body)
