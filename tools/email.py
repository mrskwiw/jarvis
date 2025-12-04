"""Email adapter with Gmail OAuth and POP support (dry-run by default)."""
from __future__ import annotations

import os
import urllib.parse
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class EmailMessage:
    subject: str
    sender: str
    body: str


@dataclass
class GmailOAuthConfig:
    client_id: str
    client_secret: str
    refresh_token: str

    @classmethod
    def from_env(cls) -> "GmailOAuthConfig":
        cid = os.environ.get("GMAIL_CLIENT_ID")
        csecret = os.environ.get("GMAIL_CLIENT_SECRET")
        refresh = os.environ.get("GMAIL_REFRESH_TOKEN")
        if not cid or not csecret or not refresh:
            raise RuntimeError("Missing Gmail OAuth env vars: GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN")
        return cls(client_id=cid, client_secret=csecret, refresh_token=refresh)


def gmail_oauth_authorize_url(client_id: str, redirect_uri: str, scopes: Optional[List[str]] = None) -> str:
    scopes = scopes or ["https://mail.google.com/"]
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",
        "prompt": "consent",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)


class EmailService:
    """Dry-run email service that can be configured for Gmail OAuth or POP.

    By default this does not perform network calls; it returns structured results
    that can be used downstream or swapped for real implementations.
    """

    def __init__(self, provider: str = "dry-run", pop_host: str = "", pop_user: str = "", dry_run: bool = True) -> None:
        self.provider = provider
        self.pop_host = pop_host
        self.pop_user = pop_user
        self.dry_run = dry_run
        self.gmail_config: Optional[GmailOAuthConfig] = None
        if provider == "gmail":
            self.gmail_config = GmailOAuthConfig.from_env()

    def send_email(self, to: str, subject: str, body: str) -> Dict[str, object]:
        if self.provider == "gmail":
            if not self.gmail_config:
                raise RuntimeError("Gmail OAuth not configured")
            return {
                "provider": "gmail",
                "to": to,
                "subject": subject,
                "body": body,
                "mode": "dry-run" if self.dry_run else "real",
            }
        if self.provider == "pop":
            return {
                "provider": "pop",
                "to": to,
                "subject": subject,
                "body": body,
                "pop_host": self.pop_host,
                "pop_user": self.pop_user,
                "mode": "dry-run" if self.dry_run else "real",
            }
        return {"provider": "dry-run", "to": to, "subject": subject, "body": body}

    def list_inbox(self) -> List[EmailMessage]:
        # Stub: return empty list
        return []

    def read_message(self, message_id: str) -> EmailMessage:
        return EmailMessage(subject="", sender="", body="")

    def draft_reply(self, to: str, subject: str, body: str) -> EmailMessage:
        return EmailMessage(subject=subject, sender=to, body=body)
