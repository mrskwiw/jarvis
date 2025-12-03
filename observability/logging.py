"""Logging helpers with redaction for sensitive fields."""
from __future__ import annotations

import logging
import re
from typing import Iterable


class RedactingFormatter(logging.Formatter):
    REDACTION = "[REDACTED]"
    PATTERNS = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in ["password", "secret", "token", "voiceprint"]
    ]

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        message = super().format(record)
        for pattern in self.PATTERNS:
            message = pattern.sub(self.REDACTION, message)
        return message


class RedactingLogger(logging.LoggerAdapter):
    def __init__(self, name: str) -> None:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(RedactingFormatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        super().__init__(logger, {})

    def process(self, msg, kwargs):  # type: ignore[override]
        return msg, kwargs
