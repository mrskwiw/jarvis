"""Optional TTS backend with simple caching."""
from __future__ import annotations

from typing import Dict


class TTSBackend:
    """Protocol-like base class."""

    def synthesize(self, text: str) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError


class SimpleTTS(TTSBackend):
    """Placeholder TTS backend that returns UTF-8 bytes of the text."""

    def synthesize(self, text: str) -> bytes:
        return text.encode("utf-8")


class CachedTTS:
    """Caches synthesized audio by text to avoid repeated work/cost."""

    def __init__(self, backend: TTSBackend | None = None) -> None:
        self.backend = backend or SimpleTTS()
        self.cache: Dict[str, bytes] = {}

    def synthesize(self, text: str) -> bytes:
        if text in self.cache:
            return self.cache[text]
        audio = self.backend.synthesize(text)
        self.cache[text] = audio
        return audio
