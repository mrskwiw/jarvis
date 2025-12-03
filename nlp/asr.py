"""ASR router supporting local and cloud backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from observability.logging import RedactingLogger


@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    source: str


class LocalWhisperASR:
    """Placeholder for a local Whisper small model."""

    def __init__(self, logger: Optional[RedactingLogger] = None) -> None:
        self.logger = logger or RedactingLogger(__name__)

    def transcribe(self, audio_frames, sample_rate: int) -> TranscriptionResult:
        # Placeholder: in production, run whisper.cpp or equivalent.
        text = ""
        for frame in audio_frames:
            text += frame.decode(errors="ignore")
        confidence = 0.65 if text else 0.0
        self.logger.debug("Local ASR produced text=%s confidence=%.2f", text, confidence)
        return TranscriptionResult(text=text.strip(), confidence=confidence, source="local_whisper")


class CloudFallbackASR:
    """Placeholder cloud ASR."""

    def __init__(self, logger: Optional[RedactingLogger] = None) -> None:
        self.logger = logger or RedactingLogger(__name__)

    def transcribe(self, audio_frames, sample_rate: int) -> TranscriptionResult:
        text = "".join(frame.decode(errors="ignore") for frame in audio_frames)
        confidence = 0.85 if text else 0.0
        self.logger.debug("Cloud ASR produced text=%s confidence=%.2f", text, confidence)
        return TranscriptionResult(text=text.strip(), confidence=confidence, source="cloud_fallback")


class ASRRouter:
    def __init__(self, local: LocalWhisperASR, cloud: CloudFallbackASR, threshold: float = 0.7) -> None:
        self.local = local
        self.cloud = cloud
        self.threshold = threshold

    def transcribe(self, audio_frames, sample_rate: int) -> TranscriptionResult:
        local_result = self.local.transcribe(audio_frames, sample_rate)
        if local_result.confidence >= self.threshold:
            return local_result
        return self.cloud.transcribe(audio_frames, sample_rate)
