"""ASR router supporting local and cloud backends."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional

from observability.logging import RedactingLogger


@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    source: str
    latency_ms: Optional[float] = None


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
        self.stream_timeout = 5.0

    def transcribe(self, audio_frames, sample_rate: int) -> TranscriptionResult:
        local_result = self.local.transcribe(audio_frames, sample_rate)
        if local_result.confidence >= self.threshold:
            return local_result
        return self.cloud.transcribe(audio_frames, sample_rate)

    async def transcribe_streaming(self, audio_source: AsyncIterator[bytes], sample_rate: int) -> TranscriptionResult:
        """Stream audio to local ASR first, then fall back to cloud with timeout."""

        local_text = await self._stream_collect(audio_source, sample_rate, prefer_cloud=False)
        if local_text.confidence >= self.threshold:
            return local_text
        return await self._stream_collect(audio_source, sample_rate, prefer_cloud=True)

    async def _stream_collect(
        self,
        audio_source: AsyncIterator[bytes],
        sample_rate: int,
        prefer_cloud: bool,
    ) -> TranscriptionResult:
        backend = self.cloud if prefer_cloud else self.local
        chunks: list[bytes] = []

        async def consume():
            async for frame in audio_source:
                chunks.append(frame)

        try:
            await asyncio.wait_for(consume(), timeout=self.stream_timeout)
        except asyncio.TimeoutError:
            pass

        result = backend.transcribe(chunks, sample_rate)
        result.source = f"{result.source}{'_stream' if prefer_cloud else '_local_stream'}"
        result.latency_ms = None  # placeholder; could be wired to wall clock if needed
        return result
