"""Voice listener with wake word detection and speaker verification."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Iterable, List, Optional

from voice.verification import SpeakerVerifier, VerificationError
from observability.logging import RedactingLogger
from observability.metrics import MetricsSink


@dataclass
class VerifiedAudio:
    """Captured audio that has passed speaker verification."""

    frames: List[bytes]
    sample_rate: int


class WakeWordDetector:
    """Lightweight wake word detection shim.

    This is intentionally simple to keep the dependency surface small. The detector
    accepts a callable that returns ``True`` when a frame contains the wake word
    (for example, an integration with Porcupine or Snowboy). If no callable is
    provided, a fallback detector inspects the decoded frame for the wake word
    string, which is primarily useful for tests.
    """

    def __init__(
        self,
        wake_word: str,
        detector: Optional[Callable[[bytes], bool]] = None,
    ) -> None:
        self.wake_word = wake_word.lower()
        self._detector = detector

    def heard(self, frame: bytes) -> bool:
        if self._detector:
            return self._detector(frame)
        # Fallback heuristic: check if the frame decodes to text containing the wake word.
        try:
            return self.wake_word in frame.decode(errors="ignore").lower()
        except Exception:
            return False


@dataclass
class ContinuousListener:
    """Continuously monitors audio frames and emits verified commands."""

    wake_detector: WakeWordDetector
    verifier: SpeakerVerifier
    audio_source: AsyncIterator[bytes]
    sample_rate: int = 16000
    logger: RedactingLogger = field(default_factory=lambda: RedactingLogger(__name__))
    metrics: MetricsSink = field(default_factory=MetricsSink)
    max_command_seconds: float = 15.0
    silence_after_frames: int = 30
    min_command_frames: int = 2
    min_speech_frames: int = 2
    energy_threshold: float = 50.0

    async def listen_for_command(self) -> VerifiedAudio:
        """Block until the wake word is heard, then capture and verify audio.

        Returns ``VerifiedAudio`` if the captured command belongs to the owner.
        Raises ``VerificationError`` when the speaker does not match.
        """

        self.logger.debug("Starting continuous listen loop")
        frames: List[bytes] = []

        async for frame in self.audio_source:
            if self.wake_detector.heard(frame):
                self.logger.info("Wake word detected; capturing command audio")
                self.metrics.increment("wake_word_detected")
                start_time = time.monotonic()
                frames = await self._capture_command_frames(initial_frame=frame)
                self.logger.debug("Captured %d frames", len(frames))
                if not self._passes_speech_guardrails(frames):
                    self.metrics.increment("speaker_rejected")
                    self.metrics.increment("speech_rejected_short")
                    self.logger.warning("Insufficient speech captured; rejecting command")
                    raise VerificationError("Insufficient speech captured for verification")
                try:
                    self.verifier.verify_owner(frames, self.sample_rate)
                    self.metrics.increment("speaker_verified")
                    latency_ms = (time.monotonic() - start_time) * 1000.0
                    self.metrics.record_timing("latency_wake_to_verify_ms", latency_ms)
                    self.logger.info("Speaker verified; emitting audio for downstream processing")
                    return VerifiedAudio(frames=frames, sample_rate=self.sample_rate)
                except VerificationError:
                    self.metrics.increment("speaker_rejected")
                    self.logger.warning("Speaker verification failed; rejecting command")
                    raise
        raise RuntimeError("Audio source closed before wake word detected")

    async def _capture_command_frames(self, initial_frame: bytes) -> List[bytes]:
        frames: List[bytes] = [initial_frame]
        silence_counter = 0
        max_frames = int(self.max_command_seconds * (self.sample_rate / 1024))

        async for frame in self.audio_source:
            frames.append(frame)
            if self._is_silent(frame):
                silence_counter += 1
            else:
                silence_counter = 0
            if silence_counter >= self.silence_after_frames:
                self.logger.debug("Detected sustained silence; stopping capture")
                break
            if len(frames) >= max_frames:
                self.logger.debug("Reached max command duration; stopping capture")
                break
        return frames

    def _passes_speech_guardrails(self, frames: List[bytes]) -> bool:
        non_silent = sum(1 for frame in frames if not self._is_silent(frame))
        return len(frames) >= self.min_command_frames and non_silent >= self.min_speech_frames

    def _is_silent(self, frame: bytes) -> bool:
        if not frame:
            return True
        try:
            # Treat frame as PCM int16; fall back to zero-check on decode failures.
            import numpy as np  # type: ignore

            pcm = np.frombuffer(frame, dtype=np.int16)
            if pcm.size == 0:
                return True
            energy = float(np.mean(np.abs(pcm)))
        except Exception:
            # Fallback: consider bytes with non-zero payload as non-silent.
            return not frame.strip(b"\x00")
        return energy < self.energy_threshold


def _porcupine_detector_factory(wake_word: str) -> Callable[[bytes], bool]:
    """Wrap pvporcupine if installed; otherwise raise a clear ImportError."""

    try:
        import pvporcupine  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised via factory error path
        raise ImportError(
            "Porcupine wake-word backend requested but 'pvporcupine' (and numpy) is not installed. "
            "Install with: pip install pvporcupine numpy"
        ) from exc

    porcupine = pvporcupine.create(keywords=[wake_word])

    def detector(frame: bytes) -> bool:
        if not frame:
            return False
        pcm = np.frombuffer(frame, dtype=np.int16)
        if pcm.size == 0:
            return False
        return porcupine.process(pcm) >= 0

    return detector


def load_wake_detector(wake_word: str, backend: str = "fallback") -> WakeWordDetector:
    """Factory for wake-word detectors.

    Supported backends:
    - "fallback" (default): simple text-based heuristic suitable for tests.
    - "porcupine": pvporcupine-based detector (requires dependency).
    """

    normalized = backend.lower()
    if normalized in ("fallback", "simple"):
        return WakeWordDetector(wake_word)
    if normalized == "porcupine":
        return WakeWordDetector(wake_word, detector=_porcupine_detector_factory(wake_word))
    raise ValueError(f"Unknown wake-word backend: {backend}")


async def audio_stream_from_queue(queue: "asyncio.Queue[bytes]") -> AsyncIterator[bytes]:
    """Utility to convert an asyncio queue into an audio source iterator."""

    while True:
        frame = await queue.get()
        if frame is None:
            break
        yield frame
