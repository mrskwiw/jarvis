import asyncio

import pytest

from observability.metrics import MetricsSink
from voice.listener import ContinuousListener, WakeWordDetector, audio_stream_from_queue
from voice.verification import VerificationError


def test_wake_word_detector_paths():
    custom = WakeWordDetector("jarvis", detector=lambda frame: frame == b"wake")
    assert custom.heard(b"wake")
    assert not custom.heard(b"other")

    fallback = WakeWordDetector("jarvis")
    assert fallback.heard(b"Hello Jarvis")

    class BadFrame:
        def decode(self, errors="ignore"):
            raise ValueError("boom")

    assert fallback.heard(BadFrame()) is False


def test_capture_stops_on_silence_and_max_frames():
    async def run_silence():
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        audio_source = audio_stream_from_queue(queue)
        listener = ContinuousListener(
            wake_detector=WakeWordDetector("jarvis"),
            verifier=None,  # not used
            audio_source=audio_source,
            silence_after_frames=2,
            sample_rate=16000,
        )
        queue.put_nowait(b"start")
        queue.put_nowait(b"\x00\x00")
        queue.put_nowait(b"\x00\x00")
        queue.put_nowait(None)
        frames = await listener._capture_command_frames(initial_frame=await queue.get())
        assert len(frames) == 3

    async def run_max_frames():
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        audio_source = audio_stream_from_queue(queue)
        listener = ContinuousListener(
            wake_detector=WakeWordDetector("jarvis"),
            verifier=None,
            audio_source=audio_source,
            sample_rate=1024,
            max_command_seconds=1,
            silence_after_frames=10,
        )
        queue.put_nowait(b"start")
        queue.put_nowait(b"more")
        queue.put_nowait(b"extra")
        queue.put_nowait(None)
        frames = await listener._capture_command_frames(initial_frame=await queue.get())
        assert len(frames) == 2

    asyncio.run(run_silence())
    asyncio.run(run_max_frames())


def test_listen_for_command_rejection_and_missing_wake():
    class RejectVerifier:
        def verify_owner(self, frames, sample_rate):
            raise VerificationError("reject")

    async def run_short_speech():
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        audio_source = audio_stream_from_queue(queue)
        metrics = MetricsSink()
        listener = ContinuousListener(
            wake_detector=WakeWordDetector("jarvis"),
            verifier=RejectVerifier(),
            audio_source=audio_source,
            silence_after_frames=1,
            metrics=metrics,
            min_speech_frames=2,
            min_command_frames=2,
        )
        queue.put_nowait(b"Jarvis wake")
        queue.put_nowait(b"\x00\x00")
        queue.put_nowait(None)
        with pytest.raises(VerificationError):
            await listener.listen_for_command()
        snapshot = metrics.snapshot()
        assert snapshot["speech_rejected_short"] == 1
        assert snapshot["speaker_rejected"] == 1

    async def run_reject():
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        audio_source = audio_stream_from_queue(queue)
        metrics = MetricsSink()
        listener = ContinuousListener(
            wake_detector=WakeWordDetector("jarvis"),
            verifier=RejectVerifier(),
            audio_source=audio_source,
            silence_after_frames=1,
            metrics=metrics,
            min_speech_frames=1,
        )
        queue.put_nowait(b"noise")
        queue.put_nowait(b"Jarvis wake")
        queue.put_nowait(b"\x00\x00")
        queue.put_nowait(None)
        with pytest.raises(VerificationError):
            await listener.listen_for_command()
        snapshot = metrics.snapshot()
        assert snapshot["wake_word_detected"] == 1
        assert snapshot["speaker_rejected"] == 1

    async def run_no_wake():
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        audio_source = audio_stream_from_queue(queue)
        listener = ContinuousListener(
            wake_detector=WakeWordDetector("jarvis"),
            verifier=RejectVerifier(),
            audio_source=audio_source,
            silence_after_frames=1,
        )
        queue.put_nowait(b"ambient")
        queue.put_nowait(None)
        with pytest.raises(RuntimeError):
            await listener.listen_for_command()

    asyncio.run(run_short_speech())
    asyncio.run(run_reject())
    asyncio.run(run_no_wake())
