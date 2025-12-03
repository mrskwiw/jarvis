import asyncio

import pytest

from voice.listener import ContinuousListener, WakeWordDetector, audio_stream_from_queue
from voice.verification import HashEmbeddingModel, SpeakerVerifier, VoiceprintStore
from observability.metrics import MetricsSink


def test_listen_for_command_captures_audio(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_VOICE_KEY", "listener-key")

    async def run():
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        audio_source = audio_stream_from_queue(queue)

        store = VoiceprintStore(tmp_path / "owner.voice")
        verifier = SpeakerVerifier(HashEmbeddingModel(), store, threshold=0.5)
        verifier.enroll_owner([b"jarvis"], sample_rate=16000)

        metrics = MetricsSink()
        listener = ContinuousListener(
            wake_detector=WakeWordDetector("jarvis"),
            verifier=verifier,
            audio_source=audio_source,
            metrics=metrics,
            silence_after_frames=2,
            max_command_seconds=1,
        )

        queue.put_nowait(b"noise frame")
        queue.put_nowait(b"Jarvis start")
        queue.put_nowait(b"execute task")
        queue.put_nowait(b"\x00\x00")
        queue.put_nowait(b"\x00\x00")
        queue.put_nowait(None)

        verified_audio = await listener.listen_for_command()

        assert len(verified_audio.frames) >= 3
        assert metrics.snapshot()["wake_word_detected"] == 1
        assert metrics.snapshot()["speaker_verified"] == 1

    asyncio.run(run())
