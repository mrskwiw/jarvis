import asyncio

from api.server import VoiceAgent
from voice.listener import audio_stream_from_queue


def test_voice_agent_process_audio(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_VOICE_KEY", "agent-key")

    async def run():
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        audio_source = audio_stream_from_queue(queue)

        agent = VoiceAgent(audio_source=audio_source, voiceprint_path=str(tmp_path / "voiceprint"))
        agent.listener.verifier.enroll_owner([b"jarvis"], sample_rate=agent.listener.sample_rate)
        agent.listener.verifier.threshold = 0.5

        queue.put_nowait(b"ambient noise")
        queue.put_nowait(b"Jarvis wake")
        queue.put_nowait(b"compose an email")
        for _ in range(agent.listener.silence_after_frames):
            queue.put_nowait(b"\x00")
        queue.put_nowait(None)

        payload = await agent.process_audio_command()

        assert payload["transcription"].text
        assert payload["transcription"].source in {"local_whisper", "cloud_fallback"}
        assert "context" in payload

    asyncio.run(run())


def test_voice_agent_health(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_VOICE_KEY", "agent-key")
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    audio_source = audio_stream_from_queue(queue)
    agent = VoiceAgent(audio_source=audio_source, voiceprint_path=str(tmp_path / "voiceprint"))

    # Initially no voiceprint file.
    health = agent.health()
    assert health["env"]["ok"] is True
    assert health["voiceprint_exists"] is False

    # After enrollment, the health should reflect presence.
    agent.listener.verifier.enroll_owner([b"jarvis"], sample_rate=agent.listener.sample_rate)
    health_after = agent.health()
    assert health_after["voiceprint_exists"] is True
