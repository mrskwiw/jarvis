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
        assert "trace" in payload and len(payload["trace"]) >= 3

    asyncio.run(run())


def test_voice_agent_health(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_VOICE_KEY", "agent-key")
    monkeypatch.delenv("JARVIS_EMBEDDING_BACKEND", raising=False)
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


def test_voice_agent_tts_optional(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_VOICE_KEY", "agent-key")
    monkeypatch.setenv("JARVIS_ENABLE_TTS", "1")
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    audio_source = audio_stream_from_queue(queue)
    agent = VoiceAgent(audio_source=audio_source, voiceprint_path=str(tmp_path / "voiceprint"))
    assert agent.tts is not None
    audio = agent.tts.synthesize("hello")
    assert audio == b"hello"
