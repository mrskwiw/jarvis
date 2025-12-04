import asyncio
import runpy
import sys
import types

import pytest

from voice import mic_cli


def test_start_mic_stream_requires_sounddevice(monkeypatch):
    monkeypatch.setattr(mic_cli, "sd", None)
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    with pytest.raises(RuntimeError):
        mic_cli._start_mic_stream(queue, sample_rate=16000, blocksize=1024)


def test_start_mic_stream_pushes_frames(monkeypatch):
    pushed = asyncio.Queue()

    class DummyStream:
        def __init__(self, callback):
            self.callback = callback
            self.started = False

        def start(self):
            self.started = True
            self.callback(b"abc", None, None, True)
            return self

    class DummySD:
        def RawInputStream(self, samplerate, blocksize, channels, dtype, callback):
            return DummyStream(callback)

    monkeypatch.setattr(mic_cli, "sd", DummySD())
    stream = mic_cli._start_mic_stream(pushed, sample_rate=16000, blocksize=1024)
    assert stream.started
    assert pushed.get_nowait() == b"abc"


def test_run_agent_handles_keyboard_interrupt(monkeypatch, capsys):
    class DummyStream:
        def __init__(self) -> None:
            self.stopped = False
            self.closed = False

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    dummy_stream = DummyStream()

    class DummyAgent:
        def __init__(self, audio_source, wake_word, voiceprint_path):
            self.listener = types.SimpleNamespace(verifier=types.SimpleNamespace(threshold=0.0))
            self.calls = 0

        async def process_audio_command(self):
            self.calls += 1
            if self.calls == 1:
                return {"transcription": types.SimpleNamespace(text="hi"), "intent": "email", "context": [1, 2, 3]}
            raise KeyboardInterrupt

    monkeypatch.setattr(mic_cli, "_start_mic_stream", lambda *_, **__: dummy_stream)
    monkeypatch.setattr(mic_cli, "VoiceAgent", DummyAgent)

    asyncio.run(mic_cli.run_agent("jarvis", "owner.voice", threshold=0.5))

    assert dummy_stream.stopped and dummy_stream.closed
    output = capsys.readouterr().out
    assert "Stopping listener." in output


def test_run_agent_file_fallback(monkeypatch, tmp_path, capsys):
    # Ensure that when audio_path is provided we do not require sounddevice
    monkeypatch.setattr(mic_cli, "sd", None)

    audio_file = tmp_path / "audio.raw"
    audio_file.write_bytes(b"wake and command")

    class DummyStream:
        pass

    class DummyAgent:
        def __init__(self, audio_source, wake_word, voiceprint_path):
            self.listener = types.SimpleNamespace(verifier=types.SimpleNamespace(threshold=0.0))
            self.calls = 0

        async def process_audio_command(self):
            self.calls += 1
            if self.calls == 1:
                return {"transcription": types.SimpleNamespace(text="hi"), "intent": "email", "context": [1]}
            raise KeyboardInterrupt

    called = {}

    def fake_stream_file(queue, audio_path, blocksize):
        called["audio_path"] = audio_path
        queue.put_nowait(b"Jarvis wake")
        queue.put_nowait(b"do task")
        queue.put_nowait(b"\x00")
        queue.put_nowait(None)

    monkeypatch.setattr(mic_cli, "_start_mic_stream", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("should not call mic")))
    monkeypatch.setattr(mic_cli, "_stream_file_to_queue", fake_stream_file)
    monkeypatch.setattr(mic_cli, "VoiceAgent", DummyAgent)

    asyncio.run(mic_cli.run_agent("jarvis", "owner.voice", threshold=0.5, audio_path=str(audio_file)))

    assert called["audio_path"] == str(audio_file)
    output = capsys.readouterr().out
    assert "Stopping listener." in output


def test_mic_cli_main_and_entrypoint(monkeypatch):
    called = {}

    async def fake_run_agent(wake_word, voiceprint_path, threshold, sample_rate, blocksize):
        called.update(
            wake_word=wake_word,
            voiceprint_path=voiceprint_path,
            threshold=threshold,
            sample_rate=sample_rate,
            blocksize=blocksize,
        )

    monkeypatch.setattr(mic_cli, "run_agent", fake_run_agent)
    monkeypatch.setattr(
        mic_cli,
        "VoiceAgent",
        lambda *args, **kwargs: types.SimpleNamespace(listener=types.SimpleNamespace(verifier=types.SimpleNamespace(threshold=0.0))),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["voice.mic_cli", "--wake-word", "hey", "--voiceprint", "vp", "--threshold", "0.5", "--sample-rate", "8000", "--blocksize", "512"],
    )
    mic_cli.main()
    assert called["wake_word"] == "hey"
    assert called["voiceprint_path"] == "vp"
    assert called["sample_rate"] == 8000
    assert called["blocksize"] == 512
