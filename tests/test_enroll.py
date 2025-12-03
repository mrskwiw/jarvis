import os
import runpy
import sys
import warnings

import pytest

from voice.enroll import enroll_from_file
from voice.verification import require_voice_key


def test_enroll_from_file_creates_voiceprint(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_VOICE_KEY", "enroll-key")
    audio_path = tmp_path / "owner.raw"
    audio_path.write_bytes(b"jarvis-owner-audio")
    voiceprint_path = tmp_path / "owner.voice"

    result = enroll_from_file(
        audio_path=str(audio_path),
        voiceprint_path=str(voiceprint_path),
        sample_rate=16000,
        chunk_size=4,
    )

    assert voiceprint_path.exists()
    assert result["frames"] > 0
    assert result["embedding_length"] > 0


def test_require_voice_key_fails_without_env(monkeypatch):
    monkeypatch.delenv("JARVIS_VOICE_KEY", raising=False)
    with pytest.raises(RuntimeError):
        require_voice_key()


def test_enroll_from_file_missing_audio_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_VOICE_KEY", "enroll-key")
    audio_path = tmp_path / "empty.raw"
    audio_path.write_bytes(b"")
    with pytest.raises(ValueError):
        enroll_from_file(audio_path=str(audio_path), voiceprint_path=str(tmp_path / "owner.voice"))


def test_enroll_main_invokes_pipeline(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_VOICE_KEY", "enroll-key")
    audio_path = tmp_path / "owner.raw"
    audio_path.write_bytes(b"frames-here")
    voiceprint_path = tmp_path / "owner.voice"

    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["voice.enroll", "--audio", str(audio_path), "--voiceprint", str(voiceprint_path)])
    from voice import enroll as enroll_module

    enroll_module.main()
    captured = capsys.readouterr()
    assert "Enrollment complete" in captured.out
    assert voiceprint_path.exists()


def test_enroll_entrypoint_runs(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("JARVIS_VOICE_KEY", "enroll-key")
    audio_path = tmp_path / "owner.raw"
    audio_path.write_bytes(b"frames")
    voiceprint_path = tmp_path / "owner.voice"
    monkeypatch.setattr(
        sys,
        "argv",
        ["voice.enroll", "--audio", str(audio_path), "--voiceprint", str(voiceprint_path)],
    )
    # Remove any existing module instance to avoid runpy warnings about preloaded modules.
    sys.modules.pop("voice.enroll", None)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        runpy.run_module("voice.enroll", run_name="__main__")
    captured = capsys.readouterr()
    assert "Enrollment complete" in captured.out
