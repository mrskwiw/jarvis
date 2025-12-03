import pytest

from voice.verification import HashEmbeddingModel, SpeakerVerifier, VerificationError, VoiceprintStore


def test_voiceprint_store_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_VOICE_KEY", "test-key")
    store = VoiceprintStore(tmp_path / "owner.voice")
    embedding = [0.1, 0.2, 0.3]

    store.save(embedding)

    assert store.exists()
    assert store.load() == embedding


def test_speaker_verifier_accepts_and_rejects(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_VOICE_KEY", "test-key")
    store = VoiceprintStore(tmp_path / "owner.voice")
    verifier = SpeakerVerifier(HashEmbeddingModel(), store, threshold=0.99)

    owner_frames = [b"owner-audio"]
    verifier.enroll_owner(owner_frames, sample_rate=16000)

    similarity = verifier.verify_owner(owner_frames, sample_rate=16000)
    assert similarity >= verifier.threshold

    with pytest.raises(VerificationError):
        verifier.verify_owner([b"intruder"], sample_rate=16000)
