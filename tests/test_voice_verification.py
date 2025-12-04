import pytest

from voice.verification import (
    HashEmbeddingModel,
    ResemblyzerEmbeddingModel,
    SpeakerVerifier,
    VerificationError,
    VoiceprintStore,
    cosine_similarity,
    load_embedding_model,
)


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


def test_hash_embedding_model_empty_frames():
    model = HashEmbeddingModel(length=4)
    embedding = model.embed([], sample_rate=16000)
    assert embedding == [0.0, 0.0, 0.0, 0.0]


def test_voiceprint_store_requires_key(monkeypatch, tmp_path):
    monkeypatch.delenv("JARVIS_VOICE_KEY", raising=False)
    store = VoiceprintStore(tmp_path / "missing.voice")
    with pytest.raises(RuntimeError):
        store.save([1.0, 2.0])


def test_speaker_verifier_enrollment_and_threshold(monkeypatch):
    class StubStore:
        def __init__(self) -> None:
            self.embedding = None

        def save(self, embedding):
            self.embedding = embedding

        def load(self):
            return self.embedding

        def exists(self):
            return self.embedding is not None

    class FakeEmbeddingModel:
        def embed(self, frames, sample_rate):
            if frames == [b"owner"]:
                return [1.0, 0.0]
            return [0.0, 1.0]

    store = StubStore()
    verifier = SpeakerVerifier(FakeEmbeddingModel(), store, threshold=0.8)
    verifier.enroll_owner([b"owner"], sample_rate=16000)

    with pytest.raises(VerificationError):
        verifier.verify_owner([b"impostor"], sample_rate=16000)

    assert store.exists()


def test_verify_owner_requires_enrollment(tmp_path):
    store = VoiceprintStore(tmp_path / "missing.voice")
    verifier = SpeakerVerifier(HashEmbeddingModel(), store)
    with pytest.raises(VerificationError):
        verifier.verify_owner([b"audio"], sample_rate=16000)


def test_cosine_similarity_edge_cases():
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 2.0])

    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_embedding_factory_defaults_and_errors(monkeypatch):
    # Default is hash
    model = load_embedding_model()
    assert isinstance(model, HashEmbeddingModel)

    # Unknown backend raises
    with pytest.raises(ValueError):
        load_embedding_model("unknown-backend")

    # Resemblyzer raises helpful ImportError when not installed
    with pytest.raises(ImportError):
        load_embedding_model("resemblyzer")
