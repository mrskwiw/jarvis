"""Speaker verification helpers."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Iterable, List, Protocol


class VerificationError(Exception):
    """Raised when speaker verification fails."""


class EmbeddingModel(Protocol):
    def embed(self, frames: Iterable[bytes], sample_rate: int) -> List[float]:
        ...  # pragma: no cover - protocol stub


class HashEmbeddingModel:
    """Simple, dependency-free embedding approximation.

    This model hashes each frame and averages the digest bytes to produce a fixed-size
    vector. It is not intended for production-grade verification but provides a clear
    interface boundary for plugging in pyannote or resemblyzer.
    """

    def __init__(self, length: int = 32) -> None:
        self.length = length

    def embed(self, frames: Iterable[bytes], sample_rate: int) -> List[float]:
        accum = [0] * self.length
        total = 0
        for frame in frames:
            digest = hashlib.sha256(frame).digest()
            for i in range(self.length):
                accum[i] += digest[i]
            total += 1
        if total == 0:
            return [0.0] * self.length
        return [value / float(total) for value in accum]


class ResemblyzerEmbeddingModel:
    """Optional real embedding backend using resemblyzer.

    This adapter is loaded only when requested, to keep the dependency optional.
    """

    def __init__(self) -> None:
        try:
            from resemblyzer import VoiceEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised via factory error path
            raise ImportError(
                "Resemblyzer backend requested but 'resemblyzer' is not installed. "
                "Install with: pip install resemblyzer"
            ) from exc
        self._encoder = VoiceEncoder()

    def embed(self, frames: Iterable[bytes], sample_rate: int) -> List[float]:
        # Resemblyzer expects a waveform; we concatenate frames and let it handle framing.
        # This keeps compatibility with the existing pipeline without changing the call site.
        import numpy as np  # type: ignore

        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive for embedding")
        if not frames:
            return []
        waveform = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        if waveform.size == 0:
            return []
        return self._encoder.embed_utterance(waveform, sample_rate=sample_rate).tolist()


def load_embedding_model(name: str = "hash") -> EmbeddingModel:
    """Factory for embedding models.

    Supported names:
    - "hash" (default): lightweight, dependency-free hashing approximation.
    - "resemblyzer": real speaker encoder; requires `pip install resemblyzer`.
    """

    normalized = name.lower()
    if normalized == "hash":
        return HashEmbeddingModel()
    if normalized == "resemblyzer":
        return ResemblyzerEmbeddingModel()
    raise ValueError(f"Unknown embedding model: {name}")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embeddings must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VoiceprintStore:
    """Stores encrypted owner embeddings on disk."""

    def __init__(self, path: str, key_env_var: str = "JARVIS_VOICE_KEY") -> None:
        self.path = path
        self.key_env_var = key_env_var

    @property
    def _key(self) -> bytes:
        return require_voice_key(self.key_env_var)

    def save(self, embedding: List[float]) -> None:
        encoded = self._encrypt(embedding)
        with open(self.path, "wb") as f:
            f.write(encoded)

    def load(self) -> List[float]:
        with open(self.path, "rb") as f:
            data = f.read()
        return self._decrypt(data)

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def _encrypt(self, embedding: List[float]) -> bytes:
        raw = ",".join(str(x) for x in embedding).encode()
        mask = self._key
        cipher = bytes(b ^ mask[i % len(mask)] for i, b in enumerate(raw))
        return base64.b64encode(cipher)

    def _decrypt(self, payload: bytes) -> List[float]:
        cipher = base64.b64decode(payload)
        mask = self._key
        raw = bytes(b ^ mask[i % len(mask)] for i, b in enumerate(cipher))
        return [float(x) for x in raw.decode().split(",") if x]


def require_voice_key(key_env_var: str = "JARVIS_VOICE_KEY") -> bytes:
    """Fail-fast helper to ensure the voice key is present and usable."""

    key = os.environ.get(key_env_var)
    if not key:
        raise RuntimeError(
            f"Missing voice encryption key env var: {key_env_var}. "
            "Generate a strong secret and export it (for example: "
            f'SET {key_env_var}=\"your-random-key\" on Windows).'
        )
    digest = hashlib.sha256(key.encode()).digest()
    return digest


@dataclass
class SpeakerVerifier:
    embedding_model: EmbeddingModel
    store: VoiceprintStore
    threshold: float = 0.8

    def enroll_owner(self, frames: Iterable[bytes], sample_rate: int) -> List[float]:
        embedding = self.embedding_model.embed(frames, sample_rate)
        self.store.save(embedding)
        return embedding

    def verify_owner(self, frames: Iterable[bytes], sample_rate: int) -> float:
        if not self.store.exists():
            raise VerificationError("Owner has not been enrolled")
        owner_embedding = self.store.load()
        candidate = self.embedding_model.embed(frames, sample_rate)
        similarity = cosine_similarity(owner_embedding, candidate)
        if similarity < self.threshold:
            raise VerificationError("Speaker does not match owner")
        return similarity
