"""CLI helper to enroll the owner voiceprint from an audio file.

Usage:
    python -m voice.enroll --audio owner.raw --voiceprint ./owner.voice --sample-rate 16000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from voice.verification import SpeakerVerifier, VoiceprintStore, load_embedding_model, require_voice_key


def _read_frames(path: Path, chunk_size: int) -> list[bytes]:
    frames: list[bytes] = []
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            frames.append(chunk)
    return frames


def enroll_from_file(
    audio_path: str,
    voiceprint_path: str,
    sample_rate: int = 16000,
    chunk_size: int = 1024,
    key_env_var: str = "JARVIS_VOICE_KEY",
) -> Dict[str, object]:
    """Enroll the owner voiceprint from a raw/PCM audio file."""

    require_voice_key(key_env_var)
    path = Path(audio_path)
    frames = _read_frames(path, chunk_size=chunk_size)
    if not frames:
        raise ValueError(f"No audio frames read from {audio_path}")

    store = VoiceprintStore(voiceprint_path, key_env_var=key_env_var)
    verifier = SpeakerVerifier(load_embedding_model(), store)
    embedding = verifier.enroll_owner(frames, sample_rate=sample_rate)

    return {
        "voiceprint_path": voiceprint_path,
        "frames": len(frames),
        "embedding_length": len(embedding),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll the JARVIS owner voiceprint.")
    parser.add_argument("--audio", required=True, help="Path to raw/PCM audio file for enrollment.")
    parser.add_argument(
        "--voiceprint",
        default="./owner.voiceprint",
        help="Where to store the encrypted voiceprint (default: ./owner.voiceprint).",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate of the audio (default: 16000).")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size when reading audio (default: 1024).")
    parser.add_argument(
        "--key-env-var",
        default="JARVIS_VOICE_KEY",
        help="Env var containing the voice encryption key (default: JARVIS_VOICE_KEY).",
    )

    args = parser.parse_args()
    result = enroll_from_file(
        audio_path=args.audio,
        voiceprint_path=args.voiceprint,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        key_env_var=args.key_env_var,
    )
    print(
        f"Enrollment complete. Stored voiceprint at {result['voiceprint_path']} "
        f"(frames: {result['frames']}, embedding length: {result['embedding_length']})."
    )


if __name__ == "__main__":
    main()
