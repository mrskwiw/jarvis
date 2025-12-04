"""CLI to stream microphone audio into the ContinuousListener/VoiceAgent pipeline.

Dependencies:
    - sounddevice (preferred) or fall back to soundfile reading.

Usage:
    python -m voice.mic_cli --wake-word jarvis --voiceprint ./owner.voice --threshold 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import threading
from typing import Optional

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sd = None  # type: ignore

from api.server import VoiceAgent
from voice.listener import audio_stream_from_queue


def _start_mic_stream(
    queue: "asyncio.Queue[bytes]",
    sample_rate: int,
    blocksize: int,
    channels: int = 1,
) -> Optional[sd.RawInputStream]:
    if sd is None:
        raise RuntimeError(
            "sounddevice is required for mic streaming but is not installed. "
            "Install with: pip install sounddevice"
        )

    def callback(indata, frames, time, status):  # type: ignore[override]
        if status:
            # We log to stderr implicitly via sounddevice status.
            pass
        queue.put_nowait(bytes(indata))

    stream = sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=blocksize,
        channels=channels,
        dtype="int16",
        callback=callback,
    )
    stream.start()
    return stream


def _stream_file_to_queue(queue: "asyncio.Queue[bytes]", audio_path: str, blocksize: int) -> None:
    with open(audio_path, "rb") as f:
        while True:
            chunk = f.read(blocksize)
            if not chunk:
                break
            queue.put_nowait(chunk)


async def run_agent(
    wake_word: str,
    voiceprint_path: str,
    threshold: float,
    sample_rate: int = 16000,
    blocksize: int = 1024,
    audio_path: Optional[str] = None,
) -> None:
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    audio_source = audio_stream_from_queue(queue)
    agent = VoiceAgent(audio_source=audio_source, wake_word=wake_word, voiceprint_path=voiceprint_path)
    agent.listener.verifier.threshold = threshold

    loop = asyncio.get_event_loop()
    stream_holder = {"stream": None, "kind": None}

    def start_stream():
        if audio_path:
            stream_holder["kind"] = "file"
            _stream_file_to_queue(queue, audio_path, blocksize)
            queue.put_nowait(None)
        else:
            stream_holder["kind"] = "mic"
            stream_holder["stream"] = _start_mic_stream(queue, sample_rate=sample_rate, blocksize=blocksize)

    thread = threading.Thread(target=start_stream, daemon=True)
    thread.start()

    print("Listening for wake word... (Ctrl+C to exit)")
    try:
        while True:
            payload = await agent.process_audio_command()
            print(f"Transcription: {payload['transcription'].text}")
            print(f"Intent: {payload.get('intent', 'n/a')}")
            print(f"Context size: {len(payload.get('context', []))}")
    except KeyboardInterrupt:
        print("Stopping listener.")
    finally:
        if stream_holder["kind"] == "mic" and stream_holder["stream"]:
            stream_holder["stream"].stop()
            stream_holder["stream"].close()
        elif stream_holder["kind"] == "file":
            thread.join(timeout=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream microphone audio into JARVIS VoiceAgent.")
    parser.add_argument("--wake-word", default="jarvis", help="Wake word to listen for (default: jarvis).")
    parser.add_argument("--voiceprint", default="./owner.voice", help="Path to the owner voiceprint file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Speaker verification threshold (default: 0.8; lower to be more permissive).",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate (default: 16000).")
    parser.add_argument("--blocksize", type=int, default=1024, help="Audio chunk size in samples (default: 1024).")
    parser.add_argument(
        "--audio",
        help="Optional fallback: stream audio from a raw/PCM file instead of microphone (sounddevice not required).",
    )

    args = parser.parse_args()
    run_kwargs = {
        "wake_word": args.wake_word,
        "voiceprint_path": args.voiceprint,
        "threshold": args.threshold,
        "sample_rate": args.sample_rate,
        "blocksize": args.blocksize,
    }
    if args.audio:
        run_kwargs["audio_path"] = args.audio

    asyncio.run(run_agent(**run_kwargs))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
