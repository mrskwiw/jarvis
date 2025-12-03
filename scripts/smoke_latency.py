"""Quick wake->intent latency smoke script."""
from __future__ import annotations

import asyncio
import time

from api.server import VoiceAgent
from voice.listener import audio_stream_from_queue


async def run_smoke():
    queue: asyncio.Queue[bytes] = asyncio.Queue()
    audio_source = audio_stream_from_queue(queue)
    agent = VoiceAgent(audio_source=audio_source, voiceprint_path="./owner.voiceprint")
    agent.listener.verifier.enroll_owner([b"jarvis"], sample_rate=agent.listener.sample_rate)
    agent.listener.verifier.threshold = 0.5

    # Simulated audio frames
    queue.put_nowait(b"ambient noise")
    queue.put_nowait(b"Jarvis wake")
    queue.put_nowait(b"compose an email")
    for _ in range(agent.listener.silence_after_frames):
        queue.put_nowait(b"\x00")
    queue.put_nowait(None)

    start = time.time()
    payload = await agent.process_audio_command()
    duration_ms = (time.time() - start) * 1000.0

    print("Wake->intent latency (wall): %.2f ms" % duration_ms)
    print("Trace spans:")
    for span in payload.get("trace", []):
        print(f" - {span['name']}: {span['duration_ms']:.2f} ms")
    print("ASR source:", payload["transcription"].source)


def main():
    asyncio.run(run_smoke())


if __name__ == "__main__":
    main()
