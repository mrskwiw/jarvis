# JARVIS Voice Agent

Voice-first personal assistant with wake-word detection, speaker verification, ASR routing, intent classification, LLM routing, and lazy-loaded tools.

## Prerequisites
- Python 3.11+
- Set a strong voice key env var before running anything that loads/saves voiceprints:
  - PowerShell: `SET JARVIS_VOICE_KEY="your-random-key"`
  - Bash: `export JARVIS_VOICE_KEY="your-random-key"`
- Rotate keys by setting a new value for `JARVIS_VOICE_KEY` and re-enrolling the owner voiceprint (old encrypted files become invalid; remove them after rotation).
- For mic streaming: `pip install sounddevice` (system audio drivers/tools may be required).

## Enrollment (owner voiceprint)
1) Record a short raw/PCM clip of your voice saying the wake word and a phrase (16 kHz, mono).
2) Enroll the owner:
```
python -m voice.enroll --audio path/to/owner.raw --voiceprint ./owner.voice --sample-rate 16000
```
This encrypts and stores the embedding at `./owner.voice` using `JARVIS_VOICE_KEY`.

## Live microphone loop
Stream mic audio through `ContinuousListener` and `VoiceAgent`:
```
python -m voice.mic_cli --wake-word jarvis --voiceprint ./owner.voice --threshold 0.8
```
- Requires `sounddevice`; if missing, the CLI will instruct you to install it.
- Adjust `--threshold` down (e.g., `0.5`) to be more permissive during testing.
- Fallback when `sounddevice` is unavailable: provide a raw/PCM file instead of the mic:
```
python -m voice.mic_cli --audio path/to/audio.raw --voiceprint ./owner.voice
```

## Testing
Unit tests cover voice verification, enrollment, listener behavior, and NLP routing:
```
pytest
```

## Roadmap
See `ROADMAP.md` for planned work on enrollment UX, ASR/TTS integrations, Anthropic routing, tools, and observability.

## Health checks
- Core requirement: `JARVIS_VOICE_KEY` must be set; `VoiceAgent.health()` reports missing envs and whether a voiceprint file exists.
