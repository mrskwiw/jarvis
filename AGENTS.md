# JARVIS Build Agent Guide

Authoritative brief for coding agents building and maintaining the JARVIS voice-first assistant. Keep this file in sync with major architectural or contract changes.

## Purpose & Scope
- Voice-first personal assistant that only obeys an enrolled owner. Pipeline: wake word → speaker verification → ASR routing → intent classification → LLM routing → lazy-loaded tools → optional TTS.
- Bias toward free/local backends first; Anthropic Agent SDK style routing (Haiku for simple, Sonnet for complex).
- Tools are lazily registered and may come from Docker-exposed services; enforce owner verification before privileged tool use.

## Codebase Map (key modules)
- `api/server.py`: Orchestrates VoiceAgent (listener, ASR router, intent classifier, LLM router, tool registry, metrics, tracing, optional TTS).
- `api/http_server.py`: Minimal HTTP API (`/health`, `/tools`, `/metrics`, `/metrics/prom`, `/enroll`, `/transcribe`, `/chat`, `/route_audio`, `/tool_call`, `/console`).
- `dialogue/controller.py`: ConversationController that hands intents + text to LLM router and returns normalized payloads.
- `nlp/asr.py`: Local Whisper stub + cloud fallback router; streaming support with timeout; confidence thresholding.
- `nlp/intent.py`: Heuristic intent classifier (labels: chat/email/call/blog/tool-needed); picks Haiku vs Sonnet by complexity.
- `nlp/router.py`: LLMRouter builds payloads, attaches tool schemas (`ToolRegistry.describe()`), adds clarification prompt when low confidence.
- `voice/listener.py`: ContinuousListener with wake detection, speech guardrails, verification gating, latency metrics.
- `voice/verification.py`: Embedding backends (hash default, optional resemblyzer), voiceprint encryption (XOR + base64) using env key, speaker verification thresholding.
- `voice/enroll.py`: CLI helpers for enrolling/rotating owner voiceprint from raw PCM.
- `voice/mic_cli.py`: Live mic/file streaming loop into ContinuousListener + VoiceAgent.
- `voice/tts.py`: Optional cached TTS backend gated by env.
- `tools/registry.py`: Lazy loader with schema registration, owner verification flag, and discovery hooks.
- `tools/email.py`, `tools/calls.py`, `tools/blogging.py`, `tools/docker_discovery.py`: Built-in tools and docker tool discovery.
- `observability/logging.py`, `observability/metrics.py`, `observability/health.py`, `observability/tracing.py`: Redacting logger, counters/timings, env audits, tool catalog health, simple tracing.
- `tests/`: Coverage across HTTP API, dialogue routing, enrollment, listener, mic CLI, NLP, tools, and verification paths.

## Runtime Contracts & Security
- Env key required: `JARVIS_VOICE_KEY` (fail fast via `require_voice_key`; encrypts voiceprints). Rotate by setting a new key and re-enrolling.
- Wake-word backend: `JARVIS_WAKE_BACKEND` (`fallback` or `porcupine`); embedding backend: `JARVIS_EMBEDDING_BACKEND` (`hash`, `resemblyzer`).
- ASR endpoint optional: `JARVIS_ASR_ENDPOINT` (enables cloud-first routing); TTS opt-in: `JARVIS_ENABLE_TTS=1`.
- Only proceed past wake detection when speaker verification succeeds; enforce minimum speech duration and silence guardrails in `ContinuousListener`.
- Do not log secrets, raw embeddings, or key material; redact user-provided content where practical.
- Owner verification flag is required for privileged tool execution; never bypass verification for tool calls.

## Tooling & Integrations
- Built-in schemas: `email`, `call`, `blog`, `pop_setup` (POP form helper). All marked free-tier-friendly.
- Docker tool discovery: `tools/docker_discovery.py` registers schemas and lazy loaders from docker-exposed metadata.
- Adding tools: register schema (description, JSON schema, free-tier flag) then register lazy factory; ensure inputs validated and owner verification enforced.

## APIs & CLIs
- Enrollment: `python -m voice.enroll --audio path/to/owner.raw --voiceprint ./owner.voice --sample-rate 16000`
- Mic loop: `python -m voice.mic_cli --wake-word jarvis --voiceprint ./owner.voice --threshold 0.8`
- HTTP server: `python -m api.http_server --port 8000` (see endpoints above). `/tools` returns catalog; `/metrics/prom` exposes Prometheus text.

## Observability
- Metrics: counters/timings via `observability.metrics`; ASR calls and wake-to-verify latency recorded.
- Health: `VoiceAgent.health()` validates env + voiceprint existence + tool catalog status.
- Tracing: `TraceRecorder` spans around listen/asr/intent/route; attach to responses.
- Logging: `RedactingLogger` with structured output; keep logs quiet in tests.

## Testing Expectations
- Default: `pytest` from repo root; covers HTTP API, dialogue routing, enrollment/verification, listener behavior, NLP routers, and tools.
- Add unit tests for every new function and error path; prefer pure, deterministic units over integration mocks.
- For new backends (wake/embedding/ASR/tools), add goldens and failure-path tests (missing envs, invalid payloads, network failures).

## Development Practices
- Codebase is Python 3.11+; prefer functional-style helpers where practical and keep side effects localized.
- Validate all inputs; return actionable error messages; fail fast when env/config is missing.
- Keep functions commented with language-appropriate, concise intent comments when logic is non-obvious.
- Use env vars for secrets/config; never hard-code credentials; keep owner-only guardrails intact.
- Maintain README per module when adding new surface areas; document API endpoints with examples.
- Track unfixed bugs in `ERROR_LOG.md` (create if absent); mark resolved bugs and completed features in the implementation document/roadmap before moving on.

## Roadmap Signals (from `ROADMAP.md`)
- Near-term: enrollment UX + env audits, replay/silence guardrails, pluggable wakeword/VAD, streaming ASR option, Anthropic tool routing, tool catalog/metadata, metrics + health checks.
- Stretch: multi-owner ACLs, edge/offline profile, privacy/ephemeral mode, billing guardrails for any paid APIs.

## Quick Start Checklist for Agents
- Export `JARVIS_VOICE_KEY` before running anything that reads/writes voiceprints.
- Enroll owner voiceprint; keep voiceprints encrypted at user-specified paths.
- Choose wake/embedding/ASR backends via env; default to free/local paths.
- Run `pytest` before/after changes; keep `VoiceAgent.health()` green; exercise `/health` and `/tools` when touching API surfaces.
