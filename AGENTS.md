# JARVIS Development Guardrails

- **Security & Ownership**
  - `JARVIS_VOICE_KEY` must be set before any voiceprint load/save; fail fast if missing. Keep rotation guidance and avoid logging key material.
  - Only accept commands from enrolled owner: speaker verification before tool calls; keep thresholds explicit and configurable.
  - Guard against replay/silence: enforce minimum speech duration, max silence, and optional challenge phrase pre-tool.

- **Enrollment**
  - Provide a clear CLI/API to enroll and rotate the owner voiceprint (`voice.enroll`); ensure docs mention 16 kHz mono raw/PCM input.
  - Never cache voiceprints unencrypted; store at user-specified path with key-based encryption.

- **Voice Pipeline**
  - Wake-word detection gates everything; allow pluggable backend (Porcupine/Snowboy wrapper).
  - Verification uses pluggable embedding model; current hash model is a stub—keep protocol stable for swapping real models.
  - Add VAD for noise/silence trimming; expose wake→verify latency metrics.

- **ASR & Routing**
  - ASR router prefers local/low-cost first; add streaming option with timeouts/retries.
  - Intent classifier feeds LLM router: route simple to Haiku, complex to Sonnet; include tool-use schema and lazy tool loading.
  - Add safety prompts: privacy (no secrets/voiceprint leakage) and cost-aware routing.

- **Tools & Integrations**
  - Tools are lazily loaded; enforce per-tool permissions and dry-run/test doubles.
  - Prefer free services/free tiers; add billing guardrails for anything paid.
  - Expose tool catalog/metadata and docker-exposed registry discovery hook.

- **APIs & UX Surfaces**
  - HTTP/WS endpoints: enroll, stream audio, list tools, fetch metrics.
  - CLI (`voice.mic_cli`) streams mic/file audio through `ContinuousListener` + `VoiceAgent`; keep clear error/help paths when deps (e.g., `sounddevice`) are missing.
  - Optional TTS path (edge-tts/Coqui) with caching.

- **Observability & Ops**
  - Structured/redacting logger with env-driven levels.
  - Metrics sink exportable (Prometheus/text); health checks for wakeword/ASR/verifier.
  - Add tracing stubs around ASR → intent → LLM → tool chain.
  - Containerization: slim runtime image; ASR sidecar profile; sample k8s manifest.

- **Testing & Quality**
  - Expand coverage on routing decisions, verifier thresholds, key rotation, tool permission failures, and wake-to-intent latency smoke tests.
  - CI: lint, tests, type checks (mypy/pyright), secrets scanning.
  - Keep entrypoints (`voice.enroll`, `voice.mic_cli`) warning-free under `runpy` and CLI invocations.

- **Priorities (from roadmap)**
  - Near-term: enrollment UX + security checks, replay/silence guardrails, pluggable wakeword/VAD, ASR streaming option, Anthropic agent wiring, tool catalog, metrics/health.
  - Stretch: multi-owner with per-owner ACLs, edge/offline mode, privacy mode (ephemeral context). 
