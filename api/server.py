"""High-level orchestration for Jarvis voice agent."""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator, Optional

from dialogue.controller import ConversationController
from nlp.asr import ASRRouter, CloudFallbackASR, LocalWhisperASR
from nlp.intent import IntentClassifier
from nlp.router import LLMRouter
from observability.health import audit_environment
from observability.logging import RedactingLogger
from observability.metrics import MetricsSink
from observability.health import tool_catalog_status
from observability.tracing import TraceRecorder
from tools.registry import ToolRegistry
from voice.listener import ContinuousListener, VerifiedAudio, load_wake_detector
from voice.verification import (
    SpeakerVerifier,
    load_embedding_model,
    VoiceprintStore,
    require_voice_key,
)
from voice.tts import CachedTTS


class VoiceAgent:
    def __init__(
        self,
        audio_source: AsyncIterator[bytes],
        wake_word: str = "jarvis",
        voiceprint_path: str = "./owner.voiceprint",
        metrics: Optional[MetricsSink] = None,
    ) -> None:
        self.logger = RedactingLogger(__name__)
        self.metrics = metrics or MetricsSink()
        env_check = audit_environment(["JARVIS_VOICE_KEY"])
        env_check.raise_if_missing()
        require_voice_key()
        embedding_backend = os.environ.get("JARVIS_EMBEDDING_BACKEND", "hash")
        wake_backend = os.environ.get("JARVIS_WAKE_BACKEND", "fallback")
        verifier = SpeakerVerifier(load_embedding_model(embedding_backend), VoiceprintStore(voiceprint_path))
        listener = ContinuousListener(
            wake_detector=load_wake_detector(wake_word, backend=wake_backend),
            verifier=verifier,
            audio_source=audio_source,
            logger=self.logger,
            metrics=self.metrics,
        )
        self.listener = listener
        asr_endpoint = os.environ.get("JARVIS_ASR_ENDPOINT")
        prefer_cloud = bool(asr_endpoint)
        self.asr = ASRRouter(
            LocalWhisperASR(self.logger),
            CloudFallbackASR(self.logger, endpoint=asr_endpoint),
            prefer_cloud=prefer_cloud,
        )
        self.tools = ToolRegistry(logger=self.logger)
        # Register minimal tool schemas for advanced tool use payloads.
        self.tools.register_schema(
            "email",
            {
                "description": "Send an email via configured provider",
                "input_schema": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["to", "subject", "body"],
                },
                "free_tier_only": True,
            },
        )
        self.tools.register_schema(
            "call",
            {
                "description": "Place a phone call",
                "input_schema": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}, "message": {"type": "string"}},
                    "required": ["to", "message"],
                },
                "free_tier_only": True,
            },
        )
        self.tools.register_schema(
            "blog",
            {
                "description": "Draft or publish a blog post",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["title", "body"],
                },
                "free_tier_only": True,
            },
        )
        self.intent_classifier = IntentClassifier()
        self.router = LLMRouter(tool_registry=self.tools)
        self.conversation = ConversationController(self.router, logger=self.logger)
        self.tts = CachedTTS() if os.environ.get("JARVIS_ENABLE_TTS") else None
        self.tracer = TraceRecorder()

    async def listen_for_command(self) -> VerifiedAudio:
        return await self.listener.listen_for_command()

    async def process_audio_command(self) -> dict:
        self.tracer.reset()
        with self.tracer.span("listen"):
            audio = await self.listen_for_command()
        with self.tracer.span("asr"):
            transcription = self.asr.transcribe(audio.frames, audio.sample_rate)
        self.metrics.increment("asr_calls")
        with self.tracer.span("intent"):
            intent = self.intent_classifier.classify(transcription.text)
        with self.tracer.span("route"):
            payload = self.conversation.respond(intent, transcription.text, tools=self.tools.names())
        payload["transcription"] = transcription
        payload["trace"] = self.tracer.export()
        return payload

    def health(self) -> dict:
        env = audit_environment(["JARVIS_VOICE_KEY"])
        tools_status = tool_catalog_status(self.tools.describe())
        return {
            "env": env.to_dict(),
            "voiceprint_exists": self.listener.verifier.store.exists(),
            "tools": tools_status,
        }
