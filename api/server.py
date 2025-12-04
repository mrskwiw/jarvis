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
from tools.email import EmailService, pop_setup_tool_form
from tools.calls import CallService
from tools.blogging import BloggingService
from tools.docker_discovery import discover_docker_tools
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
        self.tools.register("email", lambda: EmailService())
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
        self.tools.register("call", lambda: CallService())
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
        self.tools.register("blog", lambda: BloggingService())
        self.tools.register_schema(
            "pop_setup",
            {
                "description": "Collect POP email configuration via form",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "user": {"type": "string"},
                        "password": {"type": "string"},
                        "port": {"type": "number"},
                        "use_ssl": {"type": "boolean"},
                    },
                    "required": ["host", "user", "password"],
                },
                "free_tier_only": True,
            },
        )
        self.tools.register("pop_setup", pop_setup_tool_form)
        for tool in discover_docker_tools():
            name = tool.get("name")
            schema = {
                "description": tool.get("description", f"Docker tool {name}"),
                "input_schema": tool.get("input_schema", {"type": "object", "properties": {}}),
                "free_tier_only": tool.get("free_tier_only", True),
            }
            if name:
                self.tools.register_schema(name, schema)
                self.tools.register(name, lambda t=tool: t)
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

    def route_text(self, message: str) -> dict:
        self.tracer.reset()
        with self.tracer.span("intent"):
            intent = self.intent_classifier.classify(message)
        with self.tracer.span("route"):
            payload = self.conversation.respond(intent, message, tools=self.tools.names())
        payload["trace"] = self.tracer.export()
        payload["text"] = message
        payload["tool_catalog"] = self.tools.describe()
        return payload

    def health(self) -> dict:
        env = audit_environment(["JARVIS_VOICE_KEY"])
        tools_status = tool_catalog_status(self.tools.describe())
        return {
            "env": env.to_dict(),
            "voiceprint_exists": self.listener.verifier.store.exists(),
            "tools": tools_status,
        }
