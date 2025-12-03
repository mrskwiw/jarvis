"""High-level orchestration for Jarvis voice agent."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from dialogue.controller import ConversationController
from nlp.asr import ASRRouter, CloudFallbackASR, LocalWhisperASR
from nlp.intent import IntentClassifier
from nlp.router import LLMRouter
from observability.logging import RedactingLogger
from observability.metrics import MetricsSink
from tools.registry import ToolRegistry
from voice.listener import ContinuousListener, VerifiedAudio, WakeWordDetector
from voice.verification import HashEmbeddingModel, SpeakerVerifier, VoiceprintStore


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
        verifier = SpeakerVerifier(HashEmbeddingModel(), VoiceprintStore(voiceprint_path))
        listener = ContinuousListener(
            wake_detector=WakeWordDetector(wake_word),
            verifier=verifier,
            audio_source=audio_source,
            logger=self.logger,
            metrics=self.metrics,
        )
        self.listener = listener
        self.asr = ASRRouter(LocalWhisperASR(self.logger), CloudFallbackASR(self.logger))
        self.intent_classifier = IntentClassifier()
        self.router = LLMRouter()
        self.conversation = ConversationController(self.router, logger=self.logger)
        self.tools = ToolRegistry(logger=self.logger)

    async def listen_for_command(self) -> VerifiedAudio:
        return await self.listener.listen_for_command()

    async def process_audio_command(self) -> dict:
        audio = await self.listen_for_command()
        transcription = self.asr.transcribe(audio.frames, audio.sample_rate)
        self.metrics.increment("asr_calls")
        intent = self.intent_classifier.classify(transcription.text)
        payload = self.conversation.respond(intent, transcription.text, tools=self.tools.names())
        payload["transcription"] = transcription
        return payload
