from nlp.intent import IntentClassifier
from nlp.router import LLMRouter
from nlp.asr import ASRRouter, LocalWhisperASR, CloudFallbackASR
import asyncio


def test_intent_classifier_tool_and_call_and_complexity():
    classifier = IntentClassifier()

    tool_intent = classifier.classify("run the backup tool now")
    assert tool_intent.label == "tool-needed"
    assert tool_intent.model == "haiku"

    call_intent = classifier.classify("please place a call to support")
    assert call_intent.label == "call"

    complex_intent = classifier.classify("summarize and compose a detailed analysis of the report")
    assert complex_intent.complexity == "complex"
    assert complex_intent.model == "sonnet"


def test_llm_router_includes_tools_payload():
    router = LLMRouter(default_system_prompt="system")
    intent = IntentClassifier().classify("send an email")
    decision = router.normalize_request(intent, "hello", tools=["spellcheck"])
    assert decision.payload["tools"] == ["spellcheck"]
    assert decision.payload["messages"][0]["content"] == "system"


async def _collect_stream(router: ASRRouter, frames):
    async def gen():
        for frame in frames:
            yield frame

    return await router.transcribe_streaming(gen(), sample_rate=16000)


def test_asr_router_streaming_timeout_and_fallback(monkeypatch):
    local = LocalWhisperASR()
    cloud = CloudFallbackASR()
    router = ASRRouter(local, cloud, threshold=0.9)

    async def consume_no_timeout():
        return await _collect_stream(router, [b"hello", b" world"])

    result = asyncio.run(consume_no_timeout())
    # Local confidence is 0.65, so cloud is used; ensure no error and source reflects streaming.
    assert result.source in {"local_whisper_local_stream", "cloud_fallback_stream"}

    # Force timeout with no frames to ensure we still return something
    router.stream_timeout = 0.01

    async def consume_timeout():
        async def slow_gen():
            await asyncio.sleep(0.02)
            yield b"delayed"

        return await router.transcribe_streaming(slow_gen(), sample_rate=16000)

    result_timeout = asyncio.run(consume_timeout())
    assert result_timeout.text.strip() or result_timeout.confidence == 0.0
