from nlp.intent import IntentClassifier
from nlp.router import LLMRouter
from nlp.asr import ASRRouter, LocalWhisperASR, CloudFallbackASR
import asyncio
from tools.registry import ToolRegistry
import json
from types import SimpleNamespace
import urllib.request


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
    registry = ToolRegistry()
    registry.register_schema(
        "email",
        {
            "description": "Send email",
            "input_schema": {"type": "object", "properties": {"to": {"type": "string"}}, "required": ["to"]},
            "free_tier_only": True,
        },
    )
    router = LLMRouter(default_system_prompt="system", tool_registry=registry)
    intent = IntentClassifier().classify("send an email")
    decision = router.normalize_request(intent, "hello", tools=["spellcheck"])
    assert decision.payload["tools"][0]["name"] == "spellcheck"
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
    # Local confidence is 0.65, so cloud is used; ensure we get text and streaming source tag.
    assert result.text.strip()
    assert result.source == "cloud_fallback_stream"

    # Force timeout with no frames to ensure we still return something
    router.stream_timeout = 0.01

    async def consume_timeout():
        async def slow_gen():
            await asyncio.sleep(0.02)
            yield b"delayed"

        return await router.transcribe_streaming(slow_gen(), sample_rate=16000)

    result_timeout = asyncio.run(consume_timeout())
    assert result_timeout.text.strip() or result_timeout.confidence == 0.0


def test_cloud_asr_http_endpoint(monkeypatch):
    responses = []

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def read(self):
            return json.dumps(self._data).encode()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=10):
        responses.append(req)
        return FakeResponse({"text": "hello", "confidence": 0.9})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    cloud = CloudFallbackASR(endpoint="http://fake-endpoint/transcribe")
    result = cloud.transcribe([b"abc"], sample_rate=16000)
    assert result.text == "hello"
    assert result.source == "cloud_fallback_http"
    assert responses


def test_low_confidence_triggers_clarification():
    router = LLMRouter()
    intent = IntentClassifier().classify("yo")  # short/ambiguous
    decision = router.normalize_request(intent, "yo", tools=None)
    assert decision.payload.get("needs_clarification") is True
    assert any("clarify" in m["content"] for m in decision.payload["messages"])
