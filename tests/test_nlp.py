from nlp.asr import ASRRouter, CloudFallbackASR, LocalWhisperASR
from nlp.intent import IntentClassifier


def test_intent_classifier_routes_models():
    classifier = IntentClassifier()
    intent = classifier.classify("Please send an email")
    assert intent.label == "email"
    assert intent.model == "haiku"

    complex_intent = classifier.classify("Please compose and summarize a long blog post with analysis")
    assert complex_intent.label == "blog"
    assert complex_intent.complexity == "complex"
    assert complex_intent.model == "sonnet"


def test_asr_router_prefers_confident_local():
    local = LocalWhisperASR()
    cloud = CloudFallbackASR()
    router = ASRRouter(local, cloud, threshold=0.6)

    high_confidence = router.transcribe([b"hello"], sample_rate=16000)
    assert high_confidence.source == "local_whisper"

    low_confidence_router = ASRRouter(local, cloud, threshold=0.9)
    fallback = low_confidence_router.transcribe([b"hello"], sample_rate=16000)
    assert fallback.source == "cloud_fallback"
