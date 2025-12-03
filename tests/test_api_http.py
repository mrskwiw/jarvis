import base64
import http.client
import json
import os
import tempfile

import pytest

from api.http_server import run_server


@pytest.fixture(scope="module", autouse=True)
def ensure_voice_key():
    os.environ["JARVIS_VOICE_KEY"] = "test-key"
    yield


def _request(port: int, method: str, path: str, body: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    payload = json.dumps(body or {}).encode()
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn.request(method, path, body=payload if body is not None else None, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    text = data.decode()
    try:
        parsed = json.loads(text or "{}")
    except json.JSONDecodeError:
        parsed = text
    return resp.status, parsed


def test_api_health_and_tools():
    server, port, thread = run_server()
    status, body = _request(port, "GET", "/health")
    assert status == 200
    assert "env" in body

    status, body = _request(port, "GET", "/tools")
    assert status == 200
    assert "tools" in body
    server.shutdown()
    thread.join(timeout=1.0)


def test_api_enroll_and_transcribe(tmp_path):
    server, port, thread = run_server()

    audio_path = tmp_path / "owner.raw"
    audio_path.write_bytes(b"hello owner")
    voiceprint_path = tmp_path / "owner.voiceprint"

    status, body = _request(
        port,
        "POST",
        "/enroll",
        {
            "audio_path": str(audio_path),
            "voiceprint_path": str(voiceprint_path),
            "sample_rate": 16000,
        },
    )
    assert status == 200
    assert voiceprint_path.exists()

    frame = base64.b64encode(b"test frame").decode()
    status, body = _request(
        port,
        "POST",
        "/transcribe",
        {"frames": [frame], "sample_rate": 16000},
    )
    assert status == 200
    assert "text" in body
    status, prom = _request(port, "GET", "/metrics/prom")
    assert status == 200
    assert isinstance(prom, dict) or isinstance(prom, str)
    status, html = _request(port, "GET", "/console")
    assert status == 200
    if isinstance(html, str):
        assert "<h1>JARVIS Console" in html
    server.shutdown()
    thread.join(timeout=1.0)
