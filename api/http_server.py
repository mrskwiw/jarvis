"""Minimal HTTP API for JARVIS (health, tools, enroll, transcribe)."""
from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Tuple

from voice.enroll import enroll_from_file
from api.server import VoiceAgent


def _empty_audio_source():
    async def gen():
        if False:
            yield b""

    return gen()


def build_agent() -> VoiceAgent:
    # Agent is constructed with a no-op audio source; HTTP endpoints do not stream live audio.
    return VoiceAgent(audio_source=_empty_audio_source())


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class JarvisRequestHandler(BaseHTTPRequestHandler):
    agent: VoiceAgent = None  # type: ignore

    def log_message(self, format: str, *args) -> None:  # pragma: no cover - reduce test noise
        return

    def do_GET(self) -> None:  # pragma: no cover - exercised via tests
        if self.path == "/health":
            if JarvisRequestHandler.agent is None:
                JarvisRequestHandler.agent = build_agent()
            _json_response(self, 200, self.agent.health())
            return
        if self.path == "/tools":
            if JarvisRequestHandler.agent is None:
                JarvisRequestHandler.agent = build_agent()
            _json_response(self, 200, {"tools": self.agent.tools.describe()})
            return
        if self.path == "/metrics":
            if JarvisRequestHandler.agent is None:
                JarvisRequestHandler.agent = build_agent()
            _json_response(
                self,
                200,
                {"counters": self.agent.metrics.snapshot(), "timings": self.agent.metrics.timings},
            )
            return
        if self.path == "/metrics/prom":
            if JarvisRequestHandler.agent is None:
                JarvisRequestHandler.agent = build_agent()
            body = self._prometheus_metrics()
            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/console":
            if JarvisRequestHandler.agent is None:
                JarvisRequestHandler.agent = build_agent()
            html = self._render_console()
            payload = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        _json_response(self, 404, {"error": "not found"})

    def do_POST(self) -> None:  # pragma: no cover - exercised via tests
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body.decode() or "{}")
        except json.JSONDecodeError:
            _json_response(self, 400, {"error": "invalid json"})
            return

        if self.path == "/enroll":
            return self._handle_enroll(data)
        if self.path == "/transcribe":
            return self._handle_transcribe(data)
        if self.path == "/chat":
            return self._handle_chat(data)
        if self.path == "/route_audio":
            return self._handle_route_audio(data)
        if self.path == "/tool_call":
            return self._handle_tool_call(data)
        _json_response(self, 404, {"error": "not found"})

    def _handle_enroll(self, data: dict) -> None:
        if JarvisRequestHandler.agent is None:
            JarvisRequestHandler.agent = build_agent()
        try:
            audio_path = data["audio_path"]
            voiceprint_path = data.get("voiceprint_path", "./owner.voiceprint")
            sample_rate = int(data.get("sample_rate", 16000))
            chunk_size = int(data.get("chunk_size", 1024))
        except KeyError as exc:
            _json_response(self, 400, {"error": f"missing field {exc.args[0]}"})
            return

        try:
            result = enroll_from_file(
                audio_path=audio_path,
                voiceprint_path=voiceprint_path,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
            )
        except Exception as exc:  # pragma: no cover - error path
            _json_response(self, 400, {"error": str(exc)})
            return

        _json_response(self, 200, result)

    def _handle_transcribe(self, data: dict) -> None:
        if JarvisRequestHandler.agent is None:
            JarvisRequestHandler.agent = build_agent()
        try:
            frames_b64 = data.get("frames", [])
            frames = [base64.b64decode(f) for f in frames_b64]
            sample_rate = int(data.get("sample_rate", 16000))
        except Exception as exc:
            _json_response(self, 400, {"error": f"invalid payload: {exc}"})
            return
        result = self.agent.asr.transcribe(frames, sample_rate)
        _json_response(
            self,
            200,
            {"text": result.text, "confidence": result.confidence, "source": result.source},
        )

    def _handle_chat(self, data: dict) -> None:
        if JarvisRequestHandler.agent is None:
            JarvisRequestHandler.agent = build_agent()
        message = data.get("message")
        if not message:
            _json_response(self, 400, {"error": "missing field message"})
            return
        payload = self.agent.route_text(message)
        _json_response(self, 200, payload)

    def _handle_route_audio(self, data: dict) -> None:
        if JarvisRequestHandler.agent is None:
            JarvisRequestHandler.agent = build_agent()
        try:
            frames_b64 = data.get("frames", [])
            frames = [base64.b64decode(f) for f in frames_b64]
            sample_rate = int(data.get("sample_rate", 16000))
        except Exception as exc:
            _json_response(self, 400, {"error": f"invalid payload: {exc}"})
            return
        transcription = self.agent.asr.transcribe(frames, sample_rate)
        payload = self.agent.route_text(transcription.text)
        payload["transcription"] = {"text": transcription.text, "confidence": transcription.confidence, "source": transcription.source}
        _json_response(self, 200, payload)

    def _handle_tool_call(self, data: dict) -> None:
        if JarvisRequestHandler.agent is None:
            JarvisRequestHandler.agent = build_agent()
        try:
            name = data["name"]
            arguments = data.get("arguments", {})
            owner_verified = bool(data.get("owner_verified", False))
        except KeyError as exc:
            _json_response(self, 400, {"error": f"missing field {exc.args[0]}"})
            return
        try:
            result = self.agent.tools.execute(name, arguments, owner_verified=owner_verified)
        except Exception as exc:
            _json_response(self, 400, {"error": str(exc)})
            return
        _json_response(self, 200, result)

    def _prometheus_metrics(self) -> str:
        lines = []
        counters = JarvisRequestHandler.agent.metrics.snapshot()
        timings = JarvisRequestHandler.agent.metrics.timings
        for name, value in counters.items():
            lines.append(f"jarvis_counter{{name=\"{name}\"}} {value}")
        for name, samples in timings.items():
            for idx, sample in enumerate(samples):
                lines.append(f"jarvis_timing_ms{{name=\"{name}\",sample=\"{idx}\"}} {sample}")
        return "\n".join(lines) + "\n"

    def _render_console(self) -> str:
        health = self.agent.health()
        tools = self.agent.tools.describe()
        counters = self.agent.metrics.snapshot()
        return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>JARVIS Console</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
      h1 {{ margin-bottom: 0.25rem; }}
      pre {{ background: #f4f4f4; padding: 0.75rem; border-radius: 6px; }}
    </style>
  </head>
  <body>
    <h1>JARVIS Console</h1>
    <h2>Health</h2>
    <pre>{json.dumps(health, indent=2)}</pre>
    <h2>Tools</h2>
    <pre>{json.dumps(tools, indent=2)}</pre>
    <h2>Metrics</h2>
    <pre>{json.dumps(counters, indent=2)}</pre>
  </body>
</html>
"""


def run_server(host: str = "127.0.0.1", port: int = 0) -> Tuple[HTTPServer, int, threading.Thread]:
    server = HTTPServer((host, port), JarvisRequestHandler)
    bound_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, bound_port, thread


def main() -> None:  # pragma: no cover - CLI entrypoint
    import argparse

    parser = argparse.ArgumentParser(description="Start a minimal JARVIS HTTP API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args()
    server, bound_port, _thread = run_server(args.host, args.port)
    print(f"Serving JARVIS API on http://{args.host}:{bound_port}")
    try:
        _thread.join()
    except KeyboardInterrupt:
        server.shutdown()
        print("Server stopped.")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
