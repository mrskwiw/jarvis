"""Lightweight tracing stub for local observability."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Span:
    name: str
    start_ms: float
    duration_ms: float


class TraceRecorder:
    def __init__(self) -> None:
        self.spans: List[Span] = []

    def span(self, name: str):
        recorder = self

        class _SpanCtx:
            def __enter__(self_inner):
                self_inner._start = time.monotonic()
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                duration = (time.monotonic() - self_inner._start) * 1000.0
                recorder.spans.append(Span(name=name, start_ms=self_inner._start * 1000.0, duration_ms=duration))
                return False

        return _SpanCtx()

    def export(self) -> List[Dict[str, float | str]]:
        return [span.__dict__ for span in self.spans]

    def reset(self) -> None:
        self.spans.clear()
