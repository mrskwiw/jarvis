"""Lightweight metrics sink for local deployments."""
from __future__ import annotations

import time
from collections import Counter
from typing import Dict, List


class MetricsSink:
    """Collects counters for later scraping or logging."""

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self.last_updated: Dict[str, float] = {}
        self.timings: Dict[str, List[float]] = {}

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] += value
        self.last_updated[name] = time.time()

    def record_timing(self, name: str, value_ms: float) -> None:
        bucket = self.timings.setdefault(name, [])
        bucket.append(value_ms)
        self.last_updated[name] = time.time()

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counters)
