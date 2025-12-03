"""Lightweight metrics sink for local deployments."""
from __future__ import annotations

import time
from collections import Counter
from typing import Dict


class MetricsSink:
    """Collects counters for later scraping or logging."""

    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()
        self.last_updated: Dict[str, float] = {}

    def increment(self, name: str, value: int = 1) -> None:
        self.counters[name] += value
        self.last_updated[name] = time.time()

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counters)
