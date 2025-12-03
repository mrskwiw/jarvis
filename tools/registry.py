"""Tool registry with lazy loading and permission checks."""
from __future__ import annotations

from typing import Callable, Dict, Optional

from observability.logging import RedactingLogger


class PermissionError(Exception):
    pass


class ToolRegistry:
    def __init__(self, logger: Optional[RedactingLogger] = None) -> None:
        self.logger = logger or RedactingLogger(__name__)
        self._loaders: Dict[str, Callable[[], object]] = {}
        self._cache: Dict[str, object] = {}
        self._require_owner = True
        self._schemas: Dict[str, Dict[str, object]] = {}

    def register(self, name: str, loader: Callable[[], object]) -> None:
        self._loaders[name] = loader
        self.logger.debug("Registered tool %s", name)

    def register_schema(self, name: str, schema: Dict[str, object]) -> None:
        self._schemas[name] = schema

    def get(self, name: str, owner_verified: bool = False) -> object:
        if self._require_owner and not owner_verified:
            raise PermissionError("Owner verification required for tool execution")
        if name not in self._cache:
            if name not in self._loaders:
                raise KeyError(f"Unknown tool {name}")
            self._cache[name] = self._loaders[name]()
            self.logger.debug("Lazily loaded tool %s", name)
        return self._cache[name]

    def names(self):
        return list(self._loaders.keys())

    def describe(self) -> Dict[str, Dict[str, object]]:
        return {name: self._schemas.get(name, {}) for name in self._loaders.keys()}
