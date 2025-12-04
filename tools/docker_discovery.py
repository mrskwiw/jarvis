"""Docker tool discovery stub.

Reads tool definitions from an environment variable or JSON file (mounted from Docker).
This avoids needing Docker APIs while still supporting a "docker-exposed tools registry" hook.
"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional


def load_tools_from_env() -> List[Dict[str, Any]]:
    raw = os.environ.get("JARVIS_DOCKER_TOOLS")
    if not raw:
        return []
    try:
        tools = json.loads(raw)
        if not isinstance(tools, list):
            return []
        return tools
    except Exception:
        return []


def load_tools_from_file(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            tools = json.load(f)
        if not isinstance(tools, list):
            return []
        return tools
    except Exception:
        return []


def discover_docker_tools() -> List[Dict[str, Any]]:
    """Return a list of tool definitions pulled from env or file.

    Priority: env var JARVIS_DOCKER_TOOLS, then file JARVIS_DOCKER_TOOLS_PATH.
    """
    tools = load_tools_from_env()
    if tools:
        return tools
    path = os.environ.get("JARVIS_DOCKER_TOOLS_PATH")
    if path:
        return load_tools_from_file(path)
    return []
