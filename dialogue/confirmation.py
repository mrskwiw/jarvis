"""Confirmation policy for sensitive tool actions."""
from __future__ import annotations

from typing import Callable, Optional


SENSITIVE_ACTIONS = {"email": {"send", "draft"}, "call": {"place"}}


def needs_confirmation(tool: str, action: str) -> bool:
    return action in SENSITIVE_ACTIONS.get(tool, set())


def confirm_action(prompt_user: Callable[[str], str], tool: str, action: str) -> bool:
    prompt = f"Do you want to {action} using {tool}? (yes/no)"
    response = prompt_user(prompt).strip().lower()
    return response in {"yes", "y"}
