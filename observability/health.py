"""Simple environment and readiness checks."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class EnvCheck:
    present: List[str]
    missing: List[str]

    @property
    def ok(self) -> bool:
        return not self.missing

    def to_dict(self) -> Dict[str, object]:
        return {"present": self.present, "missing": self.missing, "ok": self.ok}

    def raise_if_missing(self) -> None:
        if self.missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(self.missing)}")


def audit_environment(required_vars: Iterable[str]) -> EnvCheck:
    present: List[str] = []
    missing: List[str] = []
    for var in required_vars:
        if os.environ.get(var):
            present.append(var)
        else:
            missing.append(var)
    return EnvCheck(present=present, missing=missing)
