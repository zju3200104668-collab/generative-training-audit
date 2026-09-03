"""Shared machine-readable audit results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuditResult:
    """One audit outcome suitable for logs, JSON and CI assertions."""

    name: str
    passed: bool
    summary: str
    skipped: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
