"""Trace and audit explicit solver transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Transition:
    index: int
    time_from: float
    time_to: float
    sigma_from: float | None = None
    sigma_to: float | None = None

    @property
    def delta_time(self) -> float:
        return self.time_to - self.time_from


def build_transitions(
    times: Sequence[float], sigmas: Sequence[float] | None = None
) -> list[Transition]:
    """Convert visited states into the transitions actually executed."""

    if len(times) < 2:
        raise ValueError("at least two visited states are required")
    if sigmas is not None and len(sigmas) != len(times):
        raise ValueError("times and sigmas must have the same length")
    return [
        Transition(
            index=i,
            time_from=float(times[i]),
            time_to=float(times[i + 1]),
            sigma_from=None if sigmas is None else float(sigmas[i]),
            sigma_to=None if sigmas is None else float(sigmas[i + 1]),
        )
        for i in range(len(times) - 1)
    ]


def audit_transitions(
    transitions: Sequence[Transition],
    *,
    expected_steps: int | None = None,
    expected_start: float | None = None,
    expected_end: float | None = None,
) -> list[str]:
    """Return invariant violations without assuming a specific scheduler."""

    issues: list[str] = []
    if expected_steps is not None and len(transitions) != expected_steps:
        issues.append(
            f"executed {len(transitions)} transitions, expected {expected_steps}"
        )
    if not transitions:
        return issues + ["no transitions were recorded"]
    if expected_start is not None and transitions[0].time_from != expected_start:
        issues.append(
            f"start time {transitions[0].time_from} != expected {expected_start}"
        )
    if expected_end is not None and transitions[-1].time_to != expected_end:
        issues.append(f"end time {transitions[-1].time_to} != expected {expected_end}")
    for left, right in zip(transitions, transitions[1:]):
        if left.time_to != right.time_from:
            issues.append(
                f"discontinuous trace between transitions {left.index} and {right.index}"
            )
    return issues


def format_trace(transitions: Sequence[Transition]) -> str:
    rows = ["step | time_from -> time_to | sigma_from -> sigma_to"]
    for t in transitions:
        sigma = (
            "n/a"
            if t.sigma_from is None
            else f"{t.sigma_from:.6g} -> {t.sigma_to:.6g}"
        )
        rows.append(
            f"{t.index:>4} | {t.time_from:.6g} -> {t.time_to:.6g} | {sigma}"
        )
    return "\n".join(rows)
