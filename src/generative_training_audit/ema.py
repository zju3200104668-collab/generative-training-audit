"""Minimal BF16 rounding and EMA diagnostics."""

from __future__ import annotations

import numpy as np


def bf16_round(values: np.ndarray | float) -> np.ndarray:
    """Round float32 values to BF16 precision and return them as float32.

    NumPy does not expose BF16 on every platform, so this implements
    round-to-nearest-even on the float32 bit representation.
    """

    x = np.asarray(values, dtype=np.float32)
    bits = x.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    rounded = ((bits + rounding_bias) & np.uint32(0xFFFF0000)).view(np.float32)
    return rounded


def ema_step_fp32(
    shadow: np.ndarray, parameter: np.ndarray, decay: float
) -> np.ndarray:
    """Update an FP32 EMA master without low-precision writeback."""

    _validate_decay(decay)
    s = np.asarray(shadow, dtype=np.float32)
    p = np.asarray(parameter, dtype=np.float32)
    return s + np.float32(1.0 - decay) * (p - s)


def ema_step_quantized(
    shadow: np.ndarray, parameter: np.ndarray, decay: float
) -> np.ndarray:
    """Emulate an EMA whose shadow is rounded to BF16 after every update."""

    updated = ema_step_fp32(bf16_round(shadow), parameter, decay)
    return bf16_round(updated)


def update_to_spacing_ratio(shadow: float, parameter: float, decay: float) -> float:
    """Return EMA increment divided by local BF16 spacing."""

    _validate_decay(decay)
    base = float(bf16_round(shadow))
    bits = np.asarray(base, dtype=np.float32).view(np.uint32)
    next_value = ((bits + np.uint32(0x00010000)) & np.uint32(0xFFFF0000)).view(
        np.float32
    )
    spacing = abs(float(next_value) - base)
    increment = abs((1.0 - decay) * (parameter - base))
    return increment / spacing if spacing else float("inf")


def _validate_decay(decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must satisfy 0 <= decay < 1")
