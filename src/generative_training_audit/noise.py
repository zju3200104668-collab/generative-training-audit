"""Deterministic noise generation and teacher-cache pairing checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class NoiseRecord:
    seed: int
    shape: tuple[int, ...]
    dtype: str
    fingerprint: str


def deterministic_seed(
    shard_id: int, sample_index: int, *, namespace: str = "gtaudit"
) -> int:
    """Derive a stable 64-bit seed without exposing a project-specific formula."""

    if shard_id < 0 or sample_index < 0:
        raise ValueError("shard_id and sample_index must be non-negative")
    payload = f"{namespace}:{shard_id}:{sample_index}".encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "little")


def sample_noise(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    """Generate on CPU in FP32 so cache and training can share one contract."""

    return np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)


def tensor_fingerprint(array: np.ndarray) -> str:
    """Hash shape, dtype and contiguous bytes; intended as a debug assertion."""

    x = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(x.shape).encode())
    digest.update(x.dtype.str.encode())
    digest.update(x.tobytes())
    return digest.hexdigest()


def make_noise_record(seed: int, noise: np.ndarray) -> NoiseRecord:
    x = np.asarray(noise)
    return NoiseRecord(seed, tuple(x.shape), x.dtype.str, tensor_fingerprint(x))


def assert_noise_pair(record: NoiseRecord, noise: np.ndarray) -> None:
    x = np.asarray(noise)
    errors: list[str] = []
    if tuple(x.shape) != record.shape:
        errors.append(f"shape {tuple(x.shape)} != {record.shape}")
    if x.dtype.str != record.dtype:
        errors.append(f"dtype {x.dtype.str} != {record.dtype}")
    actual = tensor_fingerprint(x)
    if actual != record.fingerprint:
        errors.append(f"fingerprint {actual[:12]} != {record.fingerprint[:12]}")
    if errors:
        raise AssertionError("noise/target pairing failed: " + "; ".join(errors))
