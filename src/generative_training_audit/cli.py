"""Command-line entry point for reproducible audit demonstrations."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np

from .ema import ema_step_fp32, ema_step_quantized, update_to_spacing_ratio
from .noise import assert_noise_pair, make_noise_record, sample_noise
from .report import AuditResult
from .scheduler import audit_transitions, build_transitions


def audit_ema() -> AuditResult:
    shadow = np.asarray([1.0], dtype=np.float32)
    parameter = np.asarray([1.01], dtype=np.float32)
    unsafe = ema_step_quantized(shadow, parameter, decay=0.999)
    safe = ema_step_fp32(shadow, parameter, decay=0.999)
    frozen = bool(np.array_equal(unsafe, shadow))
    fp32_moved = bool(not np.array_equal(safe, shadow))
    return AuditResult(
        name="bf16_ema",
        passed=frozen and fp32_moved,
        summary="detected BF16 freeze while FP32 master still updates",
        evidence={
            "bf16_shadow_changed": not frozen,
            "fp32_shadow_changed": fp32_moved,
            "update_to_spacing_ratio": update_to_spacing_ratio(1.0, 1.01, 0.999),
        },
    )


def audit_noise() -> AuditResult:
    cached = sample_noise(42, (2, 4, 8, 8))
    record = make_noise_record(42, cached)
    assert_noise_pair(record, sample_noise(42, cached.shape))
    mismatch_detected = False
    try:
        assert_noise_pair(record, sample_noise(43, cached.shape))
    except AssertionError:
        mismatch_detected = True
    return AuditResult(
        name="noise_pairing",
        passed=mismatch_detected,
        summary="exact pair passed and mismatched noise was rejected",
        evidence={
            "fingerprint_prefix": record.fingerprint[:12],
            "mismatch_detected": mismatch_detected,
        },
    )


def audit_scheduler() -> AuditResult:
    transitions = build_transitions([1.0, 0.55, 0.0], [1.0, 0.42, 0.0])
    issues = audit_transitions(
        transitions, expected_steps=2, expected_start=1.0, expected_end=0.0
    )
    return AuditResult(
        name="scheduler_trace",
        passed=not issues,
        summary="executed transitions satisfy count, endpoint and continuity invariants",
        evidence={"transition_count": len(transitions), "issues": issues},
    )


def audit_gan() -> AuditResult:
    try:
        import torch
    except ImportError:
        return AuditResult(
            name="gan_gradient",
            passed=True,
            summary="skipped: install the 'gan' extra to run the PyTorch audit",
            skipped=True,
            evidence={"skipped": True},
        )

    from .gan import discriminator_fake_loss, parameter_grad_norm

    torch.manual_seed(0)
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1)
    )
    fake = torch.randn(16, 4)
    norms: dict[str, float] = {}
    for label, broken in (("broken", True), ("correct", False)):
        discriminator.zero_grad(set_to_none=True)
        loss, _ = discriminator_fake_loss(discriminator, fake, broken=broken)
        loss.backward()
        norms[label] = parameter_grad_norm(discriminator.parameters())
    passed = norms["broken"] == 0.0 and norms["correct"] > 0.0
    return AuditResult(
        name="gan_gradient",
        passed=passed,
        summary="detached logits lose D gradients; detached samples preserve them",
        evidence={"broken_grad_norm": norms["broken"], "correct_grad_norm": norms["correct"]},
    )


def run(selected: str = "all") -> list[AuditResult]:
    audits = {
        "ema": audit_ema,
        "noise": audit_noise,
        "scheduler": audit_scheduler,
        "gan": audit_gan,
    }
    names = list(audits) if selected == "all" else [selected]
    return [audits[name]() for name in names]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run silent-failure audits for generative-model training."
    )
    parser.add_argument(
        "audit", choices=["all", "ema", "noise", "scheduler", "gan"], nargs="?", default="all"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    results = run(args.audit)
    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        for result in results:
            mark = "SKIP" if result.skipped else ("PASS" if result.passed else "FAIL")
            print(f"[{mark}] {result.name}: {result.summary}")
            for key, value in result.evidence.items():
                print(f"       {key}={value}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
