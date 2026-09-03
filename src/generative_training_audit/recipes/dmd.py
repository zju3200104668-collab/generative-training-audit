"""Model-agnostic building blocks for DMD-style training.

The functions make stop-gradient boundaries explicit. They deliberately avoid
assuming a particular diffusion parameterization, scheduler or model family.
"""

from __future__ import annotations

from collections.abc import Callable


def dmd_projection_loss(student_state, real_velocity, fake_velocity):
    """Project a detached score/velocity difference onto the student state.

    Gradients flow into ``student_state`` only. Callers are responsible for
    constructing velocities whose sign and scaling match their model's
    parameterization.
    """

    direction = (real_velocity - fake_velocity).detach()
    _require_same_shape(student_state, direction, "student_state", "direction")
    return _per_sample_mean(direction * student_state).mean()


def fake_score_matching_loss(fake_velocity, target_velocity):
    """Regression loss for the trainable fake-distribution score estimator."""

    import torch.nn.functional as functional

    _require_same_shape(fake_velocity, target_velocity, "fake_velocity", "target_velocity")
    return functional.mse_loss(fake_velocity, target_velocity.detach())


def paired_regression_loss(
    student_output,
    teacher_target,
    *,
    student_noise,
    cached_noise,
    distance: Callable | None = None,
):
    """Compute sample-wise regression only after exact noise-pair validation.

    The exact equality check is intentionally strict and is best used during
    cache validation or debugging. Production jobs may validate a stored
    fingerprint once at sample loading instead.
    """

    import torch
    import torch.nn.functional as functional

    if student_noise.shape != cached_noise.shape:
        raise AssertionError(
            f"noise shape mismatch: {tuple(student_noise.shape)} != {tuple(cached_noise.shape)}"
        )
    if student_noise.dtype != cached_noise.dtype:
        raise AssertionError(
            f"noise dtype mismatch: {student_noise.dtype} != {cached_noise.dtype}"
        )
    if not torch.equal(student_noise.detach().cpu(), cached_noise.detach().cpu()):
        raise AssertionError("teacher target is not paired with the current student noise")
    metric = distance or functional.mse_loss
    return metric(student_output, teacher_target.detach())


def _per_sample_mean(value):
    if value.ndim == 0:
        raise ValueError("expected a batch dimension")
    return value.reshape(value.shape[0], -1).mean(dim=1)


def _require_same_shape(left, right, left_name: str, right_name: str) -> None:
    if left.shape != right.shape:
        raise ValueError(
            f"{left_name} and {right_name} must match: {tuple(left.shape)} != {tuple(right.shape)}"
        )
