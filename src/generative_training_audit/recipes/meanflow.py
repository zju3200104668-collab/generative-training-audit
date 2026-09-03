"""Interval sampling and targets for MeanFlow-style conditional transport."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntervalBatch:
    """Per-sample endpoints and semantic branch masks."""

    t: object
    r: object
    h: object
    flow_matching_mask: object
    one_step_mask: object


def sample_intervals(
    batch_size: int,
    *,
    device=None,
    dtype=None,
    generator=None,
    flow_matching_probability: float = 1.0 / 3.0,
    one_step_probability: float = 1.0 / 3.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> IntervalBatch:
    """Sample FM anchors, one-step intervals and generic MeanFlow intervals.

    Branches are drawn independently per sample. This avoids integer batch
    partitioning schemes in which a branch can become impossible at batch=1.
    """

    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _validate_probabilities(flow_matching_probability, one_step_probability)
    dtype = dtype or torch.float32
    shape = (batch_size, 1, 1, 1)
    left = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    right = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    left = torch.sigmoid(left * logit_std + logit_mean)
    right = torch.sigmoid(right * logit_std + logit_mean)
    t = torch.maximum(left, right)
    r = torch.minimum(left, right)

    draw = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    fm_mask = draw < flow_matching_probability
    one_step_mask = (draw >= flow_matching_probability) & (
        draw < flow_matching_probability + one_step_probability
    )
    r = torch.where(fm_mask, t, r)
    t = torch.where(one_step_mask, torch.ones_like(t), t)
    r = torch.where(one_step_mask, torch.zeros_like(r), r)
    return IntervalBatch(
        t=t,
        r=r,
        h=t - r,
        flow_matching_mask=fm_mask,
        one_step_mask=one_step_mask,
    )


def meanflow_identity_target(velocity, interval, total_derivative):
    """Return the detached target ``v - h * d_t u``.

    ``total_derivative`` must be computed along the probability path. For a
    field ``u(z_t, t, h)`` with fixed h, this includes both explicit time and
    state transport terms: ``partial_t u + J_z(u) v``.
    """

    return (velocity - interval * total_derivative).detach()


def one_step_prediction(state_t, average_velocity, interval):
    """Integrate the predicted average velocity from t back to r."""

    return state_t - interval * average_velocity


def scale_invariant_mse(prediction, target, *, epsilon: float = 0.01):
    """Return weighted loss and raw per-sample squared error.

    The denominator is detached so normalization changes gradient scale without
    creating a trivial derivative through the adaptive weight.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    squared_error = (prediction - target).square().reshape(prediction.shape[0], -1).sum(dim=1)
    weight = 1.0 / (squared_error.detach() + epsilon)
    return (squared_error * weight).mean(), squared_error


def _validate_probabilities(flow_matching_probability: float, one_step_probability: float) -> None:
    if not 0.0 <= flow_matching_probability <= 1.0:
        raise ValueError("flow_matching_probability must be in [0, 1]")
    if not 0.0 <= one_step_probability <= 1.0:
        raise ValueError("one_step_probability must be in [0, 1]")
    if flow_matching_probability + one_step_probability > 1.0:
        raise ValueError("branch probabilities must sum to at most 1")
