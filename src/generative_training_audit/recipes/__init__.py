"""Framework-shaped, model-agnostic distillation recipes."""

from .dmd import dmd_projection_loss, fake_score_matching_loss, paired_regression_loss
from .meanflow import (
    IntervalBatch,
    meanflow_identity_target,
    one_step_prediction,
    sample_intervals,
    scale_invariant_mse,
)

__all__ = [
    "IntervalBatch",
    "dmd_projection_loss",
    "fake_score_matching_loss",
    "meanflow_identity_target",
    "one_step_prediction",
    "paired_regression_loss",
    "sample_intervals",
    "scale_invariant_mse",
]
