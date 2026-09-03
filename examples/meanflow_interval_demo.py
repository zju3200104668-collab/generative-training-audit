"""Executable MeanFlow interval sampling and identity-target example."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ImportError as error:
    raise SystemExit("Install the GAN extra first: pip install -e '.[gan]'") from error

from generative_training_audit.recipes.meanflow import (  # noqa: E402
    meanflow_identity_target,
    one_step_prediction,
    sample_intervals,
    scale_invariant_mse,
)


generator = torch.Generator().manual_seed(11)
intervals = sample_intervals(16, generator=generator)
z_hq = torch.randn(16, 4, 8, 8, generator=generator)
z_lq = torch.randn(16, 4, 8, 8, generator=generator)
z_t = (1.0 - intervals.t) * z_hq + intervals.t * z_lq
velocity = z_lq - z_hq

# In a real model, total_derivative comes from JVP along (dz/dt=v, dt/dt=1).
u = torch.zeros_like(velocity, requires_grad=True)
total_derivative = torch.zeros_like(velocity)
target = meanflow_identity_target(velocity, intervals.h, total_derivative)
loss, raw_error = scale_invariant_mse(u, target)
prediction_at_r = one_step_prediction(z_t, u, intervals.h)

print(f"FM anchors:       {int(intervals.flow_matching_mask.sum())}")
print(f"one-step samples: {int(intervals.one_step_mask.sum())}")
print(f"generic intervals:{int((~intervals.flow_matching_mask & ~intervals.one_step_mask).sum())}")
print(f"weighted loss:    {float(loss):.6f}")
print(f"raw MSE sum:      {float(raw_error.mean()):.6f}")
print(f"prediction shape: {tuple(prediction_at_r.shape)}")
