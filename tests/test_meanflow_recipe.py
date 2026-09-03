import pytest

torch = pytest.importorskip("torch")

from generative_training_audit.recipes.meanflow import (  # noqa: E402
    meanflow_identity_target,
    one_step_prediction,
    sample_intervals,
    scale_invariant_mse,
)


def test_interval_invariants_and_branch_semantics():
    generator = torch.Generator().manual_seed(123)
    batch = sample_intervals(4096, generator=generator)
    assert torch.all(batch.t >= batch.r)
    assert torch.allclose(batch.h, batch.t - batch.r)
    assert torch.all(batch.r[batch.flow_matching_mask] == batch.t[batch.flow_matching_mask])
    assert torch.all(batch.t[batch.one_step_mask] == 1)
    assert torch.all(batch.r[batch.one_step_mask] == 0)
    assert not torch.any(batch.flow_matching_mask & batch.one_step_mask)


def test_linear_path_reconstruction_with_exact_average_velocity():
    z_hq = torch.randn(4, 3)
    z_lq = torch.randn(4, 3)
    t = torch.rand(4, 1)
    z_t = (1 - t) * z_hq + t * z_lq
    velocity = z_lq - z_hq
    reconstructed = one_step_prediction(z_t, velocity, t)
    assert torch.allclose(reconstructed, z_hq, atol=1e-6)


def test_meanflow_target_is_detached_and_loss_has_gradient():
    prediction = torch.randn(3, 5, requires_grad=True)
    velocity = torch.randn(3, 5, requires_grad=True)
    derivative = torch.randn(3, 5, requires_grad=True)
    interval = torch.full((3, 1), 0.5)
    target = meanflow_identity_target(velocity, interval, derivative)
    assert not target.requires_grad
    loss, raw = scale_invariant_mse(prediction, target)
    loss.backward()
    assert prediction.grad is not None
    assert velocity.grad is None
    assert derivative.grad is None
    assert raw.shape == (3,)
