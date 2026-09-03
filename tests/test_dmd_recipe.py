import pytest

torch = pytest.importorskip("torch")

from generative_training_audit.gan import parameter_grad_norm  # noqa: E402
from generative_training_audit.recipes.dmd import (  # noqa: E402
    dmd_projection_loss,
    paired_regression_loss,
)


def test_projection_updates_student_but_not_score_estimators():
    student = torch.nn.Linear(4, 4)
    real_score = torch.nn.Linear(4, 4)
    fake_score = torch.nn.Linear(4, 4)
    source = torch.randn(8, 4)
    state = student(source)
    real_velocity = real_score(state.detach())
    fake_velocity = fake_score(state.detach())
    dmd_projection_loss(state, real_velocity, fake_velocity).backward()
    assert parameter_grad_norm(student.parameters()) > 0
    assert parameter_grad_norm(real_score.parameters()) == 0
    assert parameter_grad_norm(fake_score.parameters()) == 0


def test_paired_regression_rejects_different_noise():
    student_output = torch.randn(2, 4, requires_grad=True)
    teacher_target = torch.randn(2, 4)
    noise = torch.randn(2, 4)
    with pytest.raises(AssertionError, match="not paired"):
        paired_regression_loss(
            student_output,
            teacher_target,
            student_noise=noise,
            cached_noise=noise.clone().add_(1e-3),
        )


def test_paired_regression_updates_student_only():
    student_output = torch.randn(2, 4, requires_grad=True)
    teacher_target = torch.randn(2, 4, requires_grad=True)
    noise = torch.randn(2, 4)
    loss = paired_regression_loss(
        student_output,
        teacher_target,
        student_noise=noise,
        cached_noise=noise.clone(),
    )
    loss.backward()
    assert student_output.grad is not None
    assert teacher_target.grad is None
