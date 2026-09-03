import pytest

torch = pytest.importorskip("torch")

from generative_training_audit.gan import (  # noqa: E402
    audit_fake_branch,
    discriminator_fake_loss,
    parameter_grad_norm,
)


def make_discriminator():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1)
    )


def test_correct_fake_branch_updates_discriminator():
    discriminator = make_discriminator()
    fake = torch.randn(16, 4, requires_grad=True)
    norm = audit_fake_branch(discriminator, fake)
    assert norm > 0
    assert fake.grad is None


def test_detached_logits_have_finite_loss_but_zero_d_gradient():
    discriminator = make_discriminator()
    fake = torch.randn(16, 4)
    discriminator.zero_grad(set_to_none=True)
    loss, _ = discriminator_fake_loss(discriminator, fake, broken=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert parameter_grad_norm(discriminator.parameters()) == 0.0
