"""PyTorch gradient-path audits for adversarial training."""

from __future__ import annotations

from collections.abc import Iterable


def parameter_grad_norm(parameters: Iterable) -> float:
    """Compute an FP32 L2 norm over existing parameter gradients."""

    import torch

    squared = torch.zeros((), dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().square().sum().cpu()
    return float(squared.sqrt())


def discriminator_fake_loss(discriminator, fake, *, broken: bool = False):
    """Return fake BCE; broken=True reproduces logit-level detachment."""

    import torch
    import torch.nn.functional as functional

    fake_logits = discriminator(fake.detach())
    if broken:
        fake_logits = fake_logits.detach().requires_grad_(True)
    loss = functional.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )
    return loss, fake_logits


def audit_fake_branch(discriminator, fake) -> float:
    """Assert that fake-only BCE produces a finite, non-zero D gradient."""

    import math

    discriminator.zero_grad(set_to_none=True)
    loss, _ = discriminator_fake_loss(discriminator, fake, broken=False)
    loss.backward()
    norm = parameter_grad_norm(discriminator.parameters())
    if not math.isfinite(norm) or norm <= 0.0:
        raise AssertionError(f"fake branch has invalid D gradient norm: {norm}")
    return norm
