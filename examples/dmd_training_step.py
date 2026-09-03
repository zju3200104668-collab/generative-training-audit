"""Executable toy example of DMD-style two-optimizer gradient boundaries."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
    import torch.nn.functional as functional
except ImportError as error:
    raise SystemExit("Install the GAN extra first: pip install -e '.[gan]'") from error

from generative_training_audit.gan import parameter_grad_norm  # noqa: E402
from generative_training_audit.recipes.dmd import (  # noqa: E402
    dmd_projection_loss,
    fake_score_matching_loss,
)


torch.manual_seed(7)
student = torch.nn.Linear(8, 8)
real_score = torch.nn.Linear(8, 8).requires_grad_(False)
fake_score = torch.nn.Linear(8, 8)
condition = torch.randn(4, 8)

# Student update: the detached direction supervises the differentiable state.
x_dm = student(condition)
v_real = real_score(x_dm.detach())
v_fake = fake_score(x_dm.detach())
student_loss = dmd_projection_loss(x_dm, v_real, v_fake)
student_loss.backward()
print(f"student step: G_grad={parameter_grad_norm(student.parameters()):.6f}")
print(f"student step: fake_score_grad={parameter_grad_norm(fake_score.parameters()):.6f}")

# Fake-score update: fit the student's induced distribution on detached states.
fake_score.zero_grad(set_to_none=True)
noisy_student_state = student(condition).detach() + 0.1 * torch.randn(4, 8)
fake_velocity = fake_score(noisy_student_state)
target_velocity = -noisy_student_state  # toy flow-matching target
fake_loss = fake_score_matching_loss(fake_velocity, target_velocity)
fake_loss.backward()
print(f"fake-score step: fake_score_grad={parameter_grad_norm(fake_score.parameters()):.6f}")
