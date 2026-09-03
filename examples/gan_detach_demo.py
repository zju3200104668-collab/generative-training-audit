from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ImportError as error:
    raise SystemExit("Install the GAN extra first: pip install -e '.[gan]'") from error

from generative_training_audit.gan import (  # noqa: E402
    discriminator_fake_loss,
    parameter_grad_norm,
)


torch.manual_seed(0)
discriminator = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 1))
fake = torch.randn(16, 4)

for broken in (True, False):
    discriminator.zero_grad(set_to_none=True)
    loss, _ = discriminator_fake_loss(discriminator, fake, broken=broken)
    loss.backward()
    norm = parameter_grad_norm(discriminator.parameters())
    print(f"broken={str(broken):<5} loss={loss.item():.6f} D_grad_norm={norm:.6f}")
