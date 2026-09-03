import pytest

torch = pytest.importorskip("torch")

from generative_training_audit.torch_ema import TorchEMA  # noqa: E402


def test_shadow_is_fp32_for_low_precision_module():
    module = torch.nn.Linear(2, 1).to(dtype=torch.bfloat16)
    ema = TorchEMA(module, decay=0.9)
    assert all(shadow.dtype == torch.float32 for _, shadow in ema.named_shadow_parameters())


def test_update_and_copy_to_preserve_target_dtype():
    module = torch.nn.Linear(2, 1).to(dtype=torch.bfloat16)
    ema = TorchEMA(module, decay=0.5)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(torch.ones_like(parameter))
    ema.update(module)
    target = torch.nn.Linear(2, 1).to(dtype=torch.bfloat16)
    ema.copy_to(target)
    assert all(parameter.dtype == torch.bfloat16 for parameter in target.parameters())
