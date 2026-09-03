"""Production-shaped FP32 EMA master for PyTorch modules."""

from __future__ import annotations

from collections.abc import Iterator


class TorchEMA:
    """Track trainable parameters in FP32 regardless of model precision.

    This intentionally keeps the API small: update the shadow after optimizer
    steps, then copy it into a compatible module for evaluation or export.
    """

    def __init__(self, module, decay: float = 0.999) -> None:
        import torch

        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must satisfy 0 <= decay < 1")
        self.decay = float(decay)
        self._names: list[str] = []
        self._shadow: list[torch.Tensor] = []
        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                self._names.append(name)
                self._shadow.append(parameter.detach().float().clone())
        if not self._shadow:
            raise ValueError("module has no trainable parameters")

    def update(self, module) -> None:
        """Apply one FP32 EMA update from a compatible module."""

        current = dict(module.named_parameters())
        with _no_grad():
            for name, shadow in zip(self._names, self._shadow):
                if name not in current:
                    raise KeyError(f"parameter missing from module: {name}")
                value = current[name].detach().float()
                if value.shape != shadow.shape:
                    raise ValueError(f"shape changed for parameter: {name}")
                shadow.lerp_(value, 1.0 - self.decay)

    def copy_to(self, module) -> None:
        """Copy EMA values into a compatible module using its parameter dtype."""

        current = dict(module.named_parameters())
        with _no_grad():
            for name, shadow in zip(self._names, self._shadow):
                target = current[name]
                target.copy_(shadow.to(device=target.device, dtype=target.dtype))

    def named_shadow_parameters(self) -> Iterator[tuple[str, object]]:
        return iter(zip(self._names, self._shadow))


def _no_grad():
    import torch

    return torch.no_grad()
