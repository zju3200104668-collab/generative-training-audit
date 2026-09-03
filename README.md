# Generative Training Audit

[![tests](https://github.com/zju3200104668-collab/generative-training-audit/actions/workflows/tests.yml/badge.svg)](https://github.com/zju3200104668-collab/generative-training-audit/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Executable checks for silent failures in diffusion distillation and adversarial training. Each audit targets an invariant that scalar losses cannot prove.

This repository turns four real debugging patterns into framework-light demos and executable tests:

1. **Scheduler trace mismatch** — a configured step count does not fully describe the solver trajectory.
2. **BF16 EMA freezing** — a valid EMA update can round back to the old shadow value.
3. **Noise-pair mismatch** — cached teacher targets become invalid when regenerated noise differs.
4. **GAN gradient disconnection** — detaching fake logits removes the discriminator's fake gradient.

The code is an independent educational reimplementation. It contains no proprietary model code, data, checkpoints, prompts, infrastructure paths, or internal configuration.

## Why this repository exists

Many training bugs do not crash. Losses remain finite and dashboards look stable, while the implemented objective silently differs from the intended one. The checks here focus on invariants that can be tested before a long training run:

| Failure | Misleading signal | Decisive check |
|---|---|---|
| Scheduler mismatch | `num_inference_steps == 2` | Log actual time/sigma transitions |
| BF16 EMA freeze | EMA update runs every step | Compare update size with BF16 spacing |
| Noise mismatch | Cached target loads correctly | Hash and compare the exact noise tensor |
| GAN disconnect | Fake loss has a finite value | Backpropagate fake-only loss into D |

## Installation

```bash
git clone https://github.com/zju3200104668-collab/generative-training-audit.git
cd generative-training-audit
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

PyTorch is only required for the GAN gradient audit. The other three demos use NumPy.

## 60-second audit

```bash
gtaudit all
gtaudit all --json  # machine-readable output for CI or experiment launchers
```

Expected output:

```text
[PASS] bf16_ema: detected BF16 freeze while FP32 master still updates
       bf16_shadow_changed=False
       fp32_shadow_changed=True
       update_to_spacing_ratio=0.00128
[PASS] noise_pairing: exact pair passed and mismatched noise was rejected
       mismatch_detected=True
[PASS] scheduler_trace: executed transitions satisfy count, endpoint and continuity invariants
       transition_count=2
[PASS] gan_gradient: detached logits lose D gradients; detached samples preserve them
       broken_grad_norm=0.0
       correct_grad_norm=<positive value>
```

Run individual demonstrations and the full test suite:

```bash
python examples/ema_freeze_demo.py
python examples/noise_pairing_demo.py
python examples/scheduler_trace_demo.py
python examples/gan_detach_demo.py
pytest -q
```

## Integration examples

### Audit a configured scheduler

The adapter accepts a Diffusers-like object after `set_timesteps` and requires the common `N timesteps / N+1 sigmas` contract to be explicit:

```python
from generative_training_audit.scheduler import (
    audit_transitions,
    format_trace,
    transitions_from_scheduler,
)

scheduler.set_timesteps(num_inference_steps=2)
trace = transitions_from_scheduler(scheduler, terminal_time=0.0)
print(format_trace(trace))

issues = audit_transitions(
    trace,
    expected_steps=2,
    expected_end=0.0,
)
if issues:
    raise RuntimeError(f"unexpected solver trajectory: {issues}")
```

### Keep the EMA master in FP32

```python
import torch

from generative_training_audit.torch_ema import TorchEMA

model = model.to(dtype=torch.bfloat16)
ema = TorchEMA(model, decay=0.999)

optimizer.step()
ema.update(model)  # every shadow tensor remains FP32

ema.copy_to(eval_model)
```

### Assert teacher-cache noise pairing

```python
from generative_training_audit.noise import assert_noise_pair, make_noise_record

# Cache creation
record = make_noise_record(seed, exact_teacher_noise)

# Training startup / sample loading
assert_noise_pair(record, current_student_noise)
```

### Audit the discriminator's fake-only gradient

```python
from generative_training_audit.gan import audit_fake_branch

grad_norm = audit_fake_branch(discriminator, fake_batch)
assert grad_norm > 0
```

### Expected EMA behavior

The demo compares two update rules:

```text
unsafe: BF16 shadow <- rounded BF16 EMA update
safe:   FP32 shadow <- FP32 EMA update
```

When `(1 - decay) * |parameter - shadow|` is smaller than the representable spacing near the shadow value, the unsafe update can remain bitwise unchanged.

### Expected noise-pairing behavior

A teacher target is paired with the exact noise used to generate it. The demo persists a deterministic seed and a tensor fingerprint, then shows that a newly sampled tensor fails the pairing assertion.

### Expected GAN behavior

The broken path detaches `fake_logits`, so fake BCE still has a valid forward value but:

```text
d(fake_loss) / d(discriminator_parameters) = 0
```

The corrected path detaches the fake **sample** before passing it to the discriminator, preserving gradients into D while blocking gradients into G.

## Repository layout

```text
generative-training-audit/
├── src/generative_training_audit/
│   ├── ema.py
│   ├── gan.py
│   ├── noise.py
│   ├── report.py
│   ├── scheduler.py
│   └── torch_ema.py
├── examples/
├── tests/
├── docs/integration-checklist.md
└── .github/workflows/tests.yml
```

## Design principles

- Test gradients and trajectories, not only scalar losses.
- Record real/fake and data/regularization terms separately.
- Treat random-variable pairing as part of the dataset contract.
- Keep slow state such as EMA masters in FP32.
- State evidence boundaries explicitly; do not infer missing experiment details.

For a pre-flight review before expensive training, use the [training integration checklist](docs/integration-checklist.md).

## Scope

This is a diagnostic teaching repository, not a training framework. The scheduler utility does not reproduce every scheduler in Diffusers, and the BF16 conversion is a small NumPy emulation intended to expose rounding behavior. Validate production code against the exact framework and hardware used for training.

## CI coverage

Every push runs the suite on Python 3.10 and 3.12. The tests cover:

- bit-level BF16 rounding and the frozen-shadow counterexample;
- deterministic noise regeneration and mismatch rejection;
- transition count, endpoint and scheduler-contract validation;
- finite fake BCE with zero discriminator gradient under broken detachment;
- non-zero discriminator gradient under correct sample detachment;
- FP32 shadow storage for a BF16 PyTorch module;
- CLI text/JSON output and exit status.

## License

MIT. See [LICENSE](LICENSE).
