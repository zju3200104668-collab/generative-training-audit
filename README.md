# Generative Training Audit

Small, reproducible checks for silent failures in diffusion distillation and adversarial training.

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
git clone https://github.com/<your-account>/generative-training-audit.git
cd generative-training-audit
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

PyTorch is only required for the GAN gradient audit. The other three demos use NumPy.

## Quick start

```bash
python examples/ema_freeze_demo.py
python examples/noise_pairing_demo.py
python examples/scheduler_trace_demo.py
python examples/gan_detach_demo.py
pytest -q
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
│   └── scheduler.py
├── examples/
├── tests/
└── .github/workflows/tests.yml
```

## Design principles

- Test gradients and trajectories, not only scalar losses.
- Record real/fake and data/regularization terms separately.
- Treat random-variable pairing as part of the dataset contract.
- Keep slow state such as EMA masters in FP32.
- State evidence boundaries explicitly; do not infer missing experiment details.

## Scope

This is a diagnostic teaching repository, not a training framework. The scheduler utility does not reproduce every scheduler in Diffusers, and the BF16 conversion is a small NumPy emulation intended to expose rounding behavior. Validate production code against the exact framework and hardware used for training.

## License

MIT. See [LICENSE](LICENSE).
