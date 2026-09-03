# Training integration checklist

Use this checklist before committing compute to a long distillation or adversarial-training run.

## Solver trajectory

- [ ] Record every executed `(t_i, sigma_i) -> (t_j, sigma_j)` transition.
- [ ] Assert the number of executed transitions, not only the configured step count.
- [ ] Assert start and terminal states.
- [ ] Compare training rollout and inference rollout traces.

## EMA state

- [ ] Keep EMA master parameters in FP32.
- [ ] Log the norm of `parameter - shadow` and the effective EMA increment.
- [ ] Verify the shadow changes after a controlled parameter update.
- [ ] Test checkpoint save/load without silently downcasting the shadow.

## Cached teacher targets

- [ ] Define one deterministic seed function from stable sample identifiers.
- [ ] Generate noise in a documented device and dtype order.
- [ ] Persist seed, shape, dtype and a debug fingerprint with the target.
- [ ] Assert exact pairing before computing sample-wise regression.

## Adversarial gradients

- [ ] During the D step, detach the fake sample before D—not the fake logits after D.
- [ ] During the G step, freeze D parameters while preserving D's input Jacobian.
- [ ] Run real-only, fake-only and generator-only backward tests.
- [ ] Log real/fake losses, logits, accuracies and gradient norms separately.
- [ ] Confirm a small D update moves real and fake logits in the intended directions.

## Evidence

- [ ] Keep a minimal reproducer for every fixed failure.
- [ ] Separate observed measurements from inferred explanations.
- [ ] Record loss reduction, scale aggregation and regularization weights.
- [ ] Pair scalar metrics with qualitative inspection when visual quality matters.
