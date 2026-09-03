# Public distillation recipes

The `recipes` package is an independent, model-agnostic reconstruction of two training patterns: DMD-style distribution matching and MeanFlow-style interval learning. It is designed to expose the mathematical contracts and gradient boundaries without reproducing a private training stack.

## DMD-style two-optimizer training

A DMD-style setup contains three roles:

| Component | Role | Updated by |
|---|---|---|
| Student | Few-step generator | Distribution-matching and optional paired regression losses |
| Real score model | Reference distribution estimator | Frozen |
| Fake score model | Student-distribution estimator | Its own score/flow-matching regression loss |

The student projection loss is represented as:

```python
loss_dm = dmd_projection_loss(x_dm, v_real, v_fake)
```

Internally, `(v_real - v_fake)` is detached. The projection therefore updates the student through `x_dm` without sending the student optimizer's gradients into either score estimator. The sign and time-dependent scaling of the direction depend on the model parameterization and must be derived for the target scheduler; the helper does not guess them.

The fake score estimator receives a separate update:

```python
x_for_fake = x_student.detach()
v_fake = fake_score(x_for_fake)
loss_fake = fake_score_matching_loss(v_fake, target_velocity)
```

Detaching the student state here prevents the fake-score optimizer from modifying the generator.

### Paired regression

Sample-wise teacher regression is valid only when the teacher target and student input share the intended random-variable coupling. During debugging, use:

```python
loss_reg = paired_regression_loss(
    student_output,
    teacher_target,
    student_noise=current_noise,
    cached_noise=target_noise,
)
```

The strict tensor equality check is intentionally expensive and should normally be replaced by the repository's stored fingerprint check after the cache pipeline has been validated.

## MeanFlow-style interval training

For a conditional linear path between target and source latents,

$$
z_t=(1-t)z_{target}+t z_{source},
\qquad
v=z_{source}-z_{target},
$$

the model predicts an average velocity over `[r,t]`, where `h=t-r`. The public sampler uses three semantic branches:

1. `r=t`: zero-length Flow Matching anchors;
2. `t=1, r=0`: deployment-aligned one-step intervals;
3. `0<=r<t<=1`: generic intervals needed to learn the full field.

Branches are sampled independently per item rather than allocated with integer batch counts. At batch size one, every branch therefore retains its configured probability instead of becoming structurally impossible.

The identity target is:

$$
u_{target}=v-h\frac{d}{dt}u,
$$

where the total derivative follows the probability path. If `h` is held fixed,

$$
\frac{d}{dt}u(z_t,t,h)
=\partial_t u+J_z(u)v.
$$

The helper accepts the already-computed total derivative and detaches the complete target. This avoids presenting a framework-specific JVP implementation as universally correct.

The reconstruction from `t` to `r` is:

```python
z_r = one_step_prediction(z_t, average_velocity=u, interval=t-r)
```

For image-domain supervision, callers should mask samples according to endpoint semantics. A target-image loss is directly meaningful for intervals ending at the target endpoint (`r=0`); it should not be applied indiscriminately to arbitrary intermediate endpoints.

## Deliberate omissions

The recipes do not include:

- model- or employer-specific class names;
- private datasets, prompts, weights or cached targets;
- infrastructure paths and launch configurations;
- production learning rates, loss weights or checkpoint schedules;
- claims that one parameterization's velocity scaling applies to every scheduler.

See the executable examples and tests for the supported contracts.
