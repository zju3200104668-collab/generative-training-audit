# Changelog

All notable changes to this project are documented here.

## 0.3.0 - 2026-09-03

- Add sanitized, model-agnostic DMD and MeanFlow training recipes.
- Add explicit Student/Fake-score optimizer boundaries and paired-regression checks.
- Add per-sample MeanFlow interval sampling and endpoint-aware reconstruction helpers.
- Add executable PyTorch examples and gradient/identity tests.
- Document the mathematical assumptions and deliberate privacy omissions.

## 0.2.0 - 2026-09-03

- Add a unified `gtaudit` command with text and JSON output.
- Add a Diffusers-like scheduler trace adapter with explicit contract validation.
- Add an FP32 PyTorch EMA master suitable for BF16 modules.
- Add machine-readable audit reports and failure exit codes.
- Add CLI, scheduler integration and PyTorch EMA tests.
- Add a pre-flight training integration checklist and CI badges.

## 0.1.0 - 2026-09-03

- Add minimal reproductions for scheduler traces, BF16 EMA freezing, teacher-cache noise pairing and GAN gradient detachment.
- Add examples, tests, MIT license and GitHub Actions coverage.
