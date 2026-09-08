# Changelog

All notable CRIX releases are documented here.

## 3.0.0 — 2026-09-08

### Breaking

- API namespace moves from `/api/v2` to `/api/v3`.
- `creditScore` is now a required application field.
- `pd` changes from the v2 synthetic 12-month demonstration target to **real-data final-loan-resolution default risk**.
- Every score now includes `pdHorizon` so target semantics travel with the result.

### Model

- Replaced the 50,000-row synthetic CRIX-MonoBoost 1.0 cohort with **CRIX-MonoBoost 2.0** trained on the open LendingClub/Zenodo granting-model dataset.
- Locked primary source provenance with DOI `10.5281/zenodo.11295916` and MD5 `b019384d6bc65bf2a3e839362e4ff502`.
- Uses 786,730 originations through 2015 for training, 274,200 2016 originations for calibration and 157,119 2017 originations for out-of-time evaluation.
- Primary OOT ROC-AUC 0.6594, KS 0.2300, Brier 0.1645, log loss 0.5043.
- Replaced the hand-authored challenger with a real-data standardized logistic challenger; OOT ROC-AUC 0.6578.
- OOD support now comes from real training-distribution percentiles.
- Added explicit source/target/training metadata to the model endpoint.

### Real-world adaptation research

- Added separate UCI Taiwan credit-card behavioral adaptation model using six-month utilization, payment behavior and normalized balance growth.
- Explicitly excludes `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` from the Taiwan research artifact.
- Added legacy UCI Statlog German Credit structural benchmark.
- Added corrected UCI South German Credit structural benchmark and documents why it is preferred.
- Added dataset registry entries for gated/future Home Credit, Give Me Some Credit, FICO HELOC, Freddie Mac and Fannie Mae sources without claiming they were used.
- Added reproducible GitHub Actions training/benchmark workflow.

### Governance

- Rewrote the model card for real-data target semantics, temporal validation, transfer limitations, selection bias and production gates.
- Added primary training report and external benchmark report.
- Updated security guidance to prohibit real consumer data in the public demo and document protected-field exclusions.
- Preserved strict separation between statistical model output and `CRIX-Policy 3.0`.

### Brand / docs

- Added new CRIX v3 risk-grid wordmark.
- Rewrote README and architecture documentation around the real-data model and product-specific evidence layers.

## 2.5.0

- Stateless Fastify/OpenAPI architecture hardening.
- Synthetic CRIX-MonoBoost 1.0 retained while API/model governance surfaces were expanded.

## 1.0.0

- Initial CRIX public release.
