# Changelog

All notable CRIX releases are documented here.

## 3.1.0 — 2026-09-12

### Model governance and point-in-time research

- Added machine-checkable point-in-time feature provenance, leakage guards, granted-loans-only population conditioning and explicit reject-inference limitations.
- Added segmented calibration, deterministic bootstrap uncertainty intervals, calibration intercept/slope, PSI stability diagnostics and observed-cohort policy backtesting.
- Added immutable model-manifest / registry integrity checks with SHA-256 artifact verification.
- Added historical as-of snapshots and point-in-time backtesting infrastructure.

### Lifetime credit-risk stack

- Added research-only lifetime PD term structures with censoring-aware survival modelling and 3/6/12/24/36-month outputs.
- Added delinquency-state transition and cure/default migration research.
- Added empirical LGD research using recovery severity / timing evidence while preserving the live API's existing deterministic LGD baseline.
- Added empirical instalment EAD research and a separate revolving-credit / CCF research path.
- Kept all research models product- and horizon-specific instead of blending incompatible datasets.

### Portfolio and stress research

- Added deterministic, chunk-invariant one-factor correlated-default Monte Carlo with expected loss, unexpected loss, VaR, expected shortfall and tail-risk contributions.
- Added point-in-time macro-conditioned stress research using a frozen, provenance-tracked U.S. unemployment-rate snapshot rather than a CI-time network dependency.
- Added explicit tail-precision safeguards, including withholding 99.9% metrics when scenario support is insufficient.

### Accounting, capital and decision optimisation

- Added IFRS 9-style research staging and scenario-weighted ECL with SICR, DPD, default and cure/probation semantics.
- Added Basel-style / economic-capital research that keeps expected loss separate from unexpected/tail capital and does not claim regulatory compliance.
- Added real champion-vs-challenger governance on the same LendingClub calibration/OOT cohorts, including calibration, Brier/log-loss, PSI, bootstrap uncertainty and segment evidence.
- Added bounded deterministic portfolio optimisation with explicit infeasibility, no hidden constraint relaxation and exact toy-portfolio verification.

### Engineering and reproducibility

- Optimized the live scoring hot path around a single immutable per-request scoring context while keeping explanation work bounded to the small champion feature set.
- Added deterministic Monte Carlo reduction across chunk sizes and fail-closed validation for invalid probabilities, exposures, correlations and scenario weights.
- Hardened isolated research retraining so cold GitHub Actions caches are hydrated from verified source files instead of assuming cache hits.
- Expanded the full model/governance suite to 57 tests and kept the live `/api/v3` contract and CRIX-MonoBoost 2.0 runtime champion semantics unchanged.

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

- Added separate UCI Taiwan credit-card behavioral adaptation model using six-month utilization, payment behavior and normalized balance growth; 10-fold CV ROC-AUC **0.7606**, Brier **0.1416**, log loss **0.4464** across 30,000 clients.
- Explicitly excludes `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` from the Taiwan research artifact.
- Added legacy UCI Statlog German Credit structural benchmark; 10-fold CV ROC-AUC **0.6672**.
- Added corrected UCI South German Credit structural benchmark; 10-fold CV ROC-AUC **0.6718**, and documents why it is preferred over the legacy encoding.
- Added dataset registry entries for gated/future Home Credit, Give Me Some Credit, FICO HELOC, Freddie Mac and Fannie Mae sources without claiming they were used.
- Added reproducible GitHub Actions training/benchmark workflow with resilient, checksum-verified Zenodo download and source caching.

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
