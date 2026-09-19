# Changelog

All notable CRIX releases are documented here.

## 3.2.0 — 2026-09-19

### Runtime performance and bounded work

- Precompiled the champion execution plan and removed invariant JSON/object work from request-time scoring.
- Indexed champion trees by feature so explanation counterfactuals revisit only affected trees; deployed explanation work fell from 384 to 317 tree visits (17.45%).
- Recorded ~536,854 compiled champion ops/s and 0.003746 ms p50 champion latency on the Batch A CI host; full compiled assessment p50 was 0.018628 ms versus 0.025889 ms for the reference path.
- Added deterministic operation-count complexity gates, latency percentiles, memory evidence and real Fastify loopback load profiles.
- Final repo audit collapsed batch response accounting to one pass and removed avoidable per-subset O(n) reconstruction from the bounded exact portfolio optimiser using Gray-code incremental state.

### Portfolio dependency and tail-risk research

- Added Student-t dependence and bounded low-rank multi-factor dependence alongside the canonical one-factor Gaussian simulator.
- Added single-replay tail attribution and research-only POT/GPD EVT tail extrapolation with explicit unstable/insufficient states and finite-mean expected-shortfall rules.
- Final indexed-bucket tail replay measured ~0.80x reference speed at one quantile, ~2.42x speedup at three quantiles and ~4.03x at five quantiles, demonstrating the intended Q-scaling crossover rather than claiming a universal win.
- Final audit removed the hidden per-quantile mask scan and moved invariant backend thresholds/loadings outside the simulation chunk loop.
- Added an optional lazy CuPy backend while retaining NumPy as the canonical deterministic path. No GPU speedup/crossover claim is made in v3.2.0 because connected release infrastructure has no CUDA device.

### Model governance, explanation validation and lineage

- Added adversarial validation plus PSI, Jensen-Shannon, Wasserstein, support-breach, missingness and quantile-movement evidence.
- Historical governed shift evidence produced adversarial AUC 0.5593 with aggregate and all four champion-feature statuses passing the configured review bands.
- Added deterministic offline SHAP-vs-local-sensitivity validation. On n=512 OOT observations, mean top-3 overlap was 97.59%, sign agreement 95.33% and bounded-perturbation stability 96.94%.
- Added versioned source/harmonized/split/external dataframe contracts, immutable source checksum gating and aggregate-only failure evidence.
- Added committed tamper-evident training lineage with artifact/data/evidence/dependency hashes and read-only CI verification.

### Runtime security and observability

- Added explicit `public-demo` and fail-closed `required` auth modes, bounded current+next key rotation, constant-time digest comparison and explicit reverse-proxy trust.
- Added key-aware plus supplementary-IP quotas without placing credentials/hashes in labels.
- Added optional privacy-bounded OpenTelemetry metrics for HTTP/risk/runtime/readiness signals with no borrower IDs, raw feature values or exact PD labels.
- In-memory telemetry benchmark measured ~5.3% mean overhead and ~1.0% p95 overhead in the recorded Batch D loopback run.

### Reproducible dependencies and supply-chain evidence

- Added committed npm and hash-pinned Python research locks plus lock-provenance metadata and deliberate CI lock-drift failure tests.
- Removed an unused optbinning/ortools/protobuf research chain that carried advisories and moved Swagger UI to the patched dependency line.
- Batch D certification reported 0 known Node production vulnerabilities and 0 known vulnerabilities in the governed Python research lock.

### Compatibility

- Package release is **3.2.0** and the API contract version is now **3.2.0** under the unchanged major namespace `/api/v3`.
- OpenAPI/discovery/health/readiness and scored response envelopes expose `releaseVersion` and `apiVersion` separately; both are **3.2.0** in this release.
- CRIX-MonoBoost 2.0.0, CRIX-Policy 3.0, final-loan-resolution PD semantics and existing policy thresholds remain unchanged.
- Heavy Python/Monte Carlo/EVT/optimisation work remains outside the Fastify request path.

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
