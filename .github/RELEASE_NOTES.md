# CRIX v3.0.0 — Real-World Credit Risk

CRIX v3 is the first release whose primary risk model is trained, calibrated and evaluated on **real historical credit outcomes** rather than synthetic borrowers.

## Breaking API changes

- The versioned API namespace moves from `/api/v2` to `/api/v3`.
- `creditScore` is now a required application field.
- `pd` now means **final-loan-resolution default risk**, not the synthetic 12-month PD used by v2.x.
- Every scoring response includes `pdHorizon` so target semantics travel with the probability.

## CRIX-MonoBoost 2.0

The primary champion is trained on the open **Lending Club loan dataset for granting models** distributed through Zenodo. The research cohort is deliberately curated to application-time variables to avoid post-underwriting leakage.

- DOI: `10.5281/zenodo.11295916`
- License: CC-BY-4.0
- Source rows: **1,347,681**
- CRIX-harmonized rows: **1,269,389**
- Immutable source MD5: `b019384d6bc65bf2a3e839362e4ff502`
- Training cohort: **786,730** originations through 2015
- Calibration cohort: **274,200** 2016 originations
- Out-of-time test: **157,119** 2017 originations

Primary OOT diagnostics:

| Metric | Value |
|---|---:|
| ROC-AUC | **0.6594** |
| KS | **0.2300** |
| Brier score | **0.1645** |
| Log loss | **0.5043** |
| OOT default rate | **22.42%** |
| Logistic challenger ROC-AUC | **0.6578** |

The champion uses only mappings the source genuinely supports: debt-to-income, requested-loan-to-income, `fico_n` as the external bureau-style score, and normalized employment tenure. CRIX does not pretend missing bureau fields were present in the training cohort.

The old hand-authored challenger has also been replaced with a standardized **real-data logistic challenger** trained on the same point-in-time feature contract. Out-of-distribution support is derived from the real training distribution.

## Multi-dataset adaptation without dataset soup

v3 also adds real external evidence while keeping incompatible products and target horizons separate.

### UCI Taiwan credit-card behavior

A separate research-only `CRIX-Behavior-TW 1.0.0` model uses six-month utilization, on-time-payment behavior, delayed-payment months and normalized balance growth across **30,000** real bank clients.

- 10-fold CV ROC-AUC: **0.7606**
- Brier score: **0.1416**
- Log loss: **0.4464**
- `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` are explicitly excluded.

Its target is next-month credit-card default, so its probability is **not blended** into the personal-loan champion or policy.

### German structural benchmarks

- UCI Statlog German Credit: **0.6672** 10-fold CV ROC-AUC.
- Corrected UCI South German Credit: **0.6718** 10-fold CV ROC-AUC.

The corrected South German representation is preferred because UCI documents coding problems in the legacy Statlog representation. These datasets remain structural benchmarks rather than being forced into the LendingClub feature contract.

Home Credit, Give Me Some Credit, FICO HELOC, Freddie Mac and Fannie Mae remain explicitly registered as future/gated or product-specific sources. CRIX does not claim to have trained on data it did not actually obtain under the applicable access terms.

## Reproducibility and resilience

- Added a reproducible GitHub Actions model-development workflow.
- Added checksum-verified Zenodo downloads with bounded retries and an alternate Records API endpoint.
- Added a cache keyed to the immutable LendingClub source MD5.
- Added committed primary training and external benchmark reports/artifacts.
- Added real-training-support OOD metadata and richer `/api/v3/model` provenance.
- Stabilized npm dependency resolution consistently across CI and Render.

## Governance, API and brand

- `CRIX-Policy 3.0` remains separate from statistical model output.
- Rewritten model card documents target semantics, selection bias, transfer limits, calibration and production gates.
- Security guidance explicitly prohibits sending real consumer data to the public demo.
- Architecture and README now distinguish primary runtime modeling from product-specific research artifacts.
- Added the new CRIX v3 risk-grid wordmark and a full changelog.

## Important model-risk note

Using real historical outcomes is a substantial upgrade in model development; it is **not** production lending approval. Real deployment still requires representative institution-specific data, independent validation, fairness/proxy testing, exact performance-window governance, legal/compliance review, validated LGD/EAD, governed adverse-action reasons, monitoring and formal model-risk approval.
