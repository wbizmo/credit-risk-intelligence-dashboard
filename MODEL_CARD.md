# Model Card — CRIX-MonoBoost 2.0

## System context

CRIX API release: **v3.0.0**  
Bundled primary model: **CRIX-MonoBoost 2.0.0**  
Secondary research artifact: **CRIX-Behavior-TW 1.0.0**

The API and model versions are intentionally independent. v3 is a major API release because both the required input contract and the probability target semantics changed.

## Intended use

CRIX-MonoBoost is an engineering/model-risk demonstration showing how a real-data credit-risk model can be trained with point-in-time discipline, calibrated on later originations, challenged, explained, stress-tested and exposed through a governed API contract.

It is **not approved for real lending decisions**.

## Primary target

The primary model estimates:

> **Probability that a granted LendingClub loan ultimately resolves as charged-off/default rather than fully paid.**

This is a **final-loan-resolution probability**, spanning mixed contractual terms. It is explicitly **not a 12-month PD** and must not be relabeled as one.

The API includes `pdHorizon` on every score so downstream clients cannot safely ignore this distinction.

## Primary training source

**Lending Club loan dataset for granting models**

- Open academic distribution via Zenodo
- DOI: `10.5281/zenodo.11295916`
- License: CC-BY-4.0
- Source period: 2007–2018
- Source rows: 1,347,681
- Rows after CRIX harmonization: 1,269,389
- MD5 lock: `b019384d6bc65bf2a3e839362e4ff502`
- Curated to variables available at granting/application time, avoiding post-underwriting leakage such as grade, subgrade and realized pricing signals

## Primary feature contract

The v3 champion uses only features that can be mapped honestly from the primary cohort:

| Feature | Construction | Monotonic constraint |
|---|---|---:|
| `debtToIncome` | LendingClub `dti_n`, converted to a ratio where required | +1 |
| `loanToIncome` | `loan_amnt / revenue` | +1 |
| `creditScore` | `fico_n` | -1 |
| `employmentYears` | normalized `emp_length` | -1 |

`annualIncome` and `loanAmount` are also required by the API because `loanToIncome` is computed at runtime and because policy/LGD/EAD use them.

The broader API still accepts utilization, delinquency, inquiry, trade-age, account-count, liquidity, payment-rate, income-stability and credit-growth context. Those fields may affect policy, stress testing, confidence context or deterministic LGD logic, but CRIX does **not** represent them as champion training features when the primary dataset does not contain them.

The HTTP API intentionally does not request borrower names.

## Temporal training and evaluation

CRIX uses chronological origination cohorts rather than a random split:

| Cohort | Origination period | Rows | Default rate |
|---|---|---:|---:|
| Train | 2007-06 through 2015-12 | 786,730 | 18.17% |
| Calibration | 2016-01 through 2016-12 | 274,200 | 22.71% |
| Out-of-time test | 2017-01 through 2017-12 | 157,119 | 22.42% |

2018 is intentionally excluded from primary reported evaluation to reduce final-status maturation bias in a resolved-loan dataset.

## Primary out-of-time diagnostics

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6594 |
| KS | 0.2300 |
| Brier score | 0.1645 |
| Log loss | 0.5043 |
| OOT observations | 157,119 |
| OOT default rate | 22.42% |
| Logistic challenger ROC-AUC | 0.6578 |

These results are intentionally not presented as an “accuracy” score. They are model-development diagnostics on one historical lending population.

## Calibration

The monotonic XGBoost champion is trained on the train cohort. A separate Platt/logistic calibration layer is fitted only on 2016 originations, then evaluated on the later 2017 OOT cohort.

The committed artifact contains the calibration parameters and calibration-bin diagnostics.

## Challenger

The v3 challenger is no longer a hand-authored demonstration equation. It is a standardized logistic-regression model trained on the same real point-in-time feature contract as the champion and exported with the artifact.

Every request surfaces:

- `challengerPd`;
- absolute `disagreement`;
- confidence reduction when disagreement is material;
- `MODEL_DISAGREEMENT` when the configured threshold is crossed.

`fico_n` is used as a genuine external bureau-style feature inside both models; it is not treated as a ground-truth label or as a replacement for outcome data.

## Out-of-distribution handling

OOD support is derived from the primary training cohort. The artifact stores 1st/99th percentile support for trained features. Inputs outside support are surfaced through `outOfDistribution`, reduce confidence and emit `OUT_OF_DISTRIBUTION`.

This is intentional: schema-valid does not mean model-trustworthy.

## External adaptation / benchmark models

CRIX v3 also records product-specific real-data evidence without blending incompatible targets.

### UCI Default of Credit Card Clients — Taiwan

A separate `CRIX-Behavior-TW 1.0.0` research artifact is trained on four behaviorally defensible derived features:

- six-month mean utilization;
- six-month on-time-payment rate;
- number of delayed-payment months in the six-month observation window;
- six-month bill-balance growth normalized by credit limit.

`SEX`, `EDUCATION`, `MARRIAGE` and `AGE` are excluded.

Missing personal-loan/bureau variables are **not imputed**. Its target is **next-month credit-card default**, so its probability is not blended with CRIX-MonoBoost and does not drive the runtime policy.

### German credit benchmarks

Legacy UCI Statlog German Credit and the corrected UCI South German Credit dataset are used only as structural benchmarks on the limited overlapping loan-structure features. UCI documents coding issues in the legacy representation; the corrected South German representation is preferred.

These benchmarks are not evidence that the LendingClub champion transfers unchanged across geographies/products.

## Explainability

The API returns local reason codes from counterfactual sensitivity against real training-reference medians for trained features. These are useful for model-analysis demonstration; they are **not represented as legally sufficient adverse-action reasons**.

## Expected loss

CRIX derives:

`Expected Loss = PD × LGD × EAD`

LGD remains a deterministic engineering approximation based on leverage, income stability and liquidity buffer. EAD is the requested loan amount. Neither is institutionally validated in this repository.

Because the primary PD target is final-resolution risk, expected-loss outputs must be interpreted within the same horizon limitation.

## Decision policy

The model does not directly approve or decline applications. `CRIX-Policy 3.0` is a separate deterministic layer with explicit PD, DTI, FICO/credit-score, delinquency, loan-to-income and confidence thresholds.

The policy is deliberately versioned independently from the model.

## Stress testing

The API supports deterministic `mild` and `severe` borrower shocks, then reruns the complete model and policy lifecycle.

Only shocks affecting trained primary-model features can directly change primary PD. Other shocked contextual fields can still alter LGD, policy or flags. This limitation is deliberate and documented rather than hidden.

## Known limitations

1. The primary outcome population is historical LendingClub US p2p lending, not an institution-specific production portfolio.
2. `dti_n` is narrower than a universal bureau DTI definition; its source semantics must not be generalized silently.
3. Employment tenure is banded/normalized.
4. The champion lacks several bureau-style features that CRIX accepts at its policy/context layer.
5. Final-resolution default risk is not a fixed-horizon 12-month PD.
6. The dataset includes only granted loans, creating the usual accepted-applicant / underwriting-selection limitation.
7. Historical performance does not guarantee present-day calibration or geographic transfer.
8. No fairness conclusion can be made without the appropriate protected-attribute evaluation data and governance process.
9. LGD/EAD/pricing are engineering approximations rather than validated institution models.
10. External UCI models have different products and target horizons and are intentionally kept separate.

## Production gates

Before any real credit use, at minimum:

1. train/recalibrate on representative institution-specific historical performance data with point-in-time feature correctness;
2. define target/default definitions and observation/performance windows contractually;
3. independently validate discrimination, calibration, stability and reason-code behavior;
4. test fairness, prohibited variables and proxy risk with legal/compliance review;
5. validate selection bias and reject-inference assumptions where relevant;
6. establish model registry, approvals, signed/versioned artifacts and controlled promotion;
7. monitor drift, calibration, overrides, decision rates, complaints and performance;
8. validate LGD/EAD separately;
9. govern policy and pricing independently from model development;
10. implement auditable adverse-action processes appropriate to applicable law;
11. add production-grade authentication, authorization, audit retention and operational controls;
12. prohibit real consumer data from the public demo environment.

## Reproducibility

Primary training evidence is committed in:

- `model/artifacts/crix-monoboost-v2.json`
- `model/TRAINING_REPORT.md`

External research evidence is committed in:

- `model/artifacts/crix-behavior-tw-v1.json`
- `model/artifacts/external-benchmarks.json`
- `model/EXTERNAL_BENCHMARKS.md`

The GitHub Actions training workflow can reproduce these artifacts from public sources.
