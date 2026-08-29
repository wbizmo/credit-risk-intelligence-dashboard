# Model Card — CRIX-MonoBoost 1.0

## System context

CRIX API release: **v2.5.0**  
Bundled model: **CRIX-MonoBoost 1.0.0**

The API version and the model version are intentionally independent. v2.5 substantially changes service architecture, validation, governance surfaces and runtime hardening without pretending the underlying trained artifact was retrained.

## Intended use

CRIX-MonoBoost is an engineering/model-risk demonstration for showing how a credit-risk model can be calibrated, challenged, explained, stress-tested and exposed through a governed API contract.

It is **not approved for real lending decisions**.

## Target

The model estimates a synthetic **12-month probability of default (PD)**.

## Features

- debt-to-income ratio;
- revolving utilization;
- delinquencies in the last 24 months;
- recent credit inquiries;
- age of oldest trade;
- open account count;
- requested loan / annual income;
- employment tenure;
- liquidity buffer in months;
- on-time payment rate;
- income stability;
- recent credit growth.

The HTTP API intentionally does not request borrower names. Identity is not a model feature.

## Training and validation

The artifact is trained on 50,000 deterministic synthetic credit-performance records. Training, calibration and held-out test populations are separated. Monotonic constraints encode domain-informed directionality on selected features. A separate logistic calibration layer maps champion margins into probabilities.

Held-out snapshot:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.7334 |
| KS | 0.3405 |
| Brier score | 0.1024 |
| Log loss | 0.3452 |
| Test observations | 10,000 |
| Test default rate | 13.09% |

## Challenger

Every application is also evaluated by a transparent logistic challenger. The absolute champion/challenger PD difference is surfaced as `disagreement`. Material disagreement lowers confidence and emits a model-risk flag.

## Explainability

The API returns local reason codes derived from counterfactual sensitivity to a stable reference vector. These are useful for engineering/model-analysis demonstration; they are **not represented as legally sufficient adverse-action reasons**.

## Out-of-distribution handling

The API accepts a broad but bounded validation domain. Separately, the engine compares selected values with the synthetic training support. Inputs outside that support are returned through `outOfDistribution`, confidence is reduced, and `OUT_OF_DISTRIBUTION` is added to model-risk flags.

This is intentional: schema-valid does not mean model-trustworthy.

## Expected loss

CRIX derives:

`Expected Loss = PD × LGD × EAD`

LGD in this repository is a deterministic engineering approximation based on leverage, income stability and liquidity buffer. EAD is the requested loan amount. Neither should be treated as an institutionally validated production estimate.

## Decision policy

The model does not directly approve or decline applications. `CRIX-Policy 2.5` is a separate deterministic layer with explicit PD, DTI, delinquency, loan-to-income and confidence thresholds.

## Stress testing

The API supports deterministic `mild` and `severe` shocks that reduce income/liquidity/stability and increase leverage/utilization/credit growth, then re-run the full model and policy lifecycle.

## Production gates

Before any real credit use, at minimum:

1. train on representative historical performance data with point-in-time feature correctness;
2. document target/default definitions and observation/performance windows;
3. independently validate discrimination, calibration, stability and reason-code behaviour;
4. test fairness and prohibited/proxy-variable risks with legal/compliance review;
5. establish model registry, approvals, signed/versioned artifacts and controlled promotion;
6. monitor drift, calibration, overrides, decision rates, complaints and performance;
7. validate LGD/EAD separately;
8. govern policy and pricing independently from model development;
9. implement auditable adverse-action processes appropriate to applicable law;
10. add production authentication, authorization, audit retention and operational controls.
