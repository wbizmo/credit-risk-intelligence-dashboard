# CRIX v3 Architecture

CRIX is intentionally a **stateless backend API**. The primary risk model is the core product; HTTP is the delivery mechanism.

## Runtime boundary

```text
Client
  |
  v
Fastify API
  |-- health/readiness
  |-- OpenAPI/Swagger
  |-- strict schema validation
  |-- rate limiting
  |-- optional API-key auth
  |-- request/error controls
  |
  v
Risk Engine
  |-- CRIX-MonoBoost 2.0 primary PD
  |-- exported real-data logistic challenger
  |-- disagreement/confidence
  |-- real-training-support OOD detection
  |-- LGD/EAD/expected loss
  |-- score/grade
  |-- local reason codes
  |-- independent policy decision
  |-- deterministic stress scenarios
  |
  v
Versioned JSON model artifact
```

There is no database, Redis instance, queue, browser application, Python inference service or external inference dependency in the live system.

## Offline model-development boundary

Python is an offline build/research concern only.

```text
Public source datasets
  |
  +-- LendingClub / Zenodo
  |      -> checksum + harmonization
  |      -> chronological train / calibration / OOT split
  |      -> monotonic XGBoost champion
  |      -> Platt calibration
  |      -> logistic challenger
  |      -> compact runtime JSON artifact
  |
  +-- UCI Taiwan credit-card default
  |      -> protected-field exclusion
  |      -> reduced behavioral feature derivation
  |      -> CV benchmark
  |      -> research-only CRIX-Behavior-TW artifact
  |
  +-- UCI German credit datasets
         -> structural-only feature subset
         -> CV benchmarks
         -> no runtime probability blending
```

The live TypeScript service evaluates only the approved primary artifact directly. External benchmark artifacts remain offline evidence.

## Why v3 is a new major API

v2.x exposed a synthetic 12-month PD demonstration. v3 replaces that with a real-data model whose target is **final-loan-resolution default risk** and also introduces `creditScore` as a required input.

Keeping the old `/api/v2` namespace while silently changing those semantics would make downstream behavior ambiguous. v3 therefore uses `/api/v3` and returns `pdHorizon` on every score.

## Primary model feature contract

CRIX-MonoBoost 2.0 uses four harmonized features from the application-time-only LendingClub research cohort:

- debt-to-income ratio;
- requested-loan-to-income ratio;
- external bureau-style credit score (`fico_n`);
- employment tenure.

The wider API request contract intentionally remains richer. Context fields outside the primary champion can support policy, deterministic LGD, stress testing and future adapters without being falsely presented as trained champion features.

## Model / policy separation

The champion and challenger estimate risk. They do not make the policy decision by themselves.

`CRIX-Policy 3.0` consumes primary PD plus selected application and confidence signals and returns `APPROVE`, `REVIEW` or `DECLINE`.

This allows model retraining, risk-appetite changes and pricing changes to be governed independently.

## Dataset separation rule

The architecture explicitly rejects “more rows at any cost.” Datasets are not concatenated unless product, target horizon and feature semantics are compatible.

Current roles:

- LendingClub/Zenodo: primary personal-loan model training/calibration/OOT testing;
- Taiwan UCI: separate revolving-credit behavioral adaptation research model;
- German UCI: structural credit benchmarks;
- Home Credit / Give Me Some Credit / FICO HELOC / GSE mortgage sources: registry entries until access and product-specific governance permit real use.

This avoids training one blurred default concept across personal loans, cards, HELOCs and mortgages.

## Request lifecycle

1. Fastify assigns an opaque UUID request ID.
2. Global request/body/time limits apply.
3. Optional API-key authentication runs for `/api/v3/*`.
4. AJV validates the strict v3 request body, including `creditScore`.
5. The engine performs finite-number checks as a second trust boundary.
6. Runtime inputs are mapped into the artifact's primary feature vector.
7. The monotonic boosted champion produces a raw margin.
8. Real-data calibration converts the margin to final-resolution default probability.
9. The exported logistic challenger scores the same trained feature vector.
10. Disagreement and real-training-support OOD checks reduce confidence and emit flags.
11. LGD, EAD, expected loss, CRIX score, grade, pricing and local reasons are derived.
12. Independent policy returns approve/review/decline.
13. The response includes API, model, policy and PD-horizon information for traceability.

## External adaptation lifecycle

External research models never influence runtime policy by accident.

For Taiwan credit-card data, CRIX derives only behaviorally supportable features: utilization, payment behavior and normalized recent balance growth. Protected-attribute-adjacent fields (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`) are excluded, and missing personal-loan fields are not imputed across the whole cohort.

German credit datasets are evaluated only on their defensible structural overlap. The corrected South German representation is preferred to the legacy Statlog coding.

## Availability and readiness

`/health` is a liveness probe.

`/ready` reflects startup model-integrity validation. The server verifies artifact structure and performs a sentinel score before advertising readiness. With no external stateful runtime dependency, the primary model artifact is the principal readiness dependency.

## Security model

The public demo defaults to no API key so reviewers can exercise Swagger. A deployment can set `CRIX_API_KEY` to protect `/api/v3/*`; health/readiness/docs remain public.

Controls include body limits, rate limits, strict schemas, CORS allow-listing, Helmet, sensitive-header log redaction, constant-time API-key comparison, sanitized errors, batch bounds, finite-value guards and bounded model traversal.

The public demo must not receive real consumer credit data.

## Scaling path

A real high-volume/regulated deployment would add:

- gateway/IAM-backed authentication and tenant authorization;
- async bulk scoring behind a queue;
- immutable request/result audit storage;
- model registry and signed artifacts;
- point-in-time feature service;
- institution-specific model calibration and promotion workflows;
- telemetry for drift, calibration, decision rates and latency;
- canary/champion-challenger routing;
- fairness/proxy monitoring under legal/compliance governance;
- formal adverse-action reason governance;
- policy configuration/version approval;
- product-specific models rather than cross-product probability blending.

Those concerns are not simulated with unnecessary runtime infrastructure in this repository. v3 remains small enough to inspect and reproduce while making the production extension points explicit.
