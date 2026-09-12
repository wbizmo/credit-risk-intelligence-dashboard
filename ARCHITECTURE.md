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
  |-- request-local immutable scoring context
  |-- CRIX-MonoBoost 2.0 primary PD
  |-- exported real-data logistic challenger
  |-- disagreement/confidence
  |-- real-training-support OOD detection
  |-- LGD/EAD/expected loss
  |-- score/grade
  |-- local model reason codes
  |-- separate policy reason codes
  |-- bounded training-support counterfactuals
  |-- independent policy decision
  |-- versioned deterministic borrower sensitivity
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
  |      -> point-in-time feature provenance gate
  |      -> chronological train / calibration / OOT split
  |      -> monotonic XGBoost champion
  |      -> Platt calibration
  |      -> logistic challenger
  |      -> segmented calibration / uncertainty / stability diagnostics
  |      -> observed-cohort policy backtests
  |      -> compact runtime JSON artifact + governance evidence
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

The offline pipeline now keeps explicit provenance for every champion feature, including source fields, application-time availability and outcome-derived status. Training fails closed if a required champion feature lacks provenance or is marked outcome-derived.

The wider API request contract intentionally remains richer. Context fields outside the primary champion can support policy, deterministic LGD, borrower sensitivity testing and future adapters without being falsely presented as trained champion features.

## Point-in-time and population governance

Chronological splitting alone is not treated as a complete leakage guarantee. The model-development layer has an explicit point-in-time provenance gate so future models can prove when each feature was knowable. Timestamped feature values must be available on or before their declared `asOf` date; errors report aggregate counts rather than raw borrower rows.

The primary LendingClub cohort is also explicitly tagged as `granted-loans-only`. CRIX therefore estimates risk and performs policy backtests on the historically granted population with observed outcomes. It does not fabricate outcomes for rejected applicants and does not present accepted-only backtests as unbiased applicant-population counterfactuals.

## Model validation and stability

The training pipeline evaluates the primary OOT cohort with more than one global metric. Governance evidence includes:

- ROC-AUC, KS, Brier score and log loss;
- deterministic bootstrap confidence intervals over cached OOT predictions;
- calibration intercept/slope and calibration bins with observation/event support;
- calibration-to-OOT PD population-stability measurement;
- fixed, pre-declared borrower/loan segments such as bureau-score, DTI, loan-to-income and employment-tenure bands;
- explicit `insufficient-data` status rather than fabricated metrics for weakly supported segments.

A read-only pull-request model-validation workflow rebuilds the real-data model and verifies the generated artifact carries the governance contract. It cannot publish model artifacts or mutate the repository.

## Model / policy separation

The champion and challenger estimate risk. They do not make the policy decision by themselves.

`CRIX-Policy 3.0` consumes primary PD plus selected application and confidence signals and returns `APPROVE`, `REVIEW` or `DECLINE`.

Model explanations and policy triggers are deliberately separated in the runtime response. `reasons` are local champion-model sensitivity reasons; `policyReasons` identify the deterministic rule(s) that caused `REVIEW` or `DECLINE`. This avoids attributing a policy threshold to the PD model.

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
6. One request-local scoring context derives loan-to-income and the champion feature vector once.
7. The monotonic boosted champion produces a raw margin from that context.
8. Real-data calibration converts the margin to final-resolution default probability.
9. The exported logistic challenger scores the same trained feature vector.
10. Disagreement and real-training-support OOD checks reduce confidence and emit flags.
11. LGD, EAD, expected loss, CRIX score, grade and pricing are derived.
12. The four champion features are perturbed once against bounded training-reference values; the same evaluations feed model reason codes and lower-risk counterfactuals.
13. Independent policy evaluation returns approve/review/decline plus separate policy reasons.
14. The response includes API, model, policy and PD-horizon information for traceability.

The base feature map/vector is not repeatedly reconstructed for champion, challenger and OOD evaluation. Counterfactual rescoring remains intentionally bounded to the small trained champion feature set.

## Counterfactual semantics

Counterfactuals are model-analysis aids, not promises of approval and not legally sufficient adverse-action advice. Each returned counterfactual:

- changes only a feature the champion actually uses;
- targets the feature's training-reference value bounded by committed p01/p99 support;
- is returned only when the exact deployed model produces a lower PD;
- includes before/after PD and training-support bounds;
- never changes the original request object.

The existing `reasons` field remains a local model-sensitivity explanation. Legal adverse-action governance remains a production gate outside the public research demo.

## Borrower sensitivity versus macro stress

`/api/v3/risk/stress` is retained for compatibility, but its method is explicitly identified as `deterministic-borrower-sensitivity` with version `CRIX-Sensitivity 1.0`. `mild` and `severe` apply fixed borrower-level shocks and rerun the ordinary model/policy lifecycle.

This endpoint is **not** represented as an empirically estimated macroeconomic stress model. Context fields that are not champion features cannot directly change champion PD unless a shocked value changes a trained derived feature such as loan-to-income. A future macro stress engine must use separate, versioned statistical semantics rather than silently reinterpreting these scenarios.

## Observed-cohort policy backtesting

Offline governance can evaluate simple PD-threshold policies using cached OOT predictions and observed exposures. The artifact/report records selection rate, observed default rate, selected exposure and predicted/observed default exposure for the historically granted cohort.

These are conditional historical analyses. Rejected-applicant outcomes are explicitly marked `unsupported` until a defensible data source and reject-inference methodology exist.

## External adaptation lifecycle

External research models never influence runtime policy by accident.

For Taiwan credit-card data, CRIX derives only behaviorally supportable features: utilization, payment behavior and normalized recent balance growth. Protected-attribute-adjacent fields (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`) are excluded, and missing personal-loan fields are not imputed across the whole cohort.

German credit datasets are evaluated only on their defensible structural overlap. The corrected South German representation is preferred to the legacy Statlog coding.

## Availability and readiness

`/health` is a liveness probe.

`/ready` reflects startup model-integrity validation. The server verifies artifact structure, champion/challenger dimensions, calibration finiteness, training bounds/reference consistency and performs a sentinel score before advertising readiness. With no external stateful runtime dependency, the primary model artifact is the principal readiness dependency.

## Security model

The public demo defaults to no API key so reviewers can exercise Swagger. A deployment can set `CRIX_API_KEY` to protect `/api/v3/*`; health/readiness/docs remain public.

Controls include body limits, rate limits, strict schemas, CORS allow-listing, Helmet, sensitive-header log redaction, constant-time API-key comparison, sanitized errors, batch bounds, finite-value guards, bounded model traversal and bounded explanation work.

Model-governance reports contain aggregate metrics only. The public demo must not receive real consumer credit data.

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
