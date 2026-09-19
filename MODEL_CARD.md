# Model Card — CRIX-MonoBoost 2.0

## System context

CRIX package release: **v3.2.0**  
API namespace: **`/api/v3`**  
Bundled primary model: **CRIX-MonoBoost 2.0.0**  
Live policy: **CRIX-Policy 3.0**

The package, API, model and policy versions are intentionally independent. v3.2 expands performance evidence, tail-risk research, model-governance controls, deployment hardening and supply-chain reproducibility without changing the live v3 probability target or silently promoting research models into runtime scoring.

## Intended use

CRIX-MonoBoost is an engineering/model-risk demonstration showing how a real-data credit-risk model can be trained with point-in-time discipline, calibrated on later originations, challenged, explained, sensitivity-tested and exposed through a governed API contract.

CRIX v3.2 includes offline lifetime PD, delinquency migration, empirical LGD/EAD, Gaussian/Student-t portfolio loss, EVT/GPD tail analysis, macro-conditioned stress, IFRS 9-style ECL, Basel-style/economic capital, constrained portfolio optimisation, distribution-shift governance, explanation-fidelity validation and tamper-evident lineage.

It is **not approved for real lending decisions, accounting policy or regulatory-capital use**.

## v3.2 engineering / governance evidence

- compiled champion benchmark: **~536,854 ops/s**, p50 **0.003746 ms** on the recorded Batch A CI host;
- compiled full assessment p50: **0.018628 ms** versus **0.025889 ms** for the reference path;
- sparse explanation work: **317** tree visits versus **384** full rescoring visits (**17.45%** reduction);
- shared tail attribution: measured **~2.80×** reference speedup at three quantiles and **~4.63×** at five quantiles before the final bucket-accumulation optimization;
- drift adversarial-validation AUC: **0.5593**, with aggregate and all four feature-level statuses passing the governed review bands;
- explanation-fidelity sample n=512: **97.59%** mean top-3 overlap, **95.33%** sign agreement and **96.94%** perturbation stability;
- privacy-bounded telemetry loopback overhead: **~5.3% mean** and **~1.0% p95** with the in-memory exporter;
- Batch D supply-chain certification: **0 known Node production vulnerabilities** and **0 known vulnerabilities** in the governed Python research lock.

No GPU speedup is claimed for v3.2.0 because the connected release infrastructure has no CUDA device. CuPy remains optional and NumPy remains the canonical deterministic evidence backend.

## Primary target

The live primary model estimates:

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
- Curated to variables available at granting/application time, avoiding post-underwriting leakage such as realized outcomes/pricing signals

## Primary feature contract

The v3 champion uses only features that can be mapped honestly from the primary cohort:

| Feature | Construction | Monotonic constraint |
|---|---|---:|
| `debtToIncome` | LendingClub `dti_n`, converted to a ratio where required | +1 |
| `loanToIncome` | `loan_amnt / revenue` | +1 |
| `creditScore` | LendingClub `fico_n` | -1 |
| `employmentYears` | normalized `emp_length` | -1 |

`annualIncome` and `loanAmount` are also required by the API because `loanToIncome` is computed at runtime and because policy/runtime loss logic uses them.

The broader API accepts utilization, delinquency, inquiry, trade-age, account-count, liquidity, payment-rate, income-stability and credit-growth context. Those fields may affect policy, deterministic borrower sensitivity, confidence context or deterministic runtime LGD logic, but CRIX does **not** represent them as champion training features when the primary dataset does not contain them.

The HTTP API intentionally does not request borrower names.

## Point-in-time feature provenance

CRIX treats application-time availability as a machine-checkable model-development invariant rather than documentation alone. Each champion feature has explicit provenance metadata containing source field(s), availability semantics, allowed target and whether it is outcome- or policy-derived.

Training fails closed if a required champion feature lacks provenance metadata or is marked outcome-/policy-derived. Timestamped research inputs must also satisfy `availableAt <= asOf`; validation errors report aggregate counts rather than raw borrower rows.

| Feature | Source field(s) | Availability |
|---|---|---|
| `debtToIncome` | `dti_n` | application-time |
| `loanToIncome` | `loan_amnt`, `revenue` | application-time |
| `creditScore` | `fico_n` | application-time |
| `employmentYears` | `emp_length` | application-time |

## Population conditioning and selection bias

The primary model is trained and evaluated on **historically granted LendingClub loans with observed outcomes**. Its model-development population is explicitly tagged `granted-loans-only`.

CRIX does **not** claim that this identifies `P(default | every applicant)`. Rejected-applicant outcomes are not present in the primary cohort, so CRIX does not fabricate them and does not silently apply reject-inference methods without defensible supporting data and assumptions.

Offline policy backtests are therefore **conditional selection analyses over observed granted loans**, not causal claims about historically rejected applicants.

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

These are model-development diagnostics on one historical lending population, not an “accuracy” score.

The governance pipeline also generates:

- deterministic percentile-bootstrap confidence intervals for AUC, KS, Brier and log loss;
- calibration intercept/slope and calibration bins with observation/event support;
- calibration-to-OOT PD population stability (PSI);
- fixed segment diagnostics for bureau score, DTI, loan-to-income and employment tenure;
- `insufficient-data` instead of fabricated metrics for weakly supported segments;
- observed-cohort policy backtests from cached predictions.

### v3.2 distribution-shift governance

CRIX v3.2 complements PSI with a separate aggregate-only distribution-shift report over the approved champion feature space. A bounded standardized logistic classifier performs fixed-seed out-of-fold adversarial validation between the chronological training cohort and the later calibration+OOT population. Feature-level evidence also includes Jensen-Shannon divergence, Wasserstein distance, p01/p99 support-breach rate, missingness-rate movement and quantile movement.

The versioned `crix-distribution-shift-v1` review policy uses adversarial-AUC review bands of **0.65 warning / 0.75 material-shift**, PSI bands of **0.10 / 0.25**, Jensen-Shannon bands of **0.05 / 0.10**, and support-breach bands of **5% / 10%**. These thresholds create review evidence only: they do **not** retrain, promote, retire or replace a model automatically.

Fixed credit-score, DTI, loan-to-income and employment-tenure segments are evaluated when both populations have sufficient support; weak segments return `insufficient-data`. The report contains counts and aggregate statistics only—never borrower rows, IDs or exact feature vectors. Adversarial validation measures how distinguishable two feature populations are; it does not by itself prove that the credit model is invalid or establish causality.

On the v3.2 governed real-data validation run, the bounded adversarial classifier produced **AUC 0.5593** for training versus later calibration+OOT observations. Under the versioned review policy, the aggregate shift status was **pass**, and all four champion features (`debtToIncome`, `loanToIncome`, `creditScore`, `employmentYears`) had feature-level status **pass**. This is evidence for that historical split only, not a guarantee of present-day or cross-population stability.

## Calibration

The monotonic XGBoost champion is trained on the train cohort. A separate Platt/logistic calibration layer is fitted only on 2016 originations, then evaluated on the later 2017 OOT cohort.

The committed artifact contains calibration parameters, support bounds and diagnostics. The deployed champion is not rebound during research validation.

## Challenger governance

The runtime challenger is a standardized logistic-regression model trained on the same real point-in-time feature contract as the champion.

Every live request surfaces `challengerPd`, champion/challenger disagreement and confidence effects.

The offline research stack includes a separate research-governance comparison using the **actual embedded challenger** on the same LendingClub calibration/OOT cohorts. Promotion evidence is deliberately broader than AUC and includes calibration, Brier/log loss, PSI/stability, bootstrap uncertainty and segment diagnostics. A challenger is not promoted merely because one discrimination metric is marginally higher.

## Out-of-distribution handling

OOD support is derived from the primary training cohort. The artifact stores 1st/99th percentile support for trained features. Inputs outside support are surfaced through `outOfDistribution`, reduce confidence and emit `OUT_OF_DISTRIBUTION`.

Startup readiness checks champion feature bounds/reference values and performs a sentinel score before advertising readiness.

## External adaptation / benchmark models

CRIX records product-specific real-data evidence without blending incompatible targets.

### UCI Default of Credit Card Clients — Taiwan

A separate research artifact uses behaviorally defensible derived revolving-credit features such as utilization, payment behavior and normalized balance growth. `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` are excluded from the behavioral model.

Its target is next-month credit-card default, so its probability is not blended with CRIX-MonoBoost and does not drive the personal-loan runtime policy.

### German credit benchmarks

Legacy UCI Statlog German Credit and corrected South German Credit remain structural benchmarks on limited overlapping loan-structure features. They are not evidence that the LendingClub champion transfers unchanged across products/geographies.

## Explainability

The API returns local model reason codes from bounded counterfactual sensitivity against real training-reference values for trained features. Policy triggers are returned separately as `policyReasons`.

CRIX v3.2 adds **offline explanation-fidelity validation** in `model/explanation_validation.py`. A fixed-seed governed OOT sample is explained independently with SHAP TreeExplainer and compared with the live-compatible local champion-sensitivity method using top-k feature overlap/disagreement, absolute-rank correlation, sign/direction agreement and stability under small valid perturbations. Evidence is also aggregated by credit-score band, DTI band, loan-to-income band, PD band and in-distribution/OOD status, with `insufficient-data` for weak segments.

SHAP remains outside the Fastify request path, so this validation adds no live latency or Python dependency. The generated `model/artifacts/crix-explanation-validation-v1.json` is aggregate-only and version-bound to CRIX-MonoBoost 2.0.0; report generation fails if champion identity metadata does not match.

On the governed v3.2 OOT sample (**n=512**, fixed seed 42), the offline comparison measured mean top-3 feature overlap **0.9759**, top-3 disagreement rate **7.23%**, mean absolute-rank correlation **0.9141**, mean sign agreement **0.9533**, and mean perturbation top-3 stability **0.9694**. These values support broad consistency between the two explanation methods on this sample while leaving the semantic/legal boundaries below unchanged.

SHAP values are **model-explanation evidence only**. They are not automatically legal adverse-action reasons, do not replace deterministic `policyReasons`, and do not establish ECOA/fair-lending compliance. Counterfactuals remain model-analysis aids, not promises of approval and not legally sufficient adverse-action reasons.

## Live expected loss

The public runtime continues to derive:

`Expected Loss = PD × LGD × EAD`

where runtime LGD is a deterministic engineering approximation and runtime EAD is the requested loan amount. These live components remain intentionally separate from the newer offline empirical LGD/EAD research artifacts.

Because the live primary PD target is final-resolution risk, live expected-loss outputs inherit that horizon limitation.

## Offline lifetime-risk stack

### Historical as-of / time machine

Research snapshots enforce point-in-time availability and deterministic snapshot identity. Historical evaluation cannot use features, preprocessing information, macro values or outcomes that were unavailable at the declared as-of date.

### Lifetime PD

A censoring-aware research model produces marginal/cumulative default term structures at 3, 6, 12, 24 and 36 months. These probabilities have their own explicit horizon semantics and do **not** replace `/api/v3` `pd`.

### Delinquency transitions

A separate migration research layer models delinquency-state transitions with cures/backward movement where observed instead of forcing one-way deterioration.

### Empirical LGD

Offline LGD research uses recovery severity/timing evidence and preserves economically meaningful conventions, including the possibility that collection costs can push economic LGD above 100% where explicitly defined. It does not silently clamp away such cases.

### Empirical EAD / CCF

Offline instalment EAD research is separated from revolving-credit / CCF research. Product structure and target definitions are not blended merely for convenience.

## Correlated portfolio loss research

`model/portfolio_risk.py` implements an offline one-factor latent-default Monte Carlo engine.

Methodological invariants include:

- supplied marginal PDs remain the marginals of the simulation;
- valid/bounded correlation inputs;
- deterministic fixed-seed replay;
- deterministic results across simulation chunk sizes;
- bounded-memory chunking rather than materializing a full scenario × obligor matrix;
- explicit withholding of 99.9% tail output when scenario precision is inadequate;
- expected-shortfall/tail contribution reconciliation where additivity is required.

Heavy simulation is deliberately excluded from Fastify request handling to avoid CPU-denial-of-service risk.

## Macro-conditioned stress research

The live `/api/v3/risk/stress` endpoint remains **deterministic borrower sensitivity** (`CRIX-Sensitivity 1.0`).

The offline research stack includes a separate macro research layer with point-in-time macro joins and an empirical unemployment/default relationship demonstration. The U.S. unemployment series used by the public research run is frozen in the repository with provenance so validation is not dependent on a live FRED request.

This is research evidence, not an institutionally validated macroeconometric stress model.

## IFRS 9-style ECL research

`model/ifrs9.py` implements research semantics for:

- Stage 1 / Stage 2 / Stage 3;
- SICR and days-past-due backstops;
- default and cure/probation handling;
- marginal PD conversion;
- scenario-weighted ECL;
- effective-interest-rate-style discounting.

The module is explicitly labelled **IFRS 9-style research**. Exact accounting policy, legal interpretation, data lineage and independent validation remain production prerequisites.

## Basel-style / economic-capital research

`model/capital.py` keeps expected loss separate from unexpected/tail capital and provides IRB-inspired/economic-capital research calculations plus tail-capital contribution reconciliation.

These outputs are not represented as Basel regulatory compliance for any institution or jurisdiction.

## Portfolio decision optimisation

`model/decisioning.py` supports bounded research optimisation under explicit constraints such as budget/expected-loss limits.

The optimiser:

- never silently relaxes constraints;
- reports infeasibility explicitly;
- uses direct/simple methods where the problem structure allows them;
- treats general binary constrained allocation as combinatorial rather than pretending it is always `O(n log n)`;
- is verified against exact exhaustive solutions on toy portfolios.

## Decision policy

The live statistical model does not directly approve or decline applications. `CRIX-Policy 3.0` is a separate deterministic layer with explicit PD, DTI, credit-score, delinquency, loan-to-income and confidence thresholds.

Model retraining and risk-appetite changes can therefore be governed independently.

## Borrower sensitivity testing

The `/api/v3/risk/stress` endpoint applies fixed borrower-level mild/severe shocks and reruns the ordinary live model/policy lifecycle.

It is **not** the offline macro research engine. Context fields absent from the champion cannot directly alter champion PD unless they change a trained derived feature.

## Runtime efficiency and bounded work

The live service constructs one immutable scoring context per base application: validated numeric input, loan-to-income, feature map and champion vector. Champion, challenger, OOD, runtime loss logic and policy reuse that context.

Intentional repeated champion scoring is bounded to the small explanation feature set. Tree traversal is capped, batch size remains limited to 50, and no portfolio Monte Carlo/optimisation workload is exposed on the Fastify event loop.

## v3.2 dataframe contracts

Offline model development now fails early on versioned dataframe contracts before fitting. The primary LendingClub pipeline separately validates source schema and the harmonized frame, then independently enforces the existing point-in-time provenance gate and chronological split contract. Required controls include binary target domain, parseable chronological dates, application-time feature-as-of semantics, positive income/loan amount, finite/bounded DTI, bureau-score and employment-tenure ranges, unique source-row identity and non-overlapping train/calibration/OOT cohorts.

External Taiwan and German research datasets use product-specific contracts rather than being forced into the LendingClub schema. Contract failures report only the contract/version, failing invariant, invalid count and safe aggregate summaries; raw borrower rows and IDs are not emitted.

## v3.2 tamper-evident lineage

`model/lineage.py` defines a versioned training-run manifest that hashes the approved model artifact, dependency environment and key governance evidence while carrying source-dataset identity/checksum, split counts, random seeds, feature/data-contract versions, Python version and the governed Git revision. Optional `previousManifest` links create an append-only hash chain across approved manifests.

This is a **tamper-evident hash chain with no distributed-ledger dependency**. A successful verification proves that the checked files still match the digests recorded in the approved manifest and that the recorded chain link has not changed. It does **not** prove that the training methodology was correct, that third-party numerical libraries will reproduce bit-for-bit on every platform, or that the model is suitable for regulated use. CI verification is read-only; publication of a new approved lineage manifest remains an explicit reviewed action.

## Security / privacy boundary

The public demo must not receive real consumer-credit data or raw private portfolios.

Research reports/artifacts are aggregate or public-source evidence. Runtime protections include strict request schemas, request/body/time bounds, key-aware and supplementary-IP rate limits, CORS allow-listing, Helmet, explicit `public-demo` / fail-closed `required` authentication, bounded current+next constant-time API-key rotation, explicit proxy-hop trust, privacy-bounded OpenTelemetry, log redaction, sanitized errors, finite-number guards and model-artifact integrity checks.

A regulated deployment would require stronger IAM, tenant isolation, immutable audit retention and formal data-governance controls.

## Known limitations

1. The primary outcome population is historical LendingClub US p2p lending, not an institution-specific production portfolio.
2. `dti_n` is narrower than a universal bureau DTI definition; its source semantics must not be generalized silently.
3. Employment tenure is banded/normalized.
4. The champion lacks several bureau-style features accepted by the API policy/context layer.
5. Final-resolution default risk is not a fixed-horizon 12-month PD.
6. The primary source contains granted loans only; reject inference is unsupported.
7. Historical performance does not guarantee present-day calibration or geographic transfer.
8. No fairness conclusion can be made without appropriate protected-attribute evaluation data and governance.
9. Live LGD/EAD remain engineering baselines even though separate empirical research artifacts now exist.
10. External UCI models have different products/targets and remain separate.
11. Runtime mild/severe scenarios are deterministic borrower sensitivities, not the empirical macro research engine.
12. Counterfactuals are not legally sufficient adverse-action explanations.
13. IFRS 9-style and Basel-style modules are research implementations, not compliance approvals.
14. Public macro evidence is intentionally parsimonious and must not be generalized as a full institutional macroeconometric model.
15. Portfolio simulation/optimisation results are research outputs whose assumptions, correlations, scenario counts and constraints must be governed explicitly.

## Production gates

Before real credit/accounting/capital use, at minimum:

1. train/recalibrate on representative institution-specific historical performance data with point-in-time feature correctness;
2. define default/target/observation/performance windows contractually;
3. independently validate discrimination, calibration, uncertainty, stability, reason-code behavior and challenger promotion;
4. test fairness, prohibited variables and proxy risk with legal/compliance review;
5. validate selection bias and reject-inference assumptions where relevant;
6. establish approved/signed model registry and promotion workflows;
7. validate lifetime PD, migration, LGD, EAD, CCF and macro models separately;
8. govern IFRS 9 staging/SICR/scenario/EIR policies through accounting approval;
9. implement jurisdiction-appropriate Basel/regulatory-capital rules if regulatory outputs are required;
10. govern policy/pricing/adverse-action processes independently from model development;
11. monitor drift, calibration, overrides, decision rates, portfolio concentration and realized losses;
12. add production IAM, authorization, audit retention and operational controls;
13. prohibit real consumer data from the public demo environment.

## Reproducibility

Primary training/governance evidence:

- `model/artifacts/crix-monoboost-v2.json`
- `model/artifacts/crix-distribution-shift-v1.json`
- `model/artifacts/crix-explanation-validation-v1.json`
- `model/TRAINING_REPORT.md`
- `model/governance.py`
- `model/data_contracts.py`
- `model/drift.py`
- `model/explanation_validation.py`
- `model/lineage.py`

Lifetime-risk research:

- `model/RISK_STACK_REPORT.md`
- `model/survival.py`
- `model/transitions.py`
- `model/lgd.py`
- `model/ead.py`
- `model/time_machine.py`

Advanced offline research:

- `model/portfolio_risk.py`
- `model/macro_stress.py`
- `model/ifrs9.py`
- `model/capital.py`
- `model/decisioning.py`
- `model/challenger_governance.py`
- `model/train_advanced_risk.py`

GitHub Actions runs the complete model/governance suite and full research validation before merge. Batch D certification reached **64 passing Node tests** and **91 passing Python governance/research tests** with one optional GPU-only skip, plus real-data champion retrain equivalence, historical risk-stack reproduction, advanced-risk evidence, registry/lineage checks and clean governed dependency audits. The final v3.2 release branch adds further repo-wide optimization/regression tests and must pass the same gates before release.
