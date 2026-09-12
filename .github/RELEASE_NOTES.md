# CRIX v3.1.0 — Advanced Credit-Risk Research Stack

CRIX v3.1 extends the real-data v3 foundation into a broader, auditable credit-risk research stack while deliberately keeping the live `/api/v3` decisioning contract stable.

The deployed runtime champion is still **CRIX-MonoBoost 2.0.0** and `pd` still means **final-loan-resolution default risk**. The new lifetime, migration, LGD/EAD, portfolio, macro, IFRS 9-style, capital and optimisation capabilities are offline research modules rather than silently replacing the production-facing API semantics.

## Point-in-time governance and reproducibility

- Machine-checkable feature provenance and leakage guards.
- Granted-loans-only population conditioning with explicit reject-inference limitations.
- Segmented calibration, bootstrap uncertainty, calibration intercept/slope and PSI stability evidence.
- Historical as-of snapshots / time-machine infrastructure.
- Model registry manifests with SHA-256 artifact integrity checks.
- Cold-cache isolated research retraining that hydrates verified source files rather than assuming Actions cache hits.

## Lifetime PD, transitions, LGD and EAD

- Censoring-aware lifetime PD term structures with 3/6/12/24/36-month research horizons.
- Delinquency-state transition research with cure/backward migration rather than one-way deterioration assumptions.
- Empirical LGD research using recovery severity/timing evidence.
- Empirical instalment EAD research plus a separate revolving-credit / CCF research path.
- Product and target-horizon separation is preserved; incompatible datasets are not pooled into a synthetic universal default model.

## Correlated portfolio loss simulation

v3.1 adds a deterministic one-factor correlated-default Monte Carlo research engine with bounded-memory chunking.

It produces:

- expected loss and unexpected loss;
- loss variance;
- VaR and expected shortfall at supported confidence levels;
- tail-risk contributions that reconcile to the portfolio tail metric;
- an independent-default comparison baseline;
- deterministic replay from a fixed seed.

The counter-based random construction and fixed row-wise loss reduction make seeded results invariant to chunk size. CRIX also withholds 99.9% tail output when the number of scenarios is too small to support a meaningful estimate.

## Empirical macro stress research

The previous `/api/v3/risk/stress` endpoint remains correctly labelled **deterministic borrower sensitivity**.

v3.1 adds a separate point-in-time macro research layer. The current empirical demonstration uses a frozen, provenance-tracked U.S. unemployment-rate series aligned to the historical LendingClub research vintages. The exact macro snapshot is committed with the code so CI/research reproduction does not depend on FRED network availability at run time.

This is a research macro-conditioned stress model, not a claim of institutionally validated macroeconometric stress testing.

## IFRS 9-style ECL research

Added an accounting-research engine for:

- Stage 1 / Stage 2 / Stage 3 classification;
- SICR and days-past-due backstops;
- default and cure/probation semantics;
- scenario-weighted expected credit loss;
- marginal PD conversion from cumulative term structures;
- EIR-style discounting.

The implementation is intentionally labelled **IFRS 9-style research**. It is not represented as accounting-policy approval or regulatory compliance.

## Basel-style and economic-capital research

Added research analytics that keep expected loss separate from unexpected/tail capital, including:

- IRB-inspired capital calculations;
- economic capital from portfolio tail loss;
- tail-capital contribution reconciliation;
- exposure / LGD / PD sensitivity invariants.

These are **Basel-style / economic-capital research** outputs, not a statement of regulatory capital compliance for any jurisdiction or institution.

## Challenger governance and portfolio optimisation

The actual embedded real-data logistic challenger is now evaluated against CRIX-MonoBoost on the same LendingClub calibration/OOT cohorts rather than being judged on AUC alone.

Governance evidence includes calibration, Brier score, log loss, PSI/stability, bootstrap uncertainty, segment results and deployment-performance context.

The new deterministic portfolio optimiser supports bounded research allocation under explicit constraints such as budget and expected-loss limits. It reports infeasibility instead of silently relaxing constraints, and exact toy portfolios are checked against exhaustive enumeration.

## Engineering discipline

- Full Python model/governance suite: **57 tests**.
- CI and model-validation gates are both required before merge.
- Live scoring hot-path work remains bounded and synchronous batch scoring remains capped.
- Heavy Monte Carlo / optimisation work stays offline instead of blocking the Fastify event loop.
- Fail-closed validation covers invalid probabilities, scenario weights, correlations, exposures and non-finite inputs.
- `/api/v3` API semantics and the deployed CRIX-MonoBoost 2.0.0 champion remain unchanged.

## Important model-risk note

CRIX is still a public engineering/model-risk research system, **not an approved production lending or regulatory-capital platform**. Real deployment requires representative institution-specific data, independently validated PD/LGD/EAD and macro models, fairness/proxy testing, exact accounting/regulatory policy interpretation, governed adverse-action reasons, audit retention, authentication/authorization, monitoring and formal model-risk/legal/compliance approval.
