<p align="center">
  <img src="assets/crix-mark.svg" alt="CRIX — Credit Risk Intelligence" width="560" />
</p>

# CRIX — Credit Risk Intelligence API

CRIX is a **backend-only, stateless credit-risk decisioning and model-risk API** backed by a real-data calibrated monotonic champion, plus an offline research stack for lifetime credit risk, migration, recovery severity, exposure, portfolio loss, macro stress, accounting ECL, capital and constrained decision optimisation.

The model is the product; HTTP + OpenAPI are the live interface. Heavy portfolio/research work stays offline and does not block the Fastify runtime.

> **Model semantics:** `pd` in `/api/v3` is **final-loan-resolution default risk**, not a 12-month regulatory PD. CRIX remains a public engineering/model-risk research system and is not approved for real consumer-credit decisions, accounting policy or regulatory-capital use.

## Live API

- Base URL: `https://crix-credit-risk-intelligence.onrender.com`
- Swagger UI: `https://crix-credit-risk-intelligence.onrender.com/docs`
- OpenAPI JSON: `https://crix-credit-risk-intelligence.onrender.com/openapi.json`
- Health: `https://crix-credit-risk-intelligence.onrender.com/health`
- Readiness: `https://crix-credit-risk-intelligence.onrender.com/ready`

## Release / model versions

- Package release: **3.2.0**
- API namespace: **`/api/v3`**
- Live primary model: **CRIX-MonoBoost 2.0.0**
- Live decision policy: **CRIX-Policy 3.0**

The package and API/model versions are intentionally independent. v3.2 strengthens offline model-risk governance, explanation validation, data contracts and tamper-evident lineage without changing the deployed v3 PD target or runtime champion.

## Primary champion — real LendingClub outcomes

CRIX-MonoBoost 2.0 is trained from the open **Lending Club loan dataset for granting models**, curated to application-time variables to avoid post-underwriting leakage.

- DOI: `10.5281/zenodo.11295916`
- License: CC-BY-4.0
- Source rows: **1,347,681**
- CRIX-harmonized rows: **1,269,389**
- Source period: **2007–2018**
- Immutable source MD5: `b019384d6bc65bf2a3e839362e4ff502`
- Train: **786,730** originations through 2015
- Calibration: **274,200** 2016 originations
- OOT test: **157,119** 2017 originations
- 2018 excluded from the primary reported evaluation to reduce final-status maturation bias

Champion feature contract:

| CRIX feature | Source | Constraint |
|---|---|---|
| `debtToIncome` | `dti_n`, normalized to a ratio | risk non-decreasing |
| `loanToIncome` | `loan_amnt / revenue` | risk non-decreasing |
| `creditScore` | `fico_n` | risk non-increasing |
| `employmentYears` | normalized `emp_length` | risk non-increasing |

The wider API accepts additional context for policy, confidence, deterministic runtime LGD and sensitivity analysis, but CRIX does **not** pretend those fields were primary champion training features when the public cohort does not contain them.

## Primary OOT diagnostics

| Diagnostic | 2017 OOT value |
|---|---:|
| ROC-AUC | **0.6594** |
| KS statistic | **0.2300** |
| Brier score | **0.1645** |
| Log loss | **0.5043** |
| OOT default rate | **22.42%** |
| Logistic challenger ROC-AUC | **0.6578** |

These are model-development diagnostics, not a cosmetic “accuracy” score.

## v3.2 research stack

The research boundary is intentionally separate from the live `/api/v3` score contract.

### Point-in-time governance

- machine-checkable feature provenance and leakage guards;
- granted-loans-only population conditioning and explicit reject-inference limitations;
- historical as-of snapshots / time-machine infrastructure;
- segmented calibration and support diagnostics;
- deterministic bootstrap uncertainty intervals;
- calibration intercept/slope and PSI stability evidence;
- observed-cohort policy backtests;
- SHA-256 model-artifact manifests and registry integrity checks.

### Lifetime PD and migration

CRIX includes censoring-aware lifetime-PD research with cumulative/marginal term structures at **3, 6, 12, 24 and 36 months**, plus delinquency-state transition research that permits cures/backward migration instead of assuming deterioration is one-way.

### LGD and EAD

Offline research adds empirical recovery/severity LGD and instalment EAD modelling, plus a separate revolving-credit/CCF path where the source supports that product structure.

The live API still preserves its existing deterministic LGD and requested-amount EAD semantics; research artifacts are not silently substituted into runtime scoring.

### Correlated portfolio Monte Carlo and tail research

`model/portfolio_risk.py` keeps the canonical deterministic one-factor Gaussian simulation, while v3.2 research adds Student-t dependence, bounded low-rank multi-factor loadings, a single-replay tail-attribution algorithm, and an optional lazy CuPy backend for offline GPU experiments.

Research outputs include:

- expected loss and unexpected loss;
- VaR and expected shortfall at supported confidence levels;
- independent-default baseline comparison;
- bounded-memory tail-risk contributions;
- explicit dependency-model and backend metadata;
- explicit 99.9% precision withholding when scenario support is too small.

NumPy remains the canonical reproducible path and the legacy Gaussian seeded digest is preserved. The live TypeScript API gains no Python/CUDA dependency.

`model/evt.py` separately implements research-only Peaks-Over-Threshold GPD tail extrapolation with threshold-stability diagnostics, bounded optional bootstrap intervals, explicit insufficient/unstable states, and correct withholding of expected shortfall when the fitted GPD mean is not finite. EVT never replaces empirical Monte Carlo VaR/ES.

See [`docs/PORTFOLIO_TAIL_RISK.md`](./docs/PORTFOLIO_TAIL_RISK.md) for the dependency, GPU, attribution and EVT research contract.

### Empirical macro stress

The public `/api/v3/risk/stress` endpoint remains correctly identified as **deterministic borrower sensitivity**.

A separate offline macro research layer uses point-in-time joins and a frozen, provenance-tracked U.S. unemployment-rate snapshot aligned to historical LendingClub vintages. The snapshot is committed so CI does not depend on live FRED availability.

### IFRS 9-style ECL research

`model/ifrs9.py` implements research semantics for:

- Stage 1 / Stage 2 / Stage 3;
- SICR and DPD backstops;
- default and cure/probation handling;
- marginal PD conversion from cumulative term structures;
- scenario weighting;
- EIR-style discounting.

This is explicitly **IFRS 9-style research**, not accounting-policy approval.

### Basel-style / economic-capital research

`model/capital.py` keeps expected loss separate from unexpected/tail capital and implements IRB-inspired/economic-capital research metrics plus tail-capital contribution reconciliation.

These outputs are deliberately labelled **Basel-style / economic-capital research**, not regulatory compliance.

### Challenger governance and portfolio optimisation

The actual embedded real-data logistic challenger is evaluated against the champion on the same LendingClub calibration/OOT cohorts using more than AUC: calibration, Brier/log loss, PSI/stability, bootstrap uncertainty and segment evidence are included.

`model/decisioning.py` adds deterministic constrained portfolio optimisation. It fails explicitly on infeasible constraint sets, never silently relaxes risk limits, and exact toy portfolios are verified against exhaustive enumeration.

## External real-world evidence without dataset soup

CRIX does not concatenate incompatible products/targets just to increase row count.

- **UCI Taiwan credit-card clients** — separate behavioral/revolving-credit research; protected-attribute-adjacent `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` are excluded from the behavioral model.
- **UCI Statlog German Credit** — structural benchmark only.
- **Corrected UCI South German Credit** — preferred corrected companion benchmark.
- Home Credit, Give Me Some Credit, FICO HELOC, Freddie Mac and Fannie Mae remain registered/gated or product-specific sources rather than being falsely represented as primary training data.

See [`model/EXTERNAL_BENCHMARKS.md`](./model/EXTERNAL_BENCHMARKS.md), [`model/TRAINING_REPORT.md`](./model/TRAINING_REPORT.md) and [`model/RISK_STACK_REPORT.md`](./model/RISK_STACK_REPORT.md).

## What the live engine returns

Each `/api/v3/risk/score` result includes:

- calibrated final-resolution default probability and explicit `pdHorizon`;
- real-data logistic challenger probability and disagreement;
- confidence and model-risk flags;
- OOD signals derived from real training support;
- deterministic runtime LGD, EAD and `Expected Loss = PD × LGD × EAD`;
- CRIX score and risk grade;
- local champion reason codes and bounded lower-risk counterfactuals;
- independent policy reasons and `APPROVE`, `REVIEW` or `DECLINE` result;
- model and policy versions.

The statistical model estimates risk. The policy layer decides what to do with that risk.

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | API discovery document |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Model-integrity/readiness probe |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/openapi.json` | OpenAPI document |
| `GET` | `/api/v3/model` | Model target, provenance, diagnostics and policy metadata |
| `POST` | `/api/v3/risk/score` | Score one application |
| `POST` | `/api/v3/risk/stress` | Deterministic borrower sensitivity |
| `POST` | `/api/v3/risk/batch` | Score up to 50 applications synchronously |

## Quick start

Requirements: **Node.js 22+** and npm.

```bash
git clone https://github.com/wbizmo/credit-risk-intelligence-dashboard.git
cd credit-risk-intelligence-dashboard
npm install
npm run verify
npm run dev
```

Useful local URLs:

```text
http://localhost:3000/health
http://localhost:3000/ready
http://localhost:3000/docs
http://localhost:3000/openapi.json
```

### Score an application

```bash
curl -X POST http://localhost:3000/api/v3/risk/score \
  -H 'content-type: application/json' \
  -d '{
    "applicationId": "demo-001",
    "annualIncome": 85000,
    "debtToIncome": 0.28,
    "creditScore": 720,
    "creditUtilization": 0.30,
    "delinquencies24m": 0,
    "inquiries6m": 1,
    "oldestTradeMonths": 96,
    "openAccounts": 7,
    "loanAmount": 24000,
    "termMonths": 36,
    "employmentYears": 5,
    "cashBufferMonths": 3,
    "onTimePaymentRate": 0.98,
    "incomeStability": 0.82,
    "recentCreditGrowth": 0.08
  }'
```

If `CRIX_API_KEY` is configured, add `-H 'x-api-key: your-key'`.

## Offline research / model development

Python is offline/build-time only. The live API does **not** run Python or call an external inference service.

```bash
python -m venv model/.venv
# activate the environment for your OS
pip install -r model/requirements.txt
python model/train.py
python model/train_risk_stack.py
python model/train_advanced_risk.py
python model/challenger_governance.py
```

The model-validation workflow rebuilds the primary model, validates immutable research artifacts in an isolated workspace, runs the complete governance suite, and generates integrated advanced-risk evidence before merge.

## Security and resilience

- strict JSON schemas and `additionalProperties: false`;
- bounded numeric domains and batch size;
- 64 KiB request-body ceiling;
- per-request opaque request IDs;
- Helmet security headers;
- global and endpoint-specific rate limits;
- optional API-key authentication using constant-time comparison;
- CORS disabled unless allow-listed;
- credential-header log redaction and sanitized errors;
- finite-number checks inside the engine;
- bounded tree traversal and explanation work;
- artifact-integrity/readiness checks;
- no database or borrower persistence;
- no borrower-name field;
- heavy Monte Carlo/optimisation excluded from the public Fastify request path;
- aggregate research evidence only; no raw consumer portfolios committed.

See [`SECURITY.md`](./SECURITY.md), [`ARCHITECTURE.md`](./ARCHITECTURE.md), and [`MODEL_CARD.md`](./MODEL_CARD.md).

## Runtime architecture

```text
HTTP client
   ↓
Fastify 5
   ├─ strict validation / limits / optional API key
   ├─ health + readiness
   ├─ Swagger / OpenAPI
   └─ /api/v3 risk routes
          ↓
CRIX TypeScript risk engine
   ├─ real-data calibrated monotonic champion
   ├─ real-data logistic challenger
   ├─ OOD + confidence / disagreement
   ├─ deterministic runtime LGD / EAD / expected loss
   ├─ bounded explanations + counterfactuals
   ├─ independent policy engine
   └─ deterministic borrower sensitivity
          ↓
versioned JSON model artifact

Offline research boundary
   ├─ point-in-time / historical snapshots
   ├─ lifetime PD + transition migration
   ├─ empirical LGD / EAD / CCF
   ├─ correlated portfolio Monte Carlo
   ├─ macro-conditioned stress research
   ├─ IFRS 9-style ECL research
   ├─ Basel-style / economic-capital research
   └─ challenger governance + portfolio optimisation
```

No PostgreSQL, Redis, Python inference service or paid AI API is required at runtime.

## Disclaimer

Real historical training and a deeper research stack do **not** make CRIX a production-approved lending, accounting or capital system. Production use requires representative institution-specific data, exact target/window governance, independent validation, fairness/proxy testing, legal/compliance review, validated PD/LGD/EAD and macro models, governed adverse-action reasons, accounting/regulatory policy approval, monitoring, auditable authentication/authorization and formal model-risk governance. The public demo must not receive real consumer credit data.

## License

MIT. Third-party datasets retain their own licenses and citation requirements; dataset provenance is recorded in model artifacts and reports.
