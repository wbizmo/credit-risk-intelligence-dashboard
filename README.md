# CRIX — Credit Risk Intelligence API

CRIX is a **backend-only, stateless credit-risk decisioning API**. It estimates calibrated 12-month probability of default (PD), challenges that estimate with an interpretable benchmark, derives LGD/EAD/expected loss, generates local reason codes, surfaces model uncertainty, runs deterministic stress scenarios, and applies a separate lending-policy layer.

There is no dashboard in v2.5.0. The algorithm is the product; HTTP + OpenAPI are the interface.

## Live API

- Base URL: `https://crix-credit-risk-intelligence.onrender.com`
- Swagger UI: `https://crix-credit-risk-intelligence.onrender.com/docs`
- OpenAPI JSON: `https://crix-credit-risk-intelligence.onrender.com/openapi.json`
- Health: `https://crix-credit-risk-intelligence.onrender.com/health`
- Readiness: `https://crix-credit-risk-intelligence.onrender.com/ready`

## What the engine returns

Each score includes:

- calibrated 12-month **PD**;
- transparent challenger PD;
- champion/challenger disagreement;
- confidence score and model-risk flags;
- out-of-distribution signals;
- **LGD**, **EAD**, and `Expected Loss = PD × LGD × EAD`;
- 300–850 odds-scaled CRIX score and risk grade;
- local counterfactual reason codes;
- indicative risk-based APR;
- independent `APPROVE`, `REVIEW`, or `DECLINE` policy result;
- model and policy versions.

The statistical model estimates risk. The policy layer decides what to do with that risk. Those concerns are intentionally separate.

## API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | API discovery document |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Model-integrity/readiness probe |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/openapi.json` | OpenAPI document |
| `GET` | `/api/v2/model` | Model, calibration, diagnostics, and policy metadata |
| `POST` | `/api/v2/risk/score` | Score one application |
| `POST` | `/api/v2/risk/stress` | Re-score under mild/severe deterministic stress |
| `POST` | `/api/v2/risk/batch` | Score up to 50 applications synchronously |

The package/release version is **2.5.0**. The HTTP namespace stays at `/api/v2` so compatible 2.x changes do not churn client URLs.

## Quick start

Requirements: **Node.js 22+** and npm.

```bash
git clone https://github.com/wbizmo/credit-risk-intelligence-dashboard.git
cd credit-risk-intelligence-dashboard
npm install
npm run dev
```

The API starts on `http://localhost:3000` by default.

Useful local URLs:

```text
http://localhost:3000/health
http://localhost:3000/ready
http://localhost:3000/docs
http://localhost:3000/openapi.json
```

### Production-style local run

```bash
npm install
npm run build
npm start
```

### Configuration

No environment variables are required for the public/demo configuration. See `.env.example` for supported options.

- `PORT` — HTTP port, default `3000`.
- `HOST` — bind address, default `0.0.0.0`.
- `LOG_LEVEL` — Fastify/Pino log level, default `info`.
- `RATE_LIMIT_MAX` — default global requests/minute, default `120`.
- `CRIX_API_KEY` — optional. When set, `/api/v2/*` requires `x-api-key`.
- `CORS_ORIGIN` — optional comma-separated browser origins. CORS is disabled when empty.

## Test it

Run the same verification gates used in CI:

```bash
npm run verify
```

Or individually:

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

### Health check

```bash
curl http://localhost:3000/health
curl http://localhost:3000/ready
```

### Score an application

```bash
curl -X POST http://localhost:3000/api/v2/risk/score \
  -H 'content-type: application/json' \
  -d '{
    "applicationId": "demo-001",
    "annualIncome": 85000,
    "debtToIncome": 0.28,
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

If `CRIX_API_KEY` is configured, add:

```bash
-H 'x-api-key: your-key'
```

### Stress test

```bash
curl -X POST http://localhost:3000/api/v2/risk/stress \
  -H 'content-type: application/json' \
  -d '{
    "severity": "severe",
    "application": {
      "annualIncome": 85000,
      "debtToIncome": 0.28,
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
    }
  }'
```

For the complete request/response contract, use Swagger at `/docs` rather than copying examples from the README into client code.

## Model development

The live API does **not** run Python. Python is used only to train/export the model artifact.

```bash
cd model
python -m venv .venv
# activate the environment for your OS
pip install -r requirements.txt
python train.py
```

Training uses the mature Python ML ecosystem (XGBoost + scikit-learn, with OptBinning and SHAP available for deeper offline analysis). The resulting compact model artifact is committed under `model/artifacts/` and executed directly by the TypeScript service.

Current bundled held-out diagnostics:

| Diagnostic | Value |
|---|---:|
| ROC-AUC | 0.7334 |
| KS statistic | 0.3405 |
| Brier score | 0.1024 |
| Log loss | 0.3452 |
| Test observations | 10,000 |
| Test default rate | 13.09% |

These metrics are deliberately presented as model diagnostics, not as a cosmetic “accuracy” score.

## Security and resilience in v2.5

- strict request JSON schemas and `additionalProperties: false`;
- bounded numeric domains and synchronous batch size;
- 64 KiB request-body ceiling;
- per-request UUIDs returned as `x-request-id`;
- Helmet security headers;
- global and endpoint-specific rate limits;
- optional API-key authentication using constant-time comparison;
- CORS disabled unless explicitly allow-listed;
- redaction of credential headers from logs;
- sanitized 4xx/5xx responses with no stack traces;
- internal finite-number guards even when the engine is called outside HTTP validation;
- bounded tree traversal and model-artifact integrity checks;
- readiness fails when the bundled model cannot pass a sentinel score;
- no database and no persistence of submitted applications;
- no borrower-name field because identity is not required by the model.

See [`SECURITY.md`](./SECURITY.md), [`ARCHITECTURE.md`](./ARCHITECTURE.md), and [`MODEL_CARD.md`](./MODEL_CARD.md).

## Render free-tier caveat

This demo is intentionally deployed as a Render **Free** web service. Render currently spins down a free web service after **15 minutes without inbound traffic**. The next request wakes it and spin-up can take **about one minute**. If the first request after an idle period is slow, allow the service to wake and retry. This is a hosting-tier behaviour, not model latency.

Render also documents that free instances are intended for testing/hobby/demo workloads rather than production. See: https://render.com/docs/free

The API itself is stateless, so sleeping/restarting does not lose application data because CRIX stores no application data in the first place.

## Architecture

```text
HTTP client
   ↓
Fastify 5
   ├─ validation / limits / optional API key
   ├─ health + readiness
   ├─ Swagger / OpenAPI
   └─ /api/v2 risk routes
          ↓
CRIX TypeScript risk engine
   ├─ calibrated monotonic boosted champion
   ├─ transparent challenger
   ├─ OOD + confidence / disagreement checks
   ├─ LGD / EAD / expected loss
   ├─ local reason codes
   ├─ independent policy engine
   └─ deterministic stress engine
          ↓
versioned model artifact
```

No PostgreSQL, Redis, external inference service, or paid AI API is required.

## Disclaimer

CRIX is an engineering/model-risk demonstration. The bundled model is trained on synthetic data and **must not be used to make real consumer credit decisions**. Production use requires representative historical performance data, independent validation, fairness testing, legal/compliance review, governed adverse-action reasons, monitoring, recalibration, authentication/authorization appropriate to the deployment, and formal model-risk governance.

## License

MIT.
