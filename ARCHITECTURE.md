# CRIX Architecture

## Design goal

CRIX is an algorithm-first credit-risk laboratory. The public deployment must remain runnable without a database, queue, Python service, cloud account, or paid AI API while the repository still demonstrates a credible model-development lifecycle.

## Runtime

```text
Browser
  └─ Next.js / React dashboard
       ├─ TypeScript XGBoost tree evaluator
       │    └─ versioned model-artifact.json
       ├─ transparent logistic challenger
       ├─ underwriting policy engine
       ├─ local sensitivity explanations
       ├─ PD / LGD / EAD / EL calculations
       └─ deterministic portfolio + stress engine
```

Inference is deterministic and local. The model artifact contains only tree structure, a calibration transform, development references and validation diagnostics. No borrower data is persisted.

## Offline model development

```text
Synthetic or licensed credit-performance data
  → validation + feature engineering
  → train / calibration / test split
  → monotonic XGBoost champion
  → probability calibration
  → interpretable scorecard challenger
  → discrimination + calibration diagnostics
  → artifact export
  → browser parity tests
```

The Python training environment is deliberately not part of the Vercel runtime.

## Separation of concerns

1. **Risk model** estimates 12-month probability of default.
2. **Severity model** estimates loss given default for the demonstration exposure.
3. **Exposure layer** determines exposure at default.
4. **Economics layer** computes expected loss and indicative risk pricing.
5. **Policy layer** maps model outputs plus guardrails to APPROVE / REVIEW / DECLINE.
6. **Governance layer** exposes challenger disagreement, OOD warnings, diagnostics and reason codes.

The decision policy never changes the model's probability. This allows risk appetite to evolve independently of model retraining.

## Why no database

A database would add no value to the portfolio demonstration and would make a public deployment harder to reproduce. The dashboard generates a deterministic synthetic portfolio in memory and never claims persistence. A production lender could attach an event store and feature platform behind the same `ApplicationInput → RiskResult` contract later.

## Security and privacy posture

- no credentials or secrets are required;
- no PII is sent to a backend;
- no telemetry is required for the scoring engine;
- no model-training data is bundled except the deterministic synthetic generator;
- no external model endpoint can silently change behaviour.

## Extension points

The contract is intentionally ready for later replacement of the local artifact with an API, ONNX Runtime, a feature store, model registry, adverse-action service, streaming portfolio monitor or persistent underwriting ledger without rewriting the dashboard domain model.
