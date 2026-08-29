# CRIX — Credit Risk Intelligence & Decisioning Lab

A production-style, self-contained **credit risk algorithm + intelligence dashboard**. CRIX estimates calibrated probability of default, converts risk into a transparent score, derives expected loss, challenges the champion model with an interpretable benchmark, explains individual decisions, stress-tests a portfolio, and exposes the diagnostics a real model-risk team would care about.

The original repository was a placeholder for a Tableau analysis. It is now an algorithm-first engineering project: the dashboard exists to interrogate the risk engine.

## What makes this different

This is deliberately **not** a CRUD dashboard wrapped around a random “AI score.” The system separates statistical risk estimation from lending policy and makes uncertainty visible.

- **Monotonic gradient-boosted champion** — shallow XGBoost trees with domain-informed directional constraints.
- **Calibrated 12-month PD** — a separate calibration split maps model margins to probabilities.
- **Transparent challenger** — logistic benchmark runs alongside every score.
- **Champion/challenger disagreement** — large deltas reduce confidence and can route a case to review.
- **PD / LGD / EAD / Expected Loss** — portfolio economics use standard credit-risk components.
- **300–850 odds-scaled score** — a readable score derived from PD, never used as a fake substitute for probability.
- **Local reason codes** — counterfactual sensitivity attribution shows which inputs materially moved PD.
- **Out-of-distribution guardrails** — unusual inputs are surfaced instead of silently trusted.
- **Stress engine** — mild and severe shocks re-score the entire portfolio through the same model + policy stack.
- **Model Lab** — ROC-AUC, KS, Brier score, log loss, calibration reliability and feature gain are exposed in the UI.
- **Portfolio Lab** — exposure, expected loss, approval mix and loss concentration are explorable without a database.

## Live architecture: zero infrastructure required

```text
Next.js / React dashboard
        ↓
TypeScript risk engine
        ↓
versioned XGBoost model artifact
   ↘ transparent challenger
   ↘ reason-code engine
   ↘ policy engine
   ↘ stress engine
```

The deployed app requires **no PostgreSQL, Redis, Python server, external ML API or secret**. Training happens offline; the compact trained artifact is committed with the application and evaluated directly by the TypeScript runtime. This makes the Vercel demo deterministic, cheap and easy to reproduce.

## Model development stack

The research/training side intentionally uses the mature Python credit/ML ecosystem:

- [XGBoost](https://xgboost.readthedocs.io/) for the monotonic boosted champion;
- [scikit-learn](https://scikit-learn.org/) for calibration and validation;
- [OptBinning](https://gnpalencia.org/optbinning/) for rigorous scorecard/binning experiments;
- [SHAP](https://shap.readthedocs.io/) for offline global/local model analysis.

The browser runtime does **not** need these packages.

## Current validation snapshot

The bundled CRIX-MonoBoost 1.0 artifact was trained on 50,000 deterministic synthetic credit-performance records and evaluated on a held-out test population.

| Diagnostic | Value |
|---|---:|
| ROC-AUC | 0.7334 |
| KS statistic | 0.3405 |
| Brier score | 0.1024 |
| Log loss | 0.3452 |
| Test observations | 10,000 |
| Test default rate | 13.09% |

The metrics are intentionally realistic rather than cosmetically inflated. In risk modelling, a believable calibrated model with observable limitations is more useful than a suspicious “99% accuracy” claim.

## Decision lifecycle

1. Validate and derive the application vector.
2. Evaluate every boosted tree and calculate the champion margin.
3. Apply the held-out probability calibrator to produce PD.
4. Score the same application with the transparent challenger.
5. Calculate disagreement and out-of-distribution penalties.
6. Estimate LGD and EAD, then `EL = PD × LGD × EAD`.
7. Convert PD into the display score using good:bad odds scaling.
8. Generate local sensitivity reason codes.
9. Apply the independent policy layer to choose `APPROVE`, `REVIEW` or `DECLINE`.
10. Return model version, confidence, economics and explanations as one auditable result.

## Run it

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`.

### Verification

```bash
npm run lint
npm test
npm run build
```

## Retrain the model

The Vercel deployment does not need Python. Python is only needed when changing the model artifact.

```bash
cd model
python -m venv .venv
# activate the environment for your OS
pip install -r requirements.txt
python train.py
```

`train.py` is seeded and regenerates `lib/risk/model-artifact.json`, including model trees, calibration coefficients, feature references, validation metrics, calibration buckets and feature importance.

## Dashboard surfaces

**Overview** — exposure, expected loss, average PD, approval rate, risk distribution and top expected-loss contributors.

**Underwrite** — enter an application vector and inspect PD, CRIX score, grade, expected loss, LGD, confidence, suggested APR, champion/challenger delta and reason codes.

**Portfolio** — inspect a deterministic synthetic book and filter by policy decision.

**Model Lab** — examine discrimination, calibration and feature importance rather than treating the model as a black box.

**Stress Lab** — apply macro-style borrower shocks across the entire portfolio and compare expected-loss/PD migration.

**Methodology** — documents how the model, economics, policy and governance layers remain separate.

## Engineering principles

- deterministic model versioning;
- no hidden external inference dependency;
- strict TypeScript domain contracts;
- policy separated from prediction;
- honest model diagnostics and limitations;
- test coverage for probability bounds, monotonic behaviour, stress direction and result completeness;
- CI runs lint, tests and a production build;
- no persistence of application data in the demo.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for system boundaries and [`MODEL_CARD.md`](./MODEL_CARD.md) for intended use, validation, limitations and production gates.

## Disclaimer

CRIX is an engineering and model-risk **demonstration**, not a production underwriting system. The bundled model is trained on synthetic data and must not be used to make real consumer credit decisions. A real deployment requires representative historical performance data, legal/compliance review, fairness testing, independent model validation, monitored production calibration and governed adverse-action reasons.

## License

MIT.
