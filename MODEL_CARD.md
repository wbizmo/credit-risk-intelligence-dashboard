# Model Card — CRIX-MonoBoost 1.0

## Intended use

Educational and portfolio demonstration of credit-risk modelling and decision-system engineering. It estimates a synthetic 12-month probability of default (PD) and supports portfolio analytics, model diagnostics and stress testing.

**It is not validated or authorized for real lending decisions.**

## Model family

Champion: shallow gradient-boosted decision trees trained with monotonic constraints. Monotonicity encodes directional credit-risk priors where appropriate, such as higher debt-to-income, utilization, delinquency burden and recent credit growth not decreasing predicted risk.

Challenger: compact logistic risk model with a smaller transparent feature set. Champion/challenger disagreement contributes to the confidence signal and can trigger manual review.

## Development population

50,000 deterministic synthetic records generated from correlated consumer-credit-style distributions. The latent default function includes nonlinear interactions and stochastic noise. The split is stratified into development, calibration and held-out test populations.

The synthetic generator exists to make the project reproducible and privacy-safe. It is not a substitute for representative observed performance data.

## Current held-out metrics

- ROC-AUC: 0.7334
- KS: 0.3405
- Brier score: 0.1024
- Log loss: 0.3452
- Test records: 10,000
- Test default rate: 13.09%

These values are intentionally reported as validation diagnostics rather than being optimized to look impressive. Model usefulness depends on discrimination, calibration, stability, decision economics and governance together.

## Probability calibration

The XGBoost raw margin is mapped to PD using a logistic (Platt) calibrator fit on data not used to fit the champion. The dashboard includes a reliability view of predicted versus observed default rates across probability buckets.

## Explainability

The deployed application produces model-agnostic local sensitivity reason codes. Each input feature is replaced with a reference-development value, the application is re-scored, and the PD delta is reported. This is explicitly labelled sensitivity attribution and is not misrepresented as exact SHAP.

The training stack includes SHAP for deeper offline analysis and OptBinning for scorecard experimentation.

## Score mapping

The displayed 300–850 CRIX score is derived from calibrated PD using odds scaling with 20 points to double good:bad odds around a 600-point / 20:1 reference. The score is presentation-friendly; PD remains the primary model output.

## Expected loss

`Expected Loss = PD × LGD × EAD`

LGD in the demo is a bounded severity function of leverage, income stability and liquidity. EAD is the requested funded amount. A real implementation would use segment-specific validated LGD/EAD models.

## Key limitations

- synthetic development data;
- no protected-class fairness conclusions can be drawn;
- no bureau-specific variables or macroeconomic time series are included;
- no reject inference or sample-selection correction;
- no vintage analysis or observed backtesting exists yet;
- stress scenarios are sensitivity shocks, not economic forecasts.

## Production gates before real use

Representative historical data, data lineage, feature definitions, legal/compliance review, fairness testing, independent validation, calibration by segment, reject-inference assessment, drift thresholds, model registry/approval workflow, production monitoring, adverse-action reason governance and periodic backtesting would all be required.
