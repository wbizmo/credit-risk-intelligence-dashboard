# CRIX external real-world benchmarks

Generated 2026-09-08. These are **dataset-specific adaptation/benchmark models**, not evidence that the LendingClub champion transfers unchanged across products or geographies.

| Dataset | Product | Rows | Default rate | CV ROC-AUC | Brier | Log loss |
|---|---|---:|---:|---:|---:|---:|
| uci-taiwan-credit-card-default | revolving-credit-card | 30,000 | 22.12% | 0.7606 | 0.1416 | 0.4464 |
| uci-statlog-german-credit | consumer-instalment-credit | 1,000 | 30.00% | 0.6672 | 0.1981 | 0.5792 |
| uci-south-german-credit | consumer-instalment-credit | 1,000 | 30.00% | 0.6718 | 0.1955 | 0.5748 |

## Taiwan credit-card behavioral adaptation

`CRIX-Behavior-TW 1.0.0` is trained separately on four behaviorally defensible features derived from the 30,000-client UCI Taiwan bank dataset:

- six-month mean revolving utilization;
- six-month on-time-payment rate;
- count of delayed-payment months in the six-month window;
- six-month bill-balance growth normalized by credit limit.

`SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` are explicitly excluded. Missing personal-loan/bureau fields are **not imputed**. The model targets **next-month credit-card default**, so it is retained as a research adaptation artifact and is not blended into CRIX's personal-loan PD or policy.

## German structural benchmarks

The German datasets are used only for loan-structure benchmarking because their overlap with CRIX is limited to duration, amount, employment band, installment burden and same-bank credit count. The legacy Statlog version is retained for comparability, but the corrected South German Credit representation is preferred because UCI documents coding issues in the old representation.

## Governance rule

CRIX never concatenates these cohorts with LendingClub merely to increase row count. Product definition, target horizon, geography and feature semantics stay explicit in every artifact.
