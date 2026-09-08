# CRIX real-world training report

Generated from the immutable model artifact. This report is model-development evidence, not a production validation approval.

## Source

- Dataset: Lending Club loan dataset for granting models, version 0.1
- DOI: 10.5281/zenodo.11295916
- Source checksum (MD5): `b019384d6bc65bf2a3e839362e4ff502`
- Upstream period: 2007–2018
- Rows in source: 1,347,681
- Rows surviving CRIX harmonization: 1,269,389
- Target: final resolved loan status, charged-off/default = 1 and fully-paid = 0

## Temporal design

2018 is intentionally excluded from primary reported evaluation to reduce final-status maturation bias. No random train/test shuffle is used.

| Cohort | Origination window | Rows | Default rate |
|---|---|---:|---:|
| Train | 2007-06-01 → 2015-12-01 | 786,730 | 18.17% |
| Calibration | 2016-01-01 → 2016-12-01 | 274,200 | 22.71% |
| OOT test | 2017-01-01 → 2017-12-01 | 157,119 | 22.42% |

## Out-of-time results

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6594 |
| KS | 0.2300 |
| Brier score | 0.1645 |
| Log loss | 0.5043 |
| OOT observations | 157,119 |
| OOT default rate | 22.42% |
| Logistic challenger ROC-AUC | 0.6578 |

## Canonical CRIX features used by the champion

- `debtToIncome` ← LendingClub `dti_n` (normalized from percentage points when detected)
- `loanToIncome` ← `loan_amnt / revenue`
- `creditScore` ← LendingClub `fico_n`
- `employmentYears` ← normalized `emp_length`

The remaining CRIX application context is not falsely presented as part of the trained champion when it is absent from this public cohort.

## Dataset separation

Home Credit, Give Me Some Credit, FICO HELOC, Freddie Mac and Fannie Mae are registered as external/future product validation sources. They are not pooled into this model because their access terms, product definitions, features and/or default horizons differ.
