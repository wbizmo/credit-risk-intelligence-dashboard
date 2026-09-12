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
- Population conditioning: `granted-loans-only`

The primary outcome cohort contains historically granted loans with observed outcomes. CRIX does not fabricate outcomes for rejected applicants, so policy backtests below are conditional selection analyses over the observed granted-loan cohort rather than applicant-population counterfactuals.

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

### Bootstrap uncertainty (95% percentile intervals)

| Metric | Point | Lower | Upper |
|---|---:|---:|---:|
| auc | 0.6594 | 0.6553 | 0.6625 |
| brier | 0.1645 | 0.1641 | 0.1650 |
| logLoss | 0.5043 | 0.5032 | 0.5057 |
| ks | 0.2300 | 0.2237 | 0.2357 |

Calibration-to-OOT PD population stability index: **0.0265**.

## Point-in-time feature provenance

The champion contract fails closed if a trained feature lacks provenance metadata or is marked outcome-derived.

| Feature | Source field(s) | Availability | Outcome-derived |
|---|---|---|---|
| `debtToIncome` | dti_n | application-time | False |
| `loanToIncome` | loan_amnt, revenue | application-time | False |
| `creditScore` | fico_n | application-time | False |
| `employmentYears` | emp_length | application-time | False |

The fixed segment diagnostics stored in the artifact cover credit-score, DTI, requested-loan-to-income and employment-tenure bands. Every segment carries observation/event counts and returns `insufficient-data` rather than a fabricated metric when support is too small.

## Observed-cohort policy backtests

| Policy | Selection rate | Observed default rate | Selected exposure |
|---|---:|---:|---:|
| approveAll | 100.00% | 22.42% | $2,281,184,300 |
| pd20 | 49.77% | 14.45% | $946,700,475 |
| pd35 | 91.19% | 20.54% | $1,990,770,500 |

These backtests do **not** estimate what would have happened to historically rejected applicants. Reject inference is unsupported until a defensible rejected-applicant/outcome source and assumptions are available.

## Canonical CRIX features used by the champion

- `debtToIncome` ← LendingClub `dti_n` (normalized from percentage points when detected)
- `loanToIncome` ← `loan_amnt / revenue`
- `creditScore` ← LendingClub `fico_n`
- `employmentYears` ← normalized `emp_length`

The remaining CRIX application context is not falsely presented as part of the trained champion when it is absent from this public cohort.

## Dataset separation

Home Credit, Give Me Some Credit, FICO HELOC, Freddie Mac and Fannie Mae are registered as external/future product validation sources. They are not pooled into this model because their access terms, product definitions, features and/or default horizons differ.
