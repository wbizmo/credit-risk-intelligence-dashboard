# CRIX model registry and research risk stack

CRIX keeps the deployed `/api/v3` scoring contract separate from the research models introduced by issues #14–#19. The live API still serves **CRIX-MonoBoost 2.0.0** with final-loan-resolution PD semantics, the existing engineering LGD, and `EAD = requested loanAmount`. Lifetime PD, transitions, empirical EAD/LGD and CCF work remain separately versioned research artifacts until an explicit future API/model promotion changes that contract.

## Registry integrity chain

Every generated model artifact has a companion manifest and a repository-level registry entry. A manifest records:

- model ID, name, semantic version and artifact-schema version;
- product and target semantics;
- model training timestamp and training run ID;
- exact training-code Git commit;
- source dataset identity/checksums and licence metadata;
- feature-contract version;
- chronological/snapshot split metadata;
- random seed;
- canonical training-config hash;
- dependency-environment hash from `model/requirements.txt`;
- artifact SHA-256;
- validation metric summary;
- model-card/report references;
- explicit lifecycle status (`research`, `challenger`, `approved-demo-champion`, `retired`);
- predecessor/parent where applicable.

`model/registry.py` hashes files by streaming SHA-256 and publishes the JSON index atomically. A `(modelName, version)` or existing `modelId` cannot be silently rebound to different artifact bytes. The TypeScript runtime hashes only the deployed champion and verifies it against `crix-monoboost-v2.manifest.json`; manifest failure makes `/ready` fail closed. Arbitrary runtime model paths are not accepted.

## Reproducing the evidence

From a clean checkout with Python 3.12 and Node 22:

```bash
pip install -r model/requirements.txt
python -m unittest discover -s model -p 'test_*.py' -v
python model/train.py
python model/publish_champion_manifest.py
python model/train_risk_stack.py
npm install --no-audit --no-fund --legacy-peer-deps
npm run verify
```

The full pull-request model-validation workflow additionally pins `CRIX_TRAINING_GIT_SHA` to the PR branch head and `CRIX_TRAINING_RUN_ID` to the GitHub Actions run. It verifies every artifact/manifest digest relationship before uploading aggregate evidence. Source data is downloaded or cache-restored by immutable checksum; raw borrower/account rows are not committed.

## Historical time machine

`model/time_machine.py` distinguishes two modes:

- `point-in-time`: training/calibration labels must actually be observable by their declared label cutoffs; no future feature/value may enter a backward as-of join.
- `retrospective-resolved`: an explicitly labelled compatibility mode used only to reproduce the existing v3 resolved-loan chronological cohorts. It is not presented as a historical point-in-time training experiment.

Snapshot IDs are content/config derived and manifests contain windows, source digests and aggregate counts rather than raw records.

## Lifetime PD

`CRIX-LifetimePD 1.0.0` is a discrete-time logistic-hazard research model on a checksum-pinned LendingClub lifecycle mirror. It produces interval hazard, survival, marginal PD and cumulative PD through 36 months, including 3/6/12/24/36-month summaries where support is adequate.

The model uses origination-time DTI, requested-loan-to-income and employment tenure. LendingClub grade, policy pricing and interest rate are not used as PD predictors. `last_pymnt_d` is explicitly an endpoint proxy; the project does **not** claim that it is the exact charge-off date. Unresolved loans are censored rather than silently labelled non-default.

## Delinquency transitions

`CRIX-Transitions-Taiwan 1.0.0` uses the UCI Default of Credit Card Clients panel as a separate revolving-credit research source. It models observed monthly repayment-state migrations, including backward/cure transitions, and appends `DEFAULT` only for the observed next-month default outcome. It does not fabricate a non-default October repayment state.

The Taiwan product is never pooled into the LendingClub personal-loan champion.

## EAD and CCF

`CRIX-EAD-Installment 1.0.0` models a closed-end principal-exposure proxy conditional on default using funded amount less principal received. It compares an interpretable Ridge baseline with a nonlinear challenger under chronological evaluation.

`CRIX-CCF-Taiwan 1.0.0` is deliberately labelled a **net-balance-change CCF proxy**, not pure additional-draw CCF. The UCI source lacks a complete transaction-level draw ledger, so negative and greater-than-one proxy values are retained diagnostically rather than silently clamped. It is research evidence only.

## LGD

`CRIX-LGD 1.0.0` defines the observed severity proxy as:

```text
LGD = (EAD - (recoveries - collection_recovery_fee)) / EAD
```

The lifecycle source has aggregate recovery totals but not dated recovery cashflows. CRIX therefore records `recoveryTimingSupport = aggregate-only` and uses an explicit zero discount rate rather than manufacturing recovery dates. Values outside the usual 0–100% interval are handled according to the documented economic convention instead of being blindly clipped.

The training workflow compares an interpretable baseline with a nonlinear challenger and recommends whichever has lower out-of-time error. Model complexity is not promoted merely because it is available.

## Scope and claims

These artifacts are engineering/model-risk research, not regulatory approval, bank production validation, IFRS 9 compliance, Basel compliance, legal adverse-action tooling, or evidence that LendingClub/Taiwan estimates transfer unchanged to another geography or portfolio.

The current LendingClub champion remains conditioned on historically granted loans. Rejected-applicant outcomes are unavailable, so reject inference and applicant-population causal claims remain unsupported. The research stack is intentionally source/product specific and preserves those limitations in its manifests and reports.
