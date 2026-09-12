# CRIX Risk Stack #14–#19 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement CRIX issues #14–#19 as auditable research-model layers without changing `/api/v3` PD/LGD/EAD semantics.

**Architecture:** Keep the existing LendingClub granting champion isolated. Add a repository-native registry/integrity layer, an offline as-of snapshot engine, and separate survival/transition/LGD/EAD research models. Lifecycle LendingClub data is used only for post-origination survival/LGD/EAD research; UCI Taiwan credit-card history is used for monthly delinquency transitions and CCF research. Every artifact is immutable, versioned, source-specific, and linked through manifests.

**Tech Stack:** Python 3.12, NumPy, Pandas, scikit-learn, ucimlrepo, existing TypeScript/Fastify runtime, GitHub Actions.

**Spec:** GitHub issues #14, #15, #16, #17, #18, #19.

## Global Constraints

- Preserve CRIX v3 `/api/v3` score semantics and the final-loan-resolution champion PD.
- Do not manufacture event dates, delinquency histories, recoveries, or CCF from the granting-only Zenodo dataset.
- Do not pool product-mismatched datasets into one champion.
- Keep all research outputs de-identified and aggregate-only; no raw borrower/account rows are committed.
- Prefer pure functions and vectorized/grouped operations over frameworks, nested scans, or mutable shared state.
- Fixed seeds and immutable source checksums are required for every stochastic model or sample.
- CI must fail closed on manifest tampering, future-dated features, invalid stochastic matrices, non-monotone PD curves, invalid EAD/LGD values, or data-source mismatch.

---

### Task 1: Model registry and integrity chain (#14)

**Files:**
- Create: `model/registry.py`
- Create: `model/artifacts/registry.json`
- Create: `model/artifacts/crix-monoboost-v2.manifest.json`
- Modify: `model/train.py`
- Create: `src/risk/registry.ts`
- Modify: `src/risk/engine.ts`
- Modify: `src/risk/types.ts`

**Interfaces:**
- `sha256_file(path: Path) -> str`
- `build_manifest(...) -> dict[str, object]`
- `verify_manifest(artifact_path: Path, manifest: Mapping) -> bool`
- `update_registry_index(index_path: Path, entry: Mapping) -> None`
- TypeScript `verifyRuntimeArtifactManifest() -> boolean`

- [ ] Write failing tests for streaming SHA-256, tamper rejection, duplicate model/version digest conflicts, canonical config hashing, and atomic registry publication.
- [ ] Implement the smallest stdlib-only registry module.
- [ ] Wire `train.py` to generate the champion manifest after the artifact is complete, then atomically update the index.
- [ ] Add runtime SHA-256 verification of only the deployed champion artifact/manifest; readiness fails closed if missing or mismatched.
- [ ] Expose only safe model ID/digest/status in metadata.

### Task 2: Historical as-of engine (#15)

**Files:**
- Create: `model/time_machine.py`

**Interfaces:**
- `SnapshotSpec`
- `snapshot_id(spec, feature_contract_version) -> str`
- `backward_asof_join(...) -> np.ndarray`
- `build_snapshot_masks(...) -> SnapshotMasks`
- `snapshot_manifest(...) -> dict[str, object]`

- [ ] Write failing fixtures with future-dated values, missing dates, overlapping windows, censored outcomes, deterministic IDs, and repeated-run idempotency.
- [ ] Implement sorted `searchsorted` backward joins; never nearest/future joins.
- [ ] Support strict point-in-time mode plus an explicitly labelled retrospective-resolved mode solely to reproduce the legacy v3 split.
- [ ] Store only counts/digests/window metadata in manifests, not row data.

### Task 3: Lifetime survival PD (#16)

**Files:**
- Create: `model/survival.py`
- Create/Modify: `model/research_data.py`
- Create: `model/train_risk_stack.py`

**Interfaces:**
- `event_time_from_status(...)`
- `expand_person_period(...)`
- `fit_discrete_time_hazard(...)`
- `predict_term_structure(...)`
- `validate_term_structure(...)`

- [ ] Write failing synthetic hazard/censoring/property tests.
- [ ] Use the verified lifecycle LendingClub mirror only for event-time research; `last_pymnt_d` is recorded as an endpoint proxy, never described as an exact charge-off date.
- [ ] Train an interpretable discrete-time logistic hazard baseline with month/bucket baseline terms and point-in-time borrower/contract covariates.
- [ ] Produce hazard, survival, marginal PD, cumulative PD and 3/6/12/24/36m outputs; suppress unsupported horizons.
- [ ] Evaluate chronologically by vintage and register the research artifact separately from the v3 champion.

### Task 4: Delinquency transitions (#17)

**Files:**
- Create: `model/transitions.py`
- Modify: `model/research_data.py`

**Interfaces:**
- `map_repayment_status(code) -> str | None`
- `transition_counts(sequences, states) -> np.ndarray`
- `transition_matrix(counts, states, absorbing) -> np.ndarray`
- `propagate_distribution(matrix, initial, months) -> np.ndarray`

- [ ] Write failing fixtures for backward cures, direct default, missing intervals, zero-support rows and matrix propagation.
- [ ] Map UCI Taiwan `PAY_6 → PAY_5 → PAY_4 → PAY_3 → PAY_2 → PAY_0`; append `DEFAULT` only where the observed next-month target is default and do not fabricate an October non-default state.
- [ ] Build one-pass empirical transition counts and a stochastic matrix; preserve observed backward/cure transitions.
- [ ] Register dataset/state taxonomy/source limitations in the artifact.

### Task 5: Empirical EAD / CCF (#19)

**Files:**
- Create: `model/ead.py`
- Modify: `model/train_risk_stack.py`

**Interfaces:**
- `amortizing_balance(principal, annual_rate, term_months, elapsed_months) -> float`
- `observed_installment_ead(...) -> np.ndarray`
- `ccf_proxy(limit, prior_balance, next_balance) -> np.ndarray`
- `fit_ead_models(...) -> dict[str, object]`

- [ ] Write closed-form amortization, zero-rate, over-limit, zero-undrawn, and off-by-one tests.
- [ ] Use lifecycle LendingClub defaults for empirical installment EAD: principal outstanding proxy = funded amount minus principal received by charge-off/final default resolution, with event-month proxy supplied by the survival dataset.
- [ ] Compare an interpretable linear/Ridge baseline with a bounded nonlinear challenger only if it improves chronological holdout error.
- [ ] Use UCI Taiwan only for a clearly labelled net-balance-change CCF proxy; do not call it pure draw CCF and do not clamp negative/>1 values silently.
- [ ] Preserve v3 `ead = loanAmount` unchanged.

### Task 6: Empirical LGD (#18)

**Files:**
- Create: `model/lgd.py`
- Modify: `model/train_risk_stack.py`

**Interfaces:**
- `economic_lgd(ead, recoveries, recovery_costs, discount_factor=1.0) -> np.ndarray`
- `fit_lgd_models(...) -> dict[str, object]`

- [ ] Write exact recovery, >100% loss, zero-EAD, incomplete-window, and deterministic baseline tests.
- [ ] Define lifecycle-LendingClub EAD consistently with Task 5 and net recovery as `recoveries - collection_recovery_fee`.
- [ ] Because the chosen public mirror has aggregate recoveries rather than dated recovery cashflows, use an explicit undiscounted aggregate-recovery convention (`discountRate = 0`) and record `recoveryTimingSupport = aggregate-only`; never invent recovery dates.
- [ ] Train an interpretable baseline and challenger on defaulted loans with chronological OOT evaluation, exposure-weighted error, and segment counts.
- [ ] Preserve v3 engineering LGD as a named baseline/challenger rather than silently replacing it.

### Task 7: Research orchestration, CI and documentation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/model-validation.yml`
- Modify: `MODEL_CARD.md`
- Modify: `ARCHITECTURE.md`
- Create: `model/RISK_STACK_REPORT.md`

- [ ] Add fast unit gates for registry/time/survival/transition/EAD/LGD invariants.
- [ ] Cache the immutable lifecycle research source by SHA-256; run full research training only in model validation, never in the Node hot path.
- [ ] Upload generated research artifacts/manifests as workflow evidence and verify their hashes/registry relationships.
- [ ] Run Node verification, Python unit tests, full granting-model validation and full research-stack validation.
- [ ] Review the final diff for PII leakage, path traversal, dependency/license risk, accidental v3 semantic changes, unbounded CPU/memory work, and unsupported regulatory claims.
- [ ] Merge only with all gates green, then close #14–#19 as completed only to the extent their empirical source requirements are actually met.
