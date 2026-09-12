# CRIX Hardening Issues #7–#13 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden CRIX v3 model provenance, validation, runtime efficiency/explainability, stress semantics and policy backtesting without changing the existing v3 PD target or silently breaking `/api/v3`.

**Architecture:** Keep runtime TypeScript stateless and small. Add pure offline Python governance/backtest utilities beside the existing training code, persist only aggregate/model metadata, and reuse the current immutable JSON artifact boundary. Runtime refactors use one request-local scoring context and versioned deterministic sensitivity scenarios.

**Tech Stack:** Node 22, TypeScript 5.7, Fastify 5, Vitest 4, Python 3.12, NumPy/Pandas/scikit-learn/XGBoost.

**Spec:** GitHub issues #7, #8, #9, #10, #11, #12, #13.

## Global Constraints

- Preserve CRIX API v3 semantics: current `pd` remains final-loan-resolution default risk, not a 12-month PD.
- Keep the live service stateless; do not add Redis, a database, a queue, or a model-registry service.
- Public demo remains unsuitable for real consumer credit data.
- Prefer the smallest auditable implementation; if 10 clear lines preserve the same invariants as 30 abstract lines, use the 10.
- All new offline reports are aggregate-only and point-in-time safe.
- CPU-bound work remains bounded on the Fastify event loop.

---

### Task 1: Add governance tests first

**Files:**
- Create: `model/test_governance.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: future pure functions from `model/governance.py`.
- Produces: failing CI coverage for leakage checks, segmented metrics, PSI, bootstrap determinism and policy backtesting.

- [ ] **Step 1:** Write `unittest` cases for point-in-time feature metadata validation, future-dated rejection, duplicate calibration-bin handling, deterministic bootstrap CIs, PSI edge cases, approve-all backtest reconciliation and selection-population labels.
- [ ] **Step 2:** Extend CI with a Python governance job using Python 3.12 and `python -m unittest model/test_governance.py -v`.
- [ ] **Step 3:** Push tests-only commit and verify CI fails because `model/governance.py` does not exist.

### Task 2: Implement offline provenance, diagnostics and policy backtesting

**Files:**
- Create: `model/governance.py`
- Modify: `model/datasets.py`
- Modify: `model/train.py`
- Modify: `MODEL_CARD.md`
- Modify: `model/TRAINING_REPORT.md` generation in `model/train.py`

**Interfaces:**
- Produces: `validate_feature_provenance(...)`, `calibration_by_bins(...)`, `bootstrap_metric_interval(...)`, `population_stability_index(...)`, `backtest_pd_policy(...)`, `FeatureProvenance` metadata and `populationConditioning` artifact metadata.

- [ ] **Step 1:** Implement fail-closed feature provenance validation and as-of timestamp checks with vectorized comparisons.
- [ ] **Step 2:** Add deterministic segmented calibration helpers, minimum-support behavior, bootstrap CIs over cached predictions, and PSI with zero-mass smoothing.
- [ ] **Step 3:** Add policy backtesting over cached predictions with approve-all/PD-threshold support and explicit `granted-loans-only` conditioning.
- [ ] **Step 4:** Persist provenance/population metadata and diagnostics into the generated artifact/report without changing model scoring semantics.
- [ ] **Step 5:** Run Python governance tests and ensure they pass.

### Task 3: Add runtime tests for scoring-context reuse and explanation/stress semantics

**Files:**
- Modify: `src/risk/engine.test.ts`
- Modify: `src/app.test.ts`

**Interfaces:**
- Consumes: future runtime fields/functions from `src/risk/engine.ts` / `src/risk/types.ts` / `src/schemas.ts`.
- Produces: failing tests for deterministic sensitivity metadata, model-vs-policy reasons, counterfactual bounds/stability, no input mutation, and preserved v3 response behavior.

- [ ] **Step 1:** Add engine tests asserting model reasons are separate from policy reasons, counterfactuals stay in-domain, identical inputs are deterministic, and stress input is not mutated.
- [ ] **Step 2:** Add API tests asserting stress responses disclose deterministic sensitivity method/version and that score responses remain v3-compatible.
- [ ] **Step 3:** Push tests-only commit and confirm Node CI fails for the intended missing behavior.

### Task 4: Refactor runtime hot path and explanation governance

**Files:**
- Modify: `src/risk/engine.ts`
- Modify: `src/risk/types.ts`
- Modify: `src/schemas.ts`
- Modify: `ARCHITECTURE.md`
- Modify: `MODEL_CARD.md`

**Interfaces:**
- Produces: request-local immutable `ScoringContext`, model/policy reason separation, bounded monotonic counterfactuals, versioned deterministic sensitivity scenario metadata.

- [ ] **Step 1:** Derive/validate feature vector and shared loan-to-income once per assessment and pass a scoring context to champion/challenger/OOD/policy helpers.
- [ ] **Step 2:** Preserve existing `reasons` for model reasons and add `policyReasons` plus bounded `counterfactuals`; keep values deterministic and finite.
- [ ] **Step 3:** Move mild/severe shocks into one frozen versioned scenario table and return `method`/`scenarioVersion` metadata while retaining `/api/v3/risk/stress`.
- [ ] **Step 4:** Keep all artifact traversal bounds, batch bounds, rate limits and API-key behavior unchanged.
- [ ] **Step 5:** Run `npm run verify` through CI and ensure all tests pass.

### Task 5: Compliance, review and merge

**Files:**
- Modify docs only if verification identifies a discrepancy.

- [ ] **Step 1:** Confirm Node CI green: lint, typecheck, Vitest, bundle.
- [ ] **Step 2:** Confirm Python governance CI green.
- [ ] **Step 3:** Review diff for leakage, PII/logging, unbounded CPU loops, v3 target mislabelling, unsupported reject-inference claims and API schema drift.
- [ ] **Step 4:** Confirm no secret/private data was introduced and public-demo warnings remain.
- [ ] **Step 5:** Open PR, request/review checks, merge only after green gates.
- [ ] **Step 6:** Verify the merged `main` CI commit is green.
- [ ] **Step 7:** Verify Render deploy for the merged commit and check `/health`, `/ready`, `/openapi.json` and one safe demo score endpoint if the deployed service URL is available from the connected Render project.
