# CRIX v3.2.0 — Tail-Risk and Model-Governance Hardening

CRIX v3.2 extends the real-data v3 research stack with deeper portfolio-tail modelling and a substantially stronger offline model-governance layer while deliberately keeping the live `/api/v3` decisioning contract stable.

The deployed runtime champion remains **CRIX-MonoBoost 2.0.0**, the live policy remains **CRIX-Policy 3.0**, and `pd` still means **final-loan-resolution default risk**. None of the v3.2 research/governance work silently replaces the runtime champion or changes the API probability target.

## Portfolio tail-risk research

- Student-t dependence and bounded low-rank multi-factor dependence complement the canonical one-factor Gaussian simulator.
- Single-replay tail attribution keeps contribution accounting bounded and reconcilable.
- NumPy remains the canonical reproducible path; an optional lazy CuPy backend is available only for offline GPU experiments.
- Peaks-Over-Threshold GPD/EVT research adds threshold-stability diagnostics, bounded bootstrap uncertainty and explicit insufficient/unstable states.
- Unsupported 99.9% tail estimates continue to be withheld rather than reported with false precision.

## Distribution-shift governance

v3.2 adds a bounded adversarial classifier over the approved champion feature space and complements existing PSI evidence with Jensen-Shannon divergence, Wasserstein distance, p01/p99 support-breach rates, missingness movement, quantile movement and fixed segment diagnostics with explicit `insufficient-data` handling.

On the governed historical validation split, adversarial AUC was **0.5593** and the aggregate status plus all four champion-feature statuses were **pass**. These are historical governance diagnostics, not proof of future or cross-population stability.

## Explanation-fidelity validation

Offline SHAP TreeExplainer evidence is compared with CRIX's deterministic local champion-sensitivity method on a fixed-seed OOT sample.

For the governed **n=512** sample:

- mean top-3 overlap: **0.9759**;
- top-3 disagreement rate: **7.23%**;
- mean absolute-rank correlation: **0.9141**;
- mean sign agreement: **0.9533**;
- mean bounded-perturbation top-3 stability: **0.9694**.

SHAP remains offline and is not a Fastify dependency. SHAP agreement is model-explanation evidence only and is not represented as legal adverse-action compliance.

## Versioned data contracts

Training now fails early on explicit versioned contracts covering source schema, harmonized data, chronological splits and product-specific external research inputs.

Controls include immutable source checksum verification, binary targets, finite/ranged features, application-time `asOf` semantics, unique row identity, split isolation and the existing independent point-in-time provenance gate. Validation failures emit safe aggregate metadata rather than raw borrower rows or IDs.

## Tamper-evident lineage

The approved champion now has a committed training-run lineage manifest linking model artifact and source-data hashes, split definitions/counts and deterministic seeds, feature/data-contract versions, historical artifact-origin Git revision, v3.2 validation-environment metadata and key governance-evidence hashes.

CI verifies the lineage **read-only** and fails on artifact/report/prior-manifest tampering. Hash-chain verification is deliberately a simple tamper-evident control with no distributed-ledger dependency. Publication of new approved manifests remains an explicit reviewed action.

## Engineering discipline

- Python governance/research suite: **89 passing tests**, with one optional GPU-only skip.
- Node build, lint, typecheck, tests and runtime benchmarks remain green.
- Full real-data champion retraining and registry-integrity gates pass.
- Historical risk-stack and advanced-risk evidence continue to reproduce.
- No v3.2 Batch C change touches the live TypeScript scoring source or adds Python/SHAP to the request path.

## Important model-risk note

CRIX remains a public engineering/model-risk research system, **not an approved production lending, accounting or regulatory-capital platform**. Real deployment still requires representative institution-specific data, independently validated models, fairness/proxy testing, exact legal/accounting/regulatory policy interpretation, governed adverse-action processes, production IAM/audit controls, monitoring and formal model-risk/legal/compliance approval.
