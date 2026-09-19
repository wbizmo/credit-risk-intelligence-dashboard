# CRIX v3.2.0 — Performance, Tail Risk & Model Governance

CRIX v3.2.0 is an engineering-hardening release over the real-data v3 foundation. It makes the runtime measurably leaner, expands the offline portfolio/tail-risk stack, strengthens model-risk governance, hardens deployment/authentication and makes dependency/lineage evidence reproducible—without changing the live probability target or silently promoting a research model.

**Compatibility:** package release **3.2.0**; API contract **3.0.0** under `/api/v3`; primary champion **CRIX-MonoBoost 2.0.0**; live policy **CRIX-Policy 3.0**.

## Measured runtime evidence

- Compiled champion throughput: **~536,854 ops/s** on the recorded Batch A CI host (Node 22.23.2, AMD EPYC 7763).
- Compiled champion p50: **0.003746 ms** versus **0.004456 ms** reference.
- Full compiled assessment p50: **0.018628 ms** versus **0.025889 ms** reference.
- Sparse explanation budget: **317 tree visits** versus **384** full-rescore visits (**17.45% fewer**).
- Production rate-limit profile: **60 HTTP 200 + 20 HTTP 429**, **0** 5xx, **0** timeouts.
- Sustained scoring profile: **500/500 HTTP 200**, no failures.
- Authenticated scoring profile: **40/40 HTTP 200**.

These are CI/loopback engineering measurements, not Render cold-start, internet or end-user SLOs.

## Portfolio tail-risk research

- Gaussian and Student-t dependence with homogeneous one-factor or bounded explicit low-rank factor loadings.
- Single-replay tail attribution with disjoint indexed buckets and bounded memory.
- POT/GPD EVT tail extrapolation kept explicitly separate from empirical Monte Carlo VaR/ES.
- Final indexed-bucket benchmark evidence showed **~0.80x** reference speed at one quantile, **~2.42x** speedup at three quantiles and **~4.03x** at five quantiles.
- Final audit removes the earlier per-quantile mask scan and hoists invariant backend threshold/loading setup out of the chunk loop.
- Optional CuPy remains research-only and lazy. **No GPU speedup/crossover number is published** because the connected release infrastructure has no CUDA device; NumPy remains the canonical evidence path.

## Model governance

- Adversarial validation plus PSI, Jensen-Shannon, Wasserstein, support-breach, missingness and quantile movement.
- Governed historical adversarial AUC: **0.5593**; aggregate status and all four champion features: **pass**.
- Offline SHAP-vs-local-sensitivity sample: **n=512**, top-3 overlap **97.59%**, disagreement **7.23%**, absolute rank correlation **0.9141**, sign agreement **95.33%**, bounded-perturbation stability **96.94%**.
- Versioned dataframe contracts and immutable source-checksum gating before fitting.
- Tamper-evident lineage linking approved artifact, data, evidence, dependency environment and historical training revision.

## Runtime security and telemetry

- Explicit `public-demo` and fail-closed `required` auth modes.
- Bounded current+next API-key rotation with fixed-length SHA-256 digest comparison and constant-time equality.
- Proxy trust off by default; trusted-hop count bounded and tested against spoofed forwarding headers.
- Credential-safe/key-slot-aware rate limiting plus supplementary client-IP quotas.
- Privacy-bounded OpenTelemetry metrics with bounded labels only.
- In-memory telemetry comparison: **5.976 ms → 6.295 ms mean** (~**+5.3%**) and **10.750 ms → 10.858 ms p95** (~**+1.0%**).

## Supply chain and reproducibility

- `npm ci` from committed lock plus high-severity production audit.
- Hash-pinned Python research lock installed with `--require-hashes`.
- CI intentionally corrupts dependency inputs to prove lock drift fails closed.
- Batch D certification: **0 known Node production vulnerabilities** and **0 known vulnerabilities in the governed Python research lock**.
- The unused optbinning → ortools → protobuf chain carrying advisories was removed instead of being cosmetically waived.

## Final repo-wide audit

- Exact constrained portfolio selection no longer rebuilds every subset: bookkeeping moves from `O(n*2^n)` to Gray-code incremental `O(2^n log G)` when segment concentration is enforced, with the 24-candidate exact-search cap retained.
- Tail attribution now matches its documented `O(S*N + S*log Q + Q*N)` dominant complexity rather than retaining a hidden per-quantile chunk mask scan.
- Invariant NumPy/CuPy thresholds and low-rank factor arrays are prepared once per simulation instead of inside every chunk.
- Explanation-fidelity perturbation reuses one matrix instead of copying the full matrix for every feature.
- Chronological split overlap validation uses vectorized duplicate detection instead of million-row Python sets.
- Live batch response totals are accumulated in one pass.

## Model-risk boundary

CRIX remains a public engineering/model-risk research system, **not an approved production lending, accounting or regulatory-capital platform**. Institution-specific data, independent validation, fairness/proxy testing, legal/accounting/regulatory interpretation, governed adverse-action processes, production IAM/audit retention and formal model-risk approval remain deployment prerequisites.
