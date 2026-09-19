# Security Policy

CRIX v3 is a public engineering/model-risk demonstration. It is deliberately stateless and does not persist submitted applications.

## Runtime controls

- strict Fastify/AJV request schemas;
- unknown request fields rejected;
- bounded numeric domains;
- 64 KiB body limit;
- bounded synchronous batch size (50);
- global and endpoint-specific rate limits;
- UUID request IDs;
- Helmet security headers;
- credential-header log redaction;
- CORS disabled unless explicitly allow-listed;
- explicit `public-demo` and fail-closed `required` authentication modes;
- bounded current+next API-key rotation for `/api/v3/*`;
- SHA-256 fixed-length constant-time API-key comparison with no key/hash logging;
- reverse-proxy trust disabled by default and bounded by an explicit hop count;
- authenticated rate-limit buckets use only a bounded key slot; unauthenticated/invalid credentials fall back to client IP;
- generic 5xx responses with no stack trace disclosure;
- finite-number checks inside the model engine;
- bounded model-tree traversal and artifact integrity checks;
- no borrower-name field and no database.

## Runtime authentication profiles

CRIX uses an explicit `CRIX_AUTH_MODE`.

### `public-demo`

This preserves the public reviewer/demo posture. `/api/v3/*` is callable without credentials, while rate limits, body/time bounds and all normal validation remain active. API keys must not be configured in this mode.

### `required`

This is the hardened deployment posture. Startup fails closed if there is no acceptable high-entropy key. Configure `CRIX_API_KEYS` as a bounded comma-separated set of at most two active values so a current key and next key can overlap during rotation. Keys must be unique, at least 24 characters and non-trivial. `CRIX_API_KEY` remains a single-key compatibility alias only when `CRIX_AUTH_MODE` is explicitly set.

Every protected `/api/v3/*` request must include:

```text
x-api-key: <secret>
```

Health, readiness and API documentation remain public for probes and contract inspection. Credential values and their hashes are never used as logs or metric labels.

## Reverse proxies and client IP

Fastify proxy trust is **off by default**. `CRIX_TRUST_PROXY_HOPS` accepts only a bounded hop count (0–4). Configure the exact topology used by the deployment; do not enable blanket trust for arbitrary forwarding chains.

With trust disabled, spoofed `x-forwarded-for` input does not become the rate-limit client identity. With one trusted hop, the immediate trusted reverse proxy may supply the originating client address. Changing proxy topology requires retesting this assumption.

For a real multi-tenant product, replace the deployment-level key with gateway/IAM-backed authentication and authorization, tenant-scoped quotas, managed secret rotation and immutable audit controls.

## Data handling

Do **not** send real consumer credit data to the public demo deployment.

The project does not persist scoring requests, but public-demo infrastructure, transient process memory and operational logging are not substitutes for a regulated-data environment. Real borrower data requires an approved environment with purpose limitation, encryption, access control, retention rules, auditability and applicable legal/compliance controls.

## Training-data handling

CRIX v3 training workflows use public research datasets and do not commit the raw downloaded cohorts to the repository. Downloaded datasets live under ignored `model/data/` paths during training. The committed evidence is limited to compact trained artifacts, metrics, source identifiers, licenses and checksums/provenance.

For the primary LendingClub/Zenodo source, CRIX locks the expected MD5 before training so a silently changed upstream file cannot be treated as the same training cohort.

## Protected / identity-adjacent fields

The external Taiwan research workflow explicitly excludes `SEX`, `EDUCATION`, `MARRIAGE` and `AGE` from the reduced behavioral model.

CRIX does not infer or impute those variables into the runtime API. This exclusion is a model-design control, not a claim that fairness has been proven. Formal fairness/proxy analysis requires the appropriate legal/compliance framework and protected-attribute evaluation process.

## Model safety

CRIX v3 is trained on real historical credit data, but that does **not** make it safe or approved for production lending.

Key limitations include:

- historical and population-specific calibration;
- accepted-applicant / underwriting-selection effects;
- a primary target defined as final-loan-resolution default risk rather than fixed-horizon 12-month PD;
- incomplete bureau-feature coverage in the primary public cohort;
- non-validated LGD/EAD/pricing approximations;
- no institution-specific fairness or legal validation;
- external UCI datasets with different products and target horizons.

The Taiwan and German artifacts/benchmarks are intentionally not blended into the primary personal-loan probability.

See `MODEL_CARD.md` and `model/TRAINING_REPORT.md` for model-risk details.

## Observability privacy and cardinality

OpenTelemetry metrics are disabled by default and configured independently from scoring. When enabled, export is periodic/background; a collector failure does not fail a score request.

Allowed labels are deliberately bounded: route template, HTTP method/status class, model version, policy version, decision, fixed operation, disagreement bucket and bounded champion feature name for OOD counts. The instrumentation must not emit application/request IDs, raw IP addresses, headers, API keys or hashes, borrower feature values, exact PD/loan/income values, or arbitrary exception text.

Operational aggregates include request/error/auth/rate-limit counts, request and score/stress/batch latency, batch size, decision/OOD/low-confidence/disagreement aggregates, readiness, RSS/heap, uptime and event-loop indicators. CRIX does not add borrower-level traces or persistence.

## Dependency and CI policy

Node governed builds use the committed npm lock with `npm ci`; CI does not rewrite it. Production dependency advisories are checked separately from install with a **high** severity blocking threshold. Dev-only tooling advisories are reviewed separately rather than being treated as runtime exposure automatically.

The human-readable Python constraints remain in `model/requirements.txt`, while governed model validation installs the exact hash-pinned `model/requirements.lock`. `model/requirements.lock.meta.json` binds the lock to its source constraints and generator version, and the training-lineage manifest records the exact lock SHA-256.

Python model/research dependency audits are visible CI evidence but are triaged by reachability and environment: live-runtime, development-only, offline research-only, transitive/no-reachable-path, or no-upstream-fix. An advisory does not automatically retrain or promote a champion. Scientific dependency changes that may affect numerics require the ordinary model-validation/reproduction gates before approval.

Workflow actions use maintained major versions. Dependency/lock updates are explicit reviewed changes; CI never silently bumps dependencies or publishes a newly retrained champion.

Every change to `main` must pass lint, TypeScript checking, unit/API tests and a production build. Real-data training has a separate reproducibility workflow that downloads registered public sources, verifies immutable source checksums and validates derived evidence.

Releases are produced from the version in `package.json` only after code reaches `main`.

## Reporting

If you find a security issue, do not publish exploit details in a public issue. Contact the repository owner privately through the contact methods on the owner's GitHub profile.
