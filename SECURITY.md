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
- optional `x-api-key` protection for `/api/v3/*`;
- constant-time API-key comparison;
- generic 5xx responses with no stack trace disclosure;
- finite-number checks inside the model engine;
- bounded model-tree traversal and artifact integrity checks;
- no borrower-name field and no database.

## API key mode

Set `CRIX_API_KEY` to a high-entropy secret. When configured, every `/api/v3/*` request must include:

```text
x-api-key: <secret>
```

Health, readiness and API documentation remain public so infrastructure can probe the service and reviewers can inspect the contract.

For a real multi-tenant product, replace this simple deployment-level key with gateway/IAM-backed authentication and authorization, key rotation, tenant quotas, audit trails and managed secret controls.

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

## Dependency and CI policy

Every change to `main` must pass lint, TypeScript checking, unit/API tests and a production build. Real-data training has a separate reproducibility workflow that downloads the registered public sources, trains/evaluates the artifacts and commits only the derived evidence.

Releases are produced from the version in `package.json` only after code reaches `main`.

## Reporting

If you find a security issue, do not publish exploit details in a public issue. Contact the repository owner privately through the contact methods on the owner's GitHub profile.
