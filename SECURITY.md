# Security Policy

CRIX v2.5 is a public engineering demonstration. It is deliberately stateless and does not persist submitted applications.

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
- optional `x-api-key` protection for `/api/v2/*`;
- constant-time API-key comparison;
- generic 5xx responses with no stack trace disclosure;
- finite-number checks inside the model engine;
- bounded model-tree traversal and artifact integrity checks;
- no borrower-name field and no database.

## API key mode

Set `CRIX_API_KEY` to a high-entropy secret. When configured, every `/api/v2/*` request must include:

```text
x-api-key: <secret>
```

Health, readiness and API documentation remain public so infrastructure can probe the service and reviewers can inspect the contract.

For a real multi-tenant product, replace this simple deployment-level key with gateway/IAM-backed authentication and authorization, key rotation, tenant quotas, audit trails and secret-management controls.

## Data handling

Do not send real consumer credit data to the public demo deployment. The project does not persist requests, but public-demo infrastructure and operational logs are not a substitute for a regulated-data environment.

## Model safety

The bundled model uses synthetic data and must not be used for real lending decisions. Model-risk limitations are documented in `MODEL_CARD.md`.

## Dependency and CI policy

Every change to `main` must pass lint, TypeScript checking, unit/API tests and a production build. Releases are produced from the version in `package.json` only after code reaches `main`.

## Reporting

If you find a security issue, do not publish exploit details in a public issue. Contact the repository owner privately through the contact methods on the owner's GitHub profile.
