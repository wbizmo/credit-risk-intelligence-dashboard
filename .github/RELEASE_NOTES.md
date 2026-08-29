# CRIX v2.5.0

CRIX is now a backend-only credit-risk intelligence API.

- Removed the dashboard and all frontend/runtime UI dependencies.
- Rebuilt the service on Fastify + TypeScript.
- Added `/health` and `/ready` operational probes.
- Added full OpenAPI JSON and interactive Swagger documentation at `/docs`.
- Added single scoring, stress-testing, bounded batch scoring, and model-metadata endpoints.
- Hardened validation, rate limits, request sizing, error handling, CORS, security headers, log redaction, and optional API-key authentication.
- Added model-integrity checks, finite-input protection, bounded tree traversal, OOD flags, disagreement flags, and explicit policy/model versioning.
- Removed borrower-name input because the algorithm does not require identity.
- Expanded engine and HTTP integration tests and CI verification.
- Reworked the README with clone/run/test instructions and the Render free-tier cold-start caveat.
- Added automatic stale-branch cleanup so the repository returns to `main` only after release.

Thanks for checking it out.
