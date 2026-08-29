# CRIX v2.5 Architecture

CRIX is intentionally a **stateless backend API**. The model is the core product; HTTP is the delivery mechanism.

## Runtime boundaries

```text
Client
  |
  v
Fastify API
  |-- health/readiness
  |-- OpenAPI/Swagger
  |-- schema validation
  |-- rate limiting
  |-- optional API-key auth
  |-- request/error controls
  |
  v
Risk Engine
  |-- champion PD
  |-- challenger PD
  |-- disagreement/confidence
  |-- OOD detection
  |-- LGD/EAD/expected loss
  |-- score/grade
  |-- local reason codes
  |-- policy decision
  |-- stress scenarios
  |
  v
Versioned model artifact
```

There is no database, Redis instance, queue, browser application, or external inference dependency in the live system.

## Offline model-development boundary

Python is an offline build-time/research concern only. `model/train.py` generates deterministic synthetic performance data, trains the constrained champion, calibrates it on a separate split, calculates held-out diagnostics, and exports a compact JSON artifact.

The live TypeScript service evaluates that artifact directly. This means runtime availability is not coupled to a Python process or ML service.

## Model / policy separation

The champion and challenger estimate risk. They do not decide approval by themselves.

The policy layer consumes PD plus selected application and confidence signals and returns `APPROVE`, `REVIEW`, or `DECLINE`. Keeping policy separate allows model changes, risk-appetite changes, and pricing changes to be governed independently.

## Request lifecycle

1. Fastify assigns an opaque UUID request ID.
2. Global request/body/time limits apply.
3. Optional API-key authentication runs for `/api/v2/*`.
4. AJV validates the body against strict JSON Schema.
5. The engine performs its own finite-number checks as a second trust boundary.
6. The champion tree ensemble produces a raw margin.
7. Held-out calibration converts margin to PD.
8. The transparent challenger scores the same vector.
9. Disagreement and OOD checks reduce confidence and emit flags.
10. LGD, EAD, expected loss, score, grade, pricing and reason codes are derived.
11. Independent policy decides approve/review/decline.
12. The response includes request, model and policy versions for traceability.

## Availability and readiness

`/health` is a liveness probe. It answers without performing model work.

`/ready` reflects a startup model-integrity check. The server verifies artifact structure and executes a sentinel score before advertising readiness. With no external stateful dependencies, the model artifact is the primary runtime readiness dependency.

## Security model

The public demo defaults to no API key so reviewers can exercise Swagger immediately. A deployment can set `CRIX_API_KEY` to protect `/api/v2/*`; health/readiness/docs remain available for operations and discovery.

Controls include request-size limits, rate limits, strict schemas, CORS allow-listing, Helmet, sensitive-header log redaction, constant-time API-key comparison, sanitized errors, batch bounds, and no application persistence.

## Scaling path

The synchronous API is deliberately bounded. A real high-volume deployment would likely add:

- gateway-level authentication and quotas;
- async bulk scoring behind a queue;
- immutable request/result audit storage;
- model registry and signed artifacts;
- feature service with point-in-time correctness;
- independent policy configuration/versioning;
- telemetry for drift, calibration, decision rates and latency;
- canary/champion-challenger routing;
- formal adverse-action reason governance.

Those concerns are not simulated with unnecessary infrastructure in this repository. The v2.5 runtime stays small enough to inspect and reproduce while making its production extension points explicit.
