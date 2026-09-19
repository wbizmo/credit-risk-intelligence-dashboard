# CRIX v3.2 Performance Evidence

CRIX performance measurements are engineering evidence, not an internet or cloud-provider SLA.

## Runtime benchmark

Run on Node.js 22:

```bash
npm run benchmark:risk
```

The command warms each operation and emits machine-readable JSON with mean, p50, p90, p95 and p99 latency, throughput, heap/RSS deltas, model/runtime metadata, compiled-vs-reference evidence, feature/tree density and deterministic complexity counters.

Wall-clock timing is intentionally evidence-only. Correctness gates use deterministic operation-count invariants so noisy CI timing does not create flaky failures.

## HTTP load benchmark

Quick verification profile:

```bash
npm run benchmark:http:quick
```

Full local profile:

```bash
npm run benchmark:http
```

The harness starts the real Fastify application on a loopback ephemeral port and uses the committed `fixtures/load/v3` payloads. It covers `/health`, `/api/v3/risk/score`, `/api/v3/risk/stress`, and `/api/v3/risk/batch` at low, medium and high concurrency.

Evidence includes requests/second, p50/p95/p99 latency, status/error/timeout counts, heap/RSS before/after/peak, CPU time, event-loop delay and event-loop utilization. Additional profiles exercise invalid payload validation, the production route-rate-limit behavior, 1/10/50-item batch scaling, and a sustained score run for memory-growth evidence.

Throughput scenarios use an internal test-only route-limit multiplier so they measure request processing rather than intentional 429 responses. A separate scenario uses the production multiplier and requires observed 429s. Production configuration and public route limits are unchanged.

## Authenticated profile

Set a local or CI-only key to exercise the authenticated path:

```bash
CRIX_BENCH_API_KEY="local-benchmark-only" npm run benchmark:http
```

No benchmark credential is committed. When the variable is absent, the authenticated profile is explicitly reported as skipped.

## Interpretation

A result such as sub-10ms loopback latency is valid only for the documented machine/runtime/run. It must not be described as Render cold-start latency, internet latency, or an end-user SLO.

The synchronous batch limit remains 50. The harness does not use `Promise.all` inside scoring or increase the API batch bound; concurrency exists only in the external HTTP clients driving the server.
