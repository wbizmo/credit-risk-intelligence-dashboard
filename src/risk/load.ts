import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { monitorEventLoopDelay, performance } from "node:perf_hooks";
import { buildApp } from "../app";
import type { AppConfig } from "../config";
import { summarizeDurations } from "./benchmark-utils";
import type { ApplicationInput, StressSeverity } from "./types";

type HttpMethod = "GET" | "POST";

interface StressFixture {
  application: ApplicationInput;
  severity: StressSeverity;
}

interface ScenarioDefinition {
  name: string;
  method: HttpMethod;
  path: string;
  concurrency: number;
  requests: number;
  expectedStatuses: readonly number[];
  body?: unknown;
  apiKey?: string;
  routeRateLimitScale?: number;
  requireStatus?: number;
}

interface ScenarioResult {
  name: string;
  method: HttpMethod;
  path: string;
  concurrency: number;
  requests: number;
  elapsedMs: number;
  requestsPerSecond: number;
  opsPerSecond: number;
  meanMs: number;
  p50Ms: number;
  p90Ms: number;
  p95Ms: number;
  p99Ms: number;
  statusCounts: Record<string, number>;
  non2xxCount: number;
  server5xxCount: number;
  timeoutCount: number;
  unexpectedStatusCount: number;
  memory: {
    heapBeforeBytes: number;
    heapAfterBytes: number;
    heapPeakBytes: number;
    rssBeforeBytes: number;
    rssAfterBytes: number;
    rssPeakBytes: number;
  };
  eventLoop: {
    delayMeanMs: number;
    delayP50Ms: number;
    delayP95Ms: number;
    delayP99Ms: number;
    utilization: number;
  };
  cpu: {
    userMs: number;
    systemMs: number;
  };
}

const quick = process.argv.includes("--quick");
const fixtureRoot = resolve(process.cwd(), "fixtures", "load", "v3");
const score = JSON.parse(readFileSync(resolve(fixtureRoot, "score.json"), "utf8")) as ApplicationInput;
const stress = JSON.parse(readFileSync(resolve(fixtureRoot, "stress.json"), "utf8")) as StressFixture;
const invalidScore = JSON.parse(readFileSync(resolve(fixtureRoot, "invalid-score.json"), "utf8")) as unknown;

const baseConfig = (apiKey?: string): AppConfig => ({
  host: "127.0.0.1",
  port: 0,
  logLevel: "silent",
  rateLimitMax: 100_000,
  ...(apiKey ? { apiKey } : {}),
  corsOrigins: [],
  environment: "test",
});

const finiteMs = (nanoseconds: number): number => Number.isFinite(nanoseconds) ? nanoseconds / 1_000_000 : 0;

async function runScenario(definition: ScenarioDefinition): Promise<ScenarioResult> {
  const app = await buildApp(baseConfig(definition.apiKey), {
    routeRateLimitScale: definition.routeRateLimitScale ?? 100,
  });
  const origin = await app.listen({ host: "127.0.0.1", port: 0 });
  const histogram = monitorEventLoopDelay({ resolution: 10 });
  const eluBefore = performance.eventLoopUtilization();
  const cpuBefore = process.cpuUsage();
  const memoryBefore = process.memoryUsage();
  let heapPeakBytes = memoryBefore.heapUsed;
  let rssPeakBytes = memoryBefore.rss;
  const latencies = new Array<number>(definition.requests);
  const statusCounts: Record<string, number> = {};
  let cursor = 0;
  let timeoutCount = 0;

  histogram.enable();
  const started = performance.now();

  const worker = async (): Promise<void> => {
    while (true) {
      const index = cursor;
      cursor += 1;
      if (index >= definition.requests) return;

      const requestStarted = performance.now();
      try {
        const headers: Record<string, string> = {};
        if (definition.body !== undefined) headers["content-type"] = "application/json";
        if (definition.apiKey) headers["x-api-key"] = definition.apiKey;

        const response = await fetch(`${origin}${definition.path}`, {
          method: definition.method,
          headers,
          ...(definition.body !== undefined ? { body: JSON.stringify(definition.body) } : {}),
          signal: AbortSignal.timeout(10_000),
        });
        await response.arrayBuffer();
        statusCounts[String(response.status)] = (statusCounts[String(response.status)] ?? 0) + 1;
      } catch (error) {
        timeoutCount += 1;
        statusCounts.timeout = (statusCounts.timeout ?? 0) + 1;
        if (error instanceof Error && error.name !== "TimeoutError" && error.name !== "AbortError") {
          statusCounts.clientError = (statusCounts.clientError ?? 0) + 1;
        }
      } finally {
        latencies[index] = performance.now() - requestStarted;
        const memory = process.memoryUsage();
        heapPeakBytes = Math.max(heapPeakBytes, memory.heapUsed);
        rssPeakBytes = Math.max(rssPeakBytes, memory.rss);
      }
    }
  };

  try {
    await Promise.all(Array.from({ length: Math.min(definition.concurrency, definition.requests) }, () => worker()));
  } finally {
    histogram.disable();
  }

  const elapsedMs = performance.now() - started;
  const memoryAfter = process.memoryUsage();
  const cpu = process.cpuUsage(cpuBefore);
  const elu = performance.eventLoopUtilization(eluBefore);
  await app.close();

  const summary = summarizeDurations(latencies, elapsedMs);
  const non2xxCount = Object.entries(statusCounts)
    .filter(([status]) => /^\d+$/.test(status) && (Number(status) < 200 || Number(status) >= 300))
    .reduce((sum, [, count]) => sum + count, 0);
  const server5xxCount = Object.entries(statusCounts)
    .filter(([status]) => /^5\d\d$/.test(status))
    .reduce((sum, [, count]) => sum + count, 0);
  const unexpectedStatusCount = Object.entries(statusCounts)
    .filter(([status]) => /^\d+$/.test(status) && !definition.expectedStatuses.includes(Number(status)))
    .reduce((sum, [, count]) => sum + count, 0);

  if (server5xxCount > 0) throw new Error(`${definition.name} produced unexplained 5xx responses`);
  if (timeoutCount > 0) throw new Error(`${definition.name} produced request timeouts`);
  if (unexpectedStatusCount > 0) throw new Error(`${definition.name} produced unexpected HTTP status codes`);
  if (definition.requireStatus !== undefined && (statusCounts[String(definition.requireStatus)] ?? 0) === 0) {
    throw new Error(`${definition.name} did not produce required status ${definition.requireStatus}`);
  }

  return {
    name: definition.name,
    method: definition.method,
    path: definition.path,
    concurrency: definition.concurrency,
    requests: definition.requests,
    elapsedMs,
    requestsPerSecond: definition.requests / (elapsedMs / 1_000),
    ...summary,
    statusCounts,
    non2xxCount,
    server5xxCount,
    timeoutCount,
    unexpectedStatusCount,
    memory: {
      heapBeforeBytes: memoryBefore.heapUsed,
      heapAfterBytes: memoryAfter.heapUsed,
      heapPeakBytes,
      rssBeforeBytes: memoryBefore.rss,
      rssAfterBytes: memoryAfter.rss,
      rssPeakBytes,
    },
    eventLoop: {
      delayMeanMs: finiteMs(histogram.mean),
      delayP50Ms: finiteMs(histogram.percentile(50)),
      delayP95Ms: finiteMs(histogram.percentile(95)),
      delayP99Ms: finiteMs(histogram.percentile(99)),
      utilization: elu.utilization,
    },
    cpu: {
      userMs: cpu.user / 1_000,
      systemMs: cpu.system / 1_000,
    },
  };
}

function batchPayload(size: number): { applications: ApplicationInput[] } {
  return {
    applications: Array.from({ length: size }, (_, index) => ({
      ...score,
      applicationId: `load-v3-batch-${size}-${String(index + 1).padStart(2, "0")}`,
    })),
  };
}

const profiles = [
  { label: "low", concurrency: 1 },
  { label: "medium", concurrency: 8 },
  { label: "high", concurrency: 32 },
] as const;

const endpointDefinitions = [
  { name: "health", method: "GET" as const, path: "/health", body: undefined, requests: quick ? 80 : 500 },
  { name: "score", method: "POST" as const, path: "/api/v3/risk/score", body: score, requests: quick ? 60 : 500 },
  { name: "stress", method: "POST" as const, path: "/api/v3/risk/stress", body: stress, requests: quick ? 40 : 300 },
  { name: "batch-10", method: "POST" as const, path: "/api/v3/risk/batch", body: batchPayload(10), requests: quick ? 20 : 150 },
] as const;

const results: ScenarioResult[] = [];

for (const endpoint of endpointDefinitions) {
  for (const profile of profiles) {
    results.push(await runScenario({
      name: `${endpoint.name}-${profile.label}`,
      method: endpoint.method,
      path: endpoint.path,
      concurrency: profile.concurrency,
      requests: endpoint.requests,
      expectedStatuses: [200],
      ...(endpoint.body !== undefined ? { body: endpoint.body } : {}),
    }));
  }
}

for (const size of [1, 10, 50]) {
  results.push(await runScenario({
    name: `batch-size-${size}`,
    method: "POST",
    path: "/api/v3/risk/batch",
    concurrency: quick ? 4 : 8,
    requests: quick ? 20 : 80,
    expectedStatuses: [200],
    body: batchPayload(size),
  }));
}

results.push(await runScenario({
  name: "invalid-payload-bounded-validation",
  method: "POST",
  path: "/api/v3/risk/score",
  concurrency: 16,
  requests: quick ? 40 : 200,
  expectedStatuses: [400],
  body: invalidScore,
}));

results.push(await runScenario({
  name: "production-rate-limit-behavior",
  method: "POST",
  path: "/api/v3/risk/score",
  concurrency: 32,
  requests: 80,
  expectedStatuses: [200, 429],
  requireStatus: 429,
  routeRateLimitScale: 1,
  body: score,
}));

results.push(await runScenario({
  name: "sustained-score-memory-profile",
  method: "POST",
  path: "/api/v3/risk/score",
  concurrency: 16,
  requests: quick ? 500 : 5_000,
  expectedStatuses: [200],
  body: score,
}));

const benchmarkApiKey = process.env.CRIX_BENCH_API_KEY?.trim();
let authenticatedProfile: { status: "executed"; result: ScenarioResult } | { status: "skipped"; reason: string };

if (benchmarkApiKey) {
  authenticatedProfile = {
    status: "executed",
    result: await runScenario({
      name: "authenticated-score",
      method: "POST",
      path: "/api/v3/risk/score",
      concurrency: 8,
      requests: quick ? 40 : 200,
      expectedStatuses: [200],
      body: score,
      apiKey: benchmarkApiKey,
    }),
  };
} else {
  authenticatedProfile = {
    status: "skipped",
    reason: "Set CRIX_BENCH_API_KEY locally/CI to exercise the authenticated load profile; no credential is committed.",
  };
}

console.log(JSON.stringify({
  node: process.version,
  mode: quick ? "quick" : "full",
  fixtureVersion: "v3",
  environment: "local-loopback",
  disclaimer: "Engineering evidence only. These loopback results are not an internet, cloud-provider, cold-start, or end-user latency SLO.",
  serverSemantics: {
    synchronousScoring: true,
    publicBatchMaximum: 50,
    throughputProfilesUseInternalRateLimitScale: 100,
    productionRateLimitProfileUsesScale: 1,
  },
  results,
  authenticatedProfile,
}, null, 2));
