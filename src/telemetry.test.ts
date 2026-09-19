import { AggregationTemporality, InMemoryMetricExporter } from "@opentelemetry/sdk-metrics";
import { describe, expect, it } from "vitest";
import { createInMemoryTelemetry, createTelemetry } from "./telemetry";
import type { RiskResult } from "./risk/types";

const result: RiskResult = {
  pd: 0.12,
  pdHorizon: "final-loan-resolution",
  challengerPd: 0.15,
  disagreement: 0.03,
  lgd: 0.4,
  ead: 24_000,
  expectedLoss: 1_152,
  expectedLossRate: 0.048,
  score: 700,
  grade: "B1",
  decision: "APPROVE",
  confidence: 0.9,
  apr: 12,
  reasons: [],
  policyReasons: [],
  counterfactuals: [],
  modelVersion: "CRIX-MonoBoost 2.0.0",
  policyVersion: "CRIX-Policy 3.0",
  outOfDistribution: ["creditScore"],
  flags: [],
};

function allAttributes(exporter: InMemoryMetricExporter): Record<string, unknown>[] {
  return exporter.getMetrics().flatMap((resource) =>
    resource.scopeMetrics.flatMap((scope) =>
      scope.metrics.flatMap((metric) =>
        metric.dataPoints.map((point) => ({ ...point.attributes })),
      ),
    ),
  );
}

describe("OpenTelemetry metrics", () => {
  it("can be completely disabled", async () => {
    const telemetry = createTelemetry({ enabled: false, exportIntervalMs: 60_000 });
    expect(telemetry.enabled).toBe(false);
    telemetry.recordScore(result, 1);
    expect(await telemetry.forceFlush()).toBe(true);
  });

  it("emits bounded operational labels and no borrower/request/credential fields", async () => {
    const { telemetry, exporter } = createInMemoryTelemetry();
    telemetry.setReadiness(true, result.modelVersion);
    telemetry.recordHttp("POST", "/api/v3/risk/score", 200, 4.2);
    telemetry.recordAuthRejection("/api/v3/risk/score");
    telemetry.recordRateLimitRejection("/api/v3/risk/batch");
    telemetry.recordScore(result, 2.1);
    telemetry.recordStress({ ...result, decision: "REVIEW", flags: ["LOW_CONFIDENCE"] }, 4.5);
    telemetry.recordBatch([result, { ...result, decision: "DECLINE" }], 2, 5.5);

    expect(await telemetry.forceFlush()).toBe(true);
    const attributes = allAttributes(exporter);
    expect(attributes.length).toBeGreaterThan(0);

    const forbiddenKeys = new Set([
      "applicationId",
      "requestId",
      "ip",
      "annualIncome",
      "creditScore",
      "pd",
      "loanAmount",
      "apiKey",
      "headers",
      "exception",
      "error_message",
    ]);
    for (const attributeSet of attributes) {
      for (const key of Object.keys(attributeSet)) expect(forbiddenKeys.has(key)).toBe(false);
    }

    const serialized = JSON.stringify(attributes);
    expect(serialized).toContain("CRIX-MonoBoost 2.0.0");
    expect(serialized).toContain("CRIX-Policy 3.0");
    expect(serialized).toContain("creditScore");
    expect(serialized).not.toContain("demo-001");
    expect(serialized).not.toContain("super-secret");
    await telemetry.shutdown();
  });

  it("contains exporter failure outside request semantics", async () => {
    const exporter = new InMemoryMetricExporter(AggregationTemporality.CUMULATIVE);
    Object.defineProperty(exporter, "export", {
      value: () => {
        throw new Error("collector unavailable with sensitive-looking-token");
      },
    });
    const telemetry = createTelemetry(
      { enabled: true, exportIntervalMs: 60_000 },
      { exporter },
    );
    telemetry.recordHttp("GET", "/health", 200, 1);
    expect(await telemetry.forceFlush()).toBe(false);
    await telemetry.shutdown();
  });
});
