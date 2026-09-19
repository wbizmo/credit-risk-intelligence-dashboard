import { AggregationTemporality, InMemoryMetricExporter } from "@opentelemetry/sdk-metrics";
import { describe, expect, it } from "vitest";
import { buildApp } from "./app";
import type { AppConfig } from "./config";
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

function metricNames(exporter: InMemoryMetricExporter): string[] {
  return exporter.getMetrics().flatMap((resource) =>
    resource.scopeMetrics.flatMap((scope) => scope.metrics.map((metric) => metric.descriptor.name)),
  );
}

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

  it("keeps request IDs, borrower values, IPs and credentials out of integrated HTTP metrics", async () => {
    const apiKey = "crix-telemetry-secret-0123456789abcdef";
    const config: AppConfig = {
      host: "127.0.0.1",
      port: 0,
      logLevel: "silent",
      rateLimitMax: 1000,
      authMode: "required",
      apiKeys: [apiKey],
      trustProxyHops: 0,
      telemetryEnabled: false,
      otelExportIntervalMs: 60_000,
      corsOrigins: [],
      environment: "test",
    };
    const { telemetry, exporter } = createInMemoryTelemetry();
    const app = await buildApp(config, { telemetry });

    const denied = await app.inject({
      method: "POST",
      url: "/api/v3/risk/score",
      headers: {
        "x-api-key": "unknown-sensitive-credential-value",
        "x-forwarded-for": "203.0.113.99",
      },
      payload: { ...result, applicationId: "borrower-application-secret" },
    });
    expect(denied.statusCode).toBe(401);

    const allowed = await app.inject({
      method: "POST",
      url: "/api/v3/risk/score",
      headers: {
        "x-api-key": apiKey,
        "x-forwarded-for": "203.0.113.99",
      },
      payload: {
        applicationId: "borrower-application-secret",
        annualIncome: 85_000,
        debtToIncome: 0.28,
        creditScore: 720,
        creditUtilization: 0.3,
        delinquencies24m: 0,
        inquiries6m: 1,
        oldestTradeMonths: 96,
        openAccounts: 7,
        loanAmount: 24_000,
        termMonths: 36,
        employmentYears: 5,
        cashBufferMonths: 3,
        onTimePaymentRate: 0.98,
        incomeStability: 0.82,
        recentCreditGrowth: 0.08,
      },
    });
    expect(allowed.statusCode).toBe(200);

    expect(await telemetry.forceFlush()).toBe(true);
    const names = metricNames(exporter);
    expect(names).toContain("crix.auth.rejections");
    expect(names).toContain("crix.risk.decisions");

    const attributes = allAttributes(exporter);
    const values = attributes.flatMap((attributeSet) => Object.values(attributeSet));
    expect(values).not.toContain("borrower-application-secret");
    expect(values).not.toContain(apiKey);
    expect(values).not.toContain("unknown-sensitive-credential-value");
    expect(values).not.toContain("203.0.113.99");
    expect(values).not.toContain(85_000);
    expect(values).not.toContain(720);
    expect(values).not.toContain(24_000);

    await app.close();

    const publicConfig: AppConfig = {
      ...config,
      authMode: "public-demo",
      apiKeys: [],
    };
    const rateLimitedApp = await buildApp(publicConfig, {
      telemetry,
      routeRateLimitScale: 0.02,
    });
    const first = await rateLimitedApp.inject({
      method: "POST",
      url: "/api/v3/risk/score",
      payload: {
        annualIncome: 85_000,
        debtToIncome: 0.28,
        creditScore: 720,
        creditUtilization: 0.3,
        delinquencies24m: 0,
        inquiries6m: 1,
        oldestTradeMonths: 96,
        openAccounts: 7,
        loanAmount: 24_000,
        termMonths: 36,
        employmentYears: 5,
        cashBufferMonths: 3,
        onTimePaymentRate: 0.98,
        incomeStability: 0.82,
        recentCreditGrowth: 0.08,
      },
    });
    const second = await rateLimitedApp.inject({
      method: "POST",
      url: "/api/v3/risk/score",
      payload: {
        annualIncome: 85_000,
        debtToIncome: 0.28,
        creditScore: 720,
        creditUtilization: 0.3,
        delinquencies24m: 0,
        inquiries6m: 1,
        oldestTradeMonths: 96,
        openAccounts: 7,
        loanAmount: 24_000,
        termMonths: 36,
        employmentYears: 5,
        cashBufferMonths: 3,
        onTimePaymentRate: 0.98,
        incomeStability: 0.82,
        recentCreditGrowth: 0.08,
      },
    });
    expect(first.statusCode).toBe(200);
    expect(second.statusCode).toBe(429);
    await rateLimitedApp.close();

    expect(await telemetry.forceFlush()).toBe(true);
    expect(metricNames(exporter)).toContain("crix.rate_limit.rejections");
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
