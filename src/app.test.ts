import { describe, expect, it } from "vitest";
import { buildApp } from "./app";
import type { AppConfig } from "./config";

const application = {
  applicationId: "demo-001",
  annualIncome: 85_000,
  debtToIncome: 0.28,
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
};

const config = (apiKey?: string): AppConfig => ({
  host: "127.0.0.1",
  port: 3000,
  logLevel: "silent",
  rateLimitMax: 1000,
  ...(apiKey ? { apiKey } : {}),
  corsOrigins: [],
  environment: "test",
});

describe("CRIX HTTP API", () => {
  it("serves health and readiness probes", async () => {
    const app = await buildApp(config());
    const health = await app.inject({ method: "GET", url: "/health" });
    const ready = await app.inject({ method: "GET", url: "/ready" });
    expect(health.statusCode).toBe(200);
    expect(health.json().status).toBe("ok");
    expect(ready.statusCode).toBe(200);
    expect(ready.json().status).toBe("ready");
    await app.close();
  });

  it("publishes a valid OpenAPI document and Swagger UI", async () => {
    const app = await buildApp(config());
    const response = await app.inject({ method: "GET", url: "/openapi.json" });
    expect(response.statusCode).toBe(200);
    const spec = response.json();
    expect(spec.openapi).toBe("3.0.3");
    expect(spec.info.version).toBe("2.5.0");
    expect(spec.paths["/api/v2/risk/score"]).toBeTruthy();
    expect(spec.paths["/api/v2/risk/stress"]).toBeTruthy();
    expect(spec.paths["/api/v2/risk/batch"]).toBeTruthy();

    const docs = await app.inject({ method: "GET", url: "/docs/" });
    expect(docs.statusCode).toBe(200);
    expect(docs.headers["content-type"]).toContain("text/html");
    expect(docs.body.toLowerCase()).toContain("swagger ui");
    await app.close();
  });

  it("scores a valid application", async () => {
    const app = await buildApp(config());
    const response = await app.inject({ method: "POST", url: "/api/v2/risk/score", payload: application });
    expect(response.statusCode).toBe(200);
    const body = response.json();
    expect(body.apiVersion).toBe("2.5.0");
    expect(body.requestId).toBeTruthy();
    expect(body.result.pd).toBeGreaterThan(0);
    expect(body.result.modelVersion).toContain("CRIX-MonoBoost");
    await app.close();
  });

  it("rejects unknown fields and malformed applications", async () => {
    const app = await buildApp(config());
    const response = await app.inject({
      method: "POST",
      url: "/api/v2/risk/score",
      payload: { ...application, annualIncome: -1, unexpected: true },
    });
    expect(response.statusCode).toBe(400);
    expect(response.json().error).toBe("VALIDATION_ERROR");
    await app.close();
  });

  it("enforces optional API-key authentication with the same public health probes", async () => {
    const app = await buildApp(config("top-secret-test-key"));
    const health = await app.inject({ method: "GET", url: "/health" });
    const denied = await app.inject({ method: "POST", url: "/api/v2/risk/score", payload: application });
    const allowed = await app.inject({
      method: "POST",
      url: "/api/v2/risk/score",
      headers: { "x-api-key": "top-secret-test-key" },
      payload: application,
    });
    expect(health.statusCode).toBe(200);
    expect(denied.statusCode).toBe(401);
    expect(allowed.statusCode).toBe(200);
    await app.close();
  });

  it("enforces the synchronous batch safety bound", async () => {
    const app = await buildApp(config());
    const response = await app.inject({
      method: "POST",
      url: "/api/v2/risk/batch",
      payload: { applications: Array.from({ length: 51 }, () => application) },
    });
    expect(response.statusCode).toBe(400);
    await app.close();
  });
});
