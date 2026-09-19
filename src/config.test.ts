import { describe, expect, it } from "vitest";
import { loadConfig, MAX_ACTIVE_API_KEYS, MIN_API_KEY_LENGTH } from "./config";

const strong = (suffix: string) => `crix-test-key-${suffix}-0123456789abcdef`;

describe("CRIX runtime configuration", () => {
  it("defaults to an explicit public-demo posture with proxy trust and telemetry off", () => {
    const config = loadConfig({});
    expect(config.authMode).toBe("public-demo");
    expect(config.apiKeys).toEqual([]);
    expect(config.trustProxyHops).toBe(0);
    expect(config.telemetryEnabled).toBe(false);
  });

  it("requires an explicit auth mode whenever credentials are configured", () => {
    expect(() => loadConfig({ CRIX_API_KEY: strong("legacy") })).toThrow(/CRIX_AUTH_MODE must be explicit/);
  });

  it("fails closed when required mode has no key", () => {
    expect(() => loadConfig({ CRIX_AUTH_MODE: "required" })).toThrow(/requires at least one API key/);
  });

  it("rejects weak, duplicate and excessive rotation sets", () => {
    expect(() => loadConfig({
      CRIX_AUTH_MODE: "required",
      CRIX_API_KEYS: "tiny",
    })).toThrow(new RegExp(String(MIN_API_KEY_LENGTH)));

    const duplicate = strong("same");
    expect(() => loadConfig({
      CRIX_AUTH_MODE: "required",
      CRIX_API_KEYS: `${duplicate},${duplicate}`,
    })).toThrow(/unique/);

    const keys = Array.from({ length: MAX_ACTIVE_API_KEYS + 1 }, (_, index) => strong(String(index))).join(",");
    expect(() => loadConfig({
      CRIX_AUTH_MODE: "required",
      CRIX_API_KEYS: keys,
    })).toThrow(/At most/);
  });

  it("accepts a bounded current+next rotation set and explicit one-hop proxy trust", () => {
    const config = loadConfig({
      CRIX_AUTH_MODE: "required",
      CRIX_API_KEYS: `${strong("current")},${strong("next")}`,
      CRIX_TRUST_PROXY_HOPS: "1",
      CRIX_TELEMETRY_ENABLED: "true",
      CRIX_OTEL_METRICS_ENDPOINT: "http://collector:4318/v1/metrics",
      CRIX_OTEL_EXPORT_INTERVAL_MS: "15000",
    });
    expect(config.apiKeys).toHaveLength(2);
    expect(config.trustProxyHops).toBe(1);
    expect(config.telemetryEnabled).toBe(true);
    expect(config.otelMetricsEndpoint).toBe("http://collector:4318/v1/metrics");
    expect(config.otelExportIntervalMs).toBe(15_000);
  });

  it("rejects ambiguous public-demo credentials and invalid configuration values", () => {
    expect(() => loadConfig({
      CRIX_AUTH_MODE: "public-demo",
      CRIX_API_KEYS: strong("unused"),
    })).toThrow(/must not be configured/);
    expect(() => loadConfig({ CRIX_AUTH_MODE: "wat" })).toThrow(/public-demo or required/);
    expect(() => loadConfig({ CRIX_TELEMETRY_ENABLED: "yes" })).toThrow(/exactly true or false/);
    expect(() => loadConfig({ CRIX_OTEL_METRICS_ENDPOINT: "file:///tmp/metrics" })).toThrow(/http or https/);
  });
});
