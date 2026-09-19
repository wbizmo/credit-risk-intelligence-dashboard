export type AuthMode = "public-demo" | "required";

export interface AppConfig {
  host: string;
  port: number;
  logLevel: string;
  rateLimitMax: number;
  authMode: AuthMode;
  apiKeys: readonly string[];
  trustProxyHops: number;
  telemetryEnabled: boolean;
  otelMetricsEndpoint?: string;
  otelExportIntervalMs: number;
  corsOrigins: string[];
  environment: string;
}

export const MAX_ACTIVE_API_KEYS = 2;
export const MIN_API_KEY_LENGTH = 24;
const MAX_TRUST_PROXY_HOPS = 4;

function boundedInteger(value: string | undefined, fallback: number, min: number, max: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed >= min && parsed <= max ? parsed : fallback;
}

function strictBoolean(value: string | undefined, fallback = false): boolean {
  if (value === undefined || value.trim() === "") return fallback;
  if (value === "true") return true;
  if (value === "false") return false;
  throw new Error("Boolean configuration values must be exactly true or false.");
}

function authMode(value: string | undefined): AuthMode {
  const normalized = value?.trim() || "public-demo";
  if (normalized === "public-demo" || normalized === "required") return normalized;
  throw new Error("CRIX_AUTH_MODE must be public-demo or required.");
}

function activeKeys(env: NodeJS.ProcessEnv): readonly string[] {
  if (env.CRIX_API_KEYS && env.CRIX_API_KEY) {
    throw new Error("Configure only CRIX_API_KEYS or legacy CRIX_API_KEY, not both.");
  }
  const raw = env.CRIX_API_KEYS ?? env.CRIX_API_KEY ?? "";
  return raw.split(",").map((value) => value.trim()).filter(Boolean);
}

function validateKeys(mode: AuthMode, keys: readonly string[], authModeConfigured: boolean): void {
  if (keys.length > 0 && !authModeConfigured) {
    throw new Error("CRIX_AUTH_MODE must be explicit whenever API keys are configured.");
  }
  if (mode === "public-demo" && keys.length > 0) {
    throw new Error("API keys must not be configured in public-demo mode.");
  }
  if (mode === "required" && keys.length === 0) {
    throw new Error("CRIX_AUTH_MODE=required requires at least one API key.");
  }
  if (keys.length > MAX_ACTIVE_API_KEYS) {
    throw new Error(`At most ${MAX_ACTIVE_API_KEYS} active API keys are allowed.`);
  }
  if (new Set(keys).size !== keys.length) {
    throw new Error("Active API keys must be unique.");
  }
  for (const key of keys) {
    if (key.length < MIN_API_KEY_LENGTH || new Set(key).size < 4) {
      throw new Error(`Required-mode API keys must be at least ${MIN_API_KEY_LENGTH} characters and non-trivial.`);
    }
  }
}

export function validateRuntimeSecurityConfig(config: Pick<AppConfig, "authMode" | "apiKeys" | "trustProxyHops">): void {
  validateKeys(config.authMode, config.apiKeys, true);
  if (!Number.isInteger(config.trustProxyHops) || config.trustProxyHops < 0 || config.trustProxyHops > MAX_TRUST_PROXY_HOPS) {
    throw new Error(`trustProxyHops must be an integer from 0 to ${MAX_TRUST_PROXY_HOPS}.`);
  }
}

function optionalHttpUrl(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  if (!trimmed) return undefined;
  const parsed = new URL(trimmed);
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("CRIX_OTEL_METRICS_ENDPOINT must use http or https.");
  }
  return parsed.toString();
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  const mode = authMode(env.CRIX_AUTH_MODE);
  const apiKeys = activeKeys(env);
  validateKeys(mode, apiKeys, Boolean(env.CRIX_AUTH_MODE?.trim()));

  const corsOrigins = (env.CORS_ORIGIN ?? "")
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);

  const endpoint = optionalHttpUrl(env.CRIX_OTEL_METRICS_ENDPOINT);

  return {
    host: env.HOST?.trim() || "0.0.0.0",
    port: boundedInteger(env.PORT, 3000, 1, 65535),
    logLevel: env.LOG_LEVEL?.trim() || "info",
    rateLimitMax: boundedInteger(env.RATE_LIMIT_MAX, 120, 10, 10_000),
    authMode: mode,
    apiKeys,
    trustProxyHops: boundedInteger(env.CRIX_TRUST_PROXY_HOPS, 0, 0, MAX_TRUST_PROXY_HOPS),
    telemetryEnabled: strictBoolean(env.CRIX_TELEMETRY_ENABLED, false),
    ...(endpoint ? { otelMetricsEndpoint: endpoint } : {}),
    otelExportIntervalMs: boundedInteger(env.CRIX_OTEL_EXPORT_INTERVAL_MS, 30_000, 1_000, 300_000),
    corsOrigins,
    environment: env.NODE_ENV?.trim() || "development",
  };
}
