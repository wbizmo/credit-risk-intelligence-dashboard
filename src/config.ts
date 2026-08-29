export interface AppConfig {
  host: string;
  port: number;
  logLevel: string;
  rateLimitMax: number;
  apiKey?: string;
  corsOrigins: string[];
  environment: string;
}

function boundedInteger(value: string | undefined, fallback: number, min: number, max: number): number {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed >= min && parsed <= max ? parsed : fallback;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  const apiKey = env.CRIX_API_KEY?.trim();
  const corsOrigins = (env.CORS_ORIGIN ?? "")
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);

  return {
    host: env.HOST?.trim() || "0.0.0.0",
    port: boundedInteger(env.PORT, 3000, 1, 65535),
    logLevel: env.LOG_LEVEL?.trim() || "info",
    rateLimitMax: boundedInteger(env.RATE_LIMIT_MAX, 120, 10, 10_000),
    ...(apiKey ? { apiKey } : {}),
    corsOrigins,
    environment: env.NODE_ENV?.trim() || "development",
  };
}
